"""Durable cross-process report allocation and atomic bundle publication."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, OperationalError

from portrait_eval.database import ProjectRow, ReportPublicationRow, ReportRow
from portrait_eval.report_projection import resolve_report_payload
from portrait_eval.reporting import ReportPayload
from portrait_eval.repository import Repository


class PublicationConflict(RuntimeError):
    pass


def reserve_publication(repo: Repository, project_id: str) -> ReportPublicationRow:
    # Read-only projection work may already have opened a snapshot. Allocation uses
    # a fresh short transaction, committed before slow rendering or model activity.
    repo.session.rollback()
    for _ in range(16):
        latest = repo.session.scalar(
            select(func.max(ReportPublicationRow.ordinal)).where(
                ReportPublicationRow.project_id == project_id
            )
        )
        existing = repo.session.scalars(
            select(ReportRow.version).where(
                ReportRow.project_id == project_id, ReportRow.status == "final"
            )
        )
        old_ordinals = [
            int(version[2:])
            for version in existing
            if version.startswith("1.") and version[2:].isdigit()
        ]
        ordinal = max([latest if latest is not None else -1, *old_ordinals]) + 1
        allocation = ReportPublicationRow(project_id=project_id, ordinal=ordinal, status="reserved")
        repo.session.add(allocation)
        try:
            repo.session.commit()
            return allocation
        except (IntegrityError, OperationalError):
            repo.session.rollback()
    raise PublicationConflict("Concurrent report allocation is busy; retry publication")


def publish_final_report(
    repo: Repository,
    project_id: str,
    payload: ReportPayload,
    source: Path,
    renderer: Callable[[ReportPayload, Path, str, str], dict[str, Path]],
) -> dict[str, Any]:
    project = repo.get_project(project_id)
    expected_version, expected_status = project.version, project.status
    allocation = reserve_publication(repo, project_id)
    publication_id = allocation.id
    version = f"1.{allocation.ordinal}"
    created_at = allocation.created_at
    # Release the allocation/read transaction before invoking the slow renderer.
    repo.session.rollback()
    staging = source.parent / ".publication-staging" / publication_id
    destination = source.parent / "publications" / publication_id
    committed = False
    assets = []
    for item in payload.visual_assets:
        asset = dict(item)
        path = asset.get("path") or asset.get("source_path")
        if path and not Path(str(path)).is_absolute():
            asset["path"] = str(source.parent / str(path))
        assets.append(asset)
    payload = payload.model_copy(update={"visual_assets": assets})
    try:
        bundle = renderer(payload, staging, version, "final")
        if not {"html", "json", "csv", "docx"} <= set(bundle) or not all(
            path.is_file() and path.stat().st_size for path in bundle.values()
        ):
            raise RuntimeError("Report renderer did not produce a complete bundle")
        for path in bundle.values():
            path.resolve().relative_to(staging.resolve())
        relative_html = bundle["html"].resolve().relative_to(staging.resolve())
        # Discard any old read snapshot and acquire the writer lock before the
        # final review gate. Review writers share this project lock; SQLite also
        # serializes all writes. Rendering never holds it.
        repo.session.rollback()
        guard = repo.session.execute(
            update(ProjectRow)
            .where(
                ProjectRow.id == project_id,
                ProjectRow.version == expected_version,
                ProjectRow.status.in_([expected_status, "REPORT_FINALIZED"]),
            )
            .values(version=ProjectRow.version)
            .returning(ProjectRow.id)
            .execution_options(synchronize_session=False)
        ).scalar_one_or_none()
        if guard is None:
            raise PublicationConflict("Project pairing or state changed during publication")
        reviews = repo.list_review_items(project_id)
        mode = payload.evaluation_mode or "professional"
        if mode != "quick" and any(item["status"] == "open" for item in reviews):
            raise PublicationConflict("Resolve all review items before finalizing")
        # An older conservative projection stays safe when a rejection is later
        # approved. Newly denied evidence must not remain in the rendered payload.
        try:
            baseline = resolve_report_payload(payload, [], mode).model_dump(mode="json")
            current = resolve_report_payload(payload, reviews, mode).model_dump(mode="json")
        except ValueError as exc:
            raise PublicationConflict(
                f"Publication cannot resolve current evidence: {exc}"
            ) from exc
        baseline.pop("review_summary", None)
        current.pop("review_summary", None)
        # Quick publications retain their explicit original open-review disclosure.
        baseline.pop("limitations", None)
        current.pop("limitations", None)
        if baseline != current:
            raise PublicationConflict(
                "Review decisions changed supporting evidence during publication"
            )
        binding = payload.input_trace.get("run_binding")
        if binding is not None:
            from portrait_eval.run_binding import validate_run_binding

            try:
                validate_run_binding(repo, project_id, binding)
            except ValueError as exc:
                raise PublicationConflict(f"Report source binding changed: {exc}") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        row = ReportRow(
            project_id=project_id,
            version=version,
            status="final",
            html_path=str(destination / relative_html),
            created_at=created_at,
        )
        repo.session.add(row)
        published_allocation = repo.session.get(ReportPublicationRow, publication_id)
        if published_allocation is None:
            raise PublicationConflict("Report publication reservation no longer exists")
        published_allocation.status = "published"
        repo.get_project(project_id).status = "REPORT_FINALIZED"
        # Conventional sessions expire ORM attributes after commit. Freeze the
        # response while reads are still part of the precommit transaction.
        repo.session.flush()
        result = {
            "id": row.id,
            "version": row.version,
            "status": row.status,
            "html_path": row.html_path,
        }
        repo.session.commit()
        committed = True
        return result
    except Exception as exc:
        if committed:
            raise
        repo.session.rollback()
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        failed_allocation = repo.session.get(ReportPublicationRow, publication_id)
        if failed_allocation is not None:
            failed_allocation.status = "failed"
            failed_allocation.error = f"{type(exc).__name__}: {exc}"
            repo.session.commit()
        raise
