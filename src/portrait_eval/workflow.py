from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from portrait_eval.corroboration import CorroborationAdapter
from portrait_eval.database import ReportRow
from portrait_eval.pipeline import EvaluationPipeline
from portrait_eval.reporting import ReportPayload, render_report_bundle
from portrait_eval.repository import Repository
from portrait_eval.research import SearchProvider
from portrait_eval.vlm import VisionModelAdapter

EvaluationMode = Literal["quick", "professional"]


def _next_final_version(repo: Repository, project_id: str) -> str:
    finals = [item for item in repo.list_reports(project_id) if item["status"] == "final"]
    return f"1.{len(finals)}"


def finalize_quick_report(repo: Repository, project_id: str) -> dict[str, Any]:
    draft = repo.session.scalar(
        select(ReportRow)
        .where(ReportRow.project_id == project_id, ReportRow.status == "draft")
        .order_by(ReportRow.created_at.desc())
    )
    if draft is None:
        raise ValueError("Draft report not found")
    source = Path(draft.html_path)
    source_json = source.with_suffix(".json")
    if not source_json.is_file():
        raise ValueError("Draft report payload not found")

    payload = ReportPayload.model_validate_json(source_json.read_text(encoding="utf-8"))
    quick_note = (
        "Quick mode generated this report without requiring every review item to be resolved. "
        "Low-confidence findings remain scoped to the submitted captures and should be reviewed "
        "before external publication."
    )
    if quick_note not in payload.limitations:
        payload.limitations.append(quick_note)

    version = _next_final_version(repo, project_id)
    bundle = render_report_bundle(payload, source.parent, version, "final")
    row = repo.save_report(project_id, version, "final", str(bundle["html"]))
    project = repo.get_project(project_id)
    project.status = "REPORT_FINALIZED"
    repo.session.commit()
    return {
        "id": row.id,
        "version": row.version,
        "status": row.status,
        "html_path": row.html_path,
    }


def run_evaluation_workflow(
    session: Session,
    workspace: Path,
    project_id: str,
    mode: EvaluationMode = "quick",
    *,
    primary: VisionModelAdapter | None = None,
    reviewer: VisionModelAdapter | None = None,
    search: SearchProvider | None = None,
    corroborator: CorroborationAdapter | None = None,
) -> dict[str, Any]:
    if mode not in {"quick", "professional"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    result = EvaluationPipeline(
        session,
        workspace,
        primary=primary,
        reviewer=reviewer,
        search=search,
        corroborator=corroborator,
    ).run(project_id)
    result["mode"] = mode

    repo = Repository(session)
    if mode == "quick":
        final = finalize_quick_report(repo, project_id)
        result.update(
            {
                "status": "REPORT_FINALIZED",
                "report_id": final["id"],
                "report_path": final["html_path"],
                "report_version": final["version"],
            }
        )
        return result

    reports = repo.list_reports(project_id)
    if reports:
        latest = reports[-1]
        result["report_id"] = latest["id"]
        result["report_version"] = latest["version"]
    return result
