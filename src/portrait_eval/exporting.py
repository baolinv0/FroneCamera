from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any

from portrait_eval.repository import Repository


def sanitize_export(value: Any) -> Any:
    """Recursively retain lineage while replacing filesystem locations."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            safe_key = sanitize_export(key)
            if safe_key != key:
                safe_key = "asset:" + hashlib.sha256(str(key).encode()).hexdigest()[:16]
            result[safe_key] = sanitize_export(item)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_export(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        # Preserve HTTP evidence URLs and report-relative asset references.
        if re.fullmatch(r"https?://[^\s]+", value):
            return value
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            return "asset:" + hashlib.sha256(value.encode()).hexdigest()[:16]
        value = re.sub(r"\\\\[^\\\s]+\\[^\s\"'<>;,]+", "[redacted-path]", value)
        value = re.sub(r"[A-Za-z]:[\\/][^\s\"'<>;,]+", "[redacted-path]", value)
        value = re.sub(
            r"(?<![A-Za-z0-9:/])/(?:[^\s/\"'<>;,]+/)*[^\s/\"'<>;,]+", "[redacted-path]", value
        )
        return value
    return value


def _sanitized_pairing(pairing: dict[str, Any]) -> dict[str, Any]:
    return sanitize_export(pairing)


def export_project(repo: Repository, project_id: str, workspace: Path) -> Path:
    project = repo.get_project(project_id)
    export_dir = workspace / "projects" / project_id / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    output = export_dir / "project-export.zip"

    manifest = {
        "project": {
            "id": project.id,
            "name": project.name,
            "status": project.status,
            "version": project.version,
            "created_at": project.created_at.isoformat(),
        },
        "pairing": _sanitized_pairing(repo.get_pairing(project_id)),
    }
    analyses = repo.list_analysis(project_id)
    reviews = repo.list_review_items(project_id)
    reports = repo.list_reports(project_id)
    reports_serializable = [
        {
            **item,
            "created_at": item["created_at"].isoformat()
            if hasattr(item["created_at"], "isoformat")
            else str(item["created_at"]),
            "html_path": Path(item["html_path"]).name,
        }
        for item in reports
    ]

    snapshots = repo.list_pairing_snapshots(project_id)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(sanitize_export(manifest), ensure_ascii=False, indent=2, default=str),
        )
        archive.writestr(
            "analysis.json",
            json.dumps(sanitize_export(analyses), ensure_ascii=False, indent=2, default=str),
        )
        archive.writestr(
            "reviews.json",
            json.dumps(sanitize_export(reviews), ensure_ascii=False, indent=2, default=str),
        )
        archive.writestr(
            "pairing-snapshots.json",
            json.dumps(sanitize_export(snapshots), ensure_ascii=False, indent=2, default=str),
        )
        archive.writestr(
            "reports.json",
            json.dumps(
                sanitize_export(reports_serializable), ensure_ascii=False, indent=2, default=str
            ),
        )
    return output
