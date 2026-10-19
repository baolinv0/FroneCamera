from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from portrait_eval.repository import Repository


def _sanitized_pairing(pairing: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        "project_id": pairing["project_id"],
        "version": pairing["version"],
        "confirmed": pairing["confirmed"],
        "devices": [
            {
                "id": item["id"],
                "name": item["name"],
                "canonical_model": item.get("canonical_model"),
            }
            for item in pairing.get("devices", [])
        ],
        "groups": [],
    }
    for group in pairing.get("groups", []):
        cells = {}
        for device_id, cell in group.get("cells", {}).items():
            cells[device_id] = (
                None
                if cell is None
                else {
                    "image_id": cell["image_id"],
                    "filename": cell["filename"],
                    "width": cell["width"],
                    "height": cell["height"],
                    "exif": cell.get("exif", {}),
                }
            )
        sanitized["groups"].append(
            {
                "id": group["id"],
                "group_id": group["group_id"],
                "label": group.get("label"),
                "analyzable": group.get("analyzable", False),
                "cells": cells,
            }
        )
    return sanitized


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

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str)
        )
        archive.writestr(
            "analysis.json", json.dumps(analyses, ensure_ascii=False, indent=2, default=str)
        )
        archive.writestr(
            "reviews.json", json.dumps(reviews, ensure_ascii=False, indent=2, default=str)
        )
        archive.writestr(
            "reports.json",
            json.dumps(reports_serializable, ensure_ascii=False, indent=2, default=str),
        )
    return output
