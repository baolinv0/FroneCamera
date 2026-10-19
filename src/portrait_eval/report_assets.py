"""Serve bundle assets without exposing neighboring runs or local source paths."""

from __future__ import annotations

import html
import re
from pathlib import Path

from fastapi import HTTPException


def served_report_html(path: Path, asset_base: str, token: str | None = None) -> str:
    document = path.read_text(encoding="utf-8")
    return re.sub(
        r'((?:src|href)=")assets/([A-Za-z0-9._-]+)(")',
        lambda match: (
            match[1]
            + html.escape(asset_base + match[2] + (f"?token={token}" if token else ""), quote=True)
            + match[3]
        ),
        document,
    )


def report_asset_path(report_path: Path, filename: str, workspace: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", filename) or Path(filename).suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".avif",
    }:
        raise HTTPException(status_code=404, detail="Unknown report asset")
    assets = (report_path.parent / "assets").resolve()
    path = (assets / filename).resolve()
    if not assets.is_relative_to(workspace.resolve()) or not path.is_relative_to(assets):
        raise HTTPException(status_code=403, detail="Invalid report asset path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report asset not found")
    return path
