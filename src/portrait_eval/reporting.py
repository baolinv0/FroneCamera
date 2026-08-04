from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ReportPayload(BaseModel):
    project_name: str
    devices: list[str]
    findings: list[dict[str, Any]]
    report_title: str | None = None
    report_subtitle: str | None = None
    scene_results: list[dict[str, Any]] = Field(default_factory=list)
    visual_assets: list[dict[str, Any]] = Field(default_factory=list)
    device_profiles: list[dict[str, Any]] = Field(default_factory=list)
    device_scores: list[dict[str, Any]] = Field(default_factory=list)
    methodology_notes: list[str] = Field(default_factory=list)
    external_validation: list[dict[str, Any]] = Field(default_factory=list)
    hardware_context: list[dict[str, Any]] = Field(default_factory=list)
    attributions: list[dict[str, Any]] = Field(default_factory=list)
    capture_bias: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def render_html_report(payload: ReportPayload, output: Path, status: str = "draft") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    findings = "".join(
        f"<article><p>{html.escape(str(item['statement']))}</p>"
        f"<small>Evidence: {html.escape(', '.join(item.get('evidence', [])))}</small></article>"
        for item in payload.findings
    )
    scenes = "".join(
        f"<details><summary>{html.escape(str(scene.get('group_id', 'Scene')))}</summary>"
        f"<pre>{html.escape(json.dumps(scene, ensure_ascii=False, indent=2))}</pre></details>"
        for scene in payload.scene_results
    )
    external = "".join(
        f"<li><a href='{html.escape(str(item.get('url', '#')))}'>{html.escape(str(item.get('title', 'Source')))}</a></li>"
        for item in payload.external_validation
    )
    hardware = "".join(
        f"<li><b>{html.escape(str(item.get('device_name', 'Device')))}</b>: "
        f"<a href='{html.escape(str(item.get('url', '#')))}'>{html.escape(str(item.get('title', 'Source')))}</a> "
        f"(Tier {html.escape(str(item.get('source_tier', '?')))})</li>"
        for item in payload.hardware_context
    )
    attributions = "".join(
        f"<details><summary>{html.escape(str(item.get('primary_layer', 'indeterminate')))}</summary>"
        f"<pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre></details>"
        for item in payload.attributions
    )
    bias = "".join(
        f"<details><summary>{html.escape(str(item.get('status', 'UNRESOLVED')))}</summary><pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre></details>"
        for item in payload.capture_bias
    )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in payload.limitations)
    banner = (
        "FINAL REPORT — review gates resolved."
        if status == "final"
        else "DRAFT — verify review gates before external distribution."
    )
    banner_class = "final" if status == "final" else "draft"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(payload.project_name)}</title>
<style>body{{font-family:"Noto Sans CJK SC",Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#202124}}header{{border-bottom:4px solid #f97316}}article,details{{border:1px solid #d6d6d6;padding:16px;margin:12px 0;border-radius:8px}}small{{color:#666}}pre{{white-space:pre-wrap}}.draft{{background:#fff3cd;padding:8px}}.final{{background:#dcfae6;padding:8px}}</style></head>
<body><header><h1>{html.escape(payload.project_name)}</h1><p class="{banner_class}">{banner}</p>
<p>Devices: {html.escape(", ".join(payload.devices))}</p></header>
<section><h2>Evidence-backed findings</h2>{findings or "<p>No approved findings.</p>"}</section>
<section><h2>Scene results</h2>{scenes}</section>
<section><h2>Hardware context</h2><ul>{hardware or "<li>No hardware evidence retrieved.</li>"}</ul></section>
<section><h2>External professional corroboration</h2><ul>{external or "<li>No relevant evidence retrieved.</li>"}</ul></section>
<section><h2>Hardware–capture–reconstruction–rendering attribution</h2>{attributions or "<p>No attribution cases.</p>"}</section>
<section><h2>Capture-bias assessment</h2>{bias or "<p>Not assessed.</p>"}</section>
<section><h2>Limitations</h2><ul>{limitations}</ul></section></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output


def _asset_slug(value: Any, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return slug[:60] or fallback


def _materialize_visual_assets(payload: ReportPayload, output_dir: Path) -> ReportPayload:
    """Copy source images beside the report and persist only relative paths."""
    assets_dir = output_dir / "assets"
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(payload.visual_assets):
        asset = dict(item)
        raw_path = asset.get("path") or asset.get("source_path")
        if not raw_path:
            asset.pop("source_path", None)
            normalized.append(asset)
            continue
        source = Path(str(raw_path))
        if not source.is_absolute():
            candidate = output_dir / source
            if candidate.is_file():
                asset["path"] = candidate.relative_to(output_dir).as_posix()
                asset.pop("source_path", None)
                normalized.append(asset)
                continue
        if not source.is_file():
            asset["path"] = ""
            asset["missing"] = True
            asset.pop("source_path", None)
            normalized.append(asset)
            continue
        assets_dir.mkdir(parents=True, exist_ok=True)
        scene = _asset_slug(asset.get("scene_id") or asset.get("group_id"), "scene")
        device = _asset_slug(asset.get("device_name") or asset.get("device_id"), "device")
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        destination = assets_dir / f"{index + 1:03d}-{scene}-{device}{suffix}"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        asset["path"] = destination.relative_to(output_dir).as_posix()
        asset["filename"] = source.name
        asset.pop("source_path", None)
        normalized.append(asset)
    return payload.model_copy(update={"visual_assets": normalized})


def render_report_bundle(
    payload: ReportPayload,
    output_dir: Path,
    version: str,
    status: str,
) -> dict[str, Path]:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{status}-v{version}"
    payload = _materialize_visual_assets(payload, output_dir)
    html_path = render_html_report(payload, output_dir / f"{stem}.html", status=status)
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    csv_path = output_dir / f"{stem}-findings.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["statement", "grade", "status", "confidence", "evidence"],
        )
        writer.writeheader()
        for item in payload.findings:
            writer.writerow(
                {
                    "statement": item.get("statement", ""),
                    "grade": item.get("grade", ""),
                    "status": item.get("status", ""),
                    "confidence": item.get("confidence", ""),
                    "evidence": ";".join(item.get("evidence", [])),
                }
            )
    from portrait_eval.docx_reporting import render_docx_report

    docx_path = render_docx_report(
        payload,
        output_dir / f"{stem}.docx",
        status=status,
    )
    return {"html": html_path, "json": json_path, "csv": csv_path, "docx": docx_path}


def render_pdf_report(html_path: Path, pdf_path: Path) -> Path:
    """Render PDF when WeasyPrint is installed in the deployment image."""
    try:
        from weasyprint import HTML
    except ImportError as exc:  # optional local dependency
        raise RuntimeError("Install the 'pdf' extra to enable PDF export") from exc
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return pdf_path
