from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ReportPayload(BaseModel):
    project_name: str
    devices: list[str]
    findings: list[dict[str, Any]]
    scene_results: list[dict[str, Any]] = Field(default_factory=list)
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
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#17202a}}header{{border-bottom:2px solid #17202a}}article,details{{border:1px solid #ddd;padding:16px;margin:12px 0;border-radius:8px}}small{{color:#5d6d7e}}pre{{white-space:pre-wrap}}.draft{{background:#fff3cd;padding:8px}}.final{{background:#dcfae6;padding:8px}}</style></head>
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


def render_report_bundle(
    payload: ReportPayload,
    output_dir: Path,
    version: str,
    status: str,
) -> dict[str, Path]:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{status}-v{version}"
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
    return {"html": html_path, "json": json_path, "csv": csv_path}


def render_pdf_report(html_path: Path, pdf_path: Path) -> Path:
    """Render PDF when WeasyPrint is installed in the deployment image."""
    try:
        from weasyprint import HTML
    except ImportError as exc:  # optional local dependency
        raise RuntimeError("Install the 'pdf' extra to enable PDF export") from exc
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return pdf_path
