from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ReportPayload(BaseModel):
    project_name: str
    devices: list[str]
    findings: list[dict[str, Any]]
    schema_version: str = "0.1"
    all_claims: list[dict[str, Any]] = Field(default_factory=list)
    challenge_matrix: list[dict[str, Any]] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    review_provenance: dict[str, Any] = Field(
        default_factory=lambda: {"human_review": "pending", "calibration": "not_performed"}
    )
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

    @model_validator(mode="after")
    def validate_claim_contract(self):
        if self.schema_version == "0.2":
            scene_ids = {scene["group_id"] for scene in self.scene_results}
            for claim in self.findings + self.all_claims:
                if (
                    not claim.get("claim_id")
                    or not claim.get("claim_version")
                    or not claim.get("statement")
                ):
                    raise ValueError("Report 0.2 requires claim_id, claim_version and statement")
                if (
                    not set(
                        claim.get("supporting_scene_ids", [])
                        + claim.get("contradicting_scene_ids", [])
                    )
                    <= scene_ids
                ):
                    raise ValueError("Claim references an unknown scene")
        return self


def render_html_report(payload: ReportPayload, output: Path, status: str = "draft") -> Path:
    from portrait_eval.html_reporting import render_evidence_html

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_evidence_html(payload, status), encoding="utf-8")
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
    extra_index = 0

    def materialize(value: Any, key: str = "") -> Any:
        nonlocal extra_index
        if isinstance(value, dict):
            return {k: materialize(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [materialize(item, key) for item in value]
        if isinstance(value, str) and key.endswith("_path"):
            source = Path(value)
            if not source.is_absolute():
                return value
            if source.is_file():
                extra_index += 1
                destination = assets_dir / f"evidence-{extra_index:04d}{source.suffix.lower()}"
                assets_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                return destination.relative_to(output_dir).as_posix()
            return ""
        return value

    return payload.model_copy(
        update={"visual_assets": normalized, "scene_results": materialize(payload.scene_results)}
    )


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
