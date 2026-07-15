from pathlib import Path

from portrait_eval.reporting import ReportPayload, render_html_report


def test_report_contains_traceable_claims(tmp_path: Path) -> None:
    payload = ReportPayload(project_name="Demo", devices=["A", "B"], findings=[{"statement": "A is brighter in G001", "evidence": ["G001"]}], limitations=["Single capture"])
    output = tmp_path / "report.html"
    render_html_report(payload, output)
    text = output.read_text(encoding="utf-8")
    assert "A is brighter in G001" in text
    assert "G001" in text
    assert "Single capture" in text


def test_final_report_bundle_has_final_banner_and_preserves_draft(tmp_path: Path) -> None:
    from portrait_eval.reporting import render_report_bundle
    payload = ReportPayload(project_name="Demo", devices=["A", "B"], findings=[])
    draft = render_report_bundle(payload, tmp_path, version="0.1", status="draft")
    final = render_report_bundle(payload, tmp_path, version="1.0", status="final")
    assert "DRAFT" in draft["html"].read_text(encoding="utf-8")
    final_text = final["html"].read_text(encoding="utf-8")
    assert "FINAL REPORT" in final_text
    assert "DRAFT" not in final_text
    assert draft["html"].exists()
