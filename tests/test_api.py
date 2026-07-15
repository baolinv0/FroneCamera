from pathlib import Path

from fastapi.testclient import TestClient

from portrait_eval.api import create_app
from portrait_eval.config import Settings


def test_project_device_scan_pairing_flow(tmp_path: Path) -> None:
    device_a = tmp_path / "a"
    device_b = tmp_path / "b"
    device_a.mkdir()
    device_b.mkdir()
    from PIL import Image

    for folder, names in [(device_a, ["1.jpg", "2.jpg"]), (device_b, ["1.jpg", "2.jpg"])]:
        for name in names:
            Image.new("RGB", (16, 16), "gray").save(folder / name)

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}", workspace=tmp_path / "workspace")
    client = TestClient(create_app(settings))
    project = client.post("/api/projects", json={"name": "test"}).json()
    project_id = project["id"]
    for name, folder in [("A", device_a), ("B", device_b)]:
        response = client.post(f"/api/projects/{project_id}/devices", json={"name": name, "folder_path": str(folder)})
        assert response.status_code == 201
    response = client.post(f"/api/projects/{project_id}/scan")
    assert response.status_code == 200
    pairing = client.get(f"/api/projects/{project_id}/pairing").json()
    assert len(pairing["groups"]) == 2
    confirmed = client.post(f"/api/projects/{project_id}/pairing/confirm", json={"expected_version": pairing["version"]})
    assert confirmed.status_code == 200
    assert confirmed.json()["confirmed"] is True
    snapshots = client.get(f"/api/projects/{project_id}/pairing/snapshots")
    assert snapshots.status_code == 200
    assert snapshots.json()[0]["version"] == confirmed.json()["version"]


def test_diagnostic_asset_is_served_only_from_workspace(tmp_path: Path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'asset.db'}", workspace=tmp_path / "workspace")
    settings.prepare()
    client = TestClient(create_app(settings))
    project = client.post("/api/projects", json={"name": "asset"}).json()
    diagnostic = settings.workspace / "projects" / project["id"] / "diagnostics" / "image.jpg"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_bytes(b"jpeg-placeholder")
    from portrait_eval.database import Database
    from portrait_eval.repository import Repository
    database = Database(settings.database_url)
    with database.session_factory() as session:
        row = Repository(session).save_analysis(project["id"], "image_metrics", {"diagnostic_path": str(diagnostic)})
        analysis_id = row.id
    response = client.get(f"/api/analysis/{analysis_id}/diagnostic")
    assert response.status_code == 200
    assert response.content == b"jpeg-placeholder"


def test_finalize_report_creates_new_immutable_final_version(tmp_path: Path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'report.db'}", workspace=tmp_path / "workspace")
    settings.prepare()
    client = TestClient(create_app(settings))
    project = client.post("/api/projects", json={"name": "report"}).json()
    draft = settings.workspace / "projects" / project["id"] / "reports" / "draft-v0.1.html"
    draft.parent.mkdir(parents=True)
    draft.write_text("<p>DRAFT — verify review gates before external distribution.</p>", encoding="utf-8")
    from portrait_eval.database import Database
    from portrait_eval.repository import Repository
    database = Database(settings.database_url)
    with database.session_factory() as session:
        Repository(session).save_report(project["id"], "0.1", "draft", str(draft))
    first = client.post(f"/api/projects/{project['id']}/reports/finalize")
    assert first.status_code == 200
    assert first.json()["version"] == "1.0"
    first_path = Path(first.json()["html_path"])
    assert "FINAL REPORT" in first_path.read_text(encoding="utf-8")
    assert "DRAFT" not in first_path.read_text(encoding="utf-8")
    second = client.post(f"/api/projects/{project['id']}/reports/finalize")
    assert second.status_code == 200
    assert second.json()["version"] == "1.1"
    second_path = Path(second.json()["html_path"])
    assert second_path != first_path
    assert first_path.exists()


def test_report_pdf_endpoint_uses_registered_report(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'pdf.db'}", workspace=tmp_path / "workspace")
    settings.prepare()
    client = TestClient(create_app(settings))
    project = client.post("/api/projects", json={"name": "pdf"}).json()
    html_path = settings.workspace / "projects" / project["id"] / "reports" / "draft-v0.1.html"
    html_path.parent.mkdir(parents=True)
    html_path.write_text("<h1>report</h1>", encoding="utf-8")
    from portrait_eval.database import Database
    from portrait_eval.repository import Repository
    database = Database(settings.database_url)
    with database.session_factory() as session:
        report = Repository(session).save_report(project["id"], "0.1", "draft", str(html_path))
        report_id = report.id
    def fake_render(_html: Path, pdf: Path) -> Path:
        pdf.write_bytes(b"pdf-placeholder")
        return pdf
    monkeypatch.setattr("portrait_eval.api.render_pdf_report", fake_render)
    response = client.get(f"/api/reports/{report_id}/pdf")
    assert response.status_code == 200
    assert response.content == b"pdf-placeholder"


def test_report_html_endpoint_returns_requested_version(tmp_path: Path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'html.db'}", workspace=tmp_path / "workspace")
    settings.prepare()
    client = TestClient(create_app(settings))
    project = client.post("/api/projects", json={"name": "html"}).json()
    html_path = settings.workspace / "projects" / project["id"] / "reports" / "draft-v0.1.html"
    html_path.parent.mkdir(parents=True)
    html_path.write_text("<h1>requested-version</h1>", encoding="utf-8")
    from portrait_eval.database import Database
    from portrait_eval.repository import Repository
    database = Database(settings.database_url)
    with database.session_factory() as session:
        report = Repository(session).save_report(project["id"], "0.1", "draft", str(html_path))
    response = client.get(f"/api/reports/{report.id}/html")
    assert response.status_code == 200
    assert b"requested-version" in response.content


def test_finalize_regenerates_bundle_and_applies_rejected_attribution_review(tmp_path: Path) -> None:
    from portrait_eval.reporting import ReportPayload, render_report_bundle
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'reviewed-report.db'}", workspace=tmp_path / "workspace")
    settings.prepare()
    client = TestClient(create_app(settings))
    project = client.post("/api/projects", json={"name": "reviewed"}).json()
    report_dir = settings.workspace / "projects" / project["id"] / "reports"
    payload = ReportPayload(project_name="reviewed", devices=["A", "B"], findings=[{"statement": "Observable finding", "grade": "B", "evidence": ["G001"]}], attributions=[{"statement": "Mechanism hypothesis", "device_id": "device-a", "primary_layer": "rendering"}])
    bundle = render_report_bundle(payload, report_dir, "0.1", "draft")
    from portrait_eval.database import Database
    from portrait_eval.repository import Repository
    database = Database(settings.database_url)
    with database.session_factory() as session:
        repo = Repository(session)
        repo.save_report(project["id"], "0.1", "draft", str(bundle["html"]))
        review = repo.create_review_item(project["id"], "mechanism_attribution_review", {"statement": "Mechanism hypothesis", "device_id": "device-a"})
        review_id = review.id
    resolved = client.patch(f"/api/review-items/{review_id}", json={"status": "rejected", "note": "unsupported"})
    assert resolved.status_code == 200
    finalized = client.post(f"/api/projects/{project['id']}/reports/finalize")
    assert finalized.status_code == 200
    final_json = Path(finalized.json()["html_path"]).with_suffix(".json")
    document = final_json.read_text(encoding="utf-8")
    assert "Observable finding" in document
    assert "Mechanism hypothesis" not in document
