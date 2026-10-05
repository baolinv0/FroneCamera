import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from portrait_eval.config import Settings
from portrait_eval.database import Database
from portrait_eval.product_api import create_app
from portrait_eval.report_projection import resolve_report_payload
from portrait_eval.reporting import ReportPayload, render_report_bundle
from portrait_eval.repository import Repository
from portrait_eval.strategy import HIGHEST_LUMINANCE_OBSERVATION, infer_cross_scene_claims
from portrait_eval.workflow import finalize_quick_report
from tests.test_report_projection import docx_text, fixture_payload, setup_report


@pytest.mark.parametrize("review_shape", ["full", "nested"])
@pytest.mark.parametrize("status", ["rejected", "insufficient_evidence"])
def test_rejected_full_strategy_preserves_approved_sources_and_sibling(
    tmp_path, review_shape, status
):
    observation = {
        "claim_id": "source-1",
        "device_id": "d",
        "statement": "Approved scene observation",
        "evidence": ["G001"],
    }
    strategy = {
        "claim_id": "strategy-1",
        "device_id": "d",
        "statement": "Rejected cross-scene inference",
        "claim_type": "strategy",
        "source_claim_ids": ["source-1"],
        "supporting_claim_ids": ["source-1"],
        "evidence": ["G001"],
    }
    sibling = {
        **strategy,
        "claim_id": "strategy-2",
        "statement": "Approved independent inference",
    }
    dependent = {
        **strategy,
        "claim_id": "strategy-3",
        "statement": "Inference depending on rejected strategy",
        "source_claim_ids": ["strategy-1"],
        "supporting_claim_ids": ["strategy-1"],
    }
    payload = ReportPayload(
        project_name="Strategy target",
        devices=["d"],
        findings=[observation, strategy, sibling, dependent],
        scene_results=[
            {
                "group_id": "G001",
                "findings": [observation],
                "primary": {"observations": [observation]},
            }
        ],
    )
    client, database, project_id, _ = setup_report(tmp_path, payload)
    with database.session_factory() as session:
        repo = Repository(session)
        for approved in (observation, sibling):
            review = repo.create_review_item(project_id, "supported", approved)
            repo.resolve_review_item(review.id, "approved")
        target = strategy if review_shape == "full" else {"primary_observation": strategy}
        review = repo.create_review_item(project_id, "insufficient_scene_coverage", target)
        repo.resolve_review_item(review.id, status)
    response = client.post(f"/api/projects/{project_id}/reports/finalize")
    assert response.status_code == 200, response.text
    path = Path(response.json()["html_path"])
    published = json.loads(path.with_suffix(".json").read_text())
    assert [item["claim_id"] for item in published["findings"]] == ["source-1", "strategy-2"]
    assert published["scene_results"][0]["findings"] == [observation]
    assert published["scene_results"][0]["primary"]["observations"] == [observation]
    for content in (
        path.read_text(),
        docx_text(path.with_suffix(".docx")),
        path.with_name(path.stem + "-findings.csv").read_text(),
    ):
        assert observation["statement"] in content
        assert sibling["statement"] in content
        assert strategy["statement"] not in content
        assert dependent["statement"] not in content


@pytest.mark.parametrize("target_field", ["claim_ids", "supporting_claim_ids"])
def test_legacy_explicit_target_lists_still_deny_sources_and_their_dependents(target_field):
    observation = {"claim_id": "source-1", "device_id": "d", "statement": "Scene observation"}
    strategy = {
        "claim_id": "strategy-1",
        "device_id": "d",
        "statement": "Derived strategy",
        "source_claim_ids": ["source-1"],
    }
    unrelated = {"claim_id": "other-1", "device_id": "d", "statement": "Unrelated evidence"}
    payload = ReportPayload(
        project_name="Legacy targets", devices=["d"], findings=[observation, strategy, unrelated]
    )
    projected = resolve_report_payload(
        payload, [{"status": "rejected", "payload": {target_field: ["source-1"]}}]
    )
    assert projected.findings == [unrelated]


@pytest.mark.parametrize("review_shape", ["stable", "nested_legacy"])
def test_rejected_source_observations_remove_derived_strategy_from_every_artifact(
    tmp_path, review_shape
):
    scenes = []
    for index in range(3):
        group = f"G{index + 1:03}"
        observation = {
            "claim_id": f"o{index}",
            "device_id": "d",
            "dimension": "global_exposure",
            "statement": HIGHEST_LUMINANCE_OBSERVATION,
            "grade": "B",
            "status": "SUPPORTED",
            "confidence": 1.0,
            "model_agreement": 1.0,
            "uncertainty": 1.0,
            "independent_models": True,
            "evidence_refs": [f"metric:{group}:d:whole.luma_mean"],
            "evidence": [group],
        }
        scenes.append(
            {
                "group_id": group,
                "audit": {"status": "FULLY_COMPARABLE"},
                "metrics": {"d": {"whole": {"luma_mean": 0.8}}, "e": {"whole": {"luma_mean": 0.3}}},
                "findings": [observation],
                "primary": {"observations": [observation]},
            }
        )
    strategy = infer_cross_scene_claims(scenes)[0].model_dump(mode="json")
    strategy["evidence"] = strategy["supporting_scene_ids"]
    assert strategy["grade"] == "A" and strategy["confidence"] == 1
    payload = ReportPayload(
        project_name="Source denial", devices=["d", "e"], findings=[strategy], scene_results=scenes
    )
    client, database, project_id, _ = setup_report(tmp_path, payload)
    with database.session_factory() as session:
        repo = Repository(session)
        for index in range(3):
            target = {"claim_id": f"o{index}"}
            if review_shape == "nested_legacy":
                target = {
                    "group_id": f"G{index + 1:03}",
                    "primary_observation": {
                        "device_id": "d",
                        "dimension": "global_exposure",
                        "statement": HIGHEST_LUMINANCE_OBSERVATION,
                    },
                }
            review = repo.create_review_item(project_id, "capture_bias", target)
            repo.resolve_review_item(review.id, "rejected")
    response = client.post(f"/api/projects/{project_id}/reports/finalize")
    assert response.status_code == 200, response.text
    path = Path(response.json()["html_path"])
    assert not json.loads(path.with_suffix(".json").read_text())["findings"]
    for content in (
        path.read_text(),
        docx_text(path.with_suffix(".docx")),
        path.with_name(path.stem + "-findings.csv").read_text(),
    ):
        assert strategy["statement"] not in content


@pytest.mark.parametrize("change", ["open", "rejected", "version", "state"])
def test_professional_final_commit_rechecks_changes_during_render(tmp_path, monkeypatch, change):
    client, database, project_id, draft = setup_report(tmp_path, fixture_payload())

    def change_after_render(payload, directory, version, status):
        bundle = render_report_bundle(payload, directory, version, status)
        with database.session_factory() as session:
            repo = Repository(session)
            if change in {"open", "rejected"}:
                review = repo.create_review_item(project_id, "late_review", {"claim_id": "b"})
                if change == "rejected":
                    repo.resolve_review_item(review.id, "rejected")
            else:
                project = repo.get_project(project_id)
                if change == "version":
                    project.version += 1
                project.status = "PAIRING_REQUIRED"
                session.commit()
        return bundle

    monkeypatch.setattr("portrait_eval.api.render_report_bundle", change_after_render)
    response = client.post(f"/api/projects/{project_id}/reports/finalize")
    assert response.status_code == 409, response.text
    with database.session_factory() as session:
        repo = Repository(session)
        assert not [row for row in repo.list_reports(project_id) if row["status"] == "final"]
        assert repo.get_project(project_id).status != "REPORT_FINALIZED"
    assert not list((draft["html"].parent / "publications").glob("*"))
    assert not list((draft["html"].parent / ".publication-staging").glob("*"))


def test_open_review_added_after_initial_gate_before_projection_is_rejected(tmp_path, monkeypatch):
    client, database, project_id, _ = setup_report(tmp_path, fixture_payload())
    original = Repository.list_review_items
    inserted = False

    def insert_before_projection(self, selected_project):
        nonlocal inserted
        if selected_project == project_id and not inserted:
            inserted = True
            with database.session_factory() as session:
                Repository(session).create_review_item(project_id, "late", {"claim_id": "b"})
        return original(self, selected_project)

    monkeypatch.setattr(Repository, "list_review_items", insert_before_projection)
    response = client.post(f"/api/projects/{project_id}/reports/finalize")
    assert response.status_code == 409, response.text


@pytest.mark.parametrize("report_status", ["draft", "final"])
def test_signed_public_artifacts_hide_nested_paths_and_keep_real_assets(tmp_path, report_status):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'public.db'}",
        workspace=tmp_path / "workspace",
        api_token="admin",
        report_share_secret="share-secret",
    )
    settings.prepare()
    client = TestClient(create_app(settings))
    headers = {"Authorization": "Bearer admin"}
    project_id = client.post("/api/projects", json={"name": "Public"}, headers=headers).json()["id"]
    fixture_image = (
        Path(__file__).resolve().parents[1] / "evidence/build/task-1-example/baseline.png"
    )
    source_image = tmp_path / "private-captures" / "real-fixture.png"
    source_image.parent.mkdir()
    source_image.write_bytes(fixture_image.read_bytes())
    payload = ReportPayload(
        project_name="Public",
        devices=["d"],
        findings=[
            {
                "claim_id": "stable-id",
                "statement": f"Evidence at {source_image}",
                "evidence": ["G001"],
            }
        ],
        scene_results=[
            {
                "group_id": "G001",
                "source_path": str(source_image),
                "diagnostic": {
                    "note": f"Decoded {source_image}",
                    "windows": r"C:\private\capture.jpg",
                    "unc": r"\\secret-server\private-share\capture.jpg",
                    "sha256": "immutable-hash",
                    "evidence_url": "https://example.test/evidence/1",
                },
            }
        ],
        visual_assets=[{"source_path": str(source_image), "scene_id": "G001", "device_id": "d"}],
    )
    bundle = render_report_bundle(
        payload, settings.workspace / "projects" / project_id / "reports", "0.1", "draft"
    )
    database = Database(settings.database_url)
    with database.session_factory() as session:
        draft = Repository(session).save_report(project_id, "0.1", "draft", str(bundle["html"]))
        report = {"id": draft.id, "html_path": draft.html_path}
    if report_status == "final":
        response = client.post(f"/api/projects/{project_id}/reports/finalize", headers=headers)
        assert response.status_code == 200, response.text
        report = response.json()
    share = client.post(f"/api/reports/{report['id']}/share", headers=headers)
    assert share.status_code == 200, share.text
    public = client.get(share.json()["url"])
    assert public.status_code == 200
    public_docx = client.get(share.json()["docx_url"])
    assert public_docx.status_code == 200
    path = Path(report["html_path"])
    for content in (
        public.text,
        path.with_suffix(".json").read_text(),
        docx_text(path.with_suffix(".docx")),
        path.with_name(path.stem + "-findings.csv").read_text(),
    ):
        assert str(tmp_path) not in content
        assert "private-captures" not in content
        assert "secret-server" not in content and "private-share" not in content
        assert r"C:\private" not in content
    data = json.loads(path.with_suffix(".json").read_text())
    asset = data["visual_assets"][0]
    assert asset["path"].startswith("assets/") and not asset.get("missing")
    assert (path.parent / asset["path"]).read_bytes() == source_image.read_bytes()
    assert data["scene_results"][0]["diagnostic"]["sha256"] == "immutable-hash"
    assert (
        data["scene_results"][0]["diagnostic"]["evidence_url"] == "https://example.test/evidence/1"
    )
    assert data["findings"][0]["claim_id"] == "stable-id"


def test_sanitized_bound_draft_finalizes_with_copied_assets_and_original_binding(tmp_path):
    from portrait_eval.run_binding import validate_run_binding
    from tests.test_run_binding_acceptance import _binding
    from tests.test_task3_runtime_pipeline import setup_project

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'draft-bound.db'}", workspace=tmp_path / "workspace"
    )
    settings.prepare()
    client = TestClient(create_app(settings))
    database = Database(settings.database_url)
    with database.session_factory() as session:
        repo, project, _, paths = setup_project(tmp_path, session)
        project_id = project.id
        binding = _binding(repo, project_id)
        payload = ReportPayload(
            project_name="Bound draft",
            devices=["d"],
            findings=[
                {
                    "claim_id": "approved-1",
                    "statement": f"Observation from {paths[0]}",
                    "evidence": ["G001"],
                }
            ],
            scene_results=[
                {
                    "group_id": "G001",
                    "source_path": str(paths[0]),
                    "diagnostic": {"note": f"Opened {paths[0]}"},
                }
            ],
            visual_assets=[{"source_path": str(paths[0]), "scene_id": "G001", "device_id": "d"}],
            input_trace={"run_binding": binding},
        )
        draft = render_report_bundle(
            payload, settings.workspace / "projects" / project_id / "reports", "0.1", "draft"
        )
        data = json.loads(draft["json"].read_text())
        assert data["input_trace"]["run_binding"] == binding
        validate_run_binding(repo, project_id, data["input_trace"]["run_binding"])
        repo.save_report(project_id, "0.1", "draft", str(draft["html"]))
    for content in (
        draft["html"].read_text(),
        draft["json"].read_text(),
        docx_text(draft["docx"]),
        draft["csv"].read_text(),
    ):
        assert str(tmp_path) not in content
    draft_asset = data["visual_assets"][0]
    assert draft_asset["path"].startswith("assets/") and not draft_asset.get("missing")
    assert (draft["html"].parent / draft_asset["path"]).read_bytes() == paths[0].read_bytes()
    response = client.post(f"/api/projects/{project_id}/reports/finalize")
    assert response.status_code == 200, response.text
    final = Path(response.json()["html_path"])
    published = json.loads(final.with_suffix(".json").read_text())
    assert published["input_trace"]["run_binding"] == binding
    final_asset = published["visual_assets"][0]
    assert final_asset["path"].startswith("assets/") and not final_asset.get("missing")
    assert (final.parent / final_asset["path"]).read_bytes() == paths[0].read_bytes()
    for content in (
        final.read_text(),
        final.with_suffix(".json").read_text(),
        docx_text(final.with_suffix(".docx")),
        final.with_name(final.stem + "-findings.csv").read_text(),
    ):
        assert str(tmp_path) not in content


@pytest.mark.parametrize("mode", ["professional", "quick"])
@pytest.mark.parametrize("mutation", ["bytes", "binding"])
def test_final_publication_validates_source_binding_after_render(
    tmp_path, monkeypatch, mode, mutation
):
    from PIL import Image

    from portrait_eval.report_publication import PublicationConflict
    from tests.test_run_binding_acceptance import _binding
    from tests.test_task3_runtime_pipeline import setup_project

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'bound.db'}", workspace=tmp_path / "workspace"
    )
    settings.prepare()
    client = TestClient(create_app(settings))
    database = Database(settings.database_url)
    with database.session_factory() as session:
        repo, project, _, paths = setup_project(tmp_path, session)
        project_id = project.id
        binding = _binding(repo, project_id)
        if mutation == "binding":
            binding["pairing_sha256"] = "tampered-snapshot-hash"
        payload = ReportPayload(
            project_name="Bound", devices=[], findings=[], input_trace={"run_binding": binding}
        )
        draft = render_report_bundle(
            payload, settings.workspace / "projects" / project_id / "reports", "0.1", "draft"
        )
        repo.save_report(project_id, "0.1", "draft", str(draft["html"]))
    rendered = False

    def mutate_after_render(payload, directory, version, status):
        nonlocal rendered
        bundle = render_report_bundle(payload, directory, version, status)
        rendered = True
        if mutation == "bytes":
            Image.new("RGB", (96, 96), "red").save(paths[0])
        return bundle

    monkeypatch.setattr(
        f"portrait_eval.{'api' if mode == 'professional' else 'workflow'}.render_report_bundle",
        mutate_after_render,
    )
    if mode == "professional":
        response = client.post(f"/api/projects/{project_id}/reports/finalize")
        assert response.status_code == 409, response.text
    else:
        with (
            database.session_factory() as session,
            pytest.raises(PublicationConflict, match="binding"),
        ):
            finalize_quick_report(Repository(session), project_id)
    assert rendered
    with database.session_factory() as session:
        assert not [
            report
            for report in Repository(session).list_reports(project_id)
            if report["status"] == "final"
        ]
    assert not list((draft["html"].parent / "publications").glob("*"))
    assert not list((draft["html"].parent / ".publication-staging").glob("*"))
