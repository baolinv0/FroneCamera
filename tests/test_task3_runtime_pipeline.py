"""Adversarial integration checks against SQLite, declared files, and HTTP transports."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from portrait_eval.database import Database
from portrait_eval.dataset import propose_pairing, scan_folder
from portrait_eval.iqa_bridge import evaluate_scene
from portrait_eval.model_validation import anonymous_context
from portrait_eval.models import ModelEvaluationResult, ModelObservation
from portrait_eval.pipeline import EvaluationPipeline
from portrait_eval.repository import Repository
from portrait_eval.strategy import infer_cross_scene_claims
from portrait_eval.vlm import (
    OpenAICompatibleVisionAdapter,
    OpenAIResponsesVisionAdapter,
    VisionModelAdapter,
)


def setup_project(tmp_path: Path, session):
    repo = Repository(session)
    project = repo.create_project("runtime")
    paths = []
    for index in range(2):
        folder = tmp_path / f"real-device-{index}"
        folder.mkdir()
        path = folder / "1.png"
        Image.new("RGB", (96, 96), (80 + index * 40,) * 3).save(path)
        paths.append(path)
        repo.add_device(project.id, f"Brand Phone {index}", str(folder), f"Canonical Brand {index}")
    draft = repo.scan_and_pair(project.id)
    pairing = repo.confirm_pairing(project.id, draft["version"])
    return repo, project, pairing, paths


class ObservationAdapter(VisionModelAdapter):
    def __init__(self, statement="This output is brighter.", certainty=0.9, failures=0):
        self.statement, self.certainty, self.failures = statement, certainty, failures
        self.calls = 0

    def analyze(self, scene_id, image_paths, context):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise httpx.ConnectError("temporary transport outage")
        code = sorted(image_paths)[0]
        return ModelEvaluationResult(
            scene_id=scene_id,
            role="test",
            confidence=self.certainty,
            observations=[
                ModelObservation(
                    device_id=code,
                    dimension="face_exposure_readability",
                    statement=self.statement,
                    evidence_refs=[f"asset:{scene_id}:{code}"],
                    certainty=self.certainty,
                )
            ],
        )


def db(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'runtime.sqlite'}")
    database.create_all()
    return database


def test_actual_pipeline_calls_shared_core_and_persists_device_evidence(tmp_path, monkeypatch):
    import portrait_eval.iqa_bridge as bridge

    real = bridge.evaluate_comparison
    captured = []

    def capture(request):
        captured.append(request)
        return real(request)

    monkeypatch.setattr(bridge, "evaluate_comparison", capture)
    with db(tmp_path).session_factory() as session:
        repo, project, pairing, _ = setup_project(tmp_path, session)
        result = EvaluationPipeline(session, tmp_path / "workspace").run(project.id)
        assert result["scene_count"] == 1
        assert len(captured) == 1 and captured[0].mode == "device"
        assert captured[0].source is None and len(captured[0].candidates) == 1
        stored = repo.list_analysis(project.id, "iqa_evaluation")[0]["payload"]
        assert stored["input_trace"]["ordered_assets"]
        assert (
            stored["bridge_provenance"]["entrypoint"] == "qwen_tmqa.comparison.evaluate_comparison"
        )
        assert repo.get_pairing(project.id)["confirmed"]
        assert repo.get_pairing(project.id)["pairing_snapshot_id"]
        binding = repo.list_analysis(project.id, "evaluation_run_binding")[0]["payload"]
        assert binding["pairing_version"] == pairing["version"]
        findings = repo.list_analysis(project.id, "claim")
        assert findings and all(item["payload"]["provisional"] for item in findings)
        assert all(item["payload"]["grade"] == "C" for item in findings)


def test_opposite_statements_and_zero_certainty_cannot_be_supported(tmp_path):
    with db(tmp_path).session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        EvaluationPipeline(
            session,
            tmp_path / "workspace",
            primary=ObservationAdapter(certainty=0),
            reviewer=ObservationAdapter("This output is darker.", certainty=0),
        ).run(project.id)
        findings = repo.list_analysis(project.id, "claim")
        assert findings and all(
            item["payload"]["confidence"] == 0 and item["payload"]["grade"] == "C"
            for item in findings
        )
        review = next(
            item
            for item in repo.list_review_items(project.id)
            if item["category"] == "model_conflict"
        )
        assert review["payload"]["independent_decision"] == "opposition"
        assert review["payload"]["claim_id"] == findings[0]["payload"]["claim_id"]
        assert review["payload"]["primary_observation"]["claim_id"] == review["payload"]["claim_id"]


def test_repeated_transient_failures_keep_snapshot_and_retry_cleanly(tmp_path):
    adapter = ObservationAdapter(failures=2)
    with db(tmp_path).session_factory() as session:
        repo, project, pairing, _ = setup_project(tmp_path, session)
        pipeline = EvaluationPipeline(
            session, tmp_path / "workspace", primary=adapter, reviewer=ObservationAdapter()
        )
        for _ in range(2):
            with pytest.raises(httpx.ConnectError):
                pipeline.run(project.id)
            assert repo.get_project(project.id).status == "FAILED"
            assert repo.get_pairing(project.id)["confirmed"]
            assert not repo.list_analysis(project.id, "image_metrics")
        pipeline.run(project.id)
        assert len(repo.list_analysis(project.id, "evaluation_failure")) == 2
        assert len(repo.list_analysis(project.id, "image_metrics")) == 2
        assert len(repo.list_pairing_snapshots(project.id)) == 1
        assert (
            repo.list_analysis(project.id, "evaluation_run_binding")[0]["payload"][
                "pairing_version"
            ]
            == pairing["version"]
        )
        assert len(repo.list_analysis(project.id, "claim")) == 1


def test_changed_source_bytes_fail_before_model_call(tmp_path):
    adapter = ObservationAdapter()
    with db(tmp_path).session_factory() as session:
        repo, project, _, paths = setup_project(tmp_path, session)
        Image.new("RGB", (96, 96), "red").save(paths[0])
        with pytest.raises(ValueError, match="Source bytes changed"):
            EvaluationPipeline(session, tmp_path / "workspace", primary=adapter).run(project.id)
        assert adapter.calls == 0
        assert not repo.list_analysis(project.id, "iqa_evaluation")


def test_mutated_pairing_requires_new_confirmation(tmp_path):
    with db(tmp_path).session_factory() as session:
        repo, project, pairing, _ = setup_project(tmp_path, session)
        group = pairing["groups"][0]
        device = next(iter(group["cells"]))
        repo.update_pairing_cell(project.id, group["id"], device, None, pairing["version"])
        with pytest.raises(ValueError, match="Pairing must be confirmed"):
            EvaluationPipeline(session, tmp_path / "workspace").run(project.id)


def test_single_capture_group_has_explicit_missing_comparison(tmp_path):
    path = tmp_path / "single.png"
    Image.new("RGB", (20, 20)).save(path)
    result = evaluate_scene("single", {"A": path}, {})
    assert result["status"] == "missing_assets" and result["warnings"]


def test_equal_counts_disagreeing_ordinals_require_review(tmp_path):
    folders = []
    for device, names in (("reference", ("1.png", "2.png")), ("candidate", ("1.png", "9.png"))):
        folder = tmp_path / device
        folder.mkdir()
        for name in names:
            Image.new("RGB", (32, 32), "gray").save(folder / name)
        folders.append(scan_folder(device, folder))
    result = propose_pairing(folders)
    assert any(group.review_required for group in result.groups)
    assert all(group.matching_confidence < 0.55 for group in result.groups)
    assert not any(
        "equal_count_order" in note for group in result.groups for note in group.match_notes
    )


def test_three_supports_twenty_counters_are_disputed():
    scenes = [
        {
            "group_id": f"G{index:03d}",
            "audit": {"status": "FULLY_COMPARABLE"},
            "metrics": {
                "a": {"whole": {"luma_mean": 0.8 if index < 3 else 0.2}},
                "b": {"whole": {"luma_mean": 0.4}},
            },
        }
        for index in range(23)
    ]
    claim = next(item for item in infer_cross_scene_claims(scenes) if item.device_id == "a")
    assert claim.grade.value == "C" and claim.status == "DISPUTED"
    assert claim.confidence < 0.15 and len(claim.contradicting_scene_ids) == 20


def test_confounded_and_missing_measurements_do_not_make_strong_tendencies():
    scenes = [
        {
            "group_id": str(index),
            "audit": {"status": "COMPARABLE_WITH_CONFOUNDERS"},
            "metrics": {"a": {"whole": {"luma_mean": 0.8}}, "b": {"whole": {"luma_mean": 0.4}}},
        }
        for index in range(3)
    ]
    assert all(claim.grade.value == "C" for claim in infer_cross_scene_claims(scenes))
    scenes[0]["metrics"]["b"]["whole"] = {}
    assert all("0" not in claim.supporting_scene_ids for claim in infer_cross_scene_claims(scenes))


@pytest.mark.parametrize("transport", ["compatible", "responses"])
@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_device",
        "unknown_dimension",
        "score_999",
        "score_bool",
        "nan",
        "inf",
        "extra",
        "extra_observation",
        "undeclared_evidence",
        "hypothesis_extra",
    ],
)
def test_provider_transport_rejects_invalid_schema(tmp_path, monkeypatch, transport, mutation):
    path = tmp_path / "asset.png"
    Image.new("RGB", (32, 32), "gray").save(path)
    result = {
        "observations": [
            {
                "device_id": "A",
                "dimension": "highlight_integrity",
                "statement": "More highlight detail.",
                "evidence_refs": ["asset:G001:A"],
                "certainty": 0.8,
            }
        ],
        "scores": {},
        "hypotheses": [],
        "confidence": 0.8,
    }
    observation = result["observations"][0]
    if mutation == "unknown_device":
        observation["device_id"] = "Z"
    if mutation == "unknown_dimension":
        observation["dimension"] = "made_up"
    if mutation == "score_999":
        result["scores"] = {"highlight_integrity": 999}
    if mutation == "score_bool":
        result["scores"] = {"highlight_integrity": True}
    if mutation == "nan":
        result["confidence"] = float("nan")
    if mutation == "inf":
        observation["certainty"] = float("inf")
    if mutation == "extra":
        result["silent_extra"] = True
    if mutation == "extra_observation":
        observation["silent_extra"] = True
    if mutation == "undeclared_evidence":
        observation["evidence_refs"] = ["metric:G001:A:invented"]
    if mutation == "hypothesis_extra":
        result["hypotheses"] = [
            {
                "statement": "Possible lift",
                "evidence_refs": [],
                "alternatives": [],
                "confidence": 0.2,
                "injected": True,
            }
        ]

    def handle(request):
        if transport == "responses":
            payload = {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(result)}],
                    }
                ]
            }
        else:
            payload = {"choices": [{"message": {"content": json.dumps(result)}}]}
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("portrait_eval.vlm.httpx.post", client.post)
        adapter = (
            OpenAICompatibleVisionAdapter("https://model.test", "test", "primary")
            if transport == "compatible"
            else OpenAIResponsesVisionAdapter(
                "test-key", "test", "primary", base_url="https://model.test"
            )
        )
        with pytest.raises(ValueError):
            adapter.analyze("G001", {"A": path}, {})


@pytest.mark.parametrize("transport", ["compatible", "responses"])
def test_actual_prompt_order_and_encoded_payload_hashes(tmp_path, monkeypatch, transport):
    first, second = tmp_path / "BrandA.png", tmp_path / "BrandB.png"
    Image.new("RGB", (1800, 300), "gray").save(first)
    Image.new("RGB", (200, 100), "white").save(second)
    captured = {}
    result = {"observations": [], "scores": {}, "hypotheses": [], "confidence": 0.5}

    def handle(request):
        captured.update(json.loads(request.content))
        raw = (
            {"choices": [{"message": {"content": json.dumps(result)}}]}
            if transport == "compatible"
            else {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(result)}],
                    }
                ]
            }
        )
        return httpx.Response(200, json=raw)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("portrait_eval.vlm.httpx.post", client.post)
        adapter = (
            OpenAICompatibleVisionAdapter("https://model.test", "test", "primary")
            if transport == "compatible"
            else OpenAIResponsesVisionAdapter(
                "test-key", "test", "primary", base_url="https://model.test"
            )
        )
        output = adapter.analyze(
            "G001",
            {"B": first, "A": second},
            {
                "nested": {
                    "diagnostic_path": "/private/BrandA/image.jpg",
                    "note": "see /private/BrandA/file.png for metrics",
                }
            },
        )
    content = (
        captured["messages"][0]["content"]
        if transport == "compatible"
        else captured["input"][0]["content"]
    )
    prompt = content[0]["text"]
    assert "/private" not in prompt and "BrandA" not in prompt
    assert output.input_trace["prompt"] == prompt
    assert output.input_trace["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert [item["device_code"] for item in output.input_trace["images"]] == ["B", "A"]
    images = [item for item in content if item["type"] in {"image_url", "input_image"}]
    url = images[0]["image_url"]["url"] if transport == "compatible" else images[0]["image_url"]
    assert (
        output.input_trace["images"][0]["encoded_sha256"]
        == hashlib.sha256(base64.b64decode(url.split(",")[1])).hexdigest()
    )
    assert (
        output.input_trace["request_sha256"]
        == hashlib.sha256(
            json.dumps(
                captured, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode()
        ).hexdigest()
    )


def test_recursive_anonymization_drops_diagnostics_and_names():
    result = anonymous_context(
        {
            "nested": [
                {
                    "artifact_paths": ["/private/Phone/image.png"],
                    "note": "Brand X has observed texture",
                    "devices": {
                        "real-id": {"exif": {"Make": "Brand X"}, "whole": {"luma_mean": 0.5}}
                    },
                }
            ]
        },
        {"Brand X": "A", "real-id": "A"},
    )
    text = json.dumps(result)
    assert (
        "Brand X" not in text
        and "real-id" not in text
        and "/private" not in text
        and "Make" not in text
    )
    assert result["nested"][0]["devices"]["A"]["whole"]["luma_mean"] == 0.5


def test_worker_retries_two_real_transport_failures_then_succeeds(tmp_path, monkeypatch):
    from portrait_eval.config import Settings
    from portrait_eval.tasking import TaskService
    from portrait_eval.worker import execute_one

    database = db(tmp_path)
    adapter = ObservationAdapter(failures=2)
    monkeypatch.setattr(
        "portrait_eval.worker._adapters",
        lambda settings: (adapter, ObservationAdapter(), None, None),
    )
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'runtime.sqlite'}",
        workspace=tmp_path / "workspace",
        task_max_attempts=3,
    )
    with database.session_factory() as session:
        _repo, project, _, _ = setup_project(tmp_path, session)
        task = TaskService(session).enqueue(
            project.id, "evaluate_project", {"mode": "professional"}
        )
        task_id, project_id = task.id, project.id
    for expected_attempt, expected_status in ((1, "PENDING"), (2, "PENDING"), (3, "SUCCEEDED")):
        assert execute_one(settings)
        with database.session_factory() as session:
            task = TaskService(session).get(task_id)
            assert task.attempts == expected_attempt and task.status == expected_status
            assert Repository(session).get_pairing(project_id)["confirmed"]
            if expected_status == "PENDING":
                assert "ConnectError" in task.error
    with database.session_factory() as session:
        assert len(Repository(session).list_analysis(project_id, "evaluation_failure")) == 2
        assert TaskService(session).get(task_id).error is None


def test_worker_permanent_schema_error_stays_failed(tmp_path, monkeypatch):
    from portrait_eval.config import Settings
    from portrait_eval.tasking import TaskService
    from portrait_eval.worker import execute_one

    class InvalidAdapter(ObservationAdapter):
        def analyze(self, scene_id, image_paths, context):
            return (
                super()
                .analyze(scene_id, image_paths, context)
                .model_copy(update={"confidence": float("nan")})
            )

    database = db(tmp_path)
    monkeypatch.setattr(
        "portrait_eval.worker._adapters",
        lambda settings: (InvalidAdapter(), ObservationAdapter(), None, None),
    )
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'runtime.sqlite'}",
        workspace=tmp_path / "workspace",
        task_max_attempts=3,
    )
    with database.session_factory() as session:
        _repo, project, _, _ = setup_project(tmp_path, session)
        task_id = (
            TaskService(session)
            .enqueue(project.id, "evaluate_project", {"mode": "professional"})
            .id
        )
    assert execute_one(settings)
    with database.session_factory() as session:
        failed = TaskService(session).get(task_id)
        assert failed.status == "FAILED" and failed.attempts == 1
        assert "ValidationError" in failed.error
    assert execute_one(settings) is False


def test_actual_api_repeated_model_failure_and_success_preserve_binding(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from portrait_eval.config import Settings
    from portrait_eval.product_api import create_app

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'runtime.sqlite'}", workspace=tmp_path / "workspace"
    )
    app = create_app(settings)
    with app.state.database.session_factory() as session:
        repo, project, pairing, _ = setup_project(tmp_path, session)
        project_id = project.id
    adapter = ObservationAdapter(failures=2)
    monkeypatch.setattr(
        "portrait_eval.product_api._adapter_set",
        lambda settings: (adapter, ObservationAdapter(), None, None),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        for _ in range(2):
            response = client.post(
                f"/api/projects/{project_id}/run-full-evaluation?sync=true",
                json={"mode": "professional"},
            )
            assert response.status_code == 500
            current = client.get(f"/api/projects/{project_id}/pairing").json()
            assert current["confirmed"] and current["version"] == pairing["version"]
        response = client.post(
            f"/api/projects/{project_id}/run-full-evaluation?sync=true",
            json={"mode": "professional"},
        )
        assert response.status_code == 200
    with app.state.database.session_factory() as session:
        repo = Repository(session)
        assert len(repo.list_analysis(project_id, "evaluation_failure")) == 2
        trace = repo.list_analysis(project_id, "evaluation_run_binding")[0]["payload"]
        assert trace["pairing_version"] == pairing["version"]


def test_source_change_during_adapter_run_fails_bound_evaluation(tmp_path):
    class MutationAdapter(ObservationAdapter):
        def analyze(self, scene_id, image_paths, context):
            output = super().analyze(scene_id, image_paths, context)
            Image.new("RGB", (96, 96), "green").save(next(iter(image_paths.values())))
            return output

    with db(tmp_path).session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        with pytest.raises(ValueError, match="Source bytes changed during evaluation"):
            EvaluationPipeline(
                session,
                tmp_path / "workspace",
                primary=MutationAdapter(),
                reviewer=ObservationAdapter(),
            ).run(project.id)
        assert repo.get_project(project.id).status == "FAILED"
        assert not repo.list_reports(project.id)
        assert repo.list_analysis(project.id, "evaluation_failure")


def test_all_detected_faces_flow_from_front_metrics_to_shared_core(tmp_path, monkeypatch):
    import numpy as np

    import portrait_eval.iqa_bridge as bridge
    from portrait_eval.imaging import analyze_image

    class Cascade:
        def detectMultiScale(self, gray, **kwargs):
            return np.array([[10, 10, 30, 40], [65, 15, 25, 35], [-1, 0, 20, 20], [110, 0, 40, 30]])

    monkeypatch.setattr("portrait_eval.imaging.cv2.CascadeClassifier", lambda path: Cascade())
    image = Image.new("RGB", (128, 96), (180, 180, 180))
    from PIL import ImageDraw

    ImageDraw.Draw(image).rectangle((10, 10, 40, 50), fill=(40, 40, 40))
    paths, metrics = {}, {}
    for code in ("A", "B"):
        path = tmp_path / f"{code}.png"
        image.save(path)
        paths[code] = path
        metrics[code] = analyze_image(path)
        assert metrics[code]["face_bbox"] == [10, 10, 30, 40]
        assert len(metrics[code]["face_bboxes"]) == 2
        assert len(metrics[code]["per_face"]) == 2
        assert (
            metrics[code]["per_face"][0]["regions"]["face"]["luma_mean"]
            < metrics[code]["per_face"][1]["regions"]["face"]["luma_mean"]
        )
    captured = []
    real = bridge.evaluate_comparison

    def capture(request):
        captured.append(request)
        return real(request)

    monkeypatch.setattr(bridge, "evaluate_comparison", capture)
    evidence = bridge.evaluate_scene("multi", paths, metrics)
    assert len(captured[0].rois) == 4
    assert len({roi.person_id for roi in captured[0].rois}) == 4
    assert evidence["bridge_provenance"]["person_correspondence"] == "unverified"


def test_same_real_model_roles_do_not_create_independent_consensus(tmp_path):
    class ConfiguredAdapter(ObservationAdapter):
        model = "Same-Checkpoint"

    with db(tmp_path).session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        pipeline = EvaluationPipeline(
            session,
            tmp_path / "workspace",
            primary=ConfiguredAdapter(),
            reviewer=ConfiguredAdapter(),
        )
        pipeline.run(project.id)
        binding = repo.list_analysis(project.id, "evaluation_run_binding")[0]["payload"]
        assert not binding["independent_models"]
        assert binding["model_identities"]["primary"] == binding["model_identities"]["reviewer"]
        for row in repo.list_analysis(project.id, "claim"):
            finding = row["payload"]
            assert finding["model_agreement"] == 0
            assert not finding["independent_models"]
            assert finding["grade"] == "C"


def test_missing_primary_evidence_gets_no_model_consensus_credit(tmp_path):
    class MissingEvidence(ObservationAdapter):
        model = "primary-real"

        def analyze(self, scene_id, image_paths, context):
            result = super().analyze(scene_id, image_paths, context)
            result.observations[0].evidence_refs = []
            return result

    class Reviewer(ObservationAdapter):
        model = "reviewer-real"

    with db(tmp_path).session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        EvaluationPipeline(
            session, tmp_path / "workspace", primary=MissingEvidence(), reviewer=Reviewer()
        ).run(project.id)
        finding = repo.list_analysis(project.id, "claim")[0]["payload"]
        assert finding["independent_models"]
        assert finding["model_agreement"] == 0 and finding["grade"] == "C"


def test_reviewer_mixed_support_and_opposition_remains_conflicted(tmp_path):
    class Primary(ObservationAdapter):
        model = "primary-real"

    class Reviewer(ObservationAdapter):
        model = "reviewer-real"

        def analyze(self, scene_id, image_paths, context):
            result = super().analyze(scene_id, image_paths, context)
            opposite = result.observations[0].model_copy(
                update={"statement": "This output is darker."}
            )
            result.observations.append(opposite)
            return result

    with db(tmp_path).session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        EvaluationPipeline(
            session, tmp_path / "workspace", primary=Primary(), reviewer=Reviewer()
        ).run(project.id)
        review = next(
            item
            for item in repo.list_review_items(project.id)
            if item["category"] == "model_conflict"
        )
        assert review["payload"]["independent_decision"] == "opposition"
        assert repo.list_analysis(project.id, "claim")[0]["payload"]["model_agreement"] == 0
