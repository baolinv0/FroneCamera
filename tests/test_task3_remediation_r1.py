"""Critic3 R1 probes retained as fail-before regressions."""

from __future__ import annotations

import json

import pytest

from portrait_eval.model_validation import anonymous_context
from portrait_eval.models import ModelEvaluationResult, ModelObservation
from portrait_eval.pipeline import EvaluationPipeline
from portrait_eval.strategy import infer_cross_scene_claims
from tests.test_task3_runtime_pipeline import ObservationAdapter, db, setup_project

HIGHEST = "This output has the highest display-referred mean luminance in the matched group."
LOWEST = HIGHEST.replace("highest", "lowest")


def test_r1_01_measured_luminance_opposition_cannot_survive_as_supported_b(tmp_path):
    with db(tmp_path).session_factory() as session:
        _repo, project, pairing, _ = setup_project(tmp_path, session)
        group = pairing["groups"][0]
        first, second = list(group["cells"])

        def result(statement):
            return ModelEvaluationResult(
                scene_id=group["group_id"],
                role="probe",
                confidence=1,
                observations=[
                    ModelObservation(
                        device_id=first,
                        dimension="global_exposure",
                        statement=statement,
                        evidence_refs=[f"asset:{group['group_id']}:{first}"],
                        certainty=1,
                    )
                ],
            )

        pipeline = EvaluationPipeline(
            session,
            tmp_path / "workspace",
            primary=ObservationAdapter(),
            reviewer=ObservationAdapter(),
        )
        findings = pipeline._adjudicate_scene(
            project.id,
            group["id"],
            group["group_id"],
            {"status": "FULLY_COMPARABLE"},
            result(HIGHEST),
            result(LOWEST),
            result(LOWEST),
            {first: {"whole": {"luma_mean": 0.8}}, second: {"whole": {"luma_mean": 0.4}}},
        )
        assert findings[0]["grade"] == "C"
        assert findings[0]["status"] == "DISPUTED"
        assert findings[0]["contradicting_scene_ids"] == [group["group_id"]]


@pytest.mark.parametrize(
    "bad_finding",
    [
        {"statement": LOWEST, "grade": "C", "confidence": 0},
        {"statement": HIGHEST, "grade": "C", "confidence": 0},
        {"statement": HIGHEST, "grade": "B", "confidence": 1, "uncertainty": 0},
        {"statement": HIGHEST, "grade": "C", "confidence": 0.9, "status": "DISPUTED"},
    ],
)
def test_r1_02_invalid_direction_grade_uncertainty_claims_do_not_promote_a(bad_finding):
    scenes = [
        {
            "group_id": f"G{index:03}",
            "audit": {"status": "FULLY_COMPARABLE"},
            "metrics": {"D1": {"whole": {"luma_mean": 0.8}}, "D2": {"whole": {"luma_mean": 0.4}}},
            "findings": [
                {
                    "claim_id": f"bad-{index}",
                    "device_id": "D1",
                    "dimension": "global_exposure",
                    "model_agreement": 1,
                    "status": "SUPPORTED",
                    "independent_models": True,
                    "evidence_refs": [f"metric:G{index:03}:D1:luma_mean"],
                    **bad_finding,
                }
            ],
        }
        for index in range(3)
    ]
    claim = next(item for item in infer_cross_scene_claims(scenes) if item.device_id == "D1")
    assert claim.grade.value != "A"
    assert claim.confidence <= 0.7
    assert not claim.source_claim_ids


def test_r1_03_embedded_unc_paths_are_anonymized():
    result = anonymous_context(
        {"nested": [{"note": r"Cannot open \\secret-server\private-share\capture.jpg"}]}
    )
    text = json.dumps(result)
    assert "secret-server" not in text and "private-share" not in text and "capture.jpg" not in text


def test_r1_04_core_invalidity_blocks_model_and_scene_claims(tmp_path, monkeypatch):
    import portrait_eval.pipeline as module

    real = module.evaluate_scene

    def invalid(scene_id, paths, metrics):
        result = real(scene_id, paths, metrics)
        result["assets"][0]["state"] = "invalid"
        result["assets"][0]["reasons"] = ["image_decode_failed"]
        result["comparisons"][0]["fatal_reasons"] = ["invalid_baseline"]
        return result

    monkeypatch.setattr(module, "evaluate_scene", invalid)
    with db(tmp_path).session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        primary = ObservationAdapter()
        EvaluationPipeline(
            session, tmp_path / "workspace", primary=primary, reviewer=ObservationAdapter()
        ).run(project.id)
        assert primary.calls == 0
        assert (
            repo.list_analysis(project.id, "scene_audit")[0]["payload"]["status"]
            == "NOT_COMPARABLE"
        )
        assert not repo.list_analysis(project.id, "claim")
        assert not repo.list_analysis(project.id, "strategy_claim")
        assert any(
            item["category"] == "scene_not_comparable"
            for item in repo.list_review_items(project.id)
        )


def test_r1_04_actual_heic_captures_decode_in_both_front_and_shared_core(tmp_path):
    from PIL import Image

    from portrait_eval.repository import Repository

    database = db(tmp_path)
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("HEIC integration")
        for index, level in enumerate((100, 180)):
            folder = tmp_path / f"heic-device-{index}"
            folder.mkdir()
            Image.new("RGB", (96, 96), (level,) * 3).save(folder / "1.heic", format="HEIF")
            repo.add_device(project.id, f"HEIC device {index}", str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        EvaluationPipeline(session, tmp_path / "workspace").run(project.id)
        evidence = repo.list_analysis(project.id, "iqa_evaluation")[0]["payload"]
        assert all(asset["state"] == "valid" for asset in evidence["assets"])
        assert repo.list_analysis(project.id, "scene_audit")[0]["payload"]["iqa_evidence"]["valid"]


def test_r1_04_core_dimension_unobservability_reaches_model_context_and_claim(tmp_path):
    class SkinAdapter(ObservationAdapter):
        def __init__(self, model):
            super().__init__("Skin white balance is accurate.", certainty=1)
            self.model = model
            self.contexts = []

        def analyze(self, scene_id, image_paths, context):
            self.contexts.append(context)
            result = super().analyze(scene_id, image_paths, context)
            result.observations[0].dimension = "skin_awb"
            return result

    with db(tmp_path).session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        primary, reviewer = SkinAdapter("primary-real"), SkinAdapter("reviewer-real")
        EvaluationPipeline(session, tmp_path / "workspace", primary=primary, reviewer=reviewer).run(
            project.id
        )
        finding = repo.list_analysis(project.id, "claim")[0]["payload"]
        assert finding["confidence"] == 0 and finding["uncertainty"] == 0
        assert finding["iqa_evidence"]["state"] == "unobservable"
        assert any(
            item["category"] == "iqa_dimension_unobservable"
            for item in repo.list_review_items(project.id)
        )
        blind, validated = primary.contexts
        assert (
            blind["iqa_evidence"]["assets"][0]["dimensions"]["skin_awb"]["state"] == "unobservable"
        )
        assert "facts" not in blind["iqa_evidence"]["assets"][0]["dimensions"]["skin_awb"]
        assert "objective" in validated["iqa_evidence"]["assets"][0]
        assert all("path" not in asset for asset in validated["iqa_evidence"]["assets"])


def test_r1_02_positive_resolved_directional_sources_preserve_bounded_confidence():
    scenes = [
        {
            "group_id": f"G{index:03}",
            "audit": {"status": "FULLY_COMPARABLE"},
            "metrics": {"D1": {"whole": {"luma_mean": 0.8}}, "D2": {"whole": {"luma_mean": 0.4}}},
            "findings": [
                {
                    "claim_id": f"good-{index}",
                    "device_id": "D1",
                    "dimension": "global_exposure",
                    "statement": HIGHEST,
                    "model_agreement": 0.9,
                    "confidence": 0.9,
                    "uncertainty": 0.9,
                    "grade": "B",
                    "status": "SUPPORTED",
                    "independent_models": True,
                    "evidence_refs": [f"metric:G{index:03}:D1:luma_mean"],
                }
            ],
        }
        for index in range(3)
    ]
    claim = next(item for item in infer_cross_scene_claims(scenes) if item.device_id == "D1")
    assert 0 < claim.confidence < 1
    assert claim.source_claim_ids == ["good-0", "good-1", "good-2"]


@pytest.mark.parametrize(
    "primary_statement,reviewer_statement", [(HIGHEST, LOWEST), (LOWEST, LOWEST)]
)
def test_r1_semantic_conflicts_and_opposite_sources_in_actual_three_scene_report(
    tmp_path, monkeypatch, primary_statement, reviewer_statement
):
    from pathlib import Path

    from PIL import Image, ImageStat

    from portrait_eval.repository import Repository

    monkeypatch.setattr("portrait_eval.imaging.detect_faces", lambda image: [[12, 12, 45, 55]])

    class RealConfiguredAdapter(ObservationAdapter):
        def __init__(self, model, statement):
            super().__init__(statement, certainty=1)
            self.model = model

        def analyze(self, scene_id, image_paths, context):
            def brightness(code):
                with Image.open(image_paths[code]) as image:
                    return ImageStat.Stat(image.convert("RGB")).mean[0]

            code = max(image_paths, key=brightness)
            return ModelEvaluationResult(
                scene_id=scene_id,
                role="probe",
                confidence=1,
                observations=[
                    ModelObservation(
                        device_id=code,
                        dimension="global_exposure",
                        statement=self.statement,
                        evidence_refs=[f"asset:{scene_id}:{code}"],
                        certainty=1,
                    )
                ],
            )

    with db(tmp_path).session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("actual three scene conflict")
        for index, level in enumerate((100, 180)):
            folder = tmp_path / f"three-scene-{index}"
            folder.mkdir()
            for scene in range(1, 4):
                Image.new("RGB", (96, 96), (level,) * 3).save(folder / f"{scene}.png")
            repo.add_device(project.id, f"Device {index}", str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        result = EvaluationPipeline(
            session,
            tmp_path / "workspace",
            primary=RealConfiguredAdapter("primary-real", primary_statement),
            reviewer=RealConfiguredAdapter("reviewer-real", reviewer_statement),
        ).run(project.id)
        claims = [item["payload"] for item in repo.list_analysis(project.id, "claim")]
        assert len(claims) == 3 and all(claim["grade"] == "C" for claim in claims)
        strategies = [item["payload"] for item in repo.list_analysis(project.id, "strategy_claim")]
        assert strategies and all(
            claim["grade"] != "A" and not claim["source_claim_ids"] for claim in strategies
        )
        payload = json.loads(Path(result["json_path"]).read_text())
        assert not any(
            item["claim_id"] in {claim["claim_id"] for claim in claims}
            for item in payload["findings"]
        )
        if primary_statement == HIGHEST:
            assert all(
                claim["status"] == "DISPUTED" and claim["confidence"] == 0 for claim in claims
            )


@pytest.mark.parametrize("transport", ["compatible", "responses"])
def test_r1_03_actual_transport_prompts_redact_unc_and_serialized_context(
    tmp_path, monkeypatch, transport
):
    import httpx
    from PIL import Image

    from portrait_eval.vlm import OpenAICompatibleVisionAdapter, OpenAIResponsesVisionAdapter

    path = tmp_path / "asset.png"
    Image.new("RGB", (32, 32), "gray").save(path)
    captured = {}
    output = json.dumps({"observations": [], "scores": {}, "hypotheses": [], "confidence": 0.5})

    def handle(request):
        captured.update(json.loads(request.content))
        response = (
            {"choices": [{"message": {"content": output}}]}
            if transport == "compatible"
            else {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": output}]}
                ]
            }
        )
        return httpx.Response(200, json=response)

    unc = r"Cannot open \\secret-server\private-share\capture.jpg"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("portrait_eval.vlm.httpx.post", client.post)
        adapter = (
            OpenAICompatibleVisionAdapter("https://model.test", "probe", "primary")
            if transport == "compatible"
            else OpenAIResponsesVisionAdapter(
                "test-key", "probe", "primary", base_url="https://model.test"
            )
        )
        result = adapter.analyze(
            "G001",
            {"A": path},
            {
                "nested": [{"note": unc, "serialized_inter_stage": json.dumps({"detail": unc})}],
                "metrics": {"A": {"whole": {"luma_mean": 0.25}}},
            },
        )
    prompt = (captured["messages"] if transport == "compatible" else captured["input"])[0][
        "content"
    ][0]["text"]
    assert (
        "secret-server" not in prompt
        and "private-share" not in prompt
        and "capture.jpg" not in prompt
    )
    assert "0.25" in prompt and "luma_mean" in prompt
    assert result.input_trace["prompt"] == prompt


def test_r1_02_actual_positive_pipeline_restores_evidence_ids_and_uses_only_resolved_claims(
    tmp_path, monkeypatch
):
    from PIL import Image, ImageStat

    from portrait_eval.repository import Repository

    monkeypatch.setattr("portrait_eval.imaging.detect_faces", lambda image: [[12, 12, 45, 55]])

    class PositiveAdapter(ObservationAdapter):
        def __init__(self, model):
            super().__init__(HIGHEST, certainty=1)
            self.model = model

        def analyze(self, scene_id, image_paths, context):
            def brightness(code):
                with Image.open(image_paths[code]) as image:
                    return ImageStat.Stat(image.convert("RGB")).mean[0]

            code = max(image_paths, key=brightness)
            refs = (
                [f"metric:{scene_id}:{code}:luma_mean"]
                if "metrics" in context
                else [f"asset:{scene_id}:{code}"]
            )
            return ModelEvaluationResult(
                scene_id=scene_id,
                role="positive",
                confidence=1,
                observations=[
                    ModelObservation(
                        device_id=code,
                        dimension="global_exposure",
                        statement=HIGHEST,
                        evidence_refs=refs,
                        certainty=1,
                    )
                ],
            )

    with db(tmp_path).session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("positive three-scene metric evidence")
        for index, level in enumerate((100, 180)):
            folder = tmp_path / f"positive-{index}"
            folder.mkdir()
            for scene in range(1, 4):
                Image.new("RGB", (96, 96), (level,) * 3).save(folder / f"{scene}.png")
            repo.add_device(project.id, f"Device {index}", str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        EvaluationPipeline(
            session,
            tmp_path / "workspace",
            primary=PositiveAdapter("primary-positive"),
            reviewer=PositiveAdapter("reviewer-positive"),
        ).run(project.id)
        observations = [item["payload"] for item in repo.list_analysis(project.id, "claim")]
        assert len(observations) == 3 and all(item["grade"] == "B" for item in observations)
        assert all(item["device_id"] in item["evidence_refs"][0] for item in observations)
        strategy = repo.list_analysis(project.id, "strategy_claim")[0]["payload"]
        assert 0.8 < strategy["confidence"] < 1
        assert set(strategy["source_claim_ids"]) == {item["claim_id"] for item in observations}
