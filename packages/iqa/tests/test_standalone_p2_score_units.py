"""Provider units must not depend on the magnitude of a response value."""

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from qwen_tmqa.config import JudgeConfig, TMQAConfig
from qwen_tmqa.domain import AlphaImage, ModelEvaluation, SceneSpec
from qwen_tmqa.evaluation import EvaluationPipeline
from qwen_tmqa.image_io import encode_image_payload, file_sha256
from qwen_tmqa.judges.openai_compatible import OpenAICompatibleJudge
from qwen_tmqa.prompts import PromptRegistry, render_prompt

_SCORE_KEYS = ("tone", "color", "fidelity", "control", "preference", "overall")


@pytest.fixture
def scene(tmp_path: Path) -> SceneSpec:
    # Identical, unclipped gradients keep the objective hard gate independent
    # of the provider's scores and decision.
    pixels = np.tile(np.linspace(40, 180, 24, dtype=np.uint8)[None, :, None], (16, 1, 3))
    images = []
    for level, alpha in [("a_m100", -1), ("a_000", 0), ("a_p100", 1)]:
        path = tmp_path / f"{level}.png"
        Image.fromarray(pixels).save(path)
        images.append(AlphaImage(level=level, alpha=alpha, path=path))
    return SceneSpec(scene_id="units", baseline_path=images[1].path, alpha_images=images)


def _payload(version: str, *, score: float = 80, confidence: float = 90) -> dict:
    payload = {
        "scores": dict.fromkeys(_SCORE_KEYS, score),
        "decision": "KEEP",
        "confidence": confidence,
        "issues": [{"dimension": "tone", "severity": 1, "description": "visible issue"}],
        "rationale": "provider unit regression",
    }
    if version == "3.4":
        payload.update(
            preferred_level="a_p100",
            runner_up_level="a_m100",
            acceptable_levels=["a_p100"],
            level_scores={"a_m100": 0.6, "a_000": 0, "a_p100": 1},
            selection_confidence=1,
            baseline_improvement=1,
        )
    return payload


def _judge_config(version: str) -> JudgeConfig:
    return JudgeConfig(
        id="provider",
        role="primary",
        adapter="openai_compatible",
        model="units",
        base_url="http://test/v1",
        prompt_version=version,
    )


def _transport(raw: str) -> httpx.MockTransport:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})

    return httpx.MockTransport(handler)


def _evaluate(scene: SceneSpec, version: str, payload: dict) -> ModelEvaluation:
    trace = render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version=version,
        scene=scene,
        model_role="primary",
        objective_evidence={},
        inference_parameters={},
    )
    raw = f"```json\n{json.dumps(payload, indent=2)}\n```"
    return OpenAICompatibleJudge(_judge_config(version), transport=_transport(raw)).evaluate(trace)


@pytest.mark.parametrize("version", ["3.3", "3.4"])
@pytest.mark.parametrize("value", [0, 0.25, 1, 2, 80, 100])
def test_percent_scores_use_declared_units_for_every_value(scene, version, value):
    payload = _payload(version, score=value)
    result = _evaluate(scene, version, payload)
    assert result.available, result.error
    assert result.scores == dict.fromkeys(_SCORE_KEYS, value / 100)
    assert result.raw_response == f"```json\n{json.dumps(payload, indent=2)}\n```"
    assert result.parsed_response == payload
    assert result.issues[0].severity == 1


@pytest.mark.parametrize("version", ["3.3", "3.4"])
@pytest.mark.parametrize("value", [0, 0.25, 1, 2, 90, 100])
def test_percent_confidence_uses_declared_units_for_every_value(scene, version, value):
    payload = _payload(version, confidence=value)
    result = _evaluate(scene, version, payload)
    assert result.available, result.error
    assert result.confidence == value / 100
    assert result.parsed_response["confidence"] == value


def test_candidate_selection_fields_keep_their_declared_unit_interval(scene):
    payload = _payload("3.4", score=1, confidence=1)
    result = _evaluate(scene, "3.4", payload)
    assert result.available, result.error
    assert result.scores["overall"] == 0.01
    assert result.confidence == 0.01
    assert result.parsed_response["level_scores"] == payload["level_scores"]
    assert result.parsed_response["selection_confidence"] == 1
    assert result.parsed_response["baseline_improvement"] == 1


@pytest.mark.parametrize("version", ["3.3", "3.4"])
@pytest.mark.parametrize(
    ("score", "expected_decision"),
    [(0, "REGENERATE"), (1, "REGENERATE"), (2, "REGENERATE"), (80, "KEEP"), (100, "KEEP")],
)
def test_pipeline_decision_uses_percent_scores_without_one_point_discontinuity(
    scene, version, score, expected_decision
):
    payload = _payload(version, score=score, confidence=1)
    config = TMQAConfig(judges=[_judge_config(version)])
    result = EvaluationPipeline(
        config, transports={"provider": _transport(json.dumps(payload))}
    ).evaluate_scene(scene)
    assert next(stage for stage in result.stages if stage.stage_id == "hard_gate").status == "PASS"
    assert result.model_evaluations[0].available
    assert result.overall_score == score / 100
    assert result.model_evaluations[0].confidence == 0.01
    assert result.decision == expected_decision


@pytest.mark.parametrize("version", ["3.3", "3.4"])
@pytest.mark.parametrize("field", ["score", "confidence"])
@pytest.mark.parametrize("value", [-1, 101, True, "1", float("nan"), float("inf")])
def test_percent_boundary_keeps_invalid_provider_values_unavailable(scene, version, field, value):
    payload = _payload(version)
    if field == "score":
        payload["scores"]["overall"] = value
    else:
        payload["confidence"] = value
    result = _evaluate(scene, version, payload)
    assert not result.available
    assert result.decision == "REVIEW"
    assert result.error


@pytest.mark.parametrize(
    "value, expected", [(0, 0), (0.8, 0.8), (1, 1), (2, 0.02), (80, 0.8), (100, 1)]
)
def test_historical_v3_keeps_existing_provider_compatibility(scene, value, expected):
    # Version 3.2 does not declare units; preserve its historical compatibility
    # rather than retroactively treating its normalized responses as percent.
    result = _evaluate(scene, "3.2", _payload("3.2", score=value, confidence=value))
    assert result.available, result.error
    assert result.scores == dict.fromkeys(_SCORE_KEYS, expected)
    assert result.confidence == expected


@pytest.mark.parametrize("value", [0, 0.01, 0.8, 1])
def test_persisted_internal_scores_are_not_reinterpreted_as_provider_percent(scene, value):
    result = _evaluate(scene, "3.3", _payload("3.3", score=value * 100, confidence=value * 100))
    assert result.available, result.error
    stored = result.model_dump(mode="json")
    assert stored["scores"] == dict.fromkeys(_SCORE_KEYS, value)
    assert stored["confidence"] == value
    loaded = ModelEvaluation.model_validate_json(json.dumps(stored))
    assert loaded.scores == dict.fromkeys(_SCORE_KEYS, value)
    assert loaded.confidence == value
    assert loaded.parsed_response == result.parsed_response


def test_direct_adapter_rejects_changed_source_even_when_encoded_payload_is_identical(scene):
    trace = render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version="3.3",
        scene=scene,
        model_role="primary",
        objective_evidence={},
        inference_parameters={},
    )
    item = trace.input_manifest[0]
    path = Path(item.path)
    path.write_bytes(path.read_bytes() + b"source changed without changing PNG pixels")
    assert file_sha256(path) != item.source_sha256
    assert encode_image_payload(path, item.sent_width, item.sent_height).sha256 == item.payload_sha256
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(_payload("3.3"))}}]}
        )

    result = OpenAICompatibleJudge(
        _judge_config("3.3"), transport=httpx.MockTransport(handler)
    ).evaluate(trace)
    assert not result.available
    assert result.decision == "REVIEW"
    assert "source hash mismatch" in result.error
    assert requests == []
