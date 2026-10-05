import base64
import io
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from qwen_tmqa.config import JudgeConfig, TMQAConfig, load_config
from qwen_tmqa.dataset import discover_scenes
from qwen_tmqa.domain import ModelEvaluation, ModelIssue
from qwen_tmqa.evaluation import EvaluationPipeline, model_score_gap
from qwen_tmqa.judges.openai_compatible import OpenAICompatibleJudge, _image_data_url
from qwen_tmqa.prompts import PromptRegistry, render_prompt


def _save(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.05, value, 40, dtype=np.float32)
    image = np.tile(x[None, :, None], (28, 1, 3))
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def _dataset(root: Path) -> None:
    levels = [
        ("a_m100", -1.0, 0.25),
        ("a_m075", -0.75, 0.32),
        ("a_m050", -0.5, 0.39),
        ("a_m025", -0.25, 0.46),
        ("a_000", 0.0, 0.53),
        ("a_p025", 0.25, 0.60),
        ("a_p050", 0.5, 0.67),
        ("a_p075", 0.75, 0.80),
        ("a_p100", 1.0, 1.0),
    ]
    for level, _, value in levels:
        _save(root / level / "scene.png", value)


def test_pipeline_persists_four_independent_model_results(tmp_path: Path) -> None:
    _dataset(tmp_path)
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(tmp_path, config.dataset)[0]
    result = EvaluationPipeline(config).evaluate_scene(scene)

    assert len(result.model_evaluations) == 4
    assert {item.model_id for item in result.model_evaluations} == {
        "minicpm-prescreener",
        "qwen-primary",
        "internvl-arbiter",
        "ovis-local",
    }
    assert all(item.prompt_trace.rendered_prompt for item in result.model_evaluations)
    assert all(item.prompt_trace.input_manifest for item in result.model_evaluations)
    assert all(item.synthetic for item in result.model_evaluations)
    assert model_score_gap(result.model_evaluations, "overall") > 0
    assert result.stages[-1].stage_id == "model_judges"



def test_risky_highlight_scene_creates_reviewable_model_disagreement(tmp_path: Path) -> None:
    _dataset(tmp_path)
    Image.fromarray(np.full((28, 40, 3), 255, dtype=np.uint8)).save(tmp_path / "a_p100" / "scene.png")
    Image.fromarray(np.full((28, 40, 3), 248, dtype=np.uint8)).save(tmp_path / "a_p075" / "scene.png")
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(tmp_path, config.dataset)[0]
    result = EvaluationPipeline(config).evaluate_scene(scene)
    assert model_score_gap(result.model_evaluations, "overall") >= config.review.disagreement_threshold
    assert "model_disagreement" in result.review_reasons


def test_openai_adapter_retries_malformed_json_and_preserves_raw_response(tmp_path: Path) -> None:
    _dataset(tmp_path)
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(tmp_path, config.dataset)[0]
    trace = render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version="3.2",
        scene=scene,
        model_role="primary",
        objective_evidence={},
        inference_parameters={},
    )
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
        payload = {
            "scores": {
                "tone": 0.8,
                "color": 0.7,
                "fidelity": 0.9,
                "control": 0.85,
                "preference": 0.75,
                "overall": 0.8,
            },
            "decision": "KEEP",
            "confidence": 0.82,
            "issues": [],
            "rationale": "acceptable",
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": f"```json\n{__import__('json').dumps(payload)}\n```"
                        }
                    }
                ]
            },
        )

    judge_config = JudgeConfig(
        id="real",
        role="primary",
        adapter="openai_compatible",
        model="model",
        version="v1",
        base_url="http://test/v1",
    )
    judge = OpenAICompatibleJudge(judge_config, transport=httpx.MockTransport(handler))
    evaluation = judge.evaluate(trace)

    assert calls == 2
    assert evaluation.available is True
    assert "```json" in evaluation.raw_response
    assert evaluation.scores["overall"] == 0.8
    assert evaluation.prompt_trace.prompt_hash == trace.prompt_hash


def test_openai_adapter_normalizes_prompt_documented_100_point_scores(
    tmp_path: Path,
) -> None:
    _dataset(tmp_path)
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(tmp_path, config.dataset)[0]
    trace = render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version="3.3",
        scene=scene,
        model_role="primary",
        objective_evidence={},
        inference_parameters={},
    )

    def handler(_: httpx.Request) -> httpx.Response:
        payload = {
            "scores": {
                "tone": 80,
                "color": 70,
                "fidelity": 90,
                "control": 85,
                "preference": 75,
                "overall": 80,
            },
            "decision": "KEEP",
            "confidence": 82,
            "issues": [],
            "rationale": "acceptable",
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": __import__("json").dumps(payload)}}]},
        )

    judge = OpenAICompatibleJudge(
        JudgeConfig(
            id="hundred-point",
            role="primary",
            adapter="openai_compatible",
            model="model",
            base_url="http://test/v1",
        ),
        transport=httpx.MockTransport(handler),
    )

    evaluation = judge.evaluate(trace)

    assert evaluation.available is True
    assert evaluation.scores["overall"] == pytest.approx(0.8)
    assert evaluation.confidence == pytest.approx(0.82)


def test_model_failure_is_isolated_from_other_judges(tmp_path: Path) -> None:
    _dataset(tmp_path)
    config = TMQAConfig(
        judges=[
            JudgeConfig(
                id="bad",
                role="primary",
                adapter="openai_compatible",
                model="bad",
                base_url="http://test/v1",
            ),
            JudgeConfig(
                id="good",
                role="arbiter",
                adapter="mock",
                model="good",
                synthetic=True,
            ),
        ]
    )
    scene = discover_scenes(tmp_path, config.dataset)[0]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    pipeline = EvaluationPipeline(config, transports={"bad": httpx.MockTransport(handler)})
    result = pipeline.evaluate_scene(scene)
    assert len(result.model_evaluations) == 2
    bad = next(item for item in result.model_evaluations if item.model_id == "bad")
    good = next(item for item in result.model_evaluations if item.model_id == "good")
    assert bad.available is False and bad.error
    assert good.available is True


def test_openai_adapter_isolates_schema_invalid_json(tmp_path: Path) -> None:
    _dataset(tmp_path)
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(tmp_path, config.dataset)[0]
    trace = render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version="3.2",
        scene=scene,
        model_role="primary",
        objective_evidence={},
        inference_parameters={},
    )

    def handler(_: httpx.Request) -> httpx.Response:
        payload = {
            "scores": {"overall": 0.8},
            "decision": "MAYBE",
            "confidence": 0.8,
            "issues": [],
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": __import__("json").dumps(payload)}}
                ]
            },
        )

    judge_config = JudgeConfig(
        id="invalid-schema",
        role="primary",
        adapter="openai_compatible",
        model="model",
        version="v1",
        base_url="http://test/v1",
    )
    result = OpenAICompatibleJudge(
        judge_config,
        transport=httpx.MockTransport(handler),
    ).evaluate(trace)
    assert result.available is False
    assert result.decision == "REVIEW"
    assert "validation" in (result.error or "").lower()


def test_openai_adapter_rejects_incomplete_score_contract(tmp_path: Path) -> None:
    _dataset(tmp_path)
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(tmp_path, config.dataset)[0]
    trace = render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version="3.2",
        scene=scene,
        model_role="primary",
        objective_evidence={},
        inference_parameters={},
    )

    def handler(_: httpx.Request) -> httpx.Response:
        payload = {
            "scores": {"overall": 0.8},
            "decision": "KEEP",
            "confidence": 0.8,
            "issues": [],
            "rationale": "missing attribute scores",
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": __import__("json").dumps(payload)}}]},
        )

    judge = OpenAICompatibleJudge(
        JudgeConfig(
            id="incomplete",
            role="primary",
            adapter="openai_compatible",
            model="model",
            base_url="http://test/v1",
        ),
        transport=httpx.MockTransport(handler),
    )
    result = judge.evaluate(trace)

    assert result.available is False
    assert "missing required score keys" in (result.error or "").lower()


def test_openai_image_payload_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    array = np.zeros((20, 12, 3), dtype=np.uint8)
    array[:10, :, 0] = 255
    array[10:, :, 2] = 255
    image = Image.fromarray(array)
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)

    url = _image_data_url(str(path), width=20, height=12)
    payload = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as sent:
        assert sent.size == (20, 12)
        sent_array = np.asarray(sent.convert("RGB"), dtype=np.float32)
    horizontal_change = np.mean(
        np.abs(sent_array[:, :10].mean(axis=(0, 1)) - sent_array[:, 10:].mean(axis=(0, 1)))
    )
    vertical_change = np.mean(
        np.abs(sent_array[:6].mean(axis=(0, 1)) - sent_array[6:].mean(axis=(0, 1)))
    )
    assert horizontal_change > vertical_change * 5


def test_model_reported_fatal_issue_is_a_non_compensable_hard_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _dataset(tmp_path)
    config = TMQAConfig(
        judges=[JudgeConfig(id="fatal", role="primary", model="fatal", synthetic=True)]
    )
    scene = discover_scenes(tmp_path, config.dataset)[0]

    class FatalJudge:
        def evaluate(self, trace):
            return ModelEvaluation(
                model_id="fatal",
                model_role="primary",
                model_version="v1",
                synthetic=True,
                prompt_trace=trace,
                scores={"overall": 0.99},
                decision="REJECT",
                confidence=0.99,
                issues=[
                    ModelIssue(
                        dimension="identity",
                        severity=0.95,
                        description="face identity changed",
                        fatal=True,
                        region="face",
                    )
                ],
            )

    monkeypatch.setattr(
        "qwen_tmqa.evaluation._judge_from_config",
        lambda _config, _transport: FatalJudge(),
    )
    result = EvaluationPipeline(config).evaluate_scene(scene)

    assert result.decision == "REJECT"
    assert "fatal_risk" in result.review_reasons
    assert next(stage for stage in result.stages if stage.stage_id == "hard_gate").status == "FAIL"


def test_conflicting_model_decisions_force_human_review_even_when_scores_agree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _dataset(tmp_path)
    config = TMQAConfig(
        judges=[
            JudgeConfig(id="keep", role="primary", model="keep", synthetic=True),
            JudgeConfig(id="redo", role="arbiter", model="redo", synthetic=True),
        ]
    )
    scene = discover_scenes(tmp_path, config.dataset)[0]

    class DecisionJudge:
        def __init__(self, model_id: str):
            self.model_id = model_id

        def evaluate(self, trace):
            decision = "KEEP" if self.model_id == "keep" else "REGENERATE"
            return ModelEvaluation(
                model_id=self.model_id,
                model_role="primary" if self.model_id == "keep" else "arbiter",
                model_version="v1",
                synthetic=True,
                prompt_trace=trace,
                scores={"overall": 0.90},
                decision=decision,
                confidence=0.90,
            )

    monkeypatch.setattr(
        "qwen_tmqa.evaluation._judge_from_config",
        lambda judge_config, _transport: DecisionJudge(judge_config.id),
    )
    result = EvaluationPipeline(config).evaluate_scene(scene)

    assert result.decision == "REVIEW"
    assert "model_decision_disagreement" in result.review_reasons


def _openai_contract_trace(tmp_path: Path):
    _dataset(tmp_path)
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(tmp_path, config.dataset)[0]
    return render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version="3.3",
        scene=scene,
        model_role="primary",
        objective_evidence={},
        inference_parameters={},
    )


def _complete_openai_payload() -> dict[str, object]:
    return {
        "scores": {
            "tone": 0.8,
            "color": 0.8,
            "fidelity": 0.8,
            "control": 0.8,
            "preference": 0.8,
            "overall": 0.8,
        },
        "decision": "KEEP",
        "confidence": 0.9,
        "issues": [],
        "rationale": "Complete evidence-based response.",
    }


@pytest.mark.parametrize(
    "missing_field",
    ["scores", "decision", "confidence", "issues", "rationale"],
)
def test_openai_adapter_fails_closed_for_each_missing_required_top_level_field(
    tmp_path: Path,
    missing_field: str,
) -> None:
    trace = _openai_contract_trace(tmp_path)
    payload = _complete_openai_payload()
    payload.pop(missing_field)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": __import__("json").dumps(payload)}}
                ]
            },
        )

    judge = OpenAICompatibleJudge(
        JudgeConfig(
            id="missing-field",
            role="primary",
            adapter="openai_compatible",
            model="model",
            base_url="http://test/v1",
        ),
        transport=httpx.MockTransport(handler),
    )

    result = judge.evaluate(trace)

    assert result.available is False
    assert missing_field in (result.error or "")


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("decision", "MAYBE"),
        ("confidence", "high"),
        ("issues", {}),
        ("rationale", None),
        ("rationale", ""),
    ],
)
def test_openai_adapter_fails_closed_for_invalid_required_top_level_field(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    trace = _openai_contract_trace(tmp_path)
    payload = _complete_openai_payload()
    payload[field] = invalid_value

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": __import__("json").dumps(payload)}}
                ]
            },
        )

    judge = OpenAICompatibleJudge(
        JudgeConfig(
            id="invalid-field",
            role="primary",
            adapter="openai_compatible",
            model="model",
            base_url="http://test/v1",
        ),
        transport=httpx.MockTransport(handler),
    )

    result = judge.evaluate(trace)

    assert result.available is False
    assert "validation" in (result.error or "").lower()


@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("response_kind", ["success", "schema_failure", "request_failure"])
def test_openai_adapter_preserves_configured_synthetic_lineage(
    tmp_path: Path,
    synthetic: bool,
    response_kind: str,
) -> None:
    trace = _openai_contract_trace(tmp_path)

    def handler(_: httpx.Request) -> httpx.Response:
        if response_kind == "request_failure":
            return httpx.Response(503, text="unavailable")
        payload = _complete_openai_payload()
        if response_kind == "schema_failure":
            payload.pop("decision")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": __import__("json").dumps(payload)}}
                ]
            },
        )

    result = OpenAICompatibleJudge(
        JudgeConfig(
            id="lineage",
            role="primary",
            adapter="openai_compatible",
            model="model",
            base_url="http://test/v1",
            synthetic=synthetic,
        ),
        transport=httpx.MockTransport(handler),
    ).evaluate(trace)

    assert result.synthetic is synthetic
    assert result.available is (response_kind == "success")


def test_stricter_configured_keep_threshold_controls_decision_and_is_recorded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _dataset(tmp_path)
    config = TMQAConfig.model_validate(
        {
            "judges": [
                JudgeConfig(id="fixed", role="primary", model="fixed", synthetic=True)
            ],
            "decision_policy": {"keep_score_threshold": 0.85},
        }
    )
    scene = discover_scenes(tmp_path, config.dataset)[0]

    class FixedJudge:
        def evaluate(self, trace):
            return ModelEvaluation(
                model_id="fixed",
                model_role="primary",
                model_version="v1",
                synthetic=True,
                prompt_trace=trace,
                scores={
                    "tone": 0.8,
                    "color": 0.8,
                    "fidelity": 0.8,
                    "control": 0.8,
                    "preference": 0.8,
                    "overall": 0.8,
                },
                decision="KEEP",
                confidence=0.9,
            )

    monkeypatch.setattr(
        "qwen_tmqa.evaluation._judge_from_config",
        lambda _config, _transport: FixedJudge(),
    )

    result = EvaluationPipeline(config).evaluate_scene(scene)

    assert result.decision == "REVIEW"
    assert result.decision_provenance["version"] == config.decision_policy.version
    assert result.decision_provenance["thresholds"]["keep_score_threshold"] == 0.85
    for stage in result.stages:
        assert f"decision_policy={config.decision_policy.version}" in stage.evidence


def test_all_unavailable_judges_cannot_silently_produce_keep(tmp_path: Path) -> None:
    _dataset(tmp_path)
    config = TMQAConfig(
        judges=[
            JudgeConfig(
                id="offline",
                role="primary",
                adapter="openai_compatible",
                model="offline",
                base_url="http://test/v1",
            )
        ]
    )
    scene = discover_scenes(tmp_path, config.dataset)[0]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="offline")

    result = EvaluationPipeline(
        config,
        transports={"offline": httpx.MockTransport(handler)},
    ).evaluate_scene(scene)

    assert result.decision == "REVIEW"
    assert "judge_unavailable" in result.review_reasons
    judge_stage = next(stage for stage in result.stages if stage.stage_id == "model_judges")
    assert judge_stage.status == "WARN"
