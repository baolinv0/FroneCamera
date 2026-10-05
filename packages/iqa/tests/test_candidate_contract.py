import json

import httpx
import pytest
from test_candidate_selection_integrity import _payload, _spec, _trace

from qwen_tmqa.config import JudgeConfig, TMQAConfig
from qwen_tmqa.judges.openai_compatible import OpenAICompatibleJudge
from qwen_tmqa.prompts import PromptRegistry


def response():
    return {
        "scores": dict.fromkeys(
            ["tone", "color", "fidelity", "control", "preference", "overall"], 80
        ),
        "decision": "KEEP",
        "confidence": 90,
        "issues": [],
        "rationale": "better rendering",
        **_payload(),
    }


def evaluate(tmp_path, payload, version="3.4"):
    trace = _trace(_spec(tmp_path))
    trace.prompt_version = version
    trace.output_schema_version = (
        PromptRegistry.default().get("tmqa.sequence", version).output_schema_version
    )
    config = JudgeConfig(
        id="judge",
        role="primary",
        adapter="openai_compatible",
        model="real",
        base_url="http://test/v1",
        prompt_version=version,
    )

    def transport(_):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    return OpenAICompatibleJudge(config, transport=httpx.MockTransport(transport)).evaluate(trace)


def test_new_explicit_prompt_contract_exists_without_changing_historical_versions():
    registry = PromptRegistry.default()
    assert "preferred_level" not in registry.get("tmqa.sequence", "3.3").text
    assert "preferred_level" in registry.get("tmqa.sequence", "3.4").text
    assert registry.get("tmqa.sequence", "3.4").output_schema_version == "tmqa.sequence.v5"


def test_new_contract_round_trips_complete_candidate_preference(tmp_path):
    result = evaluate(tmp_path, response())
    assert result.available
    assert result.parsed_response["preferred_level"] == "a_p050"
    assert result.scores["overall"] == 0.8


@pytest.mark.parametrize(
    "change",
    [
        {"level_scores": {"a_000": 0.7}},
        {"baseline_improvement": 0.5},
        {"preferred_level": "unknown"},
        {"unexpected": True},
        {
            "scores": {
                "tone": 80,
                "color": 80,
                "fidelity": 80,
                "control": 80,
                "preference": 80,
                "overall": 80,
                "nonexistent": 90,
            }
        },
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"confidence": 999},
        {"confidence": True},
        {"confidence": "90"},
    ],
)
def test_new_contract_rejects_missing_invalid_or_extra_evidence(tmp_path, change):
    result = evaluate(tmp_path, {**response(), **change})
    assert not result.available
    assert result.decision == "REVIEW"
    assert result.error


def test_duplicate_judge_config_and_blank_identity_are_rejected():
    judge = JudgeConfig(id="same", role="primary", model="m")
    with pytest.raises(ValueError, match="unique"):
        TMQAConfig(judges=[judge, judge.model_copy()])
    with pytest.raises(ValueError):
        JudgeConfig(id=" ", role="primary", model="m")


@pytest.mark.parametrize(
    "bbox",
    [
        [float("nan"), 0, 1, 1],
        [0, float("inf"), 1, 1],
        [0, 0, float("-inf"), 1],
        ["0.1", 0, 1, 1],
        [0, True, 1, 1],
        [0, 0, 1],
        {"x": 0, "y": 0, "w": 1, "h": 1},
    ],
)
def test_runtime_issue_bbox_rejects_nonfinite_coerced_or_malformed_coordinates(tmp_path, bbox):
    payload = response()
    payload["issues"] = [
        {
            "dimension": "fidelity",
            "severity": 0.2,
            "description": "local detail issue",
            "bbox": bbox,
        }
    ]
    result = evaluate(tmp_path, payload)
    assert not result.available
    assert result.decision == "REVIEW"
    assert "bbox" in result.error


@pytest.mark.parametrize("bbox", [None, [0, 0.1, 0.5, 0.6], [1, 2, 3, 4]])
def test_runtime_issue_bbox_preserves_valid_finite_boxes_and_explicit_missingness(tmp_path, bbox):
    payload = response()
    payload["issues"] = [
        {
            "dimension": "fidelity",
            "severity": 0.2,
            "description": "local detail issue",
            "bbox": bbox,
        }
    ]
    result = evaluate(tmp_path, payload)
    assert result.available, result.error
    assert result.issues[0].bbox == (tuple(bbox) if bbox is not None else None)


@pytest.mark.parametrize("bbox", [[float("nan"), 0, 1, 1], ["0.1", 0, 1, 1], [0, False, 1, 1]])
def test_historical_runtime_decoder_also_rejects_invalid_bbox(tmp_path, bbox):
    payload = {
        key: value
        for key, value in response().items()
        if key in {"scores", "decision", "confidence", "issues", "rationale"}
    }
    payload["issues"] = [
        {
            "dimension": "fidelity",
            "severity": 0.2,
            "description": "local detail issue",
            "bbox": bbox,
        }
    ]
    result = evaluate(tmp_path, payload, version="3.3")
    assert not result.available
    assert "bbox" in result.error
