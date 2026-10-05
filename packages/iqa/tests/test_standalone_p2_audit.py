from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_audit_negative import _audit, _write_behavioral_evidence

from qwen_tmqa.cli import (
    calibrate_command,
    evaluate_command,
    make_example,
    simulate_human,
    visualize_command,
)


@pytest.fixture
def audit_case(tmp_path):
    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    reviews = tmp_path / "reviews.jsonl"
    calibration = tmp_path / "calibration.json"
    dashboard = tmp_path / "dashboard"
    config = Path("configs/default.yaml")
    make_example(dataset, 1)
    evaluate_command(dataset, results, config)
    simulate_human(results / "evaluations.json", reviews, config)
    calibrate_command(
        results / "evaluations.json",
        reviews,
        calibration,
        allow_synthetic=True,
        review_type="synthetic",
        config_path=config,
    )
    visualize_command(results / "evaluations.json", dashboard, config, reviews, calibration)
    evidence = tmp_path / "evidence.json"
    _write_behavioral_evidence(evidence, results / "evaluations.json")
    payload = json.loads(evidence.read_text())
    for name, check in payload["checks"].items():
        check["output"] = f"fixture output for {name}\n"
        check["output_sha256"] = hashlib.sha256(check["output"].encode()).hexdigest()
    evidence.write_text(json.dumps(payload))
    paths = (results, dashboard, reviews, calibration, evidence)
    assert all(_audit(*paths).values())
    return paths


def _mutate_model(paths, field, value):
    path = paths[0] / "evaluations.json"
    scenes = json.loads(path.read_text())
    model = scenes[0]["model_evaluations"][0]
    if value is _MISSING:
        model.pop(field)
    else:
        model[field] = value
    path.write_text(json.dumps(scenes))


_MISSING = object()


@pytest.mark.parametrize("score", [999, -0.1, 2, float("nan"), float("inf"), "0.9", True])
def test_audit_rejects_invalid_persisted_normalized_scores(audit_case, score):
    _mutate_model(
        audit_case,
        "scores",
        dict.fromkeys(["tone", "color", "fidelity", "control", "preference", "overall"], score),
    )
    assert _audit(*audit_case)["R-004"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", 1.1),
        ("confidence", float("nan")),
        ("confidence", "0.9"),
        ("confidence", True),
        ("rationale", _MISSING),
        ("rationale", ""),
        ("raw_response", _MISSING),
        ("parsed_response", []),
        ("latency_ms", -1),
        ("latency_ms", float("inf")),
        ("available", "yes"),
        ("model_version", ""),
        ("issues", [{"dimension": "tone", "severity": 999, "description": "invalid"}]),
    ],
)
def test_audit_requires_valid_complete_model_contract(audit_case, field, value):
    _mutate_model(audit_case, field, value)
    assert _audit(*audit_case)["R-004"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "g" * 64),
        ("payload_sha256", "not-a-sha256"),
        ("payload_sha256", "a" * 63 + "\n"),
        ("sha256", "b" * 64),
        ("sent_width", 0),
        ("sent_height", True),
        ("width", "128"),
        ("payload_mime", "text/plain"),
        ("payload_encoding", {}),
        ("payload_encoding", {"format": "JPEG", "quality": 999}),
        ("index", 2),
    ],
)
def test_audit_rejects_invalid_payload_lineage(audit_case, field, value):
    path = audit_case[0] / "evaluations.json"
    scenes = json.loads(path.read_text())
    scenes[0]["model_evaluations"][0]["prompt_trace"]["input_manifest"][0][field] = value
    path.write_text(json.dumps(scenes))
    assert _audit(*audit_case)["R-006"] is False


def test_audit_requires_a_manifest_for_every_model(audit_case):
    path = audit_case[0] / "evaluations.json"
    scenes = json.loads(path.read_text())
    scenes[0]["model_evaluations"][0]["prompt_trace"]["input_manifest"] = []
    path.write_text(json.dumps(scenes))
    assert _audit(*audit_case)["R-006"] is False


def test_audit_recomputes_prompt_hash(audit_case):
    path = audit_case[0] / "evaluations.json"
    scenes = json.loads(path.read_text())
    scenes[0]["model_evaluations"][0]["prompt_trace"]["rendered_prompt"] += "altered"
    path.write_text(json.dumps(scenes))
    assert _audit(*audit_case)["R-005"] is False


@pytest.mark.parametrize("change", ["digest", "output", "missing", "bool_exit"])
def test_audit_verifies_behavior_output_digest(audit_case, change):
    path = audit_case[-1]
    evidence = json.loads(path.read_text())
    check = evidence["checks"]["dashboard_behavior"]
    if change == "digest":
        check["output_sha256"] = "g" * 64
    elif change == "output":
        check["output"] += "altered"
    elif change == "missing":
        check.pop("output")
    else:
        check["exit_code"] = False
    path.write_text(json.dumps(evidence))
    assert _audit(*audit_case)["R-009"] is False


def test_audit_accepts_valid_zero_and_one_internal_scores(audit_case):
    for score in [0.0, 1.0]:
        _mutate_model(
            audit_case,
            "scores",
            dict.fromkeys(["tone", "color", "fidelity", "control", "preference", "overall"], score),
        )
        assert _audit(*audit_case)["R-004"] is True


@pytest.mark.parametrize("score", [2, 100, float("nan")])
def test_audit_rejects_invalid_scores_even_for_unavailable_models(audit_case, score):
    _mutate_model(audit_case, "available", False)
    _mutate_model(audit_case, "error", "provider unavailable")
    _mutate_model(audit_case, "scores", {"overall": score})
    assert _audit(*audit_case)["R-004"] is False


@pytest.mark.parametrize("field", ["prompt_trace", "input_manifest"])
def test_audit_null_trace_evidence_fails_closed(audit_case, field):
    path = audit_case[0] / "evaluations.json"
    scenes = json.loads(path.read_text())
    model = scenes[0]["model_evaluations"][0]
    if field == "prompt_trace":
        model[field] = None
    else:
        model["prompt_trace"][field] = None
    path.write_text(json.dumps(scenes))
    assert _audit(*audit_case)["R-006"] is False


@pytest.mark.parametrize("field", ["dashboard_behavior", "checks"])
def test_audit_null_behavior_evidence_fails_closed(audit_case, field):
    path = audit_case[-1]
    evidence = json.loads(path.read_text())
    if field == "checks":
        evidence[field] = None
    else:
        evidence["checks"][field] = None
    path.write_text(json.dumps(evidence))
    assert _audit(*audit_case)["R-009"] is False
