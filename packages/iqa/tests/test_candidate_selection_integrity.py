from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qwen_tmqa.candidate_schema import validate_candidate_preference_payload
from qwen_tmqa.config import PseudoGTConfig
from qwen_tmqa.domain import (
    AlphaImage,
    ModelEvaluation,
    ModelIssue,
    ObjectiveMetrics,
    PromptTrace,
    SceneEvaluation,
    SceneSpec,
    SequenceMetrics,
    StageResult,
)
from qwen_tmqa.lineage import (
    EvaluationRunManifest,
    SelectionLineage,
    load_split_assignments,
)
from qwen_tmqa.prompts import build_input_manifest
from qwen_tmqa.pseudo_gt import export_pseudo_gt, select_scene_pseudo_gt


def _save(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 24, 3), np.clip(value, 0, 1), dtype=np.float32)
    Image.fromarray(np.round(image * 255).astype(np.uint8)).save(path)


def _metrics(level: str, alpha: float, p50: float) -> ObjectiveMetrics:
    return ObjectiveMetrics(
        level=level,
        alpha=alpha,
        mean_luminance=p50,
        ev_mean=float(np.log2(max(p50, 1e-6))),
        p20=max(0.0, p50 - 0.15),
        p50=p50,
        p90=min(1.0, p50 + 0.25),
        clipping_ratio=0.01,
        shadow_ratio=0.05,
        contrast=0.30,
        color_drift=0.01,
        edge_similarity=0.96,
    )


def _spec(tmp_path: Path) -> SceneSpec:
    input_path = tmp_path / "inputs" / "scene.png"
    baseline = tmp_path / "levels" / "a_000" / "scene.png"
    candidate = tmp_path / "levels" / "a_p050" / "scene.png"
    _save(input_path, 0.25)
    _save(baseline, 0.35)
    _save(candidate, 0.52)
    return SceneSpec(
        scene_id="scene.png",
        source_path=input_path,
        baseline_path=baseline,
        alpha_images=[
            AlphaImage(level="a_000", alpha=0.0, path=baseline),
            AlphaImage(level="a_p050", alpha=0.5, path=candidate),
        ],
    )


def _trace(spec: SceneSpec) -> PromptTrace:
    return PromptTrace(
        prompt_id="tmqa.sequence",
        prompt_version="3.4",
        template="template",
        rendered_prompt="rendered",
        prompt_hash="0" * 64,
        variables={},
        input_manifest=build_input_manifest(spec),
        output_schema_version="tmqa.sequence.v3.4",
    )


def _payload(
    *,
    preferred: str = "a_p050",
    scores: dict[str, float] | None = None,
) -> dict[str, object]:
    level_scores = scores or {"a_000": 0.70, "a_p050": 0.88}
    return {
        "preferred_level": preferred,
        "runner_up_level": "a_000" if preferred != "a_000" else "a_p050",
        "acceptable_levels": [preferred],
        "level_scores": level_scores,
        "selection_confidence": 0.90,
        "baseline_improvement": level_scores[preferred] - level_scores["a_000"],
    }


def _judge(
    spec: SceneSpec,
    model_id: str,
    *,
    decision: str = "KEEP",
    issues: list[ModelIssue] | None = None,
    payload: dict[str, object] | None = None,
) -> ModelEvaluation:
    parsed = payload or _payload()
    return ModelEvaluation(
        model_id=model_id,
        model_role="primary",
        model_version="v1",
        available=True,
        synthetic=False,
        prompt_trace=_trace(spec),
        scores={
            "tone": 0.8,
            "color": 0.8,
            "fidelity": 0.9,
            "control": 0.8,
            "preference": 0.9,
            "overall": 0.85,
        },
        decision=decision,
        confidence=0.9,
        issues=issues or [],
        parsed_response=parsed,
    )


def _scene(spec: SceneSpec) -> SceneEvaluation:
    return SceneEvaluation(
        scene_id=spec.scene_id,
        generator="test",
        decision="KEEP",
        overall_score=0.85,
        uncertainty=0.1,
        objective_by_level={
            "a_000": _metrics("a_000", 0.0, 0.35),
            "a_p050": _metrics("a_p050", 0.5, 0.52),
        },
        sequence=SequenceMetrics(
            spearman_rho=1.0,
            violation_rate=0.0,
            smoothness_score=1.0,
            dead_zone_ratio=0.0,
            endpoint_range_ev=0.5,
            clipping_growth=0.0,
            control_score=0.95,
        ),
        stages=[
            StageResult(stage_id="integrity", label="Integrity", status="PASS", score=1.0),
            StageResult(stage_id="hard_gate", label="Hard gate", status="PASS", score=1.0),
        ],
        model_evaluations=[_judge(spec, "judge-a"), _judge(spec, "judge-b")],
        review_priority=0.1,
    )


def _lineage() -> SelectionLineage:
    return SelectionLineage(
        evaluation=EvaluationRunManifest(
            evaluation_run_id="run-123",
            evaluation_config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            dataset_version="canary-v1",
            prompt_versions=["tmqa.sequence@3.4"],
        ),
        split="train",
        split_file_sha256="c" * 64,
    )


def _select(scene: SceneEvaluation, spec: SceneSpec):
    return select_scene_pseudo_gt(
        scene,
        spec,
        PseudoGTConfig(),
        lineage=_lineage(),
    )


def test_candidate_schema_requires_complete_levels_and_consistent_fields() -> None:
    expected = {"a_000", "a_p025", "a_p050"}
    with pytest.raises(ValueError, match="level_scores"):
        validate_candidate_preference_payload(
            _payload(scores={"a_000": 0.70, "a_p050": 0.88}),
            expected_levels=expected,
            baseline_level="a_000",
        )

    invalid_runner = {
        **_payload(scores={"a_000": 0.70, "a_p025": 0.82, "a_p050": 0.88}),
        "runner_up_level": "a_000",
    }
    with pytest.raises(ValueError, match="runner_up"):
        validate_candidate_preference_payload(
            invalid_runner,
            expected_levels=expected,
            baseline_level="a_000",
        )

    invalid_improvement = {
        **_payload(scores={"a_000": 0.70, "a_p025": 0.82, "a_p050": 0.88}),
        "runner_up_level": "a_p025",
        "baseline_improvement": 0.50,
    }
    with pytest.raises(ValueError, match="baseline_improvement"):
        validate_candidate_preference_payload(
            invalid_improvement,
            expected_levels=expected,
            baseline_level="a_000",
        )


def test_selector_rejects_candidate_modified_after_evaluation(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    scene = _scene(spec)
    _save(spec.alpha_images[1].path, 0.80)

    record = _select(scene, spec)

    assert record.accepted is False
    assert "EVIDENCE_IMAGE_MISMATCH" in record.rejection_reasons


def test_selector_rejects_levels_swapped_after_evaluation(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    scene = _scene(spec)
    baseline_bytes = spec.alpha_images[0].path.read_bytes()
    candidate_bytes = spec.alpha_images[1].path.read_bytes()
    spec.alpha_images[0].path.write_bytes(candidate_bytes)
    spec.alpha_images[1].path.write_bytes(baseline_bytes)

    record = _select(scene, spec)

    assert record.accepted is False
    assert "EVIDENCE_IMAGE_MISMATCH" in record.rejection_reasons


def test_selector_rejects_input_modified_after_evaluation(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    scene = _scene(spec)
    assert spec.source_path is not None
    _save(spec.source_path, 0.65)

    record = _select(scene, spec)

    assert record.accepted is False
    assert "EVIDENCE_IMAGE_MISMATCH" in record.rejection_reasons


def test_selector_rejects_fatal_or_reject_without_score_compensation(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    scene = _scene(spec)
    fatal = ModelIssue(
        dimension="identity",
        severity=1.0,
        description="face identity changed",
        fatal=True,
    )
    scene.model_evaluations = [
        _judge(spec, "judge-fatal", issues=[fatal]),
        _judge(spec, "judge-good-a"),
        _judge(spec, "judge-good-b"),
    ]

    fatal_record = _select(scene, spec)
    assert fatal_record.accepted is False
    assert "JUDGE_FATAL" in fatal_record.rejection_reasons

    scene.model_evaluations = [
        _judge(spec, "judge-reject", decision="REJECT"),
        _judge(spec, "judge-good-a"),
        _judge(spec, "judge-good-b"),
    ]
    reject_record = _select(scene, spec)
    assert reject_record.accepted is False
    assert "JUDGE_REJECT" in reject_record.rejection_reasons


def test_selector_rejects_failed_integrity_or_scene_hard_gate(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    scene = _scene(spec)
    scene.stages[0] = StageResult(stage_id="integrity", label="Integrity", status="FAIL", score=0.0)
    integrity_record = _select(scene, spec)
    assert integrity_record.accepted is False
    assert "SCENE_INTEGRITY_FAIL" in integrity_record.rejection_reasons

    scene = _scene(spec)
    scene.stages[1] = StageResult(stage_id="hard_gate", label="Hard gate", status="FAIL", score=0.0)
    hard_gate_record = _select(scene, spec)
    assert hard_gate_record.accepted is False
    assert "SCENE_HARD_GATE_FAIL" in hard_gate_record.rejection_reasons


def test_split_file_rejects_duplicates_unassigned_and_cross_split_derivatives(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "scene_id,canonical_scene_id,group_id,split\n"
        "scene.png,scene,group,train\n"
        "scene.png,scene,group,holdout\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_split_assignments(duplicate, expected_scene_ids={"scene.png"})

    cross_split = tmp_path / "cross.csv"
    cross_split.write_text(
        "scene_id,canonical_scene_id,group_id,split\n"
        "scene_crop.png,scene,group,train\n"
        "scene_resize.png,scene,group,holdout\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical"):
        load_split_assignments(
            cross_split,
            expected_scene_ids={"scene_crop.png", "scene_resize.png"},
        )

    missing = tmp_path / "missing.csv"
    missing.write_text(
        "scene_id,canonical_scene_id,group_id,split\nother.png,other,other,train\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unassigned"):
        load_split_assignments(missing, expected_scene_ids={"scene.png"})


def test_export_is_deterministic_and_contains_lineage_hashes(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    record = _select(_scene(spec), spec)
    assert record.accepted is True

    first = tmp_path / "first"
    second = tmp_path / "second"
    export_pseudo_gt([record], first)
    export_pseudo_gt([record], second)

    first_manifest = (first / "candidate_suggestions.jsonl").read_text(encoding="utf-8")
    second_manifest = (second / "candidate_suggestions.jsonl").read_text(encoding="utf-8")
    assert first_manifest == second_manifest
    payload = json.loads(first_manifest)
    assert payload["split"] == "train"
    assert len(payload["input_sha256"]) == 64
    assert len(payload["baseline_sha256"]) == 64
    assert len(payload["selected_gt_sha256"]) == 64
    assert payload["evaluation_run_id"] == "run-123"
    assert payload["evaluation_config_sha256"] == "a" * 64
    assert payload["dataset_version"] == "canary-v1"
    assert payload["prompt_versions"] == ["tmqa.sequence@3.4"]
    assert payload["training_weight"] == 0.0
    assert payload["record_kind"] == "candidate_suggestion"


def test_selector_rejects_duplicate_judge_identities(tmp_path):
    spec = _spec(tmp_path)
    scene = _scene(spec)
    scene.model_evaluations = [_judge(spec, "one-judge"), _judge(spec, "one-judge")]
    record = _select(scene, spec)
    assert not record.accepted
    assert "DUPLICATE_JUDGE_ID" in record.rejection_reasons
    assert record.valid_judge_count <= 1


def test_historical_prompts_cannot_supply_new_candidate_contract(tmp_path):
    spec = _spec(tmp_path)
    scene = _scene(spec)
    for judge in scene.model_evaluations:
        judge.prompt_trace.prompt_version = "3.3"
    record = _select(scene, spec)
    assert not record.accepted


def test_split_metadata_must_be_explicit_and_group_closed(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text("scene_id,split\ns1,train\n")
    with pytest.raises(ValueError, match="canonical"):
        load_split_assignments(path, expected_scene_ids={"s1"})
    path.write_text(
        "scene_id,canonical_scene_id,group_id,split\ns1,c1,g1,train\ns2,c2,g1,holdout\n"
    )
    with pytest.raises(ValueError, match="group"):
        load_split_assignments(path, expected_scene_ids={"s1", "s2"})


def test_selector_cannot_count_whitespace_identity_aliases_or_empty_judges(tmp_path):
    spec = _spec(tmp_path)
    scene = _scene(spec)
    scene.model_evaluations = [_judge(spec, "same"), _judge(spec, " same ")]
    record = _select(scene, spec)
    assert not record.accepted
    assert "DUPLICATE_JUDGE_ID" in record.rejection_reasons
    scene.model_evaluations = [_judge(spec, ""), _judge(spec, "valid")]
    record = _select(scene, spec)
    assert not record.accepted
    assert "INVALID_JUDGE_ID" in record.rejection_reasons
