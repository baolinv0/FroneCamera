from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .candidate_schema import validate_candidate_preference_payload
from .config import PseudoGTConfig
from .domain import ModelEvaluation, SceneEvaluation, SceneSpec
from .image_io import file_sha256
from .lineage import SelectionLineage, verify_scene_evidence

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class PseudoGTRecord(BaseModel):
    scene_id: str
    accepted: bool
    input_path: str | None = None
    input_relpath: str
    input_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    baseline_path: str
    baseline_sha256: str = Field(min_length=64, max_length=64)
    baseline_level: str
    selected_gt_path: str | None = None
    selected_gt_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    selected_level: str | None = None
    runner_up_level: str | None = None
    selection_method: str
    valid_judge_count: int = Field(ge=0)
    vote_count: int = Field(ge=0)
    vote_ratio: float = Field(ge=0, le=1)
    mean_confidence: float = Field(ge=0, le=1)
    selection_confidence_raw: float = Field(ge=0, le=1)
    mean_improvement: float
    mean_margin: float
    hard_gate_pass: bool
    evidence_binding_pass: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    judge_ids: list[str] = Field(default_factory=list)
    split: str
    split_file_sha256: str = Field(min_length=64, max_length=64)
    evaluation_run_id: str
    evaluation_config_sha256: str = Field(min_length=64, max_length=64)
    dataset_manifest_sha256: str = Field(min_length=64, max_length=64)
    dataset_version: str
    prompt_versions: list[str]
    training_weight: float = Field(ge=0, le=1)
    selector_version: str = "candidate_selector.v1"
    record_kind: str = "candidate_suggestion"
    synthetic: bool = False


class _Preference(BaseModel):
    model_id: str
    preferred_level: str
    runner_up_level: str
    confidence: float = Field(ge=0, le=1)
    improvement: float
    margin: float


def _preference(
    evaluation: ModelEvaluation,
    levels: set[str],
    baseline: str,
    allow_synthetic: bool,
) -> _Preference | None:
    if evaluation.prompt_trace.prompt_version != "3.4":
        return None
    if not evaluation.available or (evaluation.synthetic and not allow_synthetic):
        return None
    if evaluation.decision == "REJECT" or any(issue.fatal for issue in evaluation.issues):
        return None
    try:
        evidence = validate_candidate_preference_payload(
            evaluation.parsed_response,
            expected_levels=levels,
            baseline_level=baseline,
        )
    except (TypeError, ValueError):
        return None
    scores = evidence.level_scores
    alternatives = [score for level, score in scores.items() if level != evidence.preferred_level]
    runner_score = max(alternatives, default=scores[evidence.preferred_level])
    return _Preference(
        model_id=evaluation.model_id.strip(),
        preferred_level=evidence.preferred_level,
        runner_up_level=evidence.runner_up_level,
        confidence=evidence.selection_confidence,
        improvement=evidence.baseline_improvement,
        margin=scores[evidence.preferred_level] - runner_score,
    )


def _hard_gate(
    scene: SceneEvaluation,
    selected: str,
    baseline: str,
    config: PseudoGTConfig,
) -> bool:
    candidate = scene.objective_by_level.get(selected)
    reference = scene.objective_by_level.get(baseline)
    if candidate is None or reference is None:
        return False
    return (
        candidate.clipping_ratio <= config.max_clipping_ratio
        and candidate.clipping_ratio <= reference.clipping_ratio + config.max_clipping_increase
        and candidate.shadow_ratio <= config.max_shadow_ratio
        and candidate.shadow_ratio <= reference.shadow_ratio + config.max_shadow_increase
        and candidate.color_drift <= config.max_color_drift
        and candidate.edge_similarity >= config.min_edge_similarity
    )


def _trusted_evaluations(
    scene: SceneEvaluation,
    allow_synthetic: bool,
) -> list[ModelEvaluation]:
    return [
        evaluation
        for evaluation in scene.model_evaluations
        if evaluation.available and (allow_synthetic or not evaluation.synthetic)
    ]


def select_scene_pseudo_gt(
    scene: SceneEvaluation,
    scene_spec: SceneSpec,
    config: PseudoGTConfig,
    *,
    lineage: SelectionLineage,
    baseline_level: str = "a_000",
    allow_synthetic: bool = False,
) -> PseudoGTRecord:
    levels = set(scene.objective_by_level)
    trusted = _trusted_evaluations(scene, allow_synthetic)
    duplicate_ids = len({item.model_id.strip() for item in scene.model_evaluations}) != len(
        scene.model_evaluations
    )
    unique_trusted = {item.model_id.strip(): item for item in trusted if item.model_id.strip()}
    trusted = list(unique_trusted.values())
    preferences = [
        item
        for evaluation in trusted
        if (
            item := _preference(
                evaluation,
                levels,
                baseline_level,
                allow_synthetic,
            )
        )
        is not None
    ]
    votes = Counter(item.preferred_level for item in preferences)
    selected_level: str | None = None
    vote_count = 0
    if votes:
        selected_level, vote_count = min(votes.items(), key=lambda item: (-item[1], item[0]))
    vote_ratio = vote_count / len(preferences) if preferences else 0.0
    winners = [item for item in preferences if item.preferred_level == selected_level]

    def mean(attribute: str) -> float:
        values = [float(getattr(item, attribute)) for item in winners]
        return float(np.mean(values)) if values else 0.0

    mean_confidence = mean("confidence")
    mean_improvement = mean("improvement")
    mean_margin = mean("margin")
    runner_up_level = None
    if winners:
        runner_votes = Counter(item.runner_up_level for item in winners)
        runner_up_level = min(runner_votes.items(), key=lambda item: (-item[1], item[0]))[0]

    reasons = verify_scene_evidence(scene, scene_spec, lineage.evaluation)
    evidence_binding_pass = not reasons
    if duplicate_ids:
        reasons.append("DUPLICATE_JUDGE_ID")
    if any(not item.model_id.strip() for item in scene.model_evaluations):
        reasons.append("INVALID_JUDGE_ID")
    stage_status = {stage.stage_id: stage.status for stage in scene.stages}
    if stage_status.get("integrity") == "FAIL":
        reasons.append("SCENE_INTEGRITY_FAIL")
    if stage_status.get("hard_gate") == "FAIL":
        reasons.append("SCENE_HARD_GATE_FAIL")
    if any(evaluation.decision == "REJECT" for evaluation in trusted):
        reasons.append("JUDGE_REJECT")
    if any(issue.fatal for evaluation in trusted for issue in evaluation.issues):
        reasons.append("JUDGE_FATAL")
    if scene_spec.source_path is None or not scene_spec.source_path.exists():
        reasons.append("INPUT_IMAGE_MISSING")
    if len(preferences) < config.min_judges:
        reasons.append("insufficient_valid_judges")
    if selected_level == baseline_level:
        reasons.append("baseline_is_best")
    elif selected_level is None:
        reasons.append("no_selected_level")
    if preferences and vote_ratio < config.min_vote_ratio:
        reasons.append("insufficient_consensus")
    if winners and mean_confidence < config.min_selection_confidence:
        reasons.append("low_selection_confidence")
    if winners and mean_improvement < config.min_improvement:
        reasons.append("insufficient_improvement")
    if winners and mean_margin < config.min_score_margin:
        reasons.append("insufficient_margin")

    hard_gate_pass = bool(
        selected_level
        and selected_level != baseline_level
        and _hard_gate(scene, selected_level, baseline_level, config)
    )
    if selected_level and selected_level != baseline_level and not hard_gate_pass:
        reasons.append("candidate_hard_gate")

    paths = {item.level: item.path for item in scene_spec.alpha_images}
    selected_path = paths.get(selected_level) if selected_level else None
    if selected_level and selected_path is None:
        reasons.append("selected_path_missing")

    reasons = list(dict.fromkeys(reasons))
    accepted = not reasons
    input_hash = (
        file_sha256(scene_spec.source_path)
        if scene_spec.source_path is not None and scene_spec.source_path.exists()
        else None
    )
    selected_hash = (
        file_sha256(selected_path) if selected_path is not None and selected_path.exists() else None
    )
    return PseudoGTRecord(
        scene_id=scene.scene_id,
        accepted=accepted,
        input_path=str(scene_spec.source_path) if scene_spec.source_path else None,
        input_relpath=scene.scene_id,
        input_sha256=input_hash,
        baseline_path=str(scene_spec.baseline_path),
        baseline_sha256=file_sha256(scene_spec.baseline_path),
        baseline_level=baseline_level,
        selected_gt_path=str(selected_path) if selected_path else None,
        selected_gt_sha256=selected_hash,
        selected_level=selected_level,
        runner_up_level=runner_up_level,
        selection_method="MODEL_SUGGESTION" if accepted else "EXCLUDED",
        valid_judge_count=len(preferences),
        vote_count=vote_count,
        vote_ratio=vote_ratio,
        mean_confidence=mean_confidence,
        selection_confidence_raw=mean_confidence,
        mean_improvement=mean_improvement,
        mean_margin=mean_margin,
        hard_gate_pass=hard_gate_pass,
        evidence_binding_pass=evidence_binding_pass,
        rejection_reasons=reasons,
        judge_ids=[item.model_id for item in winners],
        split=lineage.split,
        split_file_sha256=lineage.split_file_sha256,
        evaluation_run_id=lineage.evaluation.evaluation_run_id,
        evaluation_config_sha256=lineage.evaluation.evaluation_config_sha256,
        dataset_manifest_sha256=lineage.evaluation.dataset_manifest_sha256,
        dataset_version=lineage.evaluation.dataset_version,
        prompt_versions=lineage.evaluation.prompt_versions,
        training_weight=0.0,
        synthetic=scene.synthetic_experiment or any(item.synthetic for item in trusted),
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def export_pseudo_gt(
    records: list[PseudoGTRecord],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: item.scene_id)
    accepted = [item for item in ordered if item.accepted]
    _write_jsonl(
        output_dir / "candidate_suggestions.jsonl",
        [item.model_dump(mode="json") for item in accepted],
    )
    _write_jsonl(
        output_dir / "candidate_audit.jsonl",
        [item.model_dump(mode="json") for item in ordered],
    )
    levels = Counter(item.selected_level for item in accepted if item.selected_level)
    rejections: Counter[str] = Counter()
    for item in ordered:
        rejections.update(item.rejection_reasons)
    summary: dict[str, object] = {
        "scene_count": len(ordered),
        "accepted_count": len(accepted),
        "rejected_count": len(ordered) - len(accepted),
        "acceptance_rate": len(accepted) / len(ordered) if ordered else 0.0,
        "selected_level_distribution": dict(sorted(levels.items())),
        "rejection_reason_distribution": dict(sorted(rejections.items())),
        "selector_version": "candidate_selector.v1",
    }
    (output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary
