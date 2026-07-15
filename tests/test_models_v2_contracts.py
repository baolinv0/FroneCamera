from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from portrait_eval.core.models_v2 import (
    ApplicabilityStatus,
    ComparabilityStatus,
    DeviceDimensionScoreV2,
    DeviceOverallScoreV2,
    DifferenceType,
    DimensionApplicabilityV2,
    DimensionId,
    EvaluationBatchV2,
    EvidenceGradeV2,
    FaceRegionV2,
    FaceRole,
    FactCheckResultV2,
    JudgeDecisionType,
    JudgeDecisionV2,
    MatchEvidenceV2,
    MatchedSceneGroupV2,
    MechanismAttribution,
    MechanismInterpretationV2,
    ObjectiveEvidenceV2,
    PairwiseComparisonV2,
    PairwisePreference,
    ReportAllowedLevel,
    ReportEvidencePackageV2,
    RoughRankingV2,
    SceneDimensionScoreV2,
    SceneScoreStatus,
)


def make_batch() -> EvaluationBatchV2:
    return EvaluationBatchV2(
        batch_id="batch-001",
        project_id="project-001",
        device_ids=["a", "b"],
        dataset_version="dataset-v1",
        dimension_policy_version="dimensions-v2.0",
        scoring_policy_version="batch-relative-v1",
        primary_model_version="primary-v1",
        judge_model_version="judge-v1",
        report_model_version=None,
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def make_pairwise(**updates: object) -> PairwiseComparisonV2:
    payload: dict[str, object] = {
        "comparison_id": "cmp-001",
        "scene_id": "scene-001",
        "dimension_id": DimensionId.FACE_EXPOSURE_READABILITY,
        "device_a_id": "a",
        "device_b_id": "b",
        "applicability": ApplicabilityStatus.APPLICABLE,
        "preference": PairwisePreference.A_SLIGHTLY_BETTER,
        "strength": 1,
        "difference_type": DifferenceType.QUALITY,
        "confidence": 0.8,
        "observations": ["A retains more readable facial midtones."],
        "evidence_refs": ["metric:scene-001:a:face_luma"],
        "style_axes": {"face_target": "medium_high"},
        "primary_model_version": "primary-v1",
        "prompt_version": "pairwise-v1",
    }
    payload.update(updates)
    return PairwiseComparisonV2.model_validate(payload)


def make_judge(**updates: object) -> JudgeDecisionV2:
    payload: dict[str, object] = {
        "comparison_id": "cmp-001",
        "decision": JudgeDecisionType.ACCEPT,
        "accepted_preference": PairwisePreference.A_SLIGHTLY_BETTER,
        "accepted_strength": 1,
        "accepted_difference_type": DifferenceType.QUALITY,
        "revised_observations": [],
        "fact_conflicts": [],
        "unsupported_attributions": [],
        "confidence": 0.85,
        "judge_model_version": "judge-v1",
        "prompt_version": "judge-prompt-v1",
    }
    payload.update(updates)
    return JudgeDecisionV2.model_validate(payload)


def test_batch_defaults_and_round_trip() -> None:
    batch = make_batch()
    assert batch.schema_version == "2.0"
    assert batch.task_scope == "front_camera_portrait_tone_semantic_rendering"
    assert batch.scoring_scope == "batch_relative"
    assert EvaluationBatchV2.model_validate_json(batch.model_dump_json()) == batch


def test_unknown_fields_are_rejected() -> None:
    payload = make_batch().model_dump()
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        EvaluationBatchV2.model_validate(payload)


def test_two_faces_are_persisted_in_objective_evidence() -> None:
    faces = [
        FaceRegionV2(
            face_id=f"face-{index}",
            bbox_xywh=(10 * index, 20, 100, 120),
            detection_confidence=0.95,
            area_ratio=0.12,
            center_xy_normalized=(0.3 + 0.2 * index, 0.4),
            landmarks_ref=None,
            masks={"skin": f"asset-skin-{index}"},
            role=FaceRole.PRIMARY if index == 0 else FaceRole.SECONDARY,
        )
        for index in range(2)
    ]
    evidence = ObjectiveEvidenceV2(
        image_id="image-001",
        faces=faces,
        whole_image_metrics={"luma_p50": 0.4},
        region_metrics={"background": {"luma_p50": 0.2}},
        per_face_metrics={"face-0": {"luma_p50": 0.5}, "face-1": {"luma_p50": 0.45}},
        relation_metrics={"face_background_ratio": 2.0},
        artifact_metrics={"halo_proxy": 0.1},
        warnings=[],
        metric_version="metrics-v2",
    )
    assert [face.face_id for face in evidence.faces] == ["face-0", "face-1"]
    assert ObjectiveEvidenceV2.model_validate_json(evidence.model_dump_json()) == evidence


def test_match_and_confidence_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        MatchEvidenceV2(
            exif_time_score=1.1,
            perceptual_hash_score=0.9,
            scene_embedding_score=0.8,
            face_geometry_score=None,
            composition_score=0.7,
            aggregate_score=0.8,
        )
    with pytest.raises(ValidationError):
        MatchedSceneGroupV2(
            scene_id="scene-001",
            batch_id="batch-001",
            label=None,
            cells={"a": "image-a", "b": "image-b"},
            match_status="AUTO_HIGH_CONFIDENCE",
            match_confidence=-0.1,
            match_evidence={
                "exif_time_score": 0.9,
                "perceptual_hash_score": 0.9,
                "scene_embedding_score": 0.9,
                "face_geometry_score": None,
                "composition_score": 0.9,
                "aggregate_score": 0.9,
            },
            comparability_status=ComparabilityStatus.FULLY_COMPARABLE,
            confounders=[],
            confirmed_by=None,
        )


def test_score_and_objective_adjustment_bounds_are_enforced() -> None:
    base = {
        "scene_id": "scene-001",
        "dimension_id": DimensionId.HIGHLIGHT_INTEGRITY,
        "device_id": "a",
        "latent_quality": 0.5,
        "normalized_score": 80,
        "confidence": 0.9,
        "diagnostic_weight": 0.8,
        "objective_adjustment": 2,
        "final_score": 82,
        "status": SceneScoreStatus.AUTO_PASS,
        "contributing_comparison_ids": ["cmp-001"],
    }
    with pytest.raises(ValidationError):
        SceneDimensionScoreV2.model_validate({**base, "normalized_score": 101})
    with pytest.raises(ValidationError):
        SceneDimensionScoreV2.model_validate({**base, "objective_adjustment": 5.1})


def test_pairwise_preference_strength_and_devices_are_consistent() -> None:
    with pytest.raises(ValidationError, match="preference and strength"):
        make_pairwise(strength=-1)
    with pytest.raises(ValidationError, match="devices must be distinct"):
        make_pairwise(device_b_id="a")


def test_judge_semantic_fields_follow_decision() -> None:
    with pytest.raises(ValidationError, match="requires accepted semantic fields"):
        make_judge(accepted_preference=None)
    with pytest.raises(ValidationError, match="cannot carry accepted semantic fields"):
        make_judge(decision=JudgeDecisionType.REJECT)
    with pytest.raises(ValidationError, match="preference and strength"):
        make_judge(accepted_strength=-1)


def test_report_package_round_trip() -> None:
    device_dimension = DeviceDimensionScoreV2(
        batch_id="batch-001",
        device_id="a",
        dimension_id=DimensionId.FACE_EXPOSURE_READABILITY,
        aggregate_score=78,
        rank=1,
        rank_group=1,
        confidence=0.85,
        applicable_scene_count=3,
        style_position={"face_target": "medium_high"},
        supporting_scene_ids=["scene-001"],
        counterexample_scene_ids=[],
    )
    overall = DeviceOverallScoreV2(
        batch_id="batch-001",
        device_id="a",
        relative_score=76,
        rank=1,
        confidence=0.82,
        dimension_scores={DimensionId.FACE_EXPOSURE_READABILITY: 78},
    )
    mechanism = MechanismInterpretationV2(
        interpretation_id="interpretation-001",
        device_id="a",
        scene_ids=["scene-001"],
        observable_effect="The face is selectively brighter than the background.",
        strategy_inference="The output is consistent with local semantic face lifting.",
        mechanism_hypothesis=None,
        attribution=MechanismAttribution.STRATEGY_LIMITED,
        confidence=0.75,
        evidence_grade=EvidenceGradeV2.B,
        supporting_evidence_refs=["scene:scene-001"],
        alternative_explanations=["capture exposure variation"],
        report_allowed_level=ReportAllowedLevel.STRATEGY_INFERENCE,
    )
    package = ReportEvidencePackageV2(
        batch=make_batch(),
        scene_summaries=[{"scene_id": "scene-001"}],
        device_dimension_scores=[device_dimension],
        device_overall_scores=[overall],
        device_profiles=[{"device_id": "a"}],
        mechanism_interpretations=[mechanism],
        selected_visual_assets=[],
        review_notes=[],
        limitations=["Scores are batch-relative."],
        prohibited_claims=["Do not claim a proprietary ISP implementation."],
    )
    assert ReportEvidencePackageV2.model_validate_json(package.model_dump_json()) == package


def test_remaining_contracts_construct_with_schema_version() -> None:
    applicability = DimensionApplicabilityV2(
        scene_id="scene-001",
        dimension_id=DimensionId.SKIN_AWB,
        status=ApplicabilityStatus.APPLICABLE,
        score=0.9,
        reasons=["valid_skin_region"],
        required_evidence_present=True,
    )
    ranking = RoughRankingV2(
        scene_id="scene-001",
        dimension_id=DimensionId.SKIN_AWB,
        ordered_device_ids=["a", "b"],
        tie_groups=[],
        confidence=0.8,
        evidence_refs=["asset:skin-crop"],
        prompt_version="rough-v1",
        model_version="primary-v1",
    )
    fact = FactCheckResultV2(
        comparison_id="cmp-001",
        rule_version="rules-v1",
        passed_rules=["skin_mask_present"],
        failed_rules=[],
        warnings=[],
        critical_conflict=False,
        objective_support=0.9,
    )
    assert applicability.schema_version == ranking.schema_version == fact.schema_version == "2.0"
    assert make_pairwise().schema_version == make_judge().schema_version == "2.0"
