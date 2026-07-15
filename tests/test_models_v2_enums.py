from portrait_eval.core.models_v2 import (
    ApplicabilityStatus,
    ComparabilityStatus,
    DifferenceType,
    DimensionId,
    EvidenceGradeV2,
    FaceRole,
    JudgeDecisionType,
    MatchStatus,
    MechanismAttribution,
    PairwisePreference,
    ReportAllowedLevel,
    ReportState,
    SCHEMA_VERSION,
    SceneScoreStatus,
)


def test_frozen_v2_vocabulary() -> None:
    assert SCHEMA_VERSION == "2.0"
    assert [item.value for item in DimensionId] == [
        "face_exposure_readability",
        "highlight_integrity",
        "shadow_black_rendering",
        "skin_awb",
        "lighting_causality",
        "face_background_relation",
        "local_face_lift_naturalness",
        "multi_face_consistency",
        "scene_adaptability",
        "artifact_texture_control",
    ]
    assert [item.value for item in MatchStatus] == [
        "CONFIRMED_MANIFEST",
        "CONFIRMED_MANUAL",
        "AUTO_HIGH_CONFIDENCE",
        "PENDING_REVIEW",
        "UNMATCHED",
        "INVALID",
    ]
    assert [item.value for item in ComparabilityStatus] == [
        "FULLY_COMPARABLE",
        "COMPARABLE_WITH_CONFOUNDERS",
        "NOT_COMPARABLE",
    ]
    assert [item.value for item in ApplicabilityStatus] == [
        "APPLICABLE",
        "WEAKLY_APPLICABLE",
        "NOT_APPLICABLE",
        "INSUFFICIENT_EVIDENCE",
    ]
    assert [item.value for item in DifferenceType] == [
        "QUALITY",
        "STYLE_PREFERENCE",
        "MIXED",
        "INSUFFICIENT_EVIDENCE",
        "NOT_APPLICABLE",
    ]
    assert [item.value for item in PairwisePreference] == [
        "A_STRONGLY_BETTER",
        "A_SLIGHTLY_BETTER",
        "EQUIVALENT",
        "B_SLIGHTLY_BETTER",
        "B_STRONGLY_BETTER",
    ]
    assert [item.value for item in JudgeDecisionType] == [
        "ACCEPT",
        "REVISE",
        "REJECT",
        "MANUAL_REVIEW",
        "INVALID_SAMPLE",
    ]
    assert [item.value for item in MechanismAttribution] == [
        "HARDWARE_ENABLED",
        "ALGORITHM_DEFINED",
        "PREFERENCE_DRIVEN",
        "CAPTURE_LIMITED",
        "RECONSTRUCTION_LIMITED",
        "STRATEGY_LIMITED",
        "MIXED_CAUSE",
        "INDETERMINATE",
    ]
    assert [item.value for item in ReportState] == ["AUTO_REPORT", "REVIEWED_REPORT"]
    assert [item.value for item in FaceRole] == ["PRIMARY", "SECONDARY", "UNASSIGNED"]
    assert [item.value for item in SceneScoreStatus] == [
        "AUTO_PASS",
        "PASS_WITH_NOTE",
        "MANUAL_REVIEW",
        "NOT_APPLICABLE",
    ]
    assert [item.value for item in EvidenceGradeV2] == ["A", "B", "C"]
    assert [item.value for item in ReportAllowedLevel] == [
        "OBSERVABLE",
        "STRATEGY_INFERENCE",
        "INTERNAL_HYPOTHESIS",
    ]
