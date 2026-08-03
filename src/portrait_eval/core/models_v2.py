from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["2.0"] = "2.0"
UnitFloat = Annotated[float, Field(ge=0, le=1)]
Score100 = Annotated[float, Field(ge=0, le=100)]
ObjectiveAdjustment = Annotated[float, Field(ge=-5, le=5)]


class V2Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DimensionId(StrEnum):
    FACE_EXPOSURE_READABILITY = "face_exposure_readability"
    HIGHLIGHT_INTEGRITY = "highlight_integrity"
    SHADOW_BLACK_RENDERING = "shadow_black_rendering"
    SKIN_AWB = "skin_awb"
    LIGHTING_CAUSALITY = "lighting_causality"
    FACE_BACKGROUND_RELATION = "face_background_relation"
    LOCAL_FACE_LIFT_NATURALNESS = "local_face_lift_naturalness"
    MULTI_FACE_CONSISTENCY = "multi_face_consistency"
    SCENE_ADAPTABILITY = "scene_adaptability"
    ARTIFACT_TEXTURE_CONTROL = "artifact_texture_control"


class MatchStatus(StrEnum):
    CONFIRMED_MANIFEST = "CONFIRMED_MANIFEST"
    CONFIRMED_MANUAL = "CONFIRMED_MANUAL"
    AUTO_HIGH_CONFIDENCE = "AUTO_HIGH_CONFIDENCE"
    PENDING_REVIEW = "PENDING_REVIEW"
    UNMATCHED = "UNMATCHED"
    INVALID = "INVALID"


class ComparabilityStatus(StrEnum):
    FULLY_COMPARABLE = "FULLY_COMPARABLE"
    COMPARABLE_WITH_CONFOUNDERS = "COMPARABLE_WITH_CONFOUNDERS"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ApplicabilityStatus(StrEnum):
    APPLICABLE = "APPLICABLE"
    WEAKLY_APPLICABLE = "WEAKLY_APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class DifferenceType(StrEnum):
    QUALITY = "QUALITY"
    STYLE_PREFERENCE = "STYLE_PREFERENCE"
    MIXED = "MIXED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PairwisePreference(StrEnum):
    A_STRONGLY_BETTER = "A_STRONGLY_BETTER"
    A_SLIGHTLY_BETTER = "A_SLIGHTLY_BETTER"
    EQUIVALENT = "EQUIVALENT"
    B_SLIGHTLY_BETTER = "B_SLIGHTLY_BETTER"
    B_STRONGLY_BETTER = "B_STRONGLY_BETTER"


class JudgeDecisionType(StrEnum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    INVALID_SAMPLE = "INVALID_SAMPLE"


class MechanismAttribution(StrEnum):
    HARDWARE_ENABLED = "HARDWARE_ENABLED"
    ALGORITHM_DEFINED = "ALGORITHM_DEFINED"
    PREFERENCE_DRIVEN = "PREFERENCE_DRIVEN"
    CAPTURE_LIMITED = "CAPTURE_LIMITED"
    RECONSTRUCTION_LIMITED = "RECONSTRUCTION_LIMITED"
    STRATEGY_LIMITED = "STRATEGY_LIMITED"
    MIXED_CAUSE = "MIXED_CAUSE"
    INDETERMINATE = "INDETERMINATE"


class ReportState(StrEnum):
    AUTO_REPORT = "AUTO_REPORT"
    REVIEWED_REPORT = "REVIEWED_REPORT"


class FaceRole(StrEnum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    UNASSIGNED = "UNASSIGNED"


class SceneScoreStatus(StrEnum):
    AUTO_PASS = "AUTO_PASS"
    PASS_WITH_NOTE = "PASS_WITH_NOTE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceGradeV2(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class ReportAllowedLevel(StrEnum):
    OBSERVABLE = "OBSERVABLE"
    STRATEGY_INFERENCE = "STRATEGY_INFERENCE"
    INTERNAL_HYPOTHESIS = "INTERNAL_HYPOTHESIS"


PAIRWISE_STRENGTH: dict[PairwisePreference, int] = {
    PairwisePreference.A_STRONGLY_BETTER: 2,
    PairwisePreference.A_SLIGHTLY_BETTER: 1,
    PairwisePreference.EQUIVALENT: 0,
    PairwisePreference.B_SLIGHTLY_BETTER: -1,
    PairwisePreference.B_STRONGLY_BETTER: -2,
}


class EvaluationBatchV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    batch_id: str
    project_id: str
    task_scope: Literal["front_camera_portrait_tone_semantic_rendering"] = (
        "front_camera_portrait_tone_semantic_rendering"
    )
    device_ids: list[str] = Field(min_length=2)
    scoring_scope: Literal["batch_relative"] = "batch_relative"
    dataset_version: str
    dimension_policy_version: str
    scoring_policy_version: str
    primary_model_version: str
    judge_model_version: str
    report_model_version: str | None
    created_at: datetime

    @model_validator(mode="after")
    def validate_distinct_devices(self) -> EvaluationBatchV2:
        if len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("device IDs must be unique")
        return self


class MatchEvidenceV2(V2Contract):
    exif_time_score: UnitFloat
    perceptual_hash_score: UnitFloat
    scene_embedding_score: UnitFloat
    face_geometry_score: UnitFloat | None
    composition_score: UnitFloat
    aggregate_score: UnitFloat


class MatchedSceneGroupV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    scene_id: str
    batch_id: str
    label: str | None
    cells: dict[str, str | None]
    match_status: MatchStatus
    match_confidence: UnitFloat
    match_evidence: MatchEvidenceV2
    comparability_status: ComparabilityStatus
    confounders: list[str] = Field(default_factory=list)
    confirmed_by: str | None


class FaceRegionV2(V2Contract):
    face_id: str
    bbox_xywh: tuple[int, int, int, int]
    detection_confidence: UnitFloat
    area_ratio: UnitFloat
    center_xy_normalized: tuple[UnitFloat, UnitFloat]
    landmarks_ref: str | None
    masks: dict[str, str] = Field(default_factory=dict)
    role: FaceRole


class ObjectiveEvidenceV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    image_id: str
    faces: list[FaceRegionV2] = Field(default_factory=list)
    whole_image_metrics: dict[str, float] = Field(default_factory=dict)
    region_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_face_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    relation_metrics: dict[str, float] = Field(default_factory=dict)
    artifact_metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metric_version: str


class DimensionApplicabilityV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    scene_id: str
    dimension_id: DimensionId
    status: ApplicabilityStatus
    score: UnitFloat
    reasons: list[str] = Field(default_factory=list)
    required_evidence_present: bool


class RoughRankingV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    scene_id: str
    dimension_id: DimensionId
    ordered_device_ids: list[str]
    tie_groups: list[list[str]] = Field(default_factory=list)
    confidence: UnitFloat
    evidence_refs: list[str] = Field(default_factory=list)
    prompt_version: str
    model_version: str


class PairwiseComparisonV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    comparison_id: str
    scene_id: str
    dimension_id: DimensionId
    device_a_id: str
    device_b_id: str
    applicability: ApplicabilityStatus
    preference: PairwisePreference
    strength: int = Field(ge=-2, le=2)
    difference_type: DifferenceType
    confidence: UnitFloat
    observations: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    style_axes: dict[str, str | float] = Field(default_factory=dict)
    primary_model_version: str
    prompt_version: str

    @model_validator(mode="after")
    def validate_preference_strength(self) -> PairwiseComparisonV2:
        if self.strength != PAIRWISE_STRENGTH[self.preference]:
            raise ValueError("pairwise preference and strength do not agree")
        if self.device_a_id == self.device_b_id:
            raise ValueError("pairwise devices must be distinct")
        return self


class FactCheckResultV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    comparison_id: str
    rule_version: str
    passed_rules: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    critical_conflict: bool
    objective_support: UnitFloat


class JudgeDecisionV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    comparison_id: str
    decision: JudgeDecisionType
    accepted_preference: PairwisePreference | None = None
    accepted_strength: int | None = Field(default=None, ge=-2, le=2)
    accepted_difference_type: DifferenceType | None = None
    revised_observations: list[str] = Field(default_factory=list)
    fact_conflicts: list[str] = Field(default_factory=list)
    unsupported_attributions: list[str] = Field(default_factory=list)
    confidence: UnitFloat
    judge_model_version: str
    prompt_version: str

    @model_validator(mode="after")
    def validate_accepted_semantics(self) -> JudgeDecisionV2:
        accepted = (
            self.accepted_preference,
            self.accepted_strength,
            self.accepted_difference_type,
        )
        requires_result = self.decision in {
            JudgeDecisionType.ACCEPT,
            JudgeDecisionType.REVISE,
        }
        if requires_result and any(value is None for value in accepted):
            raise ValueError("accept/revise requires accepted semantic fields")
        if not requires_result and any(value is not None for value in accepted):
            raise ValueError("reject/escalate cannot carry accepted semantic fields")
        if self.accepted_preference is not None:
            expected = PAIRWISE_STRENGTH[self.accepted_preference]
            if self.accepted_strength != expected:
                raise ValueError("accepted preference and strength do not agree")
        return self


class SceneDimensionScoreV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    scene_id: str
    dimension_id: DimensionId
    device_id: str
    latent_quality: float | None
    normalized_score: Score100 | None
    confidence: UnitFloat
    diagnostic_weight: UnitFloat
    objective_adjustment: ObjectiveAdjustment | None
    final_score: Score100 | None
    status: SceneScoreStatus
    contributing_comparison_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_score_presence(self) -> SceneDimensionScoreV2:
        score_values = (
            self.latent_quality,
            self.normalized_score,
            self.objective_adjustment,
            self.final_score,
        )
        if self.status is SceneScoreStatus.NOT_APPLICABLE:
            if any(value is not None for value in score_values):
                raise ValueError("NOT_APPLICABLE must not carry numeric quality")
            if self.contributing_comparison_ids:
                raise ValueError("NOT_APPLICABLE cannot reference scoring comparisons")
            return self
        if self.status is SceneScoreStatus.MANUAL_REVIEW:
            present_count = sum(value is not None for value in score_values)
            if present_count not in {0, len(score_values)}:
                raise ValueError("MANUAL_REVIEW requires either complete or no numeric quality")
            return self
        if any(value is None for value in score_values):
            raise ValueError("scored scene status requires complete numeric quality")
        return self


class DeviceDimensionScoreV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    batch_id: str
    device_id: str
    dimension_id: DimensionId
    aggregate_score: Score100
    rank: int | None = Field(default=None, ge=1)
    rank_group: int | None = Field(default=None, ge=1)
    confidence: UnitFloat
    applicable_scene_count: int = Field(ge=0)
    style_position: dict[str, str | float] = Field(default_factory=dict)
    supporting_scene_ids: list[str] = Field(default_factory=list)
    counterexample_scene_ids: list[str] = Field(default_factory=list)


class DeviceOverallScoreV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    batch_id: str
    device_id: str
    relative_score: Score100
    rank: int = Field(ge=1)
    confidence: UnitFloat
    dimension_scores: dict[DimensionId, Score100]
    score_scope: Literal["batch_relative"] = "batch_relative"


class MechanismInterpretationV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    interpretation_id: str
    device_id: str
    scene_ids: list[str]
    observable_effect: str
    strategy_inference: str
    mechanism_hypothesis: str | None
    attribution: MechanismAttribution
    confidence: UnitFloat
    evidence_grade: EvidenceGradeV2
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    report_allowed_level: ReportAllowedLevel


class ReportEvidencePackageV2(V2Contract):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    batch: EvaluationBatchV2
    scene_summaries: list[dict[str, Any]]
    device_dimension_scores: list[DeviceDimensionScoreV2]
    device_overall_scores: list[DeviceOverallScoreV2]
    device_profiles: list[dict[str, Any]]
    mechanism_interpretations: list[MechanismInterpretationV2]
    selected_visual_assets: list[dict[str, Any]]
    review_notes: list[dict[str, Any]]
    limitations: list[str]
    prohibited_claims: list[str]
