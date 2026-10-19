from pydantic import BaseModel, Field


class CaptureBiasAssessment(BaseModel):
    status: str
    confidence: float = Field(ge=0, le=1)
    reasons: list[str]
    recommended_actions: list[str]


def assess_capture_bias(
    supporting_scene_count: int,
    comparable_scene_count: int,
    independent_support_count: int,
    independent_contradiction_count: int,
) -> CaptureBiasAssessment:
    reasons: list[str] = []
    actions: list[str] = []
    if comparable_scene_count < supporting_scene_count:
        reasons.append("one_or_more_supporting_scenes_have_comparability_warnings")
    if independent_contradiction_count > independent_support_count:
        reasons.append("professional_review_evidence_contradicts_internal_generalization")
    if supporting_scene_count < 2:
        reasons.append("single_scene_or_single_capture_evidence")
    if supporting_scene_count < 2 and independent_contradiction_count > 0:
        status = "RESHOOT_REQUIRED_FOR_GENERALIZATION"
        confidence = 0.9
    elif reasons:
        status = "POSSIBLE_CAPTURE_BIAS"
        confidence = 0.65
    elif independent_support_count >= 2 and comparable_scene_count >= 2:
        status = "LOW_CAPTURE_BIAS_RISK"
        confidence = 0.8
    else:
        status = "UNRESOLVED"
        confidence = 0.5
    if status in {"POSSIBLE_CAPTURE_BIAS", "RESHOOT_REQUIRED_FOR_GENERALIZATION"}:
        actions = [
            "repeat each affected scene at least three times",
            "alternate devices during capture rather than completing one device first",
            "lock shooting mode and record beautification, fill-light, night-mode and portrait-mode states",
            "keep subject pose, face scale and lighting timing consistent",
            "retain original uncompressed files and firmware metadata",
        ]
    return CaptureBiasAssessment(
        status=status,
        confidence=confidence,
        reasons=reasons,
        recommended_actions=actions,
    )
