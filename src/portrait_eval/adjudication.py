from portrait_eval.models import AdjudicatedClaim, ClaimCandidate, EvidenceGrade


def adjudicate_claim(claim: ClaimCandidate) -> AdjudicatedClaim:
    coverage = len(set(claim.supporting_scene_ids))
    score = (
        0.30 * claim.model_agreement
        + 0.35 * claim.objective_support
        + 0.25 * claim.scene_validity
        + 0.10 * min(coverage / 3.0, 1.0)
    )
    if claim.claim_type == "strategy" and coverage < 2:
        status = "INSUFFICIENT_SCENE_COVERAGE"
        grade = EvidenceGrade.C
    elif claim.contradicting_scene_ids and score < 0.8:
        status = "DISPUTED"
        grade = EvidenceGrade.C
    elif coverage >= 3 and score >= 0.8:
        status = "CONFIRMED"
        grade = EvidenceGrade.A
    elif score >= 0.6:
        status = "SUPPORTED"
        grade = EvidenceGrade.B
    else:
        status = "LOW_CONFIDENCE"
        grade = EvidenceGrade.C
    return AdjudicatedClaim(
        device_id=claim.device_id,
        statement=claim.statement,
        status=status,
        grade=grade,
        confidence=round(score, 3),
        supporting_scene_ids=claim.supporting_scene_ids,
        contradicting_scene_ids=claim.contradicting_scene_ids,
        alternative_explanations=claim.alternative_explanations,
    )
