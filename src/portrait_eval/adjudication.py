import hashlib
import json

from portrait_eval.models import AdjudicatedClaim, ClaimCandidate, EvidenceGrade


def stable_claim_id(
    device_id: str,
    dimension: str,
    statement: str,
    scene_ids: list[str],
    claim_type: str = "observation",
) -> str:
    identity = [
        device_id,
        dimension,
        " ".join(statement.casefold().split()),
        sorted(set(scene_ids)),
        claim_type,
    ]
    return (
        "claim_"
        + hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]
    )


def adjudicate_claim(claim: ClaimCandidate) -> AdjudicatedClaim:
    support = set(claim.supporting_scene_ids)
    counters = set(claim.contradicting_scene_ids)
    coverage = len(support - counters)
    support_ratio = coverage / max(len(support | counters), 1)
    score = (
        (
            0.30 * claim.model_agreement
            + 0.35 * claim.objective_support
            + 0.25 * claim.scene_validity
            + 0.10 * min(coverage / 3.0, 1.0)
        )
        * support_ratio
        * claim.uncertainty
    )
    if claim.claim_type == "strategy" and coverage < 2:
        status, grade = "INSUFFICIENT_SCENE_COVERAGE", EvidenceGrade.C
    elif counters and (support_ratio < 0.8 or score < 0.8):
        status, grade = "DISPUTED", EvidenceGrade.C
    elif claim.provisional:
        status, grade = "PROVISIONAL", EvidenceGrade.C
    elif (
        claim.uncertainty == 0
        or claim.scene_validity == 0
        or (claim.model_agreement == 0 and claim.objective_support == 0)
    ):
        status, grade = "LOW_CONFIDENCE", EvidenceGrade.C
    elif coverage >= 3 and score >= 0.8:
        status, grade = "CONFIRMED", EvidenceGrade.A
    elif score >= 0.6:
        status, grade = "SUPPORTED", EvidenceGrade.B
    else:
        status, grade = "LOW_CONFIDENCE", EvidenceGrade.C
    return AdjudicatedClaim(
        claim_id=claim.claim_id
        or stable_claim_id(
            claim.device_id,
            claim.dimension,
            claim.statement,
            claim.supporting_scene_ids,
            claim.claim_type,
        ),
        dimension=claim.dimension,
        claim_type=claim.claim_type,
        provisional=claim.provisional,
        evidence_refs=claim.evidence_refs,
        source_claim_ids=claim.source_claim_ids,
        device_id=claim.device_id,
        statement=claim.statement,
        status=status,
        grade=grade,
        confidence=round(score, 3),
        supporting_scene_ids=claim.supporting_scene_ids,
        contradicting_scene_ids=claim.contradicting_scene_ids,
        alternative_explanations=claim.alternative_explanations,
    )
