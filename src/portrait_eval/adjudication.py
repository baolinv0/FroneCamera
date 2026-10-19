from portrait_eval.models import AdjudicatedClaim, ClaimCandidate, EvidenceGrade


def adjudicate_claim(claim: ClaimCandidate) -> AdjudicatedClaim:
    support = set(claim.supporting_scene_ids)
    counter = set(claim.contradicting_scene_ids)
    unknown = set(claim.unknown_scene_ids)
    incomparable = set(claim.incomparable_scene_ids)
    all_scenes = support | counter | unknown | incomparable
    outcomes: dict[str, set[str]] = {}
    for scene_id in all_scenes:
        family = claim.scene_family_by_id.get(scene_id)
        if family:
            outcomes.setdefault(family, set()).add(
                "counterexample"
                if scene_id in counter
                else "support"
                if scene_id in support
                else "unknown"
                if scene_id in unknown
                else "incomparable"
            )
    collapsed = {
        family: next(
            label
            for label in ("counterexample", "support", "unknown", "incomparable")
            if label in labels
        )
        for family, labels in outcomes.items()
    }
    family_count = sum(label == "support" for label in collapsed.values())
    score = (
        0.30 * claim.model_agreement
        + 0.35 * (claim.objective_support or 0)
        + 0.25 * claim.scene_validity
        + 0.10 * min(family_count / 3, 1)
    )
    if counter or "contradicts" in claim.review_relations or claim.objective_support == 0:
        status = "DISPUTED"
    elif claim.scene_validity == 0:
        status = "NOT_COMPARABLE"
    elif claim.objective_support is None or not support:
        status = "INSUFFICIENT_EVIDENCE"
    elif claim.claim_type == "strategy" and family_count < 2:
        status = "INSUFFICIENT_SCENE_COVERAGE"
    else:
        status = "SUPPORTED_SCOPED"
    # Legacy grade is a rule label, never a calibrated confidence or general law.
    grade = EvidenceGrade.B if status == "SUPPORTED_SCOPED" else EvidenceGrade.C
    return AdjudicatedClaim(
        **claim.model_dump(exclude={"claim_type", "model_agreement", "scene_validity"}),
        status=status,
        grade=grade,
        confidence=round(score, 3),
        independent_family_count=family_count,
        unassigned_scene_ids=sorted(s for s in all_scenes if not claim.scene_family_by_id.get(s)),
        evidence_counts={
            **{
                key: sum(label == key for label in collapsed.values())
                for key in ("support", "counterexample", "unknown", "incomparable")
            },
            "total": len(collapsed),
        },
    )
