from portrait_eval.adjudication import adjudicate_claim
from portrait_eval.models import ClaimCandidate, EvidenceGrade


def test_adjudication_rejects_single_scene_strategy_generalization() -> None:
    claim = ClaimCandidate(
        device_id="a",
        statement="The device consistently lifts global shadows.",
        claim_type="strategy",
        supporting_scene_ids=["G001"],
        contradicting_scene_ids=[],
        model_agreement=1.0,
        objective_support=1.0,
        scene_validity=1.0,
    )
    result = adjudicate_claim(claim)
    assert result.grade is EvidenceGrade.C
    assert result.status == "INSUFFICIENT_SCENE_COVERAGE"
