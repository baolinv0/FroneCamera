from portrait_eval.adjudication import adjudicate_claim
from portrait_eval.models import ClaimCandidate, ModelObservation
from portrait_eval.strategy import infer_cross_scene_claims


def candidate(**updates):
    payload = dict(
        device_id="a",
        statement="higher mean",
        claim_type="strategy",
        supporting_scene_ids=["G1", "G2", "G3"],
        contradicting_scene_ids=[],
        model_agreement=1,
        objective_support=1,
        scene_validity=1,
    )
    payload.update(updates)
    return ClaimCandidate(**payload)


def test_many_counterexamples_block_confirmation_even_with_high_score():
    result = adjudicate_claim(candidate(contradicting_scene_ids=[f"X{i}" for i in range(21)]))
    assert result.status == "DISPUTED"
    assert result.grade.value == "C"


def test_unknown_support_is_not_full_objective_support():
    result = adjudicate_claim(candidate(objective_support=None))
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.objective_support is None
    assert result.score_kind == "uncalibrated_rule_score"


def test_repeated_family_is_not_independent_coverage():
    result = adjudicate_claim(
        candidate(scene_family_by_id={"G1": "same", "G2": "same", "G3": "same"})
    )
    assert result.status == "INSUFFICIENT_SCENE_COVERAGE"
    assert result.independent_family_count == 1


def test_opposite_direction_and_different_region_are_not_agreement():
    from portrait_eval.evidence import review_relation

    a = ModelObservation(
        device_id="a",
        dimension="global_exposure",
        statement="brighter",
        certainty=0.9,
        region="whole",
        attribute="encoded_luma_mean",
        direction="higher",
        comparison="peer_median",
    )
    b = a.model_copy(update={"direction": "lower", "statement": "darker"})
    assert review_relation(a, b) == "contradicts"
    assert (
        review_relation(a, b.model_copy(update={"region": "face:F002"})) == "different_proposition"
    )
    assert review_relation(a, a.model_copy(update={"direction": "unknown"})) == "unknown"
    assert review_relation(a, a) == "supports"


def test_missing_metric_does_not_become_zero_clipping_evidence():
    scenes = [
        {
            "group_id": f"G{i}",
            "scene_family": f"F{i}",
            "metrics": {"a": {"whole": {"luma_mean": 0.8}}, "b": {"whole": {"luma_mean": 0.4}}},
        }
        for i in range(3)
    ]
    claims = infer_cross_scene_claims(scenes)
    assert not any("near-white" in c.statement for c in claims)
    assert all(c.status != "CONFIRMED" for c in claims)


def test_aggregate_keeps_unknown_and_incomparable_denominators():
    scenes = []
    for i in range(5):
        metrics = {"a": {"whole": {"luma_mean": 0.8}}, "b": {"whole": {"luma_mean": 0.4}}}
        scenes.append(
            {
                "group_id": f"G{i}",
                "scene_family": f"F{i}",
                "metrics": metrics,
                "audit": {"status": "FULLY_COMPARABLE"},
            }
        )
    scenes[3]["metrics"]["a"]["whole"] = {}
    scenes[4]["audit"]["status"] = "NOT_COMPARABLE"
    claim = next(c for c in infer_cross_scene_claims(scenes) if c.device_id == "a")
    assert claim.unknown_scene_ids == ["G3"]
    assert claim.incomparable_scene_ids == ["G4"]
    assert claim.evidence_counts == {
        "support": 3,
        "counterexample": 0,
        "unknown": 1,
        "incomparable": 1,
        "total": 5,
    }


def test_claim_identity_is_bound_to_device_and_version():
    from portrait_eval.evidence import claim_identity

    a = claim_identity("a", "G1", "same", "whole", "luma", "higher", "v1")
    assert a != claim_identity("b", "G1", "same", "whole", "luma", "higher", "v1")
    assert a != claim_identity("a", "G1", "same", "whole", "luma", "higher", "v2")


def test_unknown_scene_families_are_not_counted_as_independent():
    result = adjudicate_claim(candidate())
    assert result.independent_family_count == 0
    assert result.evidence_counts["total"] == 0
    assert result.unassigned_scene_ids == ["G1", "G2", "G3"]


def test_irrelevant_comparability_override_cannot_authorize_global_measurement():
    from portrait_eval.evidence import measure_observation

    obs = ModelObservation(
        device_id="a",
        dimension="global_exposure",
        statement="Higher",
        certainty=0.9,
        region="whole",
        attribute="encoded_luma_mean",
        direction="higher",
        comparison="peer_median",
    )
    metrics = {"a": {"whole": {"luma_mean": 0.8}}, "b": {"whole": {"luma_mean": 0.4}}}
    value, _ = measure_observation(
        obs,
        metrics,
        {
            "status": "FULLY_COMPARABLE",
            "attribute_comparability": {"global_exposure": "NOT_COMPARABLE"},
        },
    )
    assert value is None


def test_mixed_family_has_one_counterexample_outcome_not_two_votes():
    result = adjudicate_claim(
        candidate(
            supporting_scene_ids=["G1"],
            contradicting_scene_ids=["G2"],
            scene_family_by_id={"G1": "room", "G2": "room"},
        )
    )
    assert result.independent_family_count == 0
    assert result.evidence_counts == {
        "support": 0,
        "counterexample": 1,
        "unknown": 0,
        "incomparable": 0,
        "total": 1,
    }


def test_cross_scene_inference_respects_dimension_comparability_override():
    scenes = [
        {
            "group_id": f"G{i}",
            "scene_family": f"F{i}",
            "audit": {
                "status": "FULLY_COMPARABLE",
                "attribute_comparability": {"global_exposure": "NOT_COMPARABLE"},
            },
            "metrics": {"a": {"whole": {"luma_mean": 0.8}}, "b": {"whole": {"luma_mean": 0.2}}},
        }
        for i in range(2)
    ]
    assert not infer_cross_scene_claims(scenes)


def test_claim_identity_distinguishes_reference_and_conditions():
    from portrait_eval.evidence import claim_identity

    args = ("a", "G1", "same statement", "whole", "encoded_luma_mean", "higher", "v1")
    assert claim_identity(*args, comparison="b", conditions=["night"]) != claim_identity(
        *args, comparison="c", conditions=["day"]
    )
