from portrait_eval.bias import assess_capture_bias


def test_confounded_single_scene_requires_reshoot_for_generalization() -> None:
    result = assess_capture_bias(
        supporting_scene_count=1,
        comparable_scene_count=0,
        independent_support_count=0,
        independent_contradiction_count=1,
    )
    assert result.status == "RESHOOT_REQUIRED_FOR_GENERALIZATION"
