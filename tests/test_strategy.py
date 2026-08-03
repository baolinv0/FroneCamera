from portrait_eval.strategy import infer_cross_scene_claims


def test_strategy_inference_requires_three_scenes_for_grade_a_or_b() -> None:
    scenes = [
        {
            "group_id": "G001",
            "metrics": {"a": {"whole": {"luma_mean": 0.8}}, "b": {"whole": {"luma_mean": 0.4}}},
        },
        {
            "group_id": "G002",
            "metrics": {"a": {"whole": {"luma_mean": 0.7}}, "b": {"whole": {"luma_mean": 0.3}}},
        },
        {
            "group_id": "G003",
            "metrics": {"a": {"whole": {"luma_mean": 0.75}}, "b": {"whole": {"luma_mean": 0.35}}},
        },
    ]
    claims = infer_cross_scene_claims(scenes)
    claim = next(item for item in claims if item.device_id == "a")
    assert claim.grade.value in {"A", "B"}
    assert len(claim.supporting_scene_ids) == 3
