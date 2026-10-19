from portrait_eval.strategy import infer_cross_scene_claims


def test_strategy_inference_scopes_evidence_to_confirmed_scene_families() -> None:
    scenes = [
        {
            "group_id": "G001",
            "scene_family": "family1",
            "metrics": {"a": {"whole": {"luma_mean": 0.8}}, "b": {"whole": {"luma_mean": 0.4}}},
        },
        {
            "group_id": "G002",
            "scene_family": "family2",
            "metrics": {"a": {"whole": {"luma_mean": 0.7}}, "b": {"whole": {"luma_mean": 0.3}}},
        },
        {
            "group_id": "G003",
            "scene_family": "family3",
            "metrics": {"a": {"whole": {"luma_mean": 0.75}}, "b": {"whole": {"luma_mean": 0.35}}},
        },
    ]
    claims = infer_cross_scene_claims(scenes)
    claim = next(item for item in claims if item.device_id == "a")
    assert claim.grade.value in {"A", "B"}
    assert len(claim.supporting_scene_ids) == 3
