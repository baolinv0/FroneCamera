from portrait_eval.hardware import HardwareFact, SourceTier, reconcile_hardware_facts


def test_official_fact_wins_but_conflict_is_preserved() -> None:
    facts = [
        HardwareFact(
            field_name="front_focus",
            value="AF",
            source_url="https://vendor.example",
            source_tier=SourceTier.OFFICIAL,
        ),
        HardwareFact(
            field_name="front_focus",
            value="fixed",
            source_url="https://review.example",
            source_tier=SourceTier.PROFESSIONAL_REVIEW,
        ),
    ]
    profile = reconcile_hardware_facts(facts)
    assert profile.values["front_focus"] == "AF"
    assert "front_focus" in profile.conflicts
