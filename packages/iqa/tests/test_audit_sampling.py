import importlib.util


def test_sampler_present():
    assert importlib.util.find_spec("qwen_tmqa.audit_sampling") is not None


def test_clean_and_risk_strata_deterministic_and_input_order_independent():
    from qwen_tmqa.audit_sampling import representative_audit_sample

    rows = [
        {
            "id": f"{split}-{i}",
            "split": "audit",
            "risk": split == "risk",
            "scene_type": "portrait" if i % 2 else "night",
        }
        for split in ("clean", "risk")
        for i in range(10)
    ]
    first = representative_audit_sample(rows, sample_size=8, seed="frozen")
    second = representative_audit_sample(list(reversed(rows)), sample_size=8, seed="frozen")
    assert first == second
    assert len(first["sample"]) == 8
    assert {x["risk"] for x in first["sample"]} == {True, False}
    assert len(first["focused_risk_queue"]) == 10
    assert first["purpose"] == "representative_audit"
    assert first["sample_design"]["seed"] == "frozen"
