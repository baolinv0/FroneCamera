from pathlib import Path

from portrait_eval.dataset import DeviceScan, PairingDraft, natural_key, propose_pairing


def test_natural_key_orders_numeric_names() -> None:
    names = ["10.jpg", "2.jpg", "1.jpg"]
    assert sorted(names, key=natural_key) == ["1.jpg", "2.jpg", "10.jpg"]


def test_pairing_preserves_missing_cells_without_shifting() -> None:
    scans = [
        DeviceScan(device_id="a", files=[Path("1.jpg"), Path("2.jpg"), Path("3.jpg")]),
        DeviceScan(device_id="b", files=[Path("1.jpg"), Path("3.jpg")]),
    ]
    draft: PairingDraft = propose_pairing(scans)
    assert len(draft.groups) == 3
    assert draft.groups[2].cells["b"] is None
    assert draft.groups[1].cells["b"] == Path("3.jpg")
