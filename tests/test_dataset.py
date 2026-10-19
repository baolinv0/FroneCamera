from pathlib import Path

from PIL import Image, ImageDraw

from portrait_eval.dataset import (
    DeviceScan,
    PairingDraft,
    natural_key,
    propose_pairing,
    scan_folder,
)


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
    assert draft.groups[1].cells["b"] is None
    assert draft.groups[2].cells["b"] == Path("3.jpg")


def test_scan_uses_dng_as_sequence_placeholder(tmp_path: Path) -> None:
    reference_folder = tmp_path / "reference"
    candidate_folder = tmp_path / "candidate"
    reference_folder.mkdir()
    candidate_folder.mkdir()
    for index in range(1, 4):
        Image.new("RGB", (20, 20), (index * 40, 10, 10)).save(reference_folder / f"{index}.jpg")
    Image.new("RGB", (20, 20), (40, 10, 10)).save(candidate_folder / "IMG_1.JPG")
    (candidate_folder / "IMG_2.DNG").write_bytes(b"raw-placeholder")
    Image.new("RGB", (20, 20), (120, 10, 10)).save(candidate_folder / "IMG_3.JPG")

    reference = scan_folder("reference", reference_folder)
    candidate = scan_folder("candidate", candidate_folder)
    draft = propose_pairing([reference, candidate])

    assert candidate.sequence_slots == [
        candidate_folder / "IMG_1.JPG",
        None,
        candidate_folder / "IMG_3.JPG",
    ]
    assert draft.groups[1].cells["candidate"] is None
    assert draft.groups[0].matching_confidence == 0.99
    assert not draft.groups[0].review_required


def test_content_alignment_finds_missing_middle_scene(tmp_path: Path) -> None:
    reference_files = []
    for index in range(3):
        path = tmp_path / f"reference_{index}.jpg"
        image = Image.new("RGB", (64, 64), "white")
        draw = ImageDraw.Draw(image)
        if index == 0:
            draw.rectangle((0, 0, 20, 63), fill="black")
        elif index == 1:
            draw.rectangle((22, 0, 42, 63), fill="black")
        else:
            draw.rectangle((44, 0, 63, 63), fill="black")
        image.save(path)
        reference_files.append(path)

    first = tmp_path / "candidate_first.jpg"
    third = tmp_path / "candidate_third.jpg"
    Image.open(reference_files[0]).save(first)
    Image.open(reference_files[2]).save(third)
    draft = propose_pairing(
        [
            DeviceScan(device_id="reference", files=reference_files),
            DeviceScan(device_id="candidate", files=[first, third]),
        ]
    )

    assert draft.groups[0].cells["candidate"] == first
    assert draft.groups[1].cells["candidate"] is None
    assert draft.groups[2].cells["candidate"] == third
