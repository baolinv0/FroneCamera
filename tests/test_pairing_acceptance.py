from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from portrait_eval.dataset import propose_pairing, scan_folder
from portrait_eval.models import DeviceScan


def _scene_scan(root: Path, device: str, scenes: list[int]) -> DeviceScan:
    folder = root / device
    folder.mkdir()
    for ordinal, scene in enumerate(scenes, start=1):
        image = Image.new("RGB", (64, 64), "white")
        draw = ImageDraw.Draw(image)
        left = (scene - 1) * 22
        draw.rectangle((left, 0, min(left + 19, 63), 63), fill="black")
        image.save(folder / f"{ordinal}.png")
    return scan_folder(device, folder)


def _gray_scan(root: Path, device: str, stems: list[str]) -> DeviceScan:
    folder = root / device
    folder.mkdir()
    for stem in stems:
        Image.new("RGB", (32, 32), "gray").save(folder / f"{stem}.png")
    return scan_folder(device, folder)


def test_one_conflicting_member_requires_review_despite_high_median(tmp_path: Path) -> None:
    scans = [_scene_scan(tmp_path, device, [1, 2, 3]) for device in ("A", "B", "C")]
    scans.append(_scene_scan(tmp_path, "D", [3, 2, 1]))

    draft = propose_pairing(scans)

    assert not draft.confirmed
    for group, ordinal in zip(draft.groups, range(1, 4), strict=True):
        assert group.cells["D"] == tmp_path / "D" / f"{ordinal}.png"
        assert group.matching_confidence == 0.99
    assert [group.review_required for group in draft.groups] == [True, False, True]
    assert "content_match_requires_review" in draft.groups[0].match_notes
    assert "content_match_requires_review" in draft.groups[2].match_notes


def test_conflict_notes_belong_only_to_affected_groups(tmp_path: Path) -> None:
    scans = [_scene_scan(tmp_path, device, [1, 2, 3]) for device in ("A", "B", "C")]
    scans.append(_scene_scan(tmp_path, "D", [3, 2, 1]))

    draft = propose_pairing(scans)

    assert [note for note in draft.groups[1].match_notes if note.startswith("D:")] == [
        "D:explicit_numeric_ordinal"
    ]
    for index in (0, 2):
        assert [note for note in draft.groups[index].match_notes if note.startswith("D:")] == [
            "D:explicit_numeric_ordinal:content_conflict"
        ]


def test_one_low_confidence_member_requires_review_despite_high_median(tmp_path: Path) -> None:
    scans = [_gray_scan(tmp_path, device, ["1", "2", "3"]) for device in ("A", "B", "C")]
    scans.append(_gray_scan(tmp_path, "D", ["scene_1", "scene_2", "scene_3"]))

    draft = propose_pairing(scans)

    assert all(group.matching_confidence == 0.99 for group in draft.groups)
    assert all(group.review_required for group in draft.groups)
    assert all("content_match_requires_review" in group.match_notes for group in draft.groups)


@pytest.mark.parametrize("candidate_stems", [["01", "02"], ["1", "2"], ["01", "03"]])
def test_zero_padded_reference_preserves_explicit_numeric_positions(
    tmp_path: Path, candidate_stems: list[str]
) -> None:
    reference = _gray_scan(tmp_path, "A", [f"{ordinal:02}" for ordinal in range(1, 10)])
    candidate = _gray_scan(tmp_path, "B", candidate_stems)

    draft = propose_pairing([reference, candidate])

    expected_cells = {int(stem) - 1: tmp_path / "B" / f"{stem}.png" for stem in candidate_stems}
    assert [group.cells["B"] for group in draft.groups] == [
        expected_cells.get(index) for index in range(9)
    ]
    assert not draft.confirmed
    for index in expected_cells:
        assert draft.groups[index].matching_confidence == 0.99
        assert "B:explicit_numeric_ordinal" in draft.groups[index].match_notes
    assert all(
        draft.groups[index].review_required for index in range(9) if index not in expected_cells
    )


@pytest.mark.parametrize("reference_stems", [["1", "9"], ["01", "09"]])
def test_arbitrary_reference_numbers_are_not_positional_proof(
    tmp_path: Path, reference_stems: list[str]
) -> None:
    reference = _gray_scan(tmp_path, "A", reference_stems)
    candidate = _gray_scan(tmp_path, "B", ["1"])

    draft = propose_pairing([reference, candidate])

    paired_group = next(group for group in draft.groups if group.cells["B"] is not None)
    assert paired_group.matching_confidence == 0.25
    assert paired_group.review_required
    assert "B:content_sequence_alignment" in paired_group.match_notes
