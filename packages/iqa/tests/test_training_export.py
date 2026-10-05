import json
from pathlib import Path

import pytest


def fixture(tmp_path):
    from qwen_tmqa.image_io import file_sha256
    from qwen_tmqa.lineage import SplitAssignment
    from qwen_tmqa.training_data import HumanConfirmation, TrainingCandidate

    source = tmp_path / "source.png"
    candidate = tmp_path / "candidate.png"
    source.write_bytes(b"actual source image bytes")
    candidate.write_bytes(b"actual selected image bytes")
    record = TrainingCandidate(
        scene_id="scene",
        candidate_id="candidate-vB",
        source_path=str(source),
        candidate_path=str(candidate),
        source_sha256=file_sha256(source),
        candidate_sha256=file_sha256(candidate),
        split="train",
        canonical_scene_id="canonical",
        group_id="group",
        eligible=True,
        judge_ids=["independent-a", "independent-b"],
        synthetic=False,
        evidence_mode="algorithm",
        lineage={"dataset_scene_ids": ["scene"], "run_id": "run"},
    )
    confirmation = HumanConfirmation(
        confirmation_id="human-confirmation",
        reviewer_id="human-a",
        scene_id="scene",
        candidate_id="candidate-vB",
        canonical_scene_id="canonical",
        group_id="group",
        source_sha256=record.source_sha256,
        candidate_sha256=record.candidate_sha256,
        confirmed=True,
        confirmation_kind="human",
        synthetic=False,
        label_scope="global_rendering_preference",
        confirmed_at="2026-10-05T00:00:00Z",
    )
    assignment = SplitAssignment(
        scene_id="scene", canonical_scene_id="canonical", group_id="group", split="train"
    )
    return record, confirmation, {"scene": assignment}


def test_formal_export_requires_real_byte_bound_human_confirmation(tmp_path):
    from qwen_tmqa.training_data import export_training_data

    record, confirmation, assignments = fixture(tmp_path)
    with pytest.raises(ValueError, match="confirmation"):
        export_training_data(
            [record], tmp_path / "unconfirmed", confirmations=[], split_assignments=assignments
        )
    assert not (tmp_path / "unconfirmed").exists()
    summary = export_training_data(
        [record],
        tmp_path / "confirmed",
        confirmations=[confirmation],
        split_assignments=assignments,
    )
    assert summary["training_count"] == 1
    item = json.loads((tmp_path / "confirmed" / "training_manifest.jsonl").read_text())
    assert item["record_kind"] == "human_confirmed_training_label"
    assert item["candidate_id"] == "candidate-vB"
    assert item["confirmation"]["confirmation_id"] == "human-confirmation"
    assert item["training_weight"] == 1.0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"split": "holdout"}, "train"),
        ({"synthetic": True}, "synthetic"),
        ({"judge_ids": ["same", "same"]}, "unique"),
        ({"eligible": False}, "eligible"),
        ({"evidence_mode": "device"}, "algorithm"),
        ({"rejection_reasons": ["fatal"]}, "eligible"),
    ],
)
def test_formal_export_cannot_bypass_selection_gates(tmp_path, change, match):
    from qwen_tmqa.training_data import export_training_data

    record, confirmation, assignments = fixture(tmp_path)
    record = record.model_copy(update=change)
    with pytest.raises(ValueError, match=match):
        export_training_data(
            [record],
            tmp_path / "export",
            confirmations=[confirmation],
            split_assignments=assignments,
        )
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"confirmed": False},
        {"synthetic": True},
        {"candidate_sha256": "0" * 64},
        {"source_sha256": "0" * 64},
        {"group_id": "another-group"},
        {"label_scope": "face_identity"},
        {"confirmation_kind": "model"},
    ],
)
def test_invalid_or_stale_confirmation_never_exports(tmp_path, change):
    from qwen_tmqa.training_data import export_training_data

    record, confirmation, assignments = fixture(tmp_path)
    confirmation = confirmation.model_copy(update=change)
    with pytest.raises(ValueError):
        export_training_data(
            [record],
            tmp_path / "export",
            confirmations=[confirmation],
            split_assignments=assignments,
        )


@pytest.mark.parametrize("field", ["source_path", "candidate_path"])
def test_bytes_changed_since_confirmation_block_export(tmp_path, field):
    from qwen_tmqa.training_data import export_training_data

    record, confirmation, assignments = fixture(tmp_path)
    Path(getattr(record, field)).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        export_training_data(
            [record],
            tmp_path / "export",
            confirmations=[confirmation],
            split_assignments=assignments,
        )


def test_export_requires_closed_explicit_canonical_groups(tmp_path):
    from qwen_tmqa.training_data import export_training_data

    record, confirmation, assignments = fixture(tmp_path)
    with pytest.raises(ValueError, match="metadata"):
        export_training_data(
            [record], tmp_path / "empty", confirmations=[confirmation], split_assignments={}
        )
    record.lineage["dataset_scene_ids"] = ["scene", "derivative"]
    with pytest.raises(ValueError, match="closed"):
        export_training_data(
            [record],
            tmp_path / "not-closed",
            confirmations=[confirmation],
            split_assignments=assignments,
        )


def test_duplicate_confirmations_are_ambiguous(tmp_path):
    from qwen_tmqa.training_data import export_training_data

    record, confirmation, assignments = fixture(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        export_training_data(
            [record],
            tmp_path / "export",
            confirmations=[confirmation, confirmation],
            split_assignments=assignments,
        )
