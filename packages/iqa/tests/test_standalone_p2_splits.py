from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_standalone_p1_run_binding as run_binding

from qwen_tmqa.lineage import load_split_assignments
from qwen_tmqa.training_data import HumanConfirmation, TrainingCandidate

policy_runs = run_binding.policy_runs


@pytest.mark.parametrize(
    "header,row,missing",
    [
        ("scene_id,split,canonical_scene_id,group_id", "scene.png,train", "canonical_scene_id"),
        ("scene_id,split,canonical_scene_id,group_id", "scene.png,train,canonical", "group_id"),
        (
            "scene_id,split,group_id,canonical_scene_id",
            "scene.png,train,group",
            "canonical_scene_id",
        ),
        ("canonical_scene_id,group_id,split,scene_id", "canonical,group,train", "scene_id"),
        ("scene_id,canonical_scene_id,group_id,split", "scene.png,canonical,group", "split"),
    ],
)
def test_truncated_required_cells_are_rejected_before_normalization(tmp_path, header, row, missing):
    splits = tmp_path / "splits.csv"
    splits.write_text(f"{header}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"split file.*row.*{missing}"):
        load_split_assignments(splits, expected_scene_ids={"scene.png"})


@pytest.mark.parametrize("field", ["scene_id", "canonical_scene_id", "group_id", "split"])
@pytest.mark.parametrize("blank", ["", " \t "])
def test_required_cells_cannot_be_blank(tmp_path, field, blank):
    row = {
        "scene_id": "scene.png",
        "canonical_scene_id": "canonical",
        "group_id": "group",
        "split": "train",
    }
    row[field] = blank
    splits = tmp_path / "splits.csv"
    with splits.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError):
        load_split_assignments(splits, expected_scene_ids={"scene.png"})


@pytest.mark.parametrize("extra", [",unexpected", ","])
def test_rows_with_extra_unnamed_cells_are_rejected(tmp_path, extra):
    splits = tmp_path / "splits.csv"
    splits.write_text(
        f"scene_id,canonical_scene_id,group_id,split\nscene.png,canonical,group,train{extra}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split file.*row.*extra columns"):
        load_split_assignments(splits, expected_scene_ids={"scene.png"})


def test_valid_csv_normalizes_whitespace_and_accepts_named_optional_columns(tmp_path):
    splits = tmp_path / "splits.csv"
    splits.write_text(
        "split,group_id,scene_id,canonical_scene_id,note\n"
        ' train , group , scene.png , canonical ,"contains,a comma"\n',
        encoding="utf-8",
    )
    assignment = load_split_assignments(splits, expected_scene_ids={"scene.png"})["scene.png"]
    assert assignment.model_dump() == {
        "scene_id": "scene.png",
        "canonical_scene_id": "canonical",
        "group_id": "group",
        "split": "train",
    }


def test_literal_none_text_is_valid_identity_metadata(tmp_path):
    splits = tmp_path / "splits.csv"
    splits.write_text(
        "scene_id,split,canonical_scene_id,group_id\nNone,train,None,None\n",
        encoding="utf-8",
    )
    assignment = load_split_assignments(splits, expected_scene_ids={"None"})["None"]
    assert assignment.scene_id == assignment.canonical_scene_id == assignment.group_id == "None"
    assert assignment.split == "train"


def _cli(*args):
    package = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(package / "src")}
    return subprocess.run(
        [sys.executable, "-m", "qwen_tmqa.cli", *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _select(policy_runs, output):
    root, paths, splits = policy_runs
    return _cli(
        "select-candidates",
        "--results",
        paths["old"]["results"],
        "--root",
        root / "levels",
        "--source-root",
        root / "inputs",
        "--config",
        paths["old"]["config"],
        "--run-manifest",
        paths["old"]["manifest"],
        "--splits",
        splits,
        "--output",
        output,
    )


def _export_confirmed(policy_runs, selected, output):
    root, _paths, splits = policy_runs
    candidate = TrainingCandidate.model_validate_json(
        (selected / "candidate_suggestions.jsonl").read_text(encoding="utf-8")
    )
    confirmation = HumanConfirmation(
        confirmation_id="human-review",
        reviewer_id="human-1",
        scene_id=candidate.scene_id,
        candidate_id=candidate.candidate_id,
        canonical_scene_id=candidate.canonical_scene_id,
        group_id=candidate.group_id,
        source_sha256=candidate.source_sha256,
        candidate_sha256=candidate.candidate_sha256,
        confirmed=True,
        confirmation_kind="human",
        synthetic=False,
        label_scope="tone_mapping_preference",
        confirmed_at="2026-10-05T00:00:00Z",
    )
    confirmations = root / "confirmations.jsonl"
    confirmations.write_text(confirmation.model_dump_json() + "\n", encoding="utf-8")
    return _cli(
        "export-training",
        "--candidates",
        selected / "candidate_suggestions.jsonl",
        "--confirmations",
        confirmations,
        "--splits",
        splits,
        "--output",
        output,
    )


def test_select_cli_rejects_truncated_metadata_before_creating_output(policy_runs):
    root, _paths, splits = policy_runs
    splits.write_text(
        "scene_id,split,canonical_scene_id,group_id\nscene.png,train\n", encoding="utf-8"
    )
    output = root / "rejected-selection"
    result = _select(policy_runs, output)
    assert result.returncode != 0, result.stdout
    assert "split file" in result.stderr and "canonical_scene_id" in result.stderr
    assert not output.exists()


def test_export_cli_rejects_truncation_even_when_literal_none_candidate_is_confirmed(policy_runs):
    root, _paths, splits = policy_runs
    splits.write_text(
        "scene_id,split,canonical_scene_id,group_id\nscene.png,train,None,None\n",
        encoding="utf-8",
    )
    selected = root / "literal-none-candidate"
    selection = _select(policy_runs, selected)
    assert selection.returncode == 0, selection.stderr
    splits.write_text(
        "scene_id,split,canonical_scene_id,group_id\nscene.png,train\n", encoding="utf-8"
    )
    output = root / "rejected-training"
    result = _export_confirmed(policy_runs, selected, output)
    assert result.returncode != 0, result.stdout
    assert "split file" in result.stderr and "canonical_scene_id" in result.stderr
    assert not output.exists()


def test_correct_csv_allows_bound_candidate_and_human_confirmed_formal_export(policy_runs):
    root, _paths, _splits = policy_runs
    selected = root / "correct-selection"
    selection = _select(policy_runs, selected)
    assert selection.returncode == 0, selection.stderr
    output = root / "correct-training"
    exported = _export_confirmed(policy_runs, selected, output)
    assert exported.returncode == 0, exported.stderr
    record = json.loads((output / "training_manifest.jsonl").read_text(encoding="utf-8"))
    assert record["training_weight"] == 1.0
    assert record["canonical_scene_id"] == "canonical"
    assert record["group_id"] == "group"
    assert record["record_kind"] == "human_confirmed_training_label"
