import json
import os
import subprocess
import sys
from pathlib import Path

from test_candidate_selection_integrity import _scene, _spec
from test_training_export import fixture

from qwen_tmqa.cli import build_parser, evaluate_command, main
from qwen_tmqa.config import load_config
from qwen_tmqa.lineage import build_evaluation_run_manifest, write_evaluation_run_manifest


def test_evaluate_writes_content_bound_run_manifest(tmp_path):
    from qwen_tmqa.cli import make_example

    dataset = tmp_path / "dataset"
    make_example(dataset, 1)
    config = Path("configs/default.yaml")
    evaluate_command(dataset, tmp_path / "results", config)
    payload = json.loads((tmp_path / "results" / "evaluation_run_manifest.json").read_text())
    assert len(payload["dataset_manifest_sha256"]) == 64
    assert len(payload["evaluation_config_sha256"]) == 64


def test_select_cli_exports_suggestions_with_zero_training_weight(tmp_path):
    spec = _spec(tmp_path)
    # The legacy discovery config describes the two test levels only.
    config = tmp_path / "config.yaml"
    config.write_text("dataset:\n  expected_levels: [a_000, a_p050]\njudges: []\n")
    scenes = tmp_path / "evaluations.json"
    scene = _scene(spec)
    scenes.write_text(json.dumps([scene.model_dump(mode="json")]))
    manifest = build_evaluation_run_manifest(
        [spec], config, load_config(config), evaluations=[scene]
    )
    manifest.prompt_versions = ["tmqa.sequence@3.4"]
    manifest_path = tmp_path / "run.json"
    write_evaluation_run_manifest(manifest_path, manifest)
    splits = tmp_path / "splits.csv"
    splits.write_text(
        "scene_id,canonical_scene_id,group_id,split\nscene.png,canonical,group,train\n"
    )
    assert (
        main(
            [
                "select-candidates",
                "--results",
                str(scenes),
                "--root",
                str(tmp_path / "levels"),
                "--source-root",
                str(tmp_path / "inputs"),
                "--config",
                str(config),
                "--run-manifest",
                str(manifest_path),
                "--splits",
                str(splits),
                "--output",
                str(tmp_path / "suggestions"),
            ]
        )
        == 0
    )
    item = json.loads((tmp_path / "suggestions" / "candidate_suggestions.jsonl").read_text())
    assert item["candidate_id"] == "a_p050"
    assert item["record_kind"] == "candidate_suggestion"
    assert item["training_weight"] == 0.0
    assert not (tmp_path / "suggestions" / "training_manifest.jsonl").exists()


def test_export_training_cli_enforces_gate_in_actual_subprocess(tmp_path):
    record, confirmation, _ = fixture(tmp_path)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(record.model_dump_json() + "\n")
    confirmations = tmp_path / "confirmations.jsonl"
    confirmations.write_text("")
    splits = tmp_path / "splits.csv"
    splits.write_text("scene_id,canonical_scene_id,group_id,split\nscene,canonical,group,train\n")
    cmd = [
        sys.executable,
        "-m",
        "qwen_tmqa.cli",
        "export-training",
        "--candidates",
        str(candidates),
        "--confirmations",
        str(confirmations),
        "--splits",
        str(splits),
        "--output",
        str(tmp_path / "out"),
    ]
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    rejected = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert rejected.returncode != 0
    assert "confirmation" in rejected.stderr
    assert not (tmp_path / "out").exists()
    confirmations.write_text(confirmation.model_dump_json() + "\n")
    accepted = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert accepted.returncode == 0, accepted.stderr
    assert (
        json.loads((tmp_path / "out" / "training_manifest.jsonl").read_text())["training_weight"]
        == 1.0
    )


def test_selection_cli_exposes_explicit_source_and_lineage_arguments():
    choices = build_parser()._subparsers._group_actions[0].choices
    assert "select-candidates" in choices
    assert "export-training" in choices
