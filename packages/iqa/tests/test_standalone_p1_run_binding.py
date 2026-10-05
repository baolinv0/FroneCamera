from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from qwen_tmqa import cli
from qwen_tmqa.config import load_config
from qwen_tmqa.domain import SceneEvaluation
from qwen_tmqa.evaluation import EvaluationPipeline
from qwen_tmqa.lineage import (
    EvaluationRunManifest,
    SelectionLineage,
    build_evaluation_run_manifest,
    load_evaluation_run_manifest,
    load_split_assignments,
)
from qwen_tmqa.pseudo_gt import select_scene_pseudo_gt
from qwen_tmqa.training_data import HumanConfirmation, TrainingCandidate, export_training_data


def _scene_digest(scene: SceneEvaluation) -> str:
    payload = json.dumps(
        scene.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@pytest.fixture
def policy_runs(tmp_path, monkeypatch):
    # Real image metrics and policy decisions; only the external judge HTTP call is simulated.
    yy, xx = np.indices((128, 128))
    gradient = 0.3 + 0.15 * np.sin(xx * 1.2) + 0.1 * np.cos(yy * 1.1)
    baseline = np.repeat(gradient[:, :, None], 3, axis=2)
    candidate = baseline * 1.2
    candidate[0, 0, :] = 1.0
    images = {
        "inputs/scene.png": baseline,
        "levels/a_000/scene.png": baseline,
        "levels/a_p050/scene.png": candidate,
    }
    for name, pixels in images.items():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(destination)

    response = {
        "scores": dict.fromkeys(
            ["tone", "color", "fidelity", "control", "preference", "overall"], 90
        ),
        "decision": "KEEP",
        "confidence": 90,
        "issues": [],
        "rationale": "candidate better",
        "preferred_level": "a_p050",
        "runner_up_level": "a_000",
        "acceptable_levels": ["a_p050"],
        "level_scores": {"a_000": 0.7, "a_p050": 0.9},
        "selection_confidence": 0.9,
        "baseline_improvement": 0.2,
    }

    def transport(_request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(response)}}]}
        )

    def pipeline(config):
        return EvaluationPipeline(
            config,
            transports={
                "judge-a": httpx.MockTransport(transport),
                "judge-b": httpx.MockTransport(transport),
            },
        )

    monkeypatch.setattr(cli, "EvaluationPipeline", pipeline)
    paths = {}
    for name, threshold in [("old", 0.45), ("new", 0.0)]:
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(
            "dataset:\n  expected_levels: [a_000, a_p050]\njudges:\n"
            "  - {id: judge-a, role: primary, adapter: openai_compatible, model: real-a, "
            'base_url: "http://test/v1", prompt_version: "3.4"}\n'
            "  - {id: judge-b, role: arbiter, adapter: openai_compatible, model: real-b, "
            'base_url: "http://test/v1", prompt_version: "3.4"}\n'
            f"decision_policy:\n  objective_fatal_clipping_ratio: {threshold}\n",
            encoding="utf-8",
        )
        output = tmp_path / name
        cli.evaluate_command(
            tmp_path / "levels", output, config_path, source_root=tmp_path / "inputs"
        )
        paths[name] = {
            "config": config_path,
            "results": output / "evaluations.json",
            "manifest": output / "evaluation_run_manifest.json",
        }

    splits = tmp_path / "splits.csv"
    splits.write_text(
        "scene_id,canonical_scene_id,group_id,split\nscene.png,canonical,group,train\n"
    )
    old_scene = cli._load_evaluations(paths["old"]["results"])[0]
    new_scene = cli._load_evaluations(paths["new"]["results"])[0]
    assert (
        next(stage.status for stage in old_scene.stages if stage.stage_id == "hard_gate") == "PASS"
    )
    assert (
        next(stage.status for stage in new_scene.stages if stage.stage_id == "hard_gate") == "FAIL"
    )
    assert all(judge.available and not judge.synthetic for judge in old_scene.model_evaluations)
    return tmp_path, paths, splits


def _select_cli(policy_runs, *, results_run="old", manifest_run="old", output="selected"):
    root, paths, splits = policy_runs
    cli.select_candidates_command(
        paths[results_run]["results"],
        root / "levels",
        root / "inputs",
        paths[manifest_run]["config"],
        paths[manifest_run]["manifest"],
        splits,
        root / output,
    )
    return root / output


def _select_direct(policy_runs, scene, manifest):
    root, paths, _splits = policy_runs
    config = load_config(paths["old"]["config"])
    specs = cli._attach_sources(
        cli.discover_scenes(root / "levels", config.dataset), root / "inputs"
    )
    return select_scene_pseudo_gt(
        scene,
        specs[0],
        config.pseudo_gt,
        lineage=SelectionLineage(evaluation=manifest, split="train", split_file_sha256="c" * 64),
    )


def test_evaluate_manifest_binds_actual_scene_content(policy_runs):
    _root, paths, _splits = policy_runs
    manifests = {}
    for name in ["old", "new"]:
        scene = cli._load_evaluations(paths[name]["results"])[0]
        manifests[name] = json.loads(paths[name]["manifest"].read_text())
        assert manifests[name].get("scene_evaluation_sha256") == {
            scene.scene_id: _scene_digest(scene)
        }
    assert (
        manifests["old"]["dataset_manifest_sha256"] == manifests["new"]["dataset_manifest_sha256"]
    )
    assert manifests["old"]["prompt_versions"] == manifests["new"]["prompt_versions"]
    assert (
        manifests["old"]["scene_evaluation_sha256"] != manifests["new"]["scene_evaluation_sha256"]
    )


def test_cli_rejects_old_pass_results_paired_with_new_fail_policy_manifest(policy_runs):
    root, _paths, _splits = policy_runs
    with pytest.raises(ValueError, match="evaluation.*mismatch.*rerun"):
        _select_cli(policy_runs, manifest_run="new")
    assert not (root / "selected").exists()


def test_direct_selector_rejects_old_results_with_new_manifest(policy_runs):
    _root, paths, _splits = policy_runs
    old_scene = cli._load_evaluations(paths["old"]["results"])[0]
    manifest = load_evaluation_run_manifest(paths["new"]["manifest"])
    record = _select_direct(policy_runs, old_scene, manifest)
    assert not record.accepted
    assert not record.evidence_binding_pass
    assert "EVIDENCE_RUN_BINDING_MISMATCH" in record.rejection_reasons


def test_correct_pairing_allows_real_candidate_and_human_confirmed_training(policy_runs):
    root, _paths, splits = policy_runs
    output = _select_cli(policy_runs)
    candidate = TrainingCandidate.model_validate_json(
        (output / "candidate_suggestions.jsonl").read_text()
    )
    confirmation = HumanConfirmation(
        confirmation_id="review-1",
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
    summary = export_training_data(
        [candidate],
        root / "training",
        confirmations=[confirmation],
        split_assignments=load_split_assignments(splits, expected_scene_ids={candidate.scene_id}),
    )
    assert summary["training_count"] == 1
    training = json.loads((root / "training" / "training_manifest.jsonl").read_text())
    assert training["training_weight"] == 1.0


def test_correct_new_pairing_preserves_failed_hard_gate(policy_runs):
    output = _select_cli(policy_runs, results_run="new", manifest_run="new")
    assert (output / "candidate_suggestions.jsonl").read_text() == ""
    audit = json.loads((output / "candidate_audit.jsonl").read_text())
    assert "SCENE_HARD_GATE_FAIL" in audit["rejection_reasons"]
    assert audit["evidence_binding_pass"]


def test_historical_manifest_loads_but_cli_requires_rerun(policy_runs):
    root, paths, _splits = policy_runs
    payload = json.loads(paths["old"]["manifest"].read_text())
    payload.pop("scene_evaluation_sha256", None)
    paths["old"]["manifest"].write_text(json.dumps(payload))
    load_evaluation_run_manifest(paths["old"]["manifest"])
    with pytest.raises(ValueError, match="binding.*rerun"):
        _select_cli(policy_runs)
    assert not (root / "selected").exists()


def test_direct_selector_rejects_historical_manifest_without_inventing_binding(policy_runs):
    _root, paths, _splits = policy_runs
    scene = cli._load_evaluations(paths["old"]["results"])[0]
    payload = json.loads(paths["old"]["manifest"].read_text())
    payload.pop("scene_evaluation_sha256", None)
    manifest = EvaluationRunManifest.model_validate(payload)
    record = _select_direct(policy_runs, scene, manifest)
    assert not record.accepted
    assert not record.evidence_binding_pass
    assert "EVIDENCE_RUN_BINDING_MISSING" in record.rejection_reasons


@pytest.mark.parametrize("field", ["score", "policy", "stage", "objective", "judge_preference"])
def test_tampered_evaluated_content_fails_cli_and_direct_api(policy_runs, field):
    _root, paths, _splits = policy_runs
    scene = cli._load_evaluations(paths["old"]["results"])[0]
    if field == "score":
        scene.overall_score = 0.99
    elif field == "policy":
        scene.decision_provenance["thresholds"]["objective_fatal_clipping_ratio"] = 0.0
    elif field == "stage":
        scene.stages[1].evidence.append("changed evaluated evidence")
    elif field == "objective":
        scene.objective_by_level["a_p050"].clipping_ratio = 0.0
    else:
        scene.model_evaluations[0].parsed_response["selection_confidence"] = 0.99
    paths["old"]["results"].write_text(json.dumps([scene.model_dump(mode="json")]))
    with pytest.raises(ValueError, match="evaluation.*mismatch.*rerun"):
        _select_cli(policy_runs)
    record = _select_direct(
        policy_runs, scene, load_evaluation_run_manifest(paths["old"]["manifest"])
    )
    assert not record.accepted
    assert "EVIDENCE_RUN_BINDING_MISMATCH" in record.rejection_reasons


@pytest.mark.parametrize("coverage", ["missing", "extra"])
def test_cli_requires_manifest_scene_coverage_to_match_results(policy_runs, coverage):
    root, paths, _splits = policy_runs
    payload = json.loads(paths["old"]["manifest"].read_text())
    scene = cli._load_evaluations(paths["old"]["results"])[0]
    binding = {scene.scene_id: _scene_digest(scene)}
    if coverage == "missing":
        binding.clear()
    else:
        binding["other.png"] = "a" * 64
    payload["scene_evaluation_sha256"] = binding
    paths["old"]["manifest"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="coverage.*rerun"):
        _select_cli(policy_runs)
    assert not (root / "selected").exists()


@pytest.mark.parametrize("digest", ["g" * 64, "a" * 63, "a" * 65, "a" * 63 + "\n"])
def test_manifest_rejects_invalid_scene_digest(policy_runs, digest):
    _root, paths, _splits = policy_runs
    payload = json.loads(paths["old"]["manifest"].read_text())
    payload["scene_evaluation_sha256"] = {"scene.png": digest}
    with pytest.raises(ValueError, match="scene_evaluation_sha256"):
        EvaluationRunManifest.model_validate(payload)


def test_binding_uses_loaded_scene_even_if_results_file_changes_after_load(
    policy_runs, monkeypatch
):
    root, paths, _splits = policy_runs
    load = cli._load_evaluations

    def load_then_replace(path: Path):
        scenes = load(path)
        path.write_bytes(paths["new"]["results"].read_bytes())
        return scenes

    monkeypatch.setattr(cli, "_load_evaluations", load_then_replace)
    with pytest.raises(ValueError, match="evaluation.*mismatch.*rerun"):
        _select_cli(policy_runs, manifest_run="new")
    assert not (root / "selected").exists()


def test_results_json_formatting_and_key_order_do_not_change_binding(policy_runs):
    _root, paths, _splits = policy_runs
    payload = json.loads(paths["old"]["results"].read_text())
    payload = [{key: scene[key] for key in reversed(scene)} for scene in payload]
    paths["old"]["results"].write_text(json.dumps(payload, indent=4, sort_keys=True))
    output = _select_cli(policy_runs)
    assert json.loads((output / "candidate_summary.json").read_text())["candidate_count"] == 1


@pytest.mark.parametrize("evaluated_ids", [[], ["scene.png", "scene.png"], ["other.png"]])
def test_manifest_factory_rejects_missing_duplicate_or_wrong_scene_coverage(
    policy_runs, evaluated_ids
):
    root, paths, _splits = policy_runs
    config_path = paths["old"]["config"]
    config = load_config(config_path)
    specs = cli._attach_sources(
        cli.discover_scenes(root / "levels", config.dataset), root / "inputs"
    )
    scene = cli._load_evaluations(paths["old"]["results"])[0]
    evaluations = [scene.model_copy(update={"scene_id": scene_id}) for scene_id in evaluated_ids]
    with pytest.raises(ValueError, match="duplicate|coverage"):
        build_evaluation_run_manifest(specs, config_path, config, evaluations=evaluations)


def test_manifest_factory_adds_binding_without_changing_existing_input_identity(policy_runs):
    root, paths, _splits = policy_runs
    config_path = paths["old"]["config"]
    config = load_config(config_path)
    specs = cli._attach_sources(
        cli.discover_scenes(root / "levels", config.dataset), root / "inputs"
    )
    scenes = cli._load_evaluations(paths["old"]["results"])
    historical = build_evaluation_run_manifest(specs, config_path, config)
    bound = build_evaluation_run_manifest(specs, config_path, config, evaluations=scenes)
    assert bound.evaluation_run_id == historical.evaluation_run_id
    assert bound.evaluation_config_sha256 == historical.evaluation_config_sha256
    assert bound.dataset_manifest_sha256 == historical.dataset_manifest_sha256
    assert bound.scene_evaluation_sha256 == {scenes[0].scene_id: _scene_digest(scenes[0])}


def _evaluate_while_yaml_changes(policy_runs, monkeypatch):
    root, paths, _splits = policy_runs
    consumed_sha = hashlib.sha256(paths["old"]["config"].read_bytes()).hexdigest()
    pipeline_factory = cli.EvaluationPipeline

    def pipeline_with_config_change(config):
        pipeline = pipeline_factory(config)
        evaluate = pipeline.evaluate_scene

        def evaluate_then_change_yaml(spec):
            scene = evaluate(spec)
            paths["old"]["config"].write_bytes(paths["new"]["config"].read_bytes())
            return scene

        monkeypatch.setattr(pipeline, "evaluate_scene", evaluate_then_change_yaml)
        return pipeline

    monkeypatch.setattr(cli, "EvaluationPipeline", pipeline_with_config_change)
    output = root / "changed-config-evaluation"
    cli.evaluate_command(
        root / "levels", output, paths["old"]["config"], source_root=root / "inputs"
    )
    paths["old"]["results"] = output / "evaluations.json"
    paths["old"]["manifest"] = output / "evaluation_run_manifest.json"
    return consumed_sha


def test_evaluate_hashes_the_config_bytes_consumed_before_inference(policy_runs, monkeypatch):
    consumed_sha = _evaluate_while_yaml_changes(policy_runs, monkeypatch)
    _root, paths, _splits = policy_runs
    manifest = load_evaluation_run_manifest(paths["old"]["manifest"])
    assert manifest.evaluation_config_sha256 == consumed_sha
    scene = cli._load_evaluations(paths["old"]["results"])[0]
    assert scene.decision_provenance["thresholds"]["objective_fatal_clipping_ratio"] == 0.45
    assert manifest.scene_evaluation_sha256 == {scene.scene_id: _scene_digest(scene)}


def test_yaml_change_during_evaluation_cannot_admit_old_pass_as_new_policy(
    policy_runs, monkeypatch
):
    _evaluate_while_yaml_changes(policy_runs, monkeypatch)
    root, _paths, _splits = policy_runs
    with pytest.raises(ValueError, match="evaluation config hash mismatch.*rerun"):
        _select_cli(policy_runs)
    assert not (root / "selected").exists()


def test_selector_hashes_the_config_snapshot_it_actually_consumes(policy_runs, monkeypatch):
    root, paths, _splits = policy_runs
    # This strict config admits no candidate at 0.9 confidence. Loading the permissive config
    # before replacing its file must not validate it against a strict config's manifest.
    strict_path = root / "strict-selector.yaml"
    strict_path.write_text(
        paths["old"]["config"].read_text() + "pseudo_gt:\n  min_selection_confidence: 0.99\n"
    )
    strict_config = load_config(strict_path)
    specs = cli._attach_sources(
        cli.discover_scenes(root / "levels", strict_config.dataset), root / "inputs"
    )
    scenes = cli._load_evaluations(paths["old"]["results"])
    manifest = build_evaluation_run_manifest(specs, strict_path, strict_config, evaluations=scenes)
    paths["old"]["manifest"].write_text(manifest.model_dump_json())
    load = cli._load_evaluations

    def load_then_replace_config(path):
        loaded = load(path)
        paths["old"]["config"].write_bytes(strict_path.read_bytes())
        return loaded

    monkeypatch.setattr(cli, "_load_evaluations", load_then_replace_config)
    with pytest.raises(ValueError, match="evaluation config hash mismatch.*rerun"):
        _select_cli(policy_runs)
    assert not (root / "selected").exists()
