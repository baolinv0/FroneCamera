from pathlib import Path


def test_shared_core_canary_exercises_both_entrypoints_without_training(tmp_path: Path) -> None:
    from scripts.run_integration_canary import run

    summary = run(tmp_path)
    assert summary["comparison_deterministic"] is True
    assert summary["front_iqa_scene_count"] == summary["front_scene_count"] == 2
    assert summary["source_byte_binding"] is True
    assert summary["privacy_export_has_host_paths"] is False
    assert summary["evidence_kind"] == "synthetic_images_and_heuristic_adapters"
    assert summary["paid_endpoint_calls"] == summary["training_runs"] == 0
