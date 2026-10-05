import json

import numpy as np
from PIL import Image


def fixture(tmp_path):
    base = np.tile(np.arange(32, dtype=np.uint8)[None, :, None], (32, 1, 3)) * 6
    a, b = tmp_path / "b.png", tmp_path / "c.png"
    Image.fromarray(base).save(a)
    Image.fromarray(base + 5).save(b)
    return {
        "scene_id": "scene",
        "group_id": "canonical",
        "mode": "algorithm",
        "baseline": {"id": "B", "path": "b.png", "encoding": "srgb"},
        "candidates": [{"id": "C", "path": "c.png", "encoding": "srgb"}],
    }


def test_cli_module_exists():
    import importlib.util

    assert importlib.util.find_spec("qwen_tmqa.comparison_cli") is not None


def test_evaluate_resolves_relative_manifest_and_writes_regression_csv(tmp_path):
    from qwen_tmqa.comparison_cli import main

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(fixture(tmp_path)))
    result = tmp_path / "result.json"
    assert main(["evaluate", "--manifest", str(manifest), "--output", str(result)]) == 0
    data = json.loads(result.read_text())
    assert data["comparisons"][0]["status"] == "REVIEW"
    assert data["assets"][0]["trace"]["path"] == str(tmp_path / "b.png")
    assert "mean_luminance_delta" in result.with_suffix(".csv").read_text()


def test_folder_cli_multiple_versions_missing_files_and_unverified_lineage(tmp_path):
    from qwen_tmqa.comparison_cli import main

    for folder in ("base", "v1", "v2", "source"):
        (tmp_path / folder / "night").mkdir(parents=True)
        Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(
            tmp_path / folder / "night" / "x.png"
        )
    Image.fromarray(np.zeros((16, 16), np.uint8)).save(tmp_path / "base" / "missing.png")
    output = tmp_path / "batch.json"
    argv = [
        "compare-versions",
        "--baseline-root",
        str(tmp_path / "base"),
        "--candidate-root",
        str(tmp_path / "v1"),
        "--candidate-root",
        str(tmp_path / "v2"),
        "--source-root",
        str(tmp_path / "source"),
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    result = json.loads(output.read_text())
    assert len(result["scenes"]) == 1
    assert len(result["scenes"][0]["comparisons"]) == 2
    assert result["scenes"][0]["comparisons"][0]["same_source_status"] == "declared_unverified"
    assert result["missing_candidates"]
    assert "relative_paths_are_correspondence_not_source_identity" in result["warnings"]
    assert output.with_suffix(".csv").exists()


def test_audit_cli_and_invalid_manifest(tmp_path):
    from qwen_tmqa.comparison_cli import main

    manifest = tmp_path / "audit.json"
    manifest.write_text(json.dumps([{"id": "clean", "risk": False}, {"id": "risk", "risk": True}]))
    output = tmp_path / "audit-out.json"
    assert (
        main(
            [
                "audit-sample",
                "--manifest",
                str(manifest),
                "--sample-size",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(json.loads(output.read_text())["focused_risk_queue"]) == 1
    manifest.write_text("{broken")
    assert (
        main(["evaluate", "--manifest", str(manifest), "--output", str(tmp_path / "bad.json")]) == 2
    )
    assert not (tmp_path / "bad.json").exists()


def test_csv_output_name_does_not_overwrite_json(tmp_path):
    from qwen_tmqa.comparison_cli import main

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(fixture(tmp_path)))
    output = tmp_path / "output.csv"
    assert main(["evaluate", "--manifest", str(manifest), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["scene_id"] == "scene"
    assert (tmp_path / "output.regression.csv").exists()
