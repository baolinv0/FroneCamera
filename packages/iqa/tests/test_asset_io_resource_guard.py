"""IQA-R2-01: actual hostile PNG dimensions must not abort an otherwise valid batch."""

import hashlib
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qwen_tmqa.asset_io import AssetDecodeError, load_comparison_asset
from qwen_tmqa.comparison import evaluate_comparison
from qwen_tmqa.comparison_cli import compare_version_folders
from qwen_tmqa.comparison_models import ComparisonAsset, ComparisonRequest


def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def guarded_png_bytes():
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 20000, 20000, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(b"\x00tiny"))
        + png_chunk(b"IEND", b"")
    )


def files(tmp_path):
    valid = tmp_path / "valid.png"
    bad = tmp_path / "oversized-header.png"
    Image.fromarray(np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)).save(valid)
    bad.write_bytes(guarded_png_bytes())
    return valid, bad


def request(valid, bad):
    return ComparisonRequest(
        scene_id="resource-guard",
        group_id="g",
        mode="algorithm",
        baseline={"id": "B", "path": valid},
        source={"id": "S", "path": valid},
        candidates=[{"id": "oversized", "path": bad}, {"id": "unaffected", "path": valid}],
    )


def test_loader_resource_guard_becomes_byte_traced_invalid_without_disabling_pillow_limit(tmp_path):
    valid, bad = files(tmp_path)
    threshold = Image.MAX_IMAGE_PIXELS
    assert threshold is not None and threshold > 0
    with pytest.raises(Image.DecompressionBombError):
        Image.open(bad)
    with pytest.raises(AssetDecodeError) as exc:
        load_comparison_asset(ComparisonAsset(id="oversized", path=bad))
    assert isinstance(exc.value.__cause__, Image.DecompressionBombError)
    assert exc.value.trace["source_bytes_sha256"] == hashlib.sha256(bad.read_bytes()).hexdigest()
    assert exc.value.trace["byte_count"] == len(bad.read_bytes())
    assert Image.MAX_IMAGE_PIXELS == threshold
    with pytest.raises(Image.DecompressionBombError):
        Image.open(bad)
    assert load_comparison_asset(ComparisonAsset(id="valid", path=valid)).pixels.shape == (
        24,
        32,
        3,
    )


def test_guarded_candidate_returns_finite_rejection_and_valid_sibling_survives(tmp_path):
    valid, bad = files(tmp_path)
    result = evaluate_comparison(request(valid, bad))
    assert result.assets[1].state == "invalid"
    assert (
        result.assets[1].trace["source_bytes_sha256"]
        == hashlib.sha256(bad.read_bytes()).hexdigest()
    )
    assert result.comparisons[0].status == "REJECT"
    assert result.comparisons[0].fatal_reasons == ["invalid_candidate"]
    assert result.comparisons[1].status == "REVIEW"
    assert result.comparisons[1].relative_metrics["pixel_mae"] == 0
    assert result.assets[2].state == "valid"
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_installed_cli_writes_reject_json_csv_and_valid_sibling_without_traceback(tmp_path):
    valid, bad = files(tmp_path)
    manifest, output = tmp_path / "manifest.json", tmp_path / "result.json"
    manifest.write_text(request(valid, bad).model_dump_json())
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    run = subprocess.run(
        ["iqa-compare", "evaluate", "--manifest", str(manifest), "--output", str(output)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "Traceback" not in run.stderr
    data = json.loads(output.read_text())
    assert data["comparisons"][0]["status"] == "REJECT"
    assert data["comparisons"][1]["status"] == "REVIEW"
    assert (
        data["assets"][1]["trace"]["source_bytes_sha256"]
        == hashlib.sha256(bad.read_bytes()).hexdigest()
    )
    csv = output.with_suffix(".csv").read_text()
    assert "oversized,REJECT" in csv
    assert "unaffected,REVIEW" in csv
    json.dumps(data, allow_nan=False)


def test_version_folder_retains_other_scene_after_resource_guard_rejection(tmp_path):
    valid, bad = files(tmp_path)
    roots = [tmp_path / name for name in ("baseline", "candidate", "source")]
    for root in roots:
        root.mkdir()
        (root / "good.png").write_bytes(valid.read_bytes())
        (root / "bad.png").write_bytes(valid.read_bytes())
    (roots[1] / "bad.png").write_bytes(bad.read_bytes())
    batch = compare_version_folders(roots[0], [roots[1]], roots[2])
    scenes = {scene["scene_id"]: scene for scene in batch["scenes"]}
    assert scenes["bad.png"]["comparisons"][0]["status"] == "REJECT"
    assert scenes["good.png"]["comparisons"][0]["status"] == "REVIEW"
    json.dumps(batch, allow_nan=False)
