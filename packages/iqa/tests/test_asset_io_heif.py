"""Actual encoded HEIF containers: Front-supported photos remain valid in shared IQA."""

import hashlib
import json

import numpy as np
import pillow_heif
import pytest
from PIL import Image, ImageOps

from qwen_tmqa.asset_io import load_comparison_asset
from qwen_tmqa.comparison import evaluate_comparison
from qwen_tmqa.comparison_models import ComparisonAsset, ComparisonRequest


@pytest.mark.parametrize("orientation", range(1, 9))
def test_actual_heic_8bit_matches_front_display_orientation(tmp_path, orientation):
    pillow_heif.register_heif_opener()
    rgb = np.random.default_rng(901).integers(0, 255, (24, 40, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / "capture.heic"
    Image.fromarray(rgb).save(path, format="HEIF", quality=-1, chroma=444, exif=exif)
    with Image.open(path) as image:
        expected = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    loaded = load_comparison_asset(ComparisonAsset(id="capture", path=path))
    np.testing.assert_array_equal(loaded.pixels, expected / 255.0)
    assert loaded.trace["source_bit_depth"] == 8
    assert loaded.trace["source_bytes_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    result = evaluate_comparison(
        ComparisonRequest(
            scene_id="s",
            group_id="g",
            mode="device",
            baseline={"id": "B", "path": path},
            candidates=[{"id": "C", "path": path}],
        )
    )
    assert all(a.state == "valid" for a in result.assets)
    assert result.comparisons[0].fatal_reasons == []
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_actual_10bit_heic_keeps_more_than_256_decoded_levels(tmp_path):
    levels = np.arange(1024, dtype=np.uint16).reshape(32, 32) * 64
    rgb = np.repeat(levels[..., None], 3, axis=2)
    path = tmp_path / "capture-10bit.heic"
    pillow_heif.from_bytes("RGB;16", (32, 32), rgb.tobytes()).save(path, quality=-1, chroma=444)
    loaded = load_comparison_asset(ComparisonAsset(id="capture10", path=path))
    assert loaded.trace["source_bit_depth"] == 10
    assert loaded.trace["dtype"] == "uint16"
    assert loaded.trace["bit_depth"] == 16
    assert np.unique(loaded.pixels[..., 0]).size > 256
    diffs = np.diff(np.unique(loaded.pixels[..., 0]))
    assert np.any((diffs > 0) & (diffs < 1 / 255))
    # Encoding loss/conversion is native codec behavior, not fabricated raw equality.
    assert np.max(np.abs(loaded.pixels - rgb / 65535.0)) < 0.005


def test_heic_version_folder_scan_and_invalid_container_are_explicit(tmp_path):
    from qwen_tmqa.comparison_cli import compare_version_folders

    pillow_heif.register_heif_opener()
    roots = [tmp_path / name for name in ("baseline", "candidate", "source")]
    rgb = np.random.default_rng(8).integers(30, 220, (24, 40, 3), dtype=np.uint8)
    for root in roots:
        root.mkdir()
        Image.fromarray(rgb).save(root / "capture.HEIC", format="HEIF", quality=-1)
    batch = compare_version_folders(roots[0], [roots[1]], roots[2])
    assert len(batch["scenes"]) == 1
    assert all(a["state"] == "valid" for a in batch["scenes"][0]["assets"])
    path = roots[1] / "capture.HEIC"
    path.write_bytes(path.read_bytes()[:16])
    batch = compare_version_folders(roots[0], [roots[1]], roots[2])
    assert batch["scenes"][0]["comparisons"][0]["status"] == "REJECT"
    assert batch["scenes"][0]["assets"][1]["state"] == "invalid"
    assert (
        batch["scenes"][0]["assets"][1]["trace"]["source_bytes_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    json.dumps(batch, allow_nan=False)
