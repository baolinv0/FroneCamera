"""R1 findings must fail against the pre-remediation decoder and request validator."""

import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from qwen_tmqa.asset_io import AssetDecodeError, load_comparison_asset
from qwen_tmqa.comparison import evaluate_comparison
from qwen_tmqa.comparison_cli import main
from qwen_tmqa.comparison_models import ComparisonAsset, ComparisonRequest


def oriented(array, orientation):
    operations = {
        1: lambda a: a,
        2: np.fliplr,
        3: lambda a: np.rot90(a, 2),
        4: np.flipud,
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.rot90(a, -1),
        7: lambda a: np.flip(np.swapaxes(a, 0, 1), axis=(0, 1)),
        8: lambda a: np.rot90(a, 1),
    }
    return operations[orientation](array)


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_tiff_orientation_once_preserves_source_precision(tmp_path, orientation, dtype):
    # Distinct positions/adjacent 16-bit values expose rotations AND quantization.
    raw = (np.arange(35).reshape(5, 7) + 1200).astype(dtype)
    path = tmp_path / f"o-{orientation}-{np.dtype(dtype).name}.tiff"
    Image.fromarray(raw).save(path, tiffinfo={274: orientation}, compression="raw")
    loaded = load_comparison_asset(ComparisonAsset(id="tiff", path=path))
    expected = np.repeat(oriented(raw, orientation)[..., None], 3, axis=2)
    np.testing.assert_array_equal(loaded.pixels, expected / np.iinfo(dtype).max)
    assert loaded.trace["bit_depth"] == np.dtype(dtype).itemsize * 8
    assert loaded.trace["exif_orientation"] == orientation
    assert loaded.trace["original_shape"] == [5, 7, 3]
    assert loaded.trace["source_bytes_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["algorithm", "device"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("explicit_asset", ["B", "C"])
def test_implicit_and_explicit_effective_roi_collision_rejected(
    tmp_path, mode, reverse, explicit_asset
):
    image = tmp_path / "image.png"
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(image)
    rois = [
        {"id": "face", "kind": "face", "person_id": "person", "bbox": [0, 0, 8, 8]},
        {
            "id": "face",
            "kind": "face",
            "person_id": "person",
            "bbox": [20, 20, 8, 8],
            "asset_id": explicit_asset,
        },
    ]
    if reverse:
        rois.reverse()
    with pytest.raises(ValidationError):
        ComparisonRequest(
            scene_id="s",
            group_id="g",
            mode=mode,
            baseline={"id": "B", "path": image},
            candidates=[{"id": "C", "path": image}],
            rois=rois,
        )


def invalid_npy(tmp_path, kind):
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((16, 16, 3), np.float64))
    contents = b"" if kind == "empty" else buffer.getvalue()[:30]
    path = tmp_path / f"{kind}.npy"
    path.write_bytes(contents)
    return path, contents


@pytest.mark.parametrize("kind", ["empty", "truncated"])
def test_invalid_npy_loader_error_keeps_exact_byte_sha(tmp_path, kind):
    path, contents = invalid_npy(tmp_path, kind)
    with pytest.raises(AssetDecodeError) as exc:
        load_comparison_asset(ComparisonAsset(id="bad", path=path, encoding="linear"))
    assert exc.value.trace["source_bytes_sha256"] == hashlib.sha256(contents).hexdigest()
    assert exc.value.trace["byte_count"] == len(contents)


@pytest.mark.parametrize("kind", ["empty", "truncated"])
def test_invalid_npy_evaluator_returns_finite_invalid_evidence(tmp_path, kind):
    path, contents = invalid_npy(tmp_path, kind)
    result = evaluate_comparison(
        ComparisonRequest(
            scene_id="s",
            group_id="g",
            mode="algorithm",
            baseline={"id": "B", "path": path},
            candidates=[{"id": "C", "path": path}],
        )
    )
    assert result.comparisons[0].status == "REJECT"
    assert all(a.state == "invalid" for a in result.assets)
    assert result.assets[0].trace["source_bytes_sha256"] == hashlib.sha256(contents).hexdigest()
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize("kind", ["empty", "truncated"])
def test_invalid_npy_cli_writes_finite_rejection_instead_of_crashing(tmp_path, kind):
    path, contents = invalid_npy(tmp_path, kind)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scene_id": "s",
                "group_id": "g",
                "mode": "algorithm",
                "baseline": {"id": "B", "path": str(path)},
                "candidates": [{"id": "C", "path": str(path)}],
            }
        )
    )
    output = tmp_path / "output.json"
    assert main(["evaluate", "--manifest", str(manifest), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["comparisons"][0]["status"] == "REJECT"
    assert (
        result["assets"][0]["trace"]["source_bytes_sha256"] == hashlib.sha256(contents).hexdigest()
    )
    assert output.with_suffix(".csv").exists()


@pytest.mark.parametrize("orientation", range(1, 9))
def test_tiff_matches_equivalent_oriented_uint16_png_comparison(tmp_path, orientation):
    raw = (np.arange(24 * 32).reshape(24, 32) * 71 + 1234).astype(np.uint16)
    tiff = tmp_path / "candidate.tiff"
    png = tmp_path / "baseline.png"
    Image.fromarray(raw).save(tiff, tiffinfo={274: orientation}, compression="raw")
    Image.fromarray(oriented(raw, orientation)).save(png)
    result = evaluate_comparison(
        ComparisonRequest(
            scene_id="orientation-equivalence",
            group_id="g",
            mode="algorithm",
            source={"id": "S", "path": png},
            baseline={"id": "B", "path": png},
            candidates=[{"id": "C", "path": tiff}],
        )
    )
    comparison = result.comparisons[0]
    assert comparison.status == "REVIEW"
    assert comparison.fatal_reasons == []
    assert comparison.relative_metrics["pixel_mae"] == 0
    assert result.assets[0].trace["oriented_shape"] == result.assets[1].trace["oriented_shape"]


@pytest.mark.parametrize("reverse", [False, True])
def test_same_id_intact_roi_cannot_hide_destroyed_person_roi(tmp_path, reverse):
    rng = np.random.default_rng(102)
    baseline = rng.integers(70, 190, (128, 128, 3), dtype=np.uint8)
    candidate = baseline.copy()
    candidate[0:8, 0:8] = 255
    b, c = tmp_path / "baseline.png", tmp_path / "candidate.png"
    Image.fromarray(baseline).save(b)
    Image.fromarray(candidate).save(c)
    original = {"id": "face", "kind": "face", "person_id": "p", "bbox": [0, 0, 8, 8]}
    data = {
        "scene_id": "s",
        "group_id": "g",
        "mode": "algorithm",
        "source": {"id": "S", "path": b},
        "baseline": {"id": "B", "path": b},
        "candidates": [{"id": "C", "path": c}],
        "rois": [original],
    }
    assert (
        "local_content_collapse:face"
        in evaluate_comparison(ComparisonRequest(**data)).comparisons[0].fatal_reasons
    )
    intact = {**original, "bbox": [64, 64, 8, 8], "asset_id": "B"}
    data["rois"] = [intact, original] if reverse else [original, intact]
    with pytest.raises(ValidationError, match="effective asset scope"):
        ComparisonRequest(**data)
