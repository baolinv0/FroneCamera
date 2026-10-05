"""Header/payload rejection happens before numpy's allocating load is called."""

import hashlib

import numpy as np
import pytest

from qwen_tmqa.asset_io import AssetDecodeError, load_comparison_asset
from qwen_tmqa.comparison import evaluate_comparison
from qwen_tmqa.comparison_models import ComparisonAsset, ComparisonRequest


def header(path, shape, dtype="|u1", version=1, payload=b""):
    with path.open("wb") as stream:
        writer = (
            np.lib.format.write_array_header_1_0
            if version == 1
            else np.lib.format.write_array_header_2_0
        )
        writer(stream, {"descr": dtype, "fortran_order": False, "shape": shape})
        stream.write(payload)


@pytest.mark.parametrize(
    "shape,dtype,payload,reason",
    [
        ((100000, 100000, 3), "|u1", b"", "numpy_decode_memory_budget_exceeded"),
        ((4000, 4000, 3), "|u1", b"", "numpy_decode_memory_budget_exceeded"),
        ((16, 16, 3), "|u1", b"x", "numpy_payload_length_mismatch"),
        ((2, 2, 3), "|u1", b"x" * 13, "numpy_payload_length_mismatch"),
        ((16, 16, 2), "|u1", b"", "image_requires_nonempty_gray_or_rgb"),
        ((16, 16, 3), "|O", b"", "unsupported_pixel_dtype"),
        ((16, 16, 3), "<u8", b"", "unsupported_integer_precision"),
    ],
)
@pytest.mark.parametrize("version", [1, 2])
def test_npy_invalid_header_and_budget_never_reach_allocating_load(
    tmp_path, monkeypatch, shape, dtype, payload, reason, version
):
    path = tmp_path / "unsafe.npy"
    header(path, shape, dtype, version, payload)
    calls = []

    def forbidden_load(*args, **kwargs):
        calls.append(True)
        raise ValueError("allocation_attempted")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(AssetDecodeError, match=reason) as exc:
        load_comparison_asset(ComparisonAsset(id="bad", path=path))
    assert calls == []
    assert exc.value.trace["source_bytes_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_oversized_npy_invalid_evidence_does_not_abort_valid_sibling(tmp_path, monkeypatch):
    valid, bad = tmp_path / "valid.npy", tmp_path / "bad.npy"
    np.save(valid, np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    header(bad, (100000, 100000, 3))
    real_load = np.load

    def guarded_load(stream, *args, **kwargs):
        if len(stream.getbuffer()) == bad.stat().st_size:
            raise ValueError("allocation_attempted")
        return real_load(stream, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    result = evaluate_comparison(
        ComparisonRequest(
            scene_id="memory",
            group_id="memory",
            mode="algorithm",
            baseline={"id": "B", "path": valid},
            candidates=[{"id": "bad", "path": bad}, {"id": "good", "path": valid}],
        )
    )
    assert result.assets[1].state == "invalid"
    assert "numpy_decode_memory_budget_exceeded" in result.assets[1].reasons[0]
    assert (
        result.assets[1].trace["source_bytes_sha256"]
        == hashlib.sha256(bad.read_bytes()).hexdigest()
    )
    assert result.assets[2].state == "valid"
    assert result.comparisons[1].relative_metrics["pixel_mae"] == 0


@pytest.mark.parametrize(
    "dtype,shape,fortran",
    [("u2", (4, 5), False), ("f8", (4, 5, 3), True), (">u2", (4, 5, 3), False)],
)
def test_normal_numpy_assets_keep_dtype_layout_and_values(tmp_path, dtype, shape, fortran):
    raw = np.full(shape, 0.123456789 if "f" in dtype else 1235, dtype=dtype)
    if fortran:
        raw = np.asfortranarray(raw)
    path = tmp_path / "valid.npy"
    np.save(path, raw)
    loaded = load_comparison_asset(ComparisonAsset(id="valid", path=path))
    expected = raw / (1 if "f" in dtype else 65535)
    if len(shape) == 2:
        expected = np.repeat(expected[..., None], 3, axis=2)
    np.testing.assert_array_equal(loaded.pixels, expected)
