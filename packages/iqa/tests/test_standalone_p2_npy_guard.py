"""S-009: NPY metadata cannot allocate before bounded payload validation."""

import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qwen_tmqa import asset_io
from qwen_tmqa.asset_io import AssetDecodeError, load_comparison_asset
from qwen_tmqa.comparison import evaluate_comparison
from qwen_tmqa.comparison_cli import main
from qwen_tmqa.comparison_models import ComparisonAsset, ComparisonRequest


def npy_header(shape=(2, 3, 3), descr="<f8", fortran_order=False, version=(1, 0), **changes):
    metadata = {"descr": descr, "fortran_order": fortran_order, "shape": shape}
    metadata.update(changes)
    text = repr(metadata).encode("utf-8" if version == (3, 0) else "latin1")
    size_format = "<H" if version == (1, 0) else "<I"
    prefix = np.lib.format.magic(*version)
    padding = -(len(prefix) + struct.calcsize(size_format) + len(text) + 1) % 64
    header = text + b" " * padding + b"\n"
    return prefix + struct.pack(size_format, len(header)) + header


def allocation_sentinel(monkeypatch):
    calls = []

    def forbidden_load(*args, **kwargs):
        calls.append((args, kwargs))
        raise MemoryError("safe sentinel: hostile header reached numpy allocation")

    monkeypatch.setattr(asset_io.np, "load", forbidden_load)
    return calls


BAD_FILES = [
    pytest.param(npy_header((100000, 100000, 3)), id="240gb-header-only"),
    pytest.param(npy_header((2, 3, 3)) + bytes(16), id="truncated-payload"),
    pytest.param(npy_header((2, 3, 3)) + bytes(145), id="trailing-payload"),
    pytest.param(npy_header(descr="|O"), id="object-dtype"),
    pytest.param(npy_header(descr=[("pixel", "|O")]), id="structured-object-dtype"),
    pytest.param(npy_header(descr=[("pixel", "<f8")]), id="structured-dtype"),
    pytest.param(npy_header(descr="<i2"), id="signed-dtype"),
    pytest.param(npy_header(descr="<u4"), id="unsupported-integer-precision"),
    pytest.param(npy_header(descr="<c8"), id="complex-dtype"),
    pytest.param(npy_header(descr="<U2"), id="unicode-dtype"),
    pytest.param(npy_header(descr="not-a-dtype"), id="invalid-descriptor"),
    pytest.param(npy_header(()), id="scalar"),
    pytest.param(npy_header((2,)), id="vector"),
    pytest.param(npy_header((2, 3, 3, 1)), id="four-dimensions"),
    pytest.param(npy_header((2, 3, 2)), id="two-channels"),
    pytest.param(npy_header((0, 3, 3)), id="empty-dimension"),
    pytest.param(npy_header((-1, 3, 3)), id="negative-dimension"),
    pytest.param(npy_header((True, 3, 3)), id="boolean-dimension"),
    pytest.param(npy_header((2.0, 3, 3)), id="noninteger-dimension"),
    pytest.param(npy_header(fortran_order="yes"), id="invalid-order"),
    pytest.param(npy_header(version=(4, 0)), id="unknown-version"),
    pytest.param(npy_header(extra=1), id="extra-header-key"),
    pytest.param(b"\x93NUMPY\x01\x00\x08\x00{1: 2}\n ", id="invalid-header-keys"),
    pytest.param(
        b"\x93NUMPY\x02\x00" + struct.pack("<I", 2**32 - 1), id="huge-declared-header-length"
    ),
    pytest.param(npy_header()[:30], id="truncated-header"),
    pytest.param(npy_header(extra="x" * 12000), id="oversized-actual-header"),
    pytest.param(npy_header((100000, 100000, 3), version=(3, 0)), id="v3-huge-shape"),
    pytest.param(npy_header(fortran_order=1, version=(3, 0)), id="v3-invalid-order"),
    pytest.param(npy_header(descr="not-a-dtype", version=(3, 0)), id="v3-invalid-descriptor"),
    pytest.param(npy_header(extra=1, version=(3, 0)), id="v3-invalid-header-keys"),
    pytest.param(npy_header((True, 3, 3), version=(3, 0)), id="v3-invalid-dimension"),
    pytest.param(npy_header(version=(3, 0)).replace(b"descr", b"\xffescr"), id="v3-invalid-utf8"),
    pytest.param(b"not numpy", id="invalid-magic"),
    pytest.param(b"", id="empty-file"),
]
BAD_FILES += [
    pytest.param(
        npy_header(descr=descr, version=version), id=f"v{version[0]}-{len(descr)}-tuple-descr"
    )
    for version in ((1, 0), (2, 0), (3, 0))
    for descr in ((), ("<u1",))
]


@pytest.mark.parametrize("contents", BAD_FILES)
def test_invalid_npy_rejected_before_numpy_allocation(tmp_path, monkeypatch, contents):
    path = tmp_path / "hostile.npy"
    path.write_bytes(contents)
    calls = allocation_sentinel(monkeypatch)
    with pytest.raises(AssetDecodeError) as failure:
        load_comparison_asset(ComparisonAsset(id="bad", path=path, encoding="linear"))
    assert calls == []
    assert failure.value.trace["source_bytes_sha256"] == hashlib.sha256(contents).hexdigest()
    assert failure.value.trace["byte_count"] == len(contents)


def valid_png(tmp_path):
    path = tmp_path / "valid.png"
    Image.fromarray(np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)).save(path)
    return path


def comparison_request(valid, bad):
    return ComparisonRequest(
        scene_id="npy-allocation-guard",
        group_id="g",
        mode="algorithm",
        baseline={"id": "B", "path": valid},
        source={"id": "S", "path": valid},
        candidates=[{"id": "hostile", "path": bad}, {"id": "good", "path": valid}],
    )


@pytest.mark.parametrize("contents", BAD_FILES)
def test_hostile_npy_isolated_and_good_candidate_continues(tmp_path, monkeypatch, contents):
    valid = valid_png(tmp_path)
    bad = tmp_path / "hostile.npy"
    bad.write_bytes(contents)
    calls = allocation_sentinel(monkeypatch)
    result = evaluate_comparison(comparison_request(valid, bad))
    assert calls == []
    assert [asset.state for asset in result.assets] == ["valid", "invalid", "valid", "valid"]
    assert result.assets[1].trace["source_bytes_sha256"] == hashlib.sha256(contents).hexdigest()
    assert result.comparisons[0].status == "REJECT"
    assert result.comparisons[0].fatal_reasons == ["invalid_candidate"]
    assert result.comparisons[1].status == "REVIEW"
    assert result.comparisons[1].relative_metrics["pixel_mae"] == 0
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_cli_writes_rejection_and_good_candidate_results(tmp_path, monkeypatch):
    valid = valid_png(tmp_path)
    bad = tmp_path / "hostile.npy"
    bad.write_bytes(npy_header((100000, 100000, 3)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(comparison_request(valid, bad).model_dump_json())
    output = tmp_path / "result.json"
    calls = allocation_sentinel(monkeypatch)
    assert main(["evaluate", "--manifest", str(manifest), "--output", str(output)]) == 0
    assert calls == []
    result = json.loads(output.read_text())
    assert [item["status"] for item in result["comparisons"]] == ["REJECT", "REVIEW"]
    csv = output.with_suffix(".csv").read_text()
    assert "hostile,REJECT" in csv
    assert "good,REVIEW" in csv
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("shape", [(3, 5), (3, 5, 3), (3, 5, 4)])
@pytest.mark.parametrize("dtype", ["u1", "<u2", ">u2", "<f2", ">f2", "<f4", ">f4", "<f8", ">f8"])
def test_allowed_npy_formats_preserve_samples(tmp_path, version, order, shape, dtype):
    dtype = np.dtype(dtype)
    values = np.arange(np.prod(shape)).reshape(shape)
    raw = np.array(
        values + 1200 if dtype.kind == "u" and dtype.itemsize == 2 else values,
        dtype=dtype,
        order=order,
    )
    if dtype.kind == "f":
        raw[...] = (values + 0.123456789) / 25
    denominator = np.iinfo(dtype).max if dtype.kind == "u" else 1
    if len(shape) == 3 and shape[-1] == 4:
        raw[..., 3] = denominator
    stream = io.BytesIO()
    np.lib.format.write_array(stream, raw, version=version, allow_pickle=False)
    data = stream.getvalue()
    path = tmp_path / "valid.npy"
    path.write_bytes(data)
    loaded = load_comparison_asset(ComparisonAsset(id="a", path=path, encoding="linear"))
    expected = np.repeat(raw[..., None], 3, axis=2) if raw.ndim == 2 else raw[..., :3]
    np.testing.assert_array_equal(loaded.pixels, expected.astype(np.float64) / denominator)
    assert loaded.pixels.flags.c_contiguous
    assert loaded.trace["bit_depth"] == dtype.itemsize * 8
    assert loaded.trace["dtype"] == str(dtype)
    assert loaded.trace["source_bytes_sha256"] == hashlib.sha256(data).hexdigest()


def test_npy_decode_uses_same_bytes_if_path_changes_after_preflight(tmp_path, monkeypatch):
    path = tmp_path / "mutable.npy"
    raw = np.arange(18, dtype=np.uint16).reshape(2, 3, 3) + 1234
    np.save(path, raw)
    original_data = path.read_bytes()
    original_load = asset_io.np.load
    snapshots = []

    def replace_path_then_load(stream, *args, **kwargs):
        # This boundary occurs after preflight in the fixed loader. A second
        # path open here would encounter the hostile replacement header.
        path.write_bytes(npy_header((100000, 100000, 3)))
        assert isinstance(stream, io.BytesIO)
        snapshots.append(stream.getvalue())
        return original_load(stream, *args, **kwargs)

    monkeypatch.setattr(asset_io.np, "load", replace_path_then_load)
    loaded = load_comparison_asset(ComparisonAsset(id="a", path=path))
    assert snapshots == [original_data]
    np.testing.assert_array_equal(loaded.pixels, raw / 65535)
    assert loaded.trace["source_bytes_sha256"] == hashlib.sha256(original_data).hexdigest()
    assert path.read_bytes() != original_data


def test_oversized_sparse_npy_rejected_before_source_read(tmp_path, monkeypatch):
    path = tmp_path / "huge-sparse.npy"
    with path.open("wb") as stream:
        stream.truncate(512 * 1024 * 1024 + 1)
    original_open = Path.open
    reads = []

    class ReadGuard:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, *args, **kwargs):
            reads.append((args, kwargs))
            raise MemoryError("safe sentinel: oversized source reached bytes read")

    def guarded_open(self, *args, **kwargs):
        stream = original_open(self, *args, **kwargs)
        return ReadGuard(stream) if self == path else stream

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(AssetDecodeError, match="numpy_source_bytes_budget_exceeded") as failure:
        load_comparison_asset(ComparisonAsset(id="bad", path=path))
    assert reads == []
    assert failure.value.trace["source_bytes_sha256"] is None
    assert failure.value.trace["byte_count"] == path.stat().st_size


def test_npy_source_growth_after_fstat_remains_bounded(tmp_path, monkeypatch):
    path = tmp_path / "growing.npy"
    np.save(path, np.zeros((2, 3, 3), dtype=np.uint8))
    initial_size = path.stat().st_size
    original_open = Path.open
    reads = []

    class GrowingSource:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, count=-1):
            reads.append(count)
            if len(reads) == 1:
                with original_open(path, "r+b") as growth:
                    growth.truncate(1024 * 1024 * 1024)
            if count < 0 or count > initial_size + 1:
                raise MemoryError("safe sentinel: file growth reached an unbounded read")
            return self.stream.read(count)

    def growing_open(self, *args, **kwargs):
        stream = original_open(self, *args, **kwargs)
        return GrowingSource(stream) if self == path else stream

    monkeypatch.setattr(Path, "open", growing_open)
    calls = allocation_sentinel(monkeypatch)
    with pytest.raises(AssetDecodeError, match="numpy_source_size_changed_during_read") as failure:
        load_comparison_asset(ComparisonAsset(id="bad", path=path))
    assert reads == [initial_size + 1]
    assert calls == []
    assert failure.value.trace["source_bytes_sha256"] is None
    assert failure.value.trace["source_snapshot_complete"] is False


def test_bounded_numpy_allocation_failure_is_invalid_evidence(tmp_path, monkeypatch):
    path = tmp_path / "valid.npy"
    np.save(path, np.zeros((2, 3, 3), dtype=np.uint8))
    contents = path.read_bytes()
    calls = allocation_sentinel(monkeypatch)
    with pytest.raises(AssetDecodeError) as failure:
        load_comparison_asset(ComparisonAsset(id="bad", path=path))
    assert len(calls) == 1
    assert isinstance(failure.value.__cause__, MemoryError)
    assert failure.value.trace["source_bytes_sha256"] == hashlib.sha256(contents).hexdigest()


def test_numpy_pixel_ceiling_matches_existing_pillow_guard(tmp_path, monkeypatch):
    path = tmp_path / "over-pixel-limit.npy"
    np.save(path, np.zeros((3, 3, 3), dtype=np.uint8))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)
    calls = allocation_sentinel(monkeypatch)
    with pytest.raises(AssetDecodeError, match="numpy_decoded_pixel_budget_exceeded"):
        load_comparison_asset(ComparisonAsset(id="bad", path=path))
    assert calls == []


def test_complete_payload_still_requires_decoded_memory_budget(tmp_path, monkeypatch):
    path = tmp_path / "complete.npy"
    np.save(path, np.zeros((2, 3, 3), dtype=np.uint8))
    monkeypatch.setattr(asset_io, "NUMPY_DECODE_MEMORY_BUDGET_BYTES", 400, raising=False)
    calls = allocation_sentinel(monkeypatch)
    with pytest.raises(AssetDecodeError, match="numpy_decode_memory_budget_exceeded"):
        load_comparison_asset(ComparisonAsset(id="bad", path=path))
    assert calls == []
