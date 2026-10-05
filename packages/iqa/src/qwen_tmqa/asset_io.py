"""Precision-preserving decoding with one EXIF convention and byte-bound trace."""

from __future__ import annotations

import ast
import hashlib
import io
import math
import os
import struct
from dataclasses import dataclass

import cv2
import numpy as np
import pillow_heif
from PIL import Image, UnidentifiedImageError

from .comparison_models import ComparisonAsset


@dataclass(frozen=True)
class LoadedComparisonAsset:
    pixels: np.ndarray
    trace: dict[str, object]


def _orient(pixels: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 2:
        return np.fliplr(pixels)
    if orientation == 3:
        return np.rot90(pixels, 2)
    if orientation == 4:
        return np.flipud(pixels)
    if orientation == 5:
        return np.swapaxes(pixels, 0, 1)
    if orientation == 6:
        return np.rot90(pixels, -1)
    if orientation == 7:
        return np.flip(np.swapaxes(pixels, 0, 1), axis=(0, 1))
    if orientation == 8:
        return np.rot90(pixels, 1)
    return pixels


class AssetDecodeError(ValueError):
    """Invalid pixels with the exact bytes read still retained as provenance."""

    def __init__(self, reason: str, trace: dict[str, object]):
        super().__init__(reason)
        self.trace = trace


# Per-asset allocation ceilings, independent of pixel quality. The decode
# estimate includes the source/raw channels and normalization/hash temporaries.
NUMPY_MAX_SOURCE_BYTES = 512 * 1024 * 1024
NUMPY_DECODE_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024
NUMPY_MAX_HEADER_BYTES = 10000


def _read_numpy_snapshot(asset: ComparisonAsset, trace: dict[str, object]) -> bytes:
    """Keep one descriptor and cap reads even if the file grows after fstat."""
    with asset.path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        trace["byte_count"] = size
        trace["numpy_source_bytes_budget"] = NUMPY_MAX_SOURCE_BYTES
        trace["source_snapshot_complete"] = False
        if size > NUMPY_MAX_SOURCE_BYTES:
            # No bytes were read, so no source hash can be asserted.
            raise ValueError("numpy_source_bytes_budget_exceeded")
        data = stream.read(size + 1)
        trace["byte_count"] = len(data)
        if len(data) > size or stream.read(1):
            raise ValueError("numpy_source_size_changed_during_read")
        trace.update(
            source_bytes_sha256=hashlib.sha256(data).hexdigest(),
            source_snapshot_complete=True,
        )
        return data


def _validate_pixel_layout(shape: tuple[int, ...], dtype: np.dtype) -> None:
    if (
        len(shape) not in (2, 3)
        or any(type(n) is not int or n <= 0 for n in shape)
        or (len(shape) == 3 and shape[2] not in (3, 4))
    ):
        raise ValueError("image_requires_nonempty_gray_or_rgb")
    if dtype.hasobject or dtype.kind not in ("u", "f") or dtype.itemsize > 8:
        raise ValueError("unsupported_pixel_dtype")
    if dtype.kind == "u" and dtype.itemsize not in (1, 2):
        raise ValueError("unsupported_integer_precision")


def _preflight_numpy(data: bytes, trace: dict[str, object]) -> None:
    """Validate the bounded header and exact payload before NumPy allocates."""
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError("unsupported_numpy_header_version")
    length_format = "<H" if version == (1, 0) else "<I"
    length_size = struct.calcsize(length_format)
    length_offset = stream.tell()
    if len(data) < length_offset + length_size:
        raise ValueError("truncated_numpy_header")
    header_length = struct.unpack_from(length_format, data, length_offset)[0]
    if header_length > NUMPY_MAX_HEADER_BYTES:
        raise ValueError("numpy_header_bytes_budget_exceeded")
    payload_offset = length_offset + length_size + header_length
    if payload_offset > len(data):
        raise ValueError("truncated_numpy_header")
    try:
        if version == (3, 0):
            # NumPy exposes public v1/v2 header readers only. V3 changes the
            # header encoding to UTF-8; parse its bounded literal directly.
            header = ast.literal_eval(
                data[length_offset + length_size : payload_offset].decode("utf-8")
            )
            if not isinstance(header, dict) or header.keys() != {"shape", "descr", "fortran_order"}:
                raise ValueError("invalid_numpy_header_keys")
            shape = header["shape"]
            if not isinstance(shape, tuple) or any(type(n) is not int for n in shape):
                raise ValueError("invalid_numpy_shape")
            if type(header["fortran_order"]) is not bool:
                raise ValueError("invalid_numpy_fortran_order")
            dtype = np.lib.format.descr_to_dtype(header["descr"])
        else:
            reader = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            shape, _fortran_order, dtype = reader(stream, max_header_size=NUMPY_MAX_HEADER_BYTES)
    except (TypeError, IndexError, RecursionError) as exc:
        raise ValueError("invalid_numpy_header") from exc
    trace.update(
        raw_shape=list(shape),
        dtype=str(dtype),
        numpy_header_version=list(version),
        numpy_memory_budget_bytes=NUMPY_DECODE_MEMORY_BUDGET_BYTES,
    )
    _validate_pixel_layout(shape, dtype)
    height, width = shape[:2]
    pixel_count = height * width
    # Match Pillow's existing decompression-bomb error ceiling without changing
    # its warning threshold or any guard on ordinary image decoders.
    if Image.MAX_IMAGE_PIXELS is not None and pixel_count > 2 * Image.MAX_IMAGE_PIXELS:
        raise ValueError("numpy_decoded_pixel_budget_exceeded")
    raw_bytes = math.prod(shape) * dtype.itemsize
    expanded_raw_bytes = pixel_count * 3 * dtype.itemsize if len(shape) == 2 else raw_bytes
    predicted_bytes = raw_bytes + expanded_raw_bytes + pixel_count * 3 * 8 * 3
    trace["numpy_predicted_decode_bytes"] = predicted_bytes
    if predicted_bytes > NUMPY_DECODE_MEMORY_BUDGET_BYTES:
        raise ValueError("numpy_decode_memory_budget_exceeded")
    if len(data) - payload_offset != raw_bytes:
        raise ValueError("numpy_payload_length_mismatch")


def load_comparison_asset(asset: ComparisonAsset) -> LoadedComparisonAsset:
    trace: dict[str, object] = {
        "asset_id": asset.id,
        "path": str(asset.path),
        "source_bytes_sha256": None,
        "byte_count": None,
        "encoding": asset.encoding,
        "declared_source_sha256": asset.source_sha256,
    }
    try:
        if asset.path.suffix.lower() == ".npy":
            data = _read_numpy_snapshot(asset, trace)
        else:
            data = asset.path.read_bytes()
            trace.update(
                source_bytes_sha256=hashlib.sha256(data).hexdigest(),
                byte_count=len(data),
            )
        return _decode(asset, data, trace)
    except (
        Image.DecompressionBombError,
        ValueError,
        OSError,
        EOFError,
        SyntaxError,
        RuntimeError,
        cv2.error,
        OverflowError,
        MemoryError,
    ) as exc:
        raise AssetDecodeError(str(exc), trace) from exc


def _neutralize_tiff_orientation(data: bytes) -> bytes:
    """Set only the decoder copy's first-IFD orientation to 1.

    OpenCV's TIFF decoder applies this tag even under IMREAD_UNCHANGED. Normalizing
    metadata before decoding makes _orient the sole orientation operation while
    retaining uint16/float channels. Source bytes/hashes remain untouched.
    """
    if data[:2] not in (b"II", b"MM"):
        return data
    endian = "<" if data[:2] == b"II" else ">"
    if len(data) < 8:
        raise ValueError("truncated_tiff_header")
    magic = struct.unpack_from(endian + "H", data, 2)[0]
    if magic == 42:
        offset = struct.unpack_from(endian + "I", data, 4)[0]
        count_format, count_size, entry_size, value_offset = "H", 2, 12, 8
        element_count_format = "I"
    elif magic == 43:
        if len(data) < 16 or struct.unpack_from(endian + "HH", data, 4) != (8, 0):
            raise ValueError("invalid_bigtiff_header")
        offset = struct.unpack_from(endian + "Q", data, 8)[0]
        count_format, count_size, entry_size, value_offset = "Q", 8, 20, 12
        element_count_format = "Q"
    else:
        return data
    if offset + count_size > len(data):
        raise ValueError("truncated_tiff_ifd")
    count = struct.unpack_from(endian + count_format, data, offset)[0]
    start = offset + count_size
    if start + count * entry_size > len(data):
        raise ValueError("truncated_tiff_ifd_entries")
    patched = None
    for index in range(count):
        entry = start + index * entry_size
        tag, field_type = struct.unpack_from(endian + "HH", data, entry)
        if tag != 274:
            continue
        element_count = struct.unpack_from(endian + element_count_format, data, entry + 4)[0]
        if field_type != 3 or element_count != 1:
            raise ValueError("invalid_tiff_orientation_tag")
        if patched is not None:
            raise ValueError("duplicate_tiff_orientation_tag")
        patched = bytearray(data)
        struct.pack_into(endian + "H", patched, entry + value_offset, 1)
    return bytes(patched) if patched is not None else data


def _decode(asset: ComparisonAsset, data: bytes, trace: dict[str, object]) -> LoadedComparisonAsset:
    orientation = 1
    native_orientation = False
    source_bit_depth = None
    warnings: list[str] = []
    if asset.path.suffix.lower() == ".npy":
        _preflight_numpy(data, trace)
        raw = np.load(io.BytesIO(data), allow_pickle=False)
        decoder = "numpy_safe"
        if not isinstance(raw, np.ndarray):
            if hasattr(raw, "close"):
                raw.close()
            raise ValueError("numpy_asset_requires_single_array")
    elif pillow_heif.is_supported(io.BytesIO(data)):
        heif = pillow_heif.open_heif(io.BytesIO(data), convert_hdr_to_8bit=False, hdr_to_16bit=True)
        raw = np.asarray(heif)
        decoder = "pillow_heif_native_precision"
        native_orientation = True
        source_bit_depth = int(heif.info["bit_depth"])
        exif_data = heif.info.get("exif")
        if exif_data:
            exif = Image.Exif()
            exif.load(exif_data)
            orientation = int(exif.get(274, 1))
        if heif.info.get("icc_profile") or heif.info.get("nclx_profile"):
            warnings.append("embedded_heif_color_profile_not_applied_declared_encoding_used")
        warnings.append("heif_container_orientation_applied_by_native_decoder_exif_not_reapplied")
    else:
        try:
            with Image.open(io.BytesIO(data)) as image:
                orientation = int(image.getexif().get(274, 1))
                if image.info.get("icc_profile"):
                    warnings.append("embedded_icc_not_applied_declared_encoding_used")
        except (UnidentifiedImageError, OSError):
            pass  # OpenCV also supports HDR formats unsupported by Pillow.
        decoder_data = _neutralize_tiff_orientation(data)
        raw = cv2.imdecode(np.frombuffer(decoder_data, np.uint8), cv2.IMREAD_UNCHANGED)
        decoder = "opencv_unchanged"
        if decoder_data != data:
            decoder = "opencv_unchanged_tiff_orientation_neutralized"
        if raw is None:
            raise ValueError("image_decode_failed")
        if raw.ndim == 3 and raw.shape[2] in (3, 4):
            raw = raw[..., [2, 1, 0, 3] if raw.shape[2] == 4 else [2, 1, 0]]
    trace.update(
        {
            "dtype": str(raw.dtype),
            "bit_depth": raw.dtype.itemsize * 8,
            "raw_shape": list(raw.shape),
            "decoder": decoder,
            "exif_orientation": orientation,
        }
    )
    if orientation not in range(1, 9):
        raise ValueError("unsupported_exif_orientation")
    _validate_pixel_layout(raw.shape, raw.dtype)
    if raw.ndim == 2:
        raw = np.repeat(raw[..., None], 3, axis=2)
    dtype = str(raw.dtype)
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite_pixels")
    denominator = float(np.iinfo(raw.dtype).max) if raw.dtype.kind == "u" else 1.0
    if raw.shape[2] == 4:
        if not np.all(raw[..., 3] == denominator):
            raise ValueError("nonopaque_alpha_requires_explicit_compositing")
        raw = raw[..., :3]
        warnings.append("opaque_alpha_removed")
    pixels = raw.astype(np.float64) / denominator
    if pixels.min() < 0 or (asset.encoding == "srgb" and pixels.max() > 1):
        raise ValueError("pixels_outside_declared_encoding_range")
    # Numerical ceilings prevent overflow while retaining ordinary linear HDR ranges.
    if pixels.max() > 1e12:
        raise ValueError("linear_dynamic_range_exceeds_numeric_guard")
    original_shape = list(pixels.shape)
    pixels = np.ascontiguousarray(pixels if native_orientation else _orient(pixels, orientation))
    return LoadedComparisonAsset(
        pixels,
        {
            **trace,
            "asset_id": asset.id,
            "path": str(asset.path),
            "source_bytes_sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
            "dtype": dtype,
            "bit_depth": np.dtype(dtype).itemsize * 8,
            "source_bit_depth": source_bit_depth or np.dtype(dtype).itemsize * 8,
            "encoding": asset.encoding,
            "normalization_denominator": denominator,
            "original_shape": original_shape,
            "oriented_shape": list(pixels.shape),
            "exif_orientation": orientation,
            "orientation_convention": "heif_native_display_oriented"
            if native_orientation
            else "exif_transposed",
            "decoder": decoder,
            "warnings": warnings,
            "decoded_pixels_sha256": hashlib.sha256(pixels.astype("<f8").tobytes()).hexdigest(),
            "declared_source_sha256": asset.source_sha256,
        },
    )
