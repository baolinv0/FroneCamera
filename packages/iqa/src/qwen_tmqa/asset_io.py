"""Precision-preserving decoding with one EXIF convention and byte-bound trace."""

from __future__ import annotations

import hashlib
import io
import math
import struct
from dataclasses import dataclass

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageCms, UnidentifiedImageError

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


# Includes raw/expanded channels and normalization/orientation/hash temporaries.
# This is an allocation safety limit per NPY asset, not a pixel quality threshold.
NUMPY_DECODE_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024


def _validate_pixel_layout(shape: tuple[int, ...], dtype: np.dtype) -> None:
    if (
        len(shape) not in (2, 3)
        or any(type(n) is not int or n <= 0 for n in shape)
        or (len(shape) == 3 and shape[2] not in (3, 4))
    ):
        raise ValueError("image_requires_nonempty_gray_or_rgb")
    if dtype.kind not in ("u", "f") or dtype.itemsize > 8:
        raise ValueError("unsupported_pixel_dtype")
    if dtype.kind == "u" and dtype.itemsize not in (1, 2):
        raise ValueError("unsupported_integer_precision")


def _preflight_numpy(data: bytes, trace: dict[str, object]) -> None:
    """Read the bounded header, reject unsafe dimensions and missing payload first."""
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError("unsupported_numpy_header_version")
    # Supported scalar numeric dtypes have ASCII descriptors even in v3 headers;
    # structured/unicode dtypes are refused below. Use only public NumPy readers.
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    shape, _fortran, dtype = reader(stream, max_header_size=10000)
    trace.update(
        {
            "raw_shape": list(shape),
            "dtype": str(dtype),
            "numpy_memory_budget_bytes": NUMPY_DECODE_MEMORY_BUDGET_BYTES,
        }
    )
    _validate_pixel_layout(shape, dtype)
    raw_bytes = math.prod(shape) * dtype.itemsize
    height, width = shape[:2]
    expanded_raw_bytes = height * width * 3 * dtype.itemsize if len(shape) == 2 else raw_bytes
    predicted_bytes = raw_bytes + expanded_raw_bytes + height * width * 3 * 8 * 3
    trace["numpy_predicted_decode_bytes"] = predicted_bytes
    if predicted_bytes > NUMPY_DECODE_MEMORY_BUDGET_BYTES:
        raise ValueError("numpy_decode_memory_budget_exceeded")
    if len(data) - stream.tell() != raw_bytes:
        raise ValueError("numpy_payload_length_mismatch")


def _normalize_embedded_icc(raw: np.ndarray, profile: bytes, warnings: list[str]) -> np.ndarray:
    # Pillow/LittleCMS RGB transforms are 8-bit only. Refuse higher precision rather
    # than dropping meaningful samples. The declared path never invokes this conversion.
    if raw.dtype != np.uint8:
        warnings.append("embedded_icc_precision_unsupported")
        raise ValueError("embedded_icc_precision_unsupported")
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        target = ImageCms.createProfile("sRGB")
        mode = "L" if raw.ndim == 2 else "RGBA" if raw.shape[2] == 4 else "RGB"
        converted = ImageCms.profileToProfile(
            Image.fromarray(raw), source, target, outputMode="RGB"
        )
        result = np.asarray(converted).copy()
        # Opaque-alpha validation still runs on source values below.
        if mode == "RGBA":
            result = np.concatenate((result, raw[..., 3:4]), axis=2)
    except (ImageCms.PyCMSError, ValueError, OSError, TypeError) as exc:
        warnings.append("embedded_icc_normalization_failed")
        raise ValueError("embedded_icc_normalization_failed") from exc
    warnings.append("embedded_icc_transformed_to_srgb")
    return result


def load_comparison_asset(asset: ComparisonAsset) -> LoadedComparisonAsset:
    data = asset.path.read_bytes()
    trace: dict[str, object] = {
        "asset_id": asset.id,
        "path": str(asset.path),
        "source_bytes_sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
        "encoding": asset.encoding,
        "color_policy": asset.color_policy,
        "declared_source_sha256": asset.source_sha256,
    }
    try:
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
    trace["warnings"] = warnings
    icc_profile = None
    icc_profile_present = False
    heif_color_profile = None
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
        icc_profile = heif.info.get("icc_profile")
        icc_profile_present = "icc_profile" in heif.info
        heif_color_profile = heif.info.get("nclx_profile")
        if asset.color_policy == "declared" and (icc_profile or heif_color_profile):
            warnings.append("embedded_heif_color_profile_not_applied_declared_encoding_used")
        warnings.append("heif_container_orientation_applied_by_native_decoder_exif_not_reapplied")
    else:
        try:
            with Image.open(io.BytesIO(data)) as image:
                orientation = int(image.getexif().get(274, 1))
                icc_profile = image.info.get("icc_profile")
                icc_profile_present = "icc_profile" in image.info
                if icc_profile and asset.color_policy == "declared":
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
    if asset.color_policy == "embedded_to_srgb":
        # Pillow keeps the ICC key with None for corrupt compression and b'' for
        # an empty profile. Neither is an unprofiled source or safe sRGB evidence.
        if icc_profile_present and not icc_profile:
            warnings.append("embedded_icc_normalization_failed")
            raise ValueError("embedded_icc_normalization_failed")
        if icc_profile and heif_color_profile:
            warnings.append("embedded_heif_multiple_color_profiles_unsupported")
            raise ValueError("embedded_heif_multiple_color_profiles_unsupported")
        if icc_profile:
            raw = _normalize_embedded_icc(raw, icc_profile, warnings)
            decoder += "_icc_to_srgb"
        elif heif_color_profile:
            # libheif already converts the known YCbCr matrix and range to RGB.
            # BT.709 primaries + IEC 61966-2-1 transfer are exactly sRGB, so no
            # precision-losing ICC round trip is needed (including 10/12-bit).
            if (
                heif_color_profile.get("color_primaries") == 1
                and heif_color_profile.get("transfer_characteristics") == 13
                and heif_color_profile.get("matrix_coefficients") in (0, 1, 6)
                and heif_color_profile.get("full_range_flag") in (0, 1)
            ):
                warnings.append("embedded_heif_srgb_native_rgb")
            else:
                warnings.append("embedded_heif_color_normalization_unsupported")
                raise ValueError("embedded_heif_color_normalization_unsupported")
        else:
            warnings.append("unprofiled_color_assumed_srgb")
    if raw.ndim == 2:
        raw = np.repeat(raw[..., None], 3, axis=2)
    if raw.ndim != 3 or raw.shape[2] not in (3, 4) or min(raw.shape[:2]) == 0:
        raise ValueError("image_requires_nonempty_gray_or_rgb")
    dtype = str(raw.dtype)
    if raw.dtype.kind not in ("u", "f") or raw.dtype.itemsize > 8:
        raise ValueError("unsupported_pixel_dtype")
    if raw.dtype.kind == "u" and raw.dtype.itemsize not in (1, 2):
        raise ValueError("unsupported_integer_precision")
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
            "asset_id": asset.id,
            "path": str(asset.path),
            "source_bytes_sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
            "dtype": dtype,
            "bit_depth": np.dtype(dtype).itemsize * 8,
            "source_bit_depth": source_bit_depth or np.dtype(dtype).itemsize * 8,
            "encoding": asset.encoding,
            "color_policy": asset.color_policy,
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
