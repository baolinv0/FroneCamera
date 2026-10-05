"""Portable real ICC transforms; no system ICC files or precision downgrade."""

import hashlib
import struct
import zlib

import cv2
import numpy as np
import pillow_heif
import pytest
from PIL import Image, ImageCms, ImageOps
from qwen_tmqa.asset_io import AssetDecodeError, load_comparison_asset
from qwen_tmqa.comparison_models import ComparisonAsset

from portrait_eval.iqa_bridge import evaluate_scene, model_evidence_context, scene_evidence_audit
from portrait_eval.pixel_io import load_normalized_rgb


def adobe_rgb_profile():
    """Build an Adobe RGB matrix/TRC profile from LittleCMS's built-in sRGB.

    Adobe RGB (1998) D50-adapted primaries and gamma 563/256 define the color
    transform. Runtime-generated metadata is retained; no vendor profile copied.
    """
    data = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    colorants = {
        b"rXYZ": (0.6097559, 0.3111242, 0.0194811),
        b"gXYZ": (0.2052401, 0.6256560, 0.0608902),
        b"bXYZ": (0.1492240, 0.0632197, 0.7448387),
    }
    for index in range(struct.unpack_from(">I", data, 128)[0]):
        entry = 132 + index * 12
        tag, offset, _ = struct.unpack_from(">4sII", data, entry)
        if tag in colorants:
            struct.pack_into(">iii", data, offset + 8, *(round(x * 65536) for x in colorants[tag]))
        elif tag in (b"rTRC", b"gTRC", b"bTRC"):
            data[offset : offset + 14] = struct.pack(">4sIIH", b"curv", 0, 1, 563)
            struct.pack_into(">I", data, entry + 8, 14)
    return ImageCms.ImageCmsProfile(__import__("io").BytesIO(data))


def color_pair(tmp_path):
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    adobe = adobe_rgb_profile()
    original = Image.new("RGB", (32, 32), (64, 160, 64))
    converted = ImageCms.profileToProfile(original, srgb, adobe, outputMode="RGB")
    restored = ImageCms.profileToProfile(converted, adobe, srgb, outputMode="RGB")
    first, second = tmp_path / "srgb.png", tmp_path / "adobe.png"
    original.save(first, icc_profile=srgb.tobytes())
    converted.save(second, icc_profile=adobe.tobytes())
    assert original.getpixel((0, 0)) != converted.getpixel((0, 0))
    return first, second, np.asarray(restored) / 255.0


def test_front_embedded_icc_measurements_use_actual_srgb_transform(tmp_path):
    first, second, expected = color_pair(tmp_path)
    a, _ = load_normalized_rgb(first)
    b, trace = load_normalized_rgb(second)
    np.testing.assert_array_equal(b, expected)
    np.testing.assert_allclose(a, b, atol=1 / 255)
    assert trace["color_policy"] == "embedded_to_srgb"
    assert "embedded_icc_transformed_to_srgb" in trace["warnings"]
    evidence = evaluate_scene("color", {"A": first, "B": second}, {})
    assert abs(evidence["comparisons"][0]["relative_metrics"]["mean_luminance_delta"]) < 0.003
    assert scene_evidence_audit(evidence, ["A", "B"])["valid"]
    for context in (
        model_evidence_context(evidence, measurements=True),
        model_evidence_context(evidence, measurements=False),
    ):
        assert "embedded_icc_transformed_to_srgb" in context["assets"][1]["color_warnings"]
        assert "embedded_icc_transformed_to_srgb" in context["warnings"]


def test_declared_encoding_default_remains_untransformed(tmp_path):
    _, second, expected = color_pair(tmp_path)
    loaded = load_comparison_asset(ComparisonAsset(id="a", path=second))
    with Image.open(second) as image:
        encoded = np.asarray(image) / 255.0
    np.testing.assert_array_equal(loaded.pixels, encoded)
    assert np.max(np.abs(loaded.pixels - expected)) > 0.02
    assert loaded.trace["color_policy"] == "declared"
    assert "embedded_icc_not_applied_declared_encoding_used" in loaded.trace["warnings"]


def test_invalid_icc_refused_and_audit_and_model_receive_warning(tmp_path):
    path = tmp_path / "bad-profile.png"
    Image.new("RGB", (16, 16), (80, 100, 120)).save(path, icc_profile=b"not an ICC profile")
    with pytest.raises(AssetDecodeError, match="embedded_icc_normalization_failed") as exc:
        load_normalized_rgb(path)
    assert exc.value.trace["source_bytes_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = evaluate_scene("color", {"A": path, "B": path}, {})
    audit = scene_evidence_audit(evidence, ["A", "B"])
    assert not audit["valid"]
    assert "embedded_icc_normalization_failed" in audit["color_warnings"]
    assert (
        "embedded_icc_normalization_failed"
        in model_evidence_context(evidence, measurements=True)["assets"][0]["color_warnings"]
    )


@pytest.mark.parametrize("compressed_profile", [zlib.compress(b""), b"not-zlib"])
def test_front_empty_or_corrupt_png_iccp_is_refused_with_provenance(tmp_path, compressed_profile):
    normal, malformed = tmp_path / "normal.png", tmp_path / "malformed-profile.png"
    Image.new("RGB", (16, 16), (80, 100, 120)).save(normal)
    original = normal.read_bytes()
    chunk_data = b"profile\x00\x00" + compressed_profile
    chunk = (
        struct.pack(">I", len(chunk_data))
        + b"iCCP"
        + chunk_data
        + struct.pack(">I", zlib.crc32(b"iCCP" + chunk_data) & 0xFFFFFFFF)
    )
    # Insert an actual PNG profile chunk immediately after IHDR.
    malformed.write_bytes(original[:33] + chunk + original[33:])
    with Image.open(malformed) as image:
        assert "icc_profile" in image.info
        assert not image.info["icc_profile"]

    # The shared declared-encoding contract still ignores embedded color metadata.
    declared = load_comparison_asset(ComparisonAsset(id="declared", path=malformed))
    np.testing.assert_array_equal(declared.pixels, load_normalized_rgb(normal)[0])
    assert declared.trace["color_policy"] == "declared"

    with pytest.raises(AssetDecodeError, match="embedded_icc_normalization_failed") as exc:
        load_normalized_rgb(malformed)
    assert (
        exc.value.trace["source_bytes_sha256"] == hashlib.sha256(malformed.read_bytes()).hexdigest()
    )
    assert "embedded_icc_normalization_failed" in exc.value.trace["warnings"]
    assert "unprofiled_color_assumed_srgb" not in exc.value.trace["warnings"]
    evidence = evaluate_scene("malformed-profile", {"A": normal, "B": malformed}, {})
    assert evidence["assets"][0]["state"] == "valid"
    assert evidence["assets"][1]["state"] == "invalid"
    audit = scene_evidence_audit(evidence, ["A", "B"])
    assert not audit["valid"]
    assert "embedded_icc_normalization_failed" in audit["color_warnings"]
    assert (
        "embedded_icc_normalization_failed"
        in model_evidence_context(evidence, measurements=False)["warnings"]
    )


def test_front_high_precision_embedded_icc_fails_closed_without_uint8_conversion(tmp_path):
    path = tmp_path / "16bit-profile.png"
    Image.fromarray(np.array([[1234, 1235], [30000, 40000]], dtype=np.uint16)).save(
        path, icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    )
    declared = load_comparison_asset(ComparisonAsset(id="a", path=path))
    assert declared.trace["bit_depth"] == 16
    assert declared.pixels[0, 0, 0] != declared.pixels[0, 1, 0]
    with pytest.raises(AssetDecodeError, match="embedded_icc_precision_unsupported"):
        load_normalized_rgb(path)


def test_front_unprofiled_uint16_and_float_keep_precision(tmp_path):
    rgb = np.array([[[1234, 1235, 1236], [1235, 1236, 1237]]], dtype=np.uint16)
    path = tmp_path / "16bit.png"
    cv2.imwrite(str(path), rgb[..., ::-1])
    actual, trace = load_normalized_rgb(path)
    np.testing.assert_array_equal(actual, rgb / 65535)
    assert trace["bit_depth"] == 16
    path = tmp_path / "float.npy"
    rgb = np.array([[[0.123456789, 0.4, 0.8]]], dtype=np.float64)
    np.save(path, rgb)
    np.testing.assert_array_equal(load_normalized_rgb(path)[0], rgb)


@pytest.mark.parametrize("bit_depth", [8, 10])
def test_front_native_heif_srgb_color_keeps_precision_and_orientation(tmp_path, bit_depth):
    path = tmp_path / "srgb.heic"
    if bit_depth == 8:
        pillow_heif.register_heif_opener()
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(np.arange(24 * 40 * 3, dtype=np.uint8).reshape(24, 40, 3)).save(
            path, format="HEIF", quality=-1, chroma=444, exif=exif
        )
    else:
        rgb = np.repeat(
            (np.arange(1024, dtype=np.uint16).reshape(32, 32) * 64)[..., None], 3, axis=2
        )
        pillow_heif.from_bytes("RGB;16", (32, 32), rgb.tobytes()).save(path, quality=-1, chroma=444)
    declared = load_comparison_asset(ComparisonAsset(id="a", path=path))
    front, trace = load_normalized_rgb(path)
    np.testing.assert_array_equal(front, declared.pixels)
    assert trace["source_bit_depth"] == bit_depth
    assert "embedded_heif_srgb_native_rgb" in trace["warnings"]
    if bit_depth == 8:
        with Image.open(path) as image:
            np.testing.assert_array_equal(
                front, np.asarray(ImageOps.exif_transpose(image).convert("RGB")) / 255
            )
    else:
        assert np.unique(front[..., 0]).size > 256


def test_front_actual_heif_unsupported_nclx_profile_fails_closed(tmp_path):
    path = tmp_path / "rec2020.heic"
    pillow_heif.register_heif_opener()
    Image.new("RGB", (32, 32), (90, 120, 160)).save(
        path, format="HEIF", color_primaries=9, transfer_characteristic=16, matrix_coefficients=9
    )
    with pytest.raises(AssetDecodeError, match="embedded_heif_color_normalization_unsupported"):
        load_normalized_rgb(path)


def test_front_actual_high_precision_heif_icc_is_refused(tmp_path):
    path = tmp_path / "icc10.heic"
    rgb = np.repeat((np.arange(1024, dtype=np.uint16).reshape(32, 32) * 64)[..., None], 3, axis=2)
    pillow_heif.from_bytes("RGB;16", (32, 32), rgb.tobytes()).save(
        path,
        quality=-1,
        chroma=444,
        icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes(),
    )
    with pytest.raises(AssetDecodeError, match="embedded_icc_precision_unsupported"):
        load_normalized_rgb(path)


def test_front_conflicting_heif_profile_metadata_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "conflicting.heic"
    pillow_heif.register_heif_opener()
    Image.new("RGB", (32, 32), (80, 100, 120)).save(
        path, format="HEIF", icc_profile=adobe_rgb_profile().tobytes()
    )
    # The encoder normally writes a single profile. Retain real decoded pixels
    # and ICC bytes while simulating additional, contradictory decoder metadata.
    native = pillow_heif.open_heif(path, convert_hdr_to_8bit=False, hdr_to_16bit=True)
    native.info["nclx_profile"] = {
        "color_primaries": 1,
        "transfer_characteristics": 13,
        "matrix_coefficients": 6,
        "full_range_flag": 1,
    }
    monkeypatch.setattr(pillow_heif, "open_heif", lambda *args, **kwargs: native)
    with pytest.raises(AssetDecodeError, match="embedded_heif_multiple_color_profiles_unsupported"):
        load_normalized_rgb(path)
