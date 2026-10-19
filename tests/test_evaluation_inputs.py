from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps

from portrait_eval import imaging


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_exif_orientations_are_applied_once(tmp_path, orientation):
    path = tmp_path / "oriented.png"
    image = Image.fromarray(np.arange(6 * 9 * 3, dtype=np.uint8).reshape(6, 9, 3))
    exif = image.getexif()
    exif[274] = orientation
    image.save(path, exif=exif)
    with Image.open(path) as opened:
        expected = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    pixels, record = imaging.load_analysis_rgb(path)
    assert np.array_equal(pixels, expected)
    assert record["exif_orientation"] == orientation
    assert record["output_space"] == "sRGB"


def test_icc_conversion_and_unknown_color_state(tmp_path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "tagged.png"
    Image.new("RGB", (16, 16), (50, 120, 200)).save(path, icc_profile=profile)
    pixels, record = imaging.load_analysis_rgb(path)
    assert pixels[0, 0].tolist() == [50, 120, 200]
    assert record["color_status"] == "converted_icc"
    assert record["color_claims_allowed"] is True
    Image.new("RGB", (16, 16)).save(path, icc_profile=b"corrupt")
    _, record = imaging.load_analysis_rgb(path)
    assert record["color_status"] == "invalid_icc"
    assert record["color_claims_allowed"] is False
    Image.new("RGB", (16, 16)).save(path)
    _, record = imaging.load_analysis_rgb(path)
    assert record["color_status"] == "assumed_srgb"
    assert record["color_claims_allowed"] is False


def test_encoded_and_linear_luminance_have_different_definitions():
    values = imaging.compute_image_metrics(np.full((8, 8, 3), 128, dtype=np.uint8))
    assert values["encoded_luma_mean"] == pytest.approx(128 / 255, abs=1e-6)
    assert values["linear_luminance_mean"] == pytest.approx(
        ((128 / 255 + 0.055) / 1.055) ** 2.4, abs=1e-6
    )
    assert values["luma_mean"] == values["encoded_luma_mean"]


def test_multiframe_selection_is_recorded(tmp_path):
    path = tmp_path / "multi.tiff"
    Image.new("RGB", (8, 8), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (8, 8), "blue")]
    )
    pixels, record = imaging.load_analysis_rgb(path)
    assert pixels[0, 0].tolist() == [255, 0, 0]
    assert record["selected_frame"] == 0
    assert record["frame_count"] == 2
    assert record["hdr_analysis"] == "sdr_base_only"


def test_every_face_has_a_lossless_native_crop(tmp_path, monkeypatch):
    path = tmp_path / "faces.png"
    pixels = np.random.default_rng(7).integers(0, 255, (100, 150, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    monkeypatch.setattr(imaging, "detect_faces", lambda image: [[10, 10, 40, 40], [90, 20, 20, 30]])
    result = imaging.analyze_image(path, tmp_path / "artifacts")
    assert len(result["faces"]) == 2
    for face in result["faces"]:
        x, y, w, h = face["bbox_xywh"]
        with Image.open(face["native_crop_path"]) as crop:
            assert np.array_equal(np.asarray(crop), pixels[y : y + h, x : x + w])
        assert face["role"] == "UNASSIGNED"
    assert result["rendering"]["input_sha256"]
    assert Path(result["analysis_path"]).exists()


def test_model_preview_uses_the_same_color_conversion(tmp_path):
    import base64

    from portrait_eval.vlm import encode_image_data_url

    path = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), (80, 120, 160)).save(path)
    encoded = encode_image_data_url(path)
    preview = Image.open(BytesIO(base64.b64decode(encoded.split(",")[1])))
    assert preview.info.get("icc_profile")


def test_dark_rendering_does_not_classify_capture_as_low_light():
    audit = imaging.audit_scene(
        {
            key: {
                "whole": {
                    "luma_mean": 0.1,
                    "luma_p90": 0.2,
                    "luma_p10": 0.01,
                    "highlight_clip_ratio": 0,
                }
            }
            for key in ("a", "b")
        }
    )
    assert "low_light" not in audit["suggested_tags"]
    assert audit["challenge"]["primary"] == "unknown"


def test_background_excludes_every_face_and_position_role_is_preserved(tmp_path, monkeypatch):
    path = tmp_path / "group.png"
    Image.new("RGB", (150, 100), (80, 100, 120)).save(path)
    annotations = [
        {"bbox_xywh": [10, 10, 30, 30], "role": "PRIMARY", "position": "foreground"},
        {"bbox_xywh": [100, 20, 30, 30], "role": "SECONDARY", "position": "rear"},
    ]
    result = imaging.analyze_image(path, tmp_path / "crops", face_annotations=annotations)
    assert result["faces"][1]["position"] == "rear"
    assert result["regions"]["background"]["pixel_count"] < 15000 - 2 * 2000


def test_all_raw_slots_remain_explicit_when_every_source_has_them(tmp_path):
    from portrait_eval.dataset import propose_pairing, scan_folder

    scans = []
    for device in ["a", "b"]:
        folder = tmp_path / device
        folder.mkdir()
        Image.new("RGB", (8, 8)).save(folder / "1.png")
        (folder / "2.dng").write_bytes(b"placeholder")
        Image.new("RGB", (8, 8)).save(folder / "3.png")
        scans.append(scan_folder(device, folder))
    draft = propose_pairing(scans)
    assert len(draft.groups) == 3
    assert all(value is None for value in draft.groups[1].cells.values())


def test_same_color_in_srgb_and_display_p3_converts_within_two_code_values(tmp_path):
    import base64

    profile = ImageCms.ImageCmsProfile(
        BytesIO(base64.b64decode((Path(__file__).parent / "p3_profile.txt").read_text()))
    )
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    original = Image.new("RGB", (24, 24), (80, 140, 180))
    p3 = ImageCms.profileToProfile(
        original, srgb, profile, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC
    )
    a, b = tmp_path / "s.png", tmp_path / "p.png"
    original.save(a, icc_profile=srgb.tobytes())
    p3.save(b, icc_profile=profile.tobytes())
    left, _ = imaging.load_analysis_rgb(a)
    right, record = imaging.load_analysis_rgb(b)
    assert np.max(np.abs(left.astype(int) - right.astype(int))) <= 2
    assert record["color_claims_allowed"] is True


def test_native_face_model_input_is_lossless_and_keeps_evidence_mapping(tmp_path):
    import base64

    from portrait_eval.vlm import encode_image_data_url

    path = tmp_path / "crop.png"
    pixels = np.random.default_rng(4).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    url = encode_image_data_url(path, lossless=True)
    assert url.startswith("data:image/png;base64,")
    decoded = np.asarray(Image.open(BytesIO(base64.b64decode(url.split(",")[1]))))
    assert np.array_equal(decoded, pixels)


def test_source_precision_is_not_inferred_from_decoded_rgb_mode(tmp_path):
    import struct
    import zlib

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    path = tmp_path / "rgb16.png"
    ihdr = struct.pack(">IIBBBBB", 4, 4, 16, 2, 0, 0, 0)
    rows = b"".join(b"\0" + struct.pack(">HHH", 1000, 2000, 3000) * 4 for _ in range(4))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    _, record = imaging.load_analysis_rgb(path)
    assert record["source_bit_depth"] == 16
    assert record["output_bit_depth"] == 8


def test_raw_heavy_source_cannot_crash_alignment_or_drop_other_assets(tmp_path):
    from portrait_eval.dataset import propose_pairing, scan_folder

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for name in ["a.png", "d.png"]:
        Image.new("RGB", (8, 8)).save(a / name)
    for name in ["b.dng", "c.dng"]:
        (a / name).write_bytes(b"raw")
    for name in ["p.png", "q.png", "r.png"]:
        Image.new("RGB", (8, 8)).save(b / name)
    draft = propose_pairing([scan_folder("a", a), scan_folder("b", b)])
    assert len(draft.groups) == 4
    assert sum(cell is not None for group in draft.groups for cell in group.cells.values()) == 5
    assert all(group.review_required for group in draft.groups)
