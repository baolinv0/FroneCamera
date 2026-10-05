import numpy as np
from PIL import Image


def test_loader_present():
    import importlib.util

    assert importlib.util.find_spec("qwen_tmqa.asset_io") is not None


def test_uint16_float_and_exif_preserved(tmp_path):
    from qwen_tmqa.asset_io import load_comparison_asset
    from qwen_tmqa.comparison_models import ComparisonAsset

    a = np.array([[1234, 1235], [40000, 65535]], dtype=np.uint16)
    path = tmp_path / "16.png"
    Image.fromarray(a).save(path)
    loaded = load_comparison_asset(ComparisonAsset(id="a", path=path, encoding="srgb"))
    assert loaded.trace["bit_depth"] == 16
    assert loaded.pixels[0, 0, 0] != loaded.pixels[0, 1, 0]
    assert loaded.pixels[1, 0, 0] == a[1, 0] / 65535
    f = np.array([[0.123456789, 2.0], [0.23, 0.01]], np.float64)
    path = tmp_path / "linear.npy"
    np.save(path, f)
    loaded = load_comparison_asset(ComparisonAsset(id="f", path=path, encoding="linear"))
    assert loaded.pixels[0, 0, 0] == f[0, 0]
    assert loaded.pixels[0, 1, 0] == 2
    assert loaded.trace["dtype"] == "float64"
    path = tmp_path / "rotate.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(np.zeros((4, 9, 3), dtype=np.uint8)).save(path, exif=exif)
    loaded = load_comparison_asset(ComparisonAsset(id="e", path=path, encoding="srgb"))
    assert loaded.pixels.shape == (9, 4, 3)
    assert loaded.trace["exif_orientation"] == 6
    assert loaded.trace["orientation_convention"] == "exif_transposed"


def test_rgb16_and_all_orientation_conventions(tmp_path):
    import cv2

    from qwen_tmqa.asset_io import load_comparison_asset
    from qwen_tmqa.comparison_models import ComparisonAsset

    rgb = np.array([[[1000, 2000, 3000], [1001, 2001, 3001]]], np.uint16)
    path = tmp_path / "rgb16.png"
    cv2.imwrite(str(path), rgb[..., ::-1])
    loaded = load_comparison_asset(ComparisonAsset(id="rgb", path=path))
    np.testing.assert_array_equal(loaded.pixels, rgb.astype(float) / 65535)
    for orientation in range(1, 9):
        a = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        path = tmp_path / f"orientation-{orientation}.png"
        exif = Image.Exif()
        exif[274] = orientation
        Image.fromarray(a).save(path, exif=exif)
        from PIL import ImageOps

        with Image.open(path) as image:
            expected = np.asarray(ImageOps.exif_transpose(image)) / 255
        actual = load_comparison_asset(ComparisonAsset(id="e", path=path))
        np.testing.assert_array_equal(actual.pixels, expected)


def test_disguised_numpy_archive_is_invalid_evidence(tmp_path):
    from qwen_tmqa.comparison import evaluate_comparison
    from qwen_tmqa.comparison_models import ComparisonRequest

    path = tmp_path / "not-image.npy"
    with path.open("wb") as f:
        np.savez(f, data=np.zeros((16, 16, 3)))
    out = evaluate_comparison(
        ComparisonRequest(
            scene_id="s",
            group_id="g",
            mode="device",
            baseline={"id": "B", "path": path},
            candidates=[{"id": "C", "path": path}],
        )
    )
    assert out.assets[0].state == "invalid"
    assert out.assets[0].trace["source_bytes_sha256"]
