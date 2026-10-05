from pathlib import Path

import numpy as np
from PIL import Image

from qwen_tmqa.config import DatasetConfig
from qwen_tmqa.dataset import discover_scenes
from qwen_tmqa.image_io import file_sha256, image_dimensions, load_rgb
from qwen_tmqa.metrics import compute_objective_metrics, compute_sequence_metrics

LEVELS = {
    "a_m100": -1.0,
    "a_m075": -0.75,
    "a_m050": -0.5,
    "a_m025": -0.25,
    "a_000": 0.0,
    "a_p025": 0.25,
    "a_p050": 0.5,
    "a_p075": 0.75,
    "a_p100": 1.0,
}


def _save(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.05, value, 48, dtype=np.float32)
    image = np.tile(x[None, :, None], (32, 1, 3))
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def test_discover_scenes_parses_nine_alpha_levels(tmp_path: Path) -> None:
    for level, alpha in LEVELS.items():
        _save(tmp_path / level / "scene.png", 0.45 + 0.25 * alpha)
    scenes = discover_scenes(tmp_path, DatasetConfig(generator="control-light"))
    assert len(scenes) == 1
    assert scenes[0].generator == "control-light"
    assert [item.alpha for item in scenes[0].alpha_images] == list(LEVELS.values())
    assert scenes[0].baseline_path == tmp_path / "a_000" / "scene.png"


def test_discover_scenes_ignores_non_image_files(tmp_path: Path) -> None:
    for level in LEVELS:
        _save(tmp_path / level / "scene.png", 0.5)
    (tmp_path / "a_000" / "notes.json").write_text("{}", encoding="utf-8")
    scenes = discover_scenes(tmp_path, DatasetConfig())
    assert [scene.scene_id for scene in scenes] == ["scene.png"]


def test_image_dimensions_follow_exif_orientation_like_image_loading(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (20, 10), color=(20, 40, 60))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    loaded = load_rgb(path)
    assert image_dimensions(path) == (10, 20)
    assert loaded.shape[:2] == (20, 10)


def test_manifest_hash_and_dimensions_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _save(path, 0.5)
    image = load_rgb(path)
    assert image.shape == (32, 48, 3)
    assert file_sha256(path) == file_sha256(path)
    assert len(file_sha256(path)) == 64


def test_objective_metrics_detect_clipping_and_brightness() -> None:
    dark = np.full((20, 20, 3), 0.02, dtype=np.float32)
    bright = np.ones((20, 20, 3), dtype=np.float32)
    dark_metrics = compute_objective_metrics(
        dark,
        alpha=-1,
        level="a_m100",
        baseline=dark,
    )
    bright_metrics = compute_objective_metrics(
        bright,
        alpha=1,
        level="a_p100",
        baseline=dark,
    )
    assert dark_metrics.ev_mean < bright_metrics.ev_mean
    assert bright_metrics.clipping_ratio == 1.0
    assert dark_metrics.shadow_ratio > 0.9
    assert bright_metrics.color_drift < 1e-6


def test_sequence_metrics_detect_reversal_and_dead_zone() -> None:
    baseline = np.full((16, 16, 3), 0.3, dtype=np.float32)
    monotonic = [
        compute_objective_metrics(
            np.full((16, 16, 3), value, dtype=np.float32),
            alpha=alpha,
            level=str(index),
            baseline=baseline,
        )
        for index, (alpha, value) in enumerate(
            [(-1, 0.1), (-0.5, 0.2), (0, 0.3), (0.5, 0.45), (1, 0.6)]
        )
    ]
    bad = [
        compute_objective_metrics(
            np.full((16, 16, 3), value, dtype=np.float32),
            alpha=alpha,
            level=str(index),
            baseline=baseline,
        )
        for index, (alpha, value) in enumerate(
            [(-1, 0.1), (-0.5, 0.2), (0, 0.3), (0.5, 0.28), (1, 0.28)]
        )
    ]
    good_result = compute_sequence_metrics(monotonic)
    bad_result = compute_sequence_metrics(bad)
    assert good_result.spearman_rho > 0.99
    assert good_result.violation_rate == 0
    assert bad_result.violation_rate > 0
    assert bad_result.dead_zone_ratio > 0
    assert good_result.control_score > bad_result.control_score
