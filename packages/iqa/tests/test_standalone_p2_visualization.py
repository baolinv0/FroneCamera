from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qwen_tmqa.config import TMQAConfig
from qwen_tmqa.domain import AlphaImage, SceneEvaluation, SceneSpec
from qwen_tmqa.evaluation import EvaluationPipeline
from qwen_tmqa.visualization import generate_dashboard

LEVELS = [("a_m100", -1.0), ("a_000", 0.0), ("a_p100", 1.0)]


def _evaluations(root: Path, scene_ids: tuple[str, ...]) -> list[SceneEvaluation]:
    evaluations = []
    pipeline = EvaluationPipeline(TMQAConfig(judges=[]))
    for scene_index, scene_id in enumerate(scene_ids):
        scene_dir = root / str(scene_index)
        scene_dir.mkdir(parents=True)
        source = scene_dir / "source.png"
        Image.new("RGB", (24, 20), (35 + scene_index * 80,) * 3).save(source)
        alphas = []
        for level, alpha in LEVELS:
            path = scene_dir / f"{level}.png"
            value = 55 + scene_index * 80 + round(alpha * 20)
            Image.new("RGB", (24, 20), (value,) * 3).save(path)
            alphas.append(AlphaImage(level=level, alpha=alpha, path=path))
        evaluations.append(
            pipeline.evaluate_scene(
                SceneSpec(
                    scene_id=scene_id,
                    baseline_path=scene_dir / "a_000.png",
                    source_path=source,
                    alpha_images=alphas,
                )
            )
        )
    return evaluations


def _read_payload(output: Path, name: str) -> dict:
    return json.loads((output / "data" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "scene_ids",
    [
        ("shot A.png", "shot_A.png"),
        ("folder/shot.png", "folder_shot.png"),
        ("目录/画面.png", "路径/场景.png"),
        ("../outside.png", ".._outside.png"),
        (".", ".."),
        ("a" * 300 + "A.png", "a" * 300 + "B.png"),
    ],
)
def test_dashboard_keeps_distinct_scene_images_in_reviewer_payload(
    tmp_path: Path, scene_ids: tuple[str, str]
) -> None:
    scenes = _evaluations(tmp_path / "inputs", scene_ids)
    output = tmp_path / "dashboard"

    generate_dashboard(scenes, output, title="Collision regression", review_queue=scenes)

    reviewer = _read_payload(output, "reviewer_payload.json")
    engineering = _read_payload(output, "engineering_evaluations.json")
    all_urls = [image["url"] for scene in reviewer["scenes"] for image in scene["images"]]
    assert len(set(all_urls)) == len(all_urls), "distinct review images must have distinct URLs"

    copied_and_diff_urls = []
    for evaluation, review_scene, engineering_scene in zip(
        scenes, reviewer["scenes"], engineering["scenes"], strict=True
    ):
        assert review_scene["scene_id"] == evaluation.scene_id
        visuals = engineering_scene["visual_paths"]
        assert len(review_scene["images"]) == 4
        for item, review_image in zip(evaluation.input_manifest, review_scene["images"], strict=True):
            key = f"{item.index}:{item.level}"
            url = review_image["url"]
            assert url == visuals[key]
            assert re.fullmatch(r"[A-Za-z0-9_./-]+", url)
            assert not {".", ".."}.intersection(Path(url).parts)
            assert (output / url).resolve().is_relative_to(output.resolve())
            with Image.open(item.path) as original, Image.open(output / url) as copied:
                assert copied.size == original.size
                assert np.allclose(np.asarray(copied), np.asarray(original), atol=2)
            if item.role in {"source", "candidate"}:
                diff_url = visuals[f"diff:{key}"]
                assert diff_url != url
                with Image.open(output / diff_url) as diff:
                    assert diff.size == (24, 20)
                    assert diff.mode == "RGB"
        copied_and_diff_urls.extend(value for key, value in visuals.items() if key != "baseline_key")

    assert len(copied_and_diff_urls) == 14
    assert len(set(copied_and_diff_urls)) == 14
    assert len(list((output / "assets" / "images").rglob("*.jpg"))) == 14
    assert "__EMBEDDED_REVIEWER_DATA__" not in (output / "review.html").read_text(encoding="utf-8")


def test_dashboard_asset_urls_are_stable_across_order_and_regeneration(tmp_path: Path) -> None:
    scenes = _evaluations(tmp_path / "inputs", ("shot A.png", "shot_A.png"))
    outputs = [tmp_path / "first", tmp_path / "second"]
    generate_dashboard(scenes, outputs[0], title="First", review_queue=scenes)
    generate_dashboard(scenes[::-1], outputs[1], title="Second", review_queue=scenes[::-1])
    by_scene = []
    for output in outputs:
        reviewer = _read_payload(output, "reviewer_payload.json")
        by_scene.append({scene["scene_id"]: scene["images"] for scene in reviewer["scenes"]})
    assert by_scene[0] == by_scene[1]
    original_bytes = {
        image["url"]: (outputs[0] / image["url"]).read_bytes()
        for scene in by_scene[0].values()
        for image in scene
    }

    generate_dashboard(scenes, outputs[0], title="Again", review_queue=scenes)

    assert original_bytes == {url: (outputs[0] / url).read_bytes() for url in original_bytes}


def test_dashboard_rejects_duplicate_scene_ids_before_writing(tmp_path: Path) -> None:
    scenes = _evaluations(tmp_path / "inputs", ("duplicate.png", "duplicate.png"))
    output = tmp_path / "dashboard"

    with pytest.raises(ValueError, match="duplicate scene_id"):
        generate_dashboard(scenes, output, title="Duplicate", review_queue=scenes)

    assert not output.exists()


@pytest.mark.parametrize("duplicate_level", ["a_000", "a 000", "other-level"])
def test_dashboard_rejects_duplicate_manifest_indices_before_writing(
    tmp_path: Path, duplicate_level: str
) -> None:
    scenes = _evaluations(tmp_path / "inputs", ("scene.png",))
    scenes[0].input_manifest[-1] = scenes[0].input_manifest[-1].model_copy(
        update={"index": 3, "level": duplicate_level}
    )
    output = tmp_path / "dashboard"

    with pytest.raises(ValueError, match="duplicate image index"):
        generate_dashboard(scenes, output, title="Duplicate", review_queue=scenes)

    assert not output.exists()
