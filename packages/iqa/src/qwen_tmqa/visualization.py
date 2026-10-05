from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .domain import ReliabilityResult, SceneEvaluation
from .prompts import PromptRegistry, prompt_version_diff


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _scene_manifest(scene: SceneEvaluation):
    if scene.input_manifest:
        return scene.input_manifest
    if scene.model_evaluations:
        return scene.model_evaluations[0].prompt_trace.input_manifest
    return []


def _copy_visual_assets(scene: SceneEvaluation, output_dir: Path) -> dict[str, str]:
    visual_paths: dict[str, str] = {}
    manifest = _scene_manifest(scene)
    if not manifest:
        return visual_paths
    scene_dir = output_dir / "assets" / "images" / _safe_name(scene.scene_id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    baseline_array: np.ndarray | None = None
    baseline_key: str | None = None
    for item in manifest:
        source = Path(item.path)
        destination = scene_dir / f"{item.index:02d}_{_safe_name(item.level)}.jpg"
        with Image.open(source) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.save(destination, format="JPEG", quality=92)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        key = f"{item.index}:{item.level}"
        visual_paths[key] = destination.relative_to(output_dir).as_posix()
        if item.role == "baseline":
            baseline_array = array
            baseline_key = key
    if baseline_array is not None:
        for item in manifest:
            if item.role not in {"candidate", "source"}:
                continue
            source = Path(item.path)
            with Image.open(source) as image:
                candidate = (
                    np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
                    / 255.0
                )
            if candidate.shape != baseline_array.shape:
                continue
            difference = np.mean(np.abs(candidate - baseline_array), axis=2)
            scale = max(float(np.percentile(difference, 99)), 1e-6)
            heat = np.clip(difference / scale, 0, 1)
            heat_rgb = np.stack([heat, np.sqrt(heat), 1 - heat], axis=2)
            destination = scene_dir / f"diff_{item.index:02d}_{_safe_name(item.level)}.jpg"
            Image.fromarray(np.round(heat_rgb * 255).astype(np.uint8)).save(
                destination, format="JPEG", quality=92
            )
            visual_paths[f"diff:{item.index}:{item.level}"] = destination.relative_to(
                output_dir
            ).as_posix()
    if baseline_key:
        visual_paths["baseline_key"] = baseline_key
    return visual_paths


def _scene_payload(scene: SceneEvaluation, output_dir: Path) -> dict[str, Any]:
    payload = scene.model_dump(mode="json")
    payload["visual_paths"] = _copy_visual_assets(scene, output_dir)
    registry = PromptRegistry.default()
    for model in payload["model_evaluations"]:
        version = model["prompt_trace"]["prompt_version"]
        if version != "3.1":
            model["prompt_diff"] = prompt_version_diff(
                registry,
                model["prompt_trace"]["prompt_id"],
                "3.1",
                version,
            )
        else:
            model["prompt_diff"] = ""
    return payload


def _summary(scenes: list[SceneEvaluation]) -> dict[str, Any]:
    decisions: dict[str, int] = {
        key: 0 for key in ["KEEP", "REGENERATE", "REVIEW", "REJECT"]
    }
    for scene in scenes:
        decisions[scene.decision] += 1
    scores = [scene.overall_score for scene in scenes]
    return {
        "scene_count": len(scenes),
        "decisions": decisions,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "p10_score": float(np.quantile(scores, 0.1)) if scores else 0.0,
        "low_confidence_count": sum(scene.uncertainty >= 0.5 for scene in scenes),
        "synthetic": bool(scenes) and all(scene.synthetic_experiment for scene in scenes),
    }


def _reviewer_scene(engineering_scene: dict[str, Any]) -> dict[str, Any]:
    manifest = engineering_scene.get("input_manifest", [])
    if not manifest and engineering_scene.get("model_evaluations"):
        manifest = engineering_scene["model_evaluations"][0]["prompt_trace"][
            "input_manifest"
        ]
    visuals = engineering_scene["visual_paths"]
    images = []
    for item in manifest:
        key = f"{item['index']}:{item['level']}"
        images.append(
            {
                "index": item["index"],
                "role": item["role"],
                "alpha": item["alpha"],
                "level": item["level"],
                "url": visuals.get(key, ""),
            }
        )
    return {"scene_id": engineering_scene["scene_id"], "images": images}


def generate_dashboard(
    scenes: list[SceneEvaluation],
    output_dir: Path,
    *,
    title: str,
    review_queue: list[SceneEvaluation] | None = None,
    reliability: list[ReliabilityResult] | None = None,
    pairwise_mean_gap: dict[str, float] | None = None,
    calibration_disclosure: dict[str, Any] | None = None,
) -> Path:
    if not scenes:
        raise ValueError("at least one scene evaluation is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    engineering_scenes = [_scene_payload(scene, output_dir) for scene in scenes]
    queue_ids = [scene.scene_id for scene in (review_queue or [])]
    engineering_payload = {
        "title": title,
        "summary": _summary(scenes),
        "scenes": engineering_scenes,
        "review_queue": queue_ids,
        "reliability": [item.model_dump(mode="json") for item in (reliability or [])],
        "pairwise_mean_gap": pairwise_mean_gap or {},
        "calibration": calibration_disclosure or {},
    }
    engineering_by_id = {scene["scene_id"]: scene for scene in engineering_scenes}
    reviewer_payload = {
        "title": title,
        "scenes": [
            _reviewer_scene(engineering_by_id[scene_id])
            for scene_id in queue_ids
            if scene_id in engineering_by_id
        ],
    }
    reveal_payload = {
        scene_id: {
            "scene_id": scene_id,
            "model_evaluations": engineering_by_id[scene_id]["model_evaluations"],
        }
        for scene_id in queue_ids
        if scene_id in engineering_by_id
    }

    engineering_text = json.dumps(engineering_payload, indent=2, ensure_ascii=False)
    reviewer_text = json.dumps(reviewer_payload, indent=2, ensure_ascii=False)
    (output_dir / "data" / "engineering_evaluations.json").write_text(
        engineering_text, encoding="utf-8"
    )
    (output_dir / "data" / "reviewer_payload.json").write_text(
        reviewer_text, encoding="utf-8"
    )
    (output_dir / "data" / "reveal_payload.json").write_text(
        json.dumps(reveal_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "data" / "review_queue.json").write_text(
        json.dumps(queue_ids, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    assets_dir = Path(__file__).with_name("assets")
    engineering_template = (assets_dir / "dashboard.html").read_text(encoding="utf-8")
    engineering_rendered = engineering_template.replace("__TITLE__", html.escape(title))
    engineering_rendered = engineering_rendered.replace(
        "__EMBEDDED_DATA__", engineering_text.replace("</", "<\\/")
    )
    index_path = output_dir / "index.html"
    index_path.write_text(engineering_rendered, encoding="utf-8")

    reviewer_template = (assets_dir / "review.html").read_text(encoding="utf-8")
    reviewer_rendered = reviewer_template.replace("__TITLE__", html.escape(title))
    reviewer_rendered = reviewer_rendered.replace(
        "__EMBEDDED_REVIEWER_DATA__", reviewer_text.replace("</", "<\\/")
    )
    (output_dir / "review.html").write_text(reviewer_rendered, encoding="utf-8")
    return index_path
