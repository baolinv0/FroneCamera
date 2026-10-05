from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_tmqa.config import TMQAConfig, VisualizationConfig
from qwen_tmqa.dataset import discover_scenes
from qwen_tmqa.evaluation import EvaluationPipeline
from qwen_tmqa.visualization import generate_dashboard

LEVELS = [
    ("a_m100", -1.0),
    ("a_m075", -0.75),
    ("a_m050", -0.5),
    ("a_m025", -0.25),
    ("a_000", 0.0),
    ("a_p025", 0.25),
    ("a_p050", 0.5),
    ("a_p075", 0.75),
    ("a_p100", 1.0),
]


def _save(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((20, 28, 3), np.clip(value, 0, 1), dtype=np.float32)
    Image.fromarray(np.round(image * 255).astype(np.uint8)).save(path)


def test_zero_judge_pipeline_still_generates_engineering_and_reviewer_images(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    for level, alpha in LEVELS:
        _save(dataset / level / "scene.png", 0.45 + 0.2 * alpha)

    config = TMQAConfig(
        judges=[],
        visualization=VisualizationConfig(title="Zero Judge TMQA"),
    )
    spec = discover_scenes(dataset, config.dataset)[0]
    evaluation = EvaluationPipeline(config).evaluate_scene(spec)

    assert evaluation.model_evaluations == []
    assert len(evaluation.input_manifest) == 9
    assert evaluation.decision == "REVIEW"
    assert "judge_unavailable" in evaluation.review_reasons
    assert evaluation.uncertainty >= 0.5
    model_stage = next(
        stage for stage in evaluation.stages if stage.stage_id == "model_judges"
    )
    assert model_stage.status == "WARN"
    assert "available=0" in model_stage.evidence

    output = tmp_path / "dashboard"
    generate_dashboard(
        [evaluation],
        output,
        title=config.visualization.title,
        review_queue=[evaluation],
    )

    engineering = json.loads(
        (output / "data" / "engineering_evaluations.json").read_text(
            encoding="utf-8"
        )
    )
    reviewer = json.loads(
        (output / "data" / "reviewer_payload.json").read_text(encoding="utf-8")
    )

    assert engineering["scenes"][0]["model_evaluations"] == []
    assert len(engineering["scenes"][0]["input_manifest"]) == 9
    assert engineering["scenes"][0]["decision"] == "REVIEW"
    assert "judge_unavailable" in engineering["scenes"][0]["review_reasons"]
    assert len(reviewer["scenes"][0]["images"]) == 9
    assert len(list((output / "assets" / "images").rglob("*.jpg"))) >= 9
