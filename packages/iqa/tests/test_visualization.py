import json
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_tmqa.config import load_config
from qwen_tmqa.dataset import discover_scenes
from qwen_tmqa.domain import ReliabilityResult
from qwen_tmqa.evaluation import EvaluationPipeline
from qwen_tmqa.review import select_review_queue
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
    image = np.full((24, 32, 3), np.clip(value, 0, 1), dtype=np.float32)
    Image.fromarray(np.round(image * 255).astype(np.uint8)).save(path)


def _evaluations(tmp_path: Path):
    for scene_index in range(2):
        for level, alpha in LEVELS:
            _save(
                tmp_path / "dataset" / level / f"scene_{scene_index}.png",
                0.45 + 0.25 * alpha,
            )
    config = load_config(Path("configs/default.yaml"))
    scenes = discover_scenes(tmp_path / "dataset", config.dataset)
    evaluations = [EvaluationPipeline(config).evaluate_scene(scene) for scene in scenes]
    return config, evaluations


def test_engineering_dashboard_contains_required_debug_views_and_prompt_data(tmp_path: Path) -> None:
    config, evaluations = _evaluations(tmp_path)
    queue = select_review_queue(evaluations, config.review)
    output = tmp_path / "dashboard"
    generate_dashboard(evaluations, output, title="Test Dashboard", review_queue=queue)

    html = (output / "index.html").read_text(encoding="utf-8")
    data = json.loads(
        (output / "data" / "engineering_evaluations.json").read_text(encoding="utf-8")
    )

    for marker in [
        'data-view="overview"',
        'data-view="scene"',
        'data-view="stages"',
        'data-view="models"',
        'data-view="prompt"',
        'data-view="queue"',
        'data-view="reliability"',
    ]:
        assert marker in html
    for prompt_tab in [
        "Rendered Prompt",
        "Template",
        "Input Manifest",
        "Version Diff",
        "Raw Response",
        "Parsed JSON",
    ]:
        assert prompt_tab in html
    assert "Qwen3-VL-8B-Instruct" in html
    assert "人工盲审" not in html
    assert 'data-view="review"' not in html
    first_model = data["scenes"][0]["model_evaluations"][0]
    assert first_model["prompt_trace"]["rendered_prompt"]
    assert first_model["prompt_trace"]["input_manifest"]
    assert "raw_response" in first_model
    assert (output / "data" / "review_queue.json").exists()


def test_engineering_dashboard_renders_complete_existing_evidence(tmp_path: Path) -> None:
    _, evaluations = _evaluations(tmp_path)
    output = tmp_path / "dashboard"
    generate_dashboard(
        evaluations,
        output,
        title="Test Dashboard",
        review_queue=evaluations,
        reliability=[
            ReliabilityResult(
                model_id="qwen-primary",
                sample_count=2,
                overall_mae=0.1,
                decision_agreement=0.5,
                dimension_mae={"tone": 0.2, "overall": 0.1},
                fusion_weight=0.6,
            )
        ],
        pairwise_mean_gap={"qwen-primary__internvl-arbiter": 0.12},
    )

    html = (output / "index.html").read_text(encoding="utf-8")

    for marker in [
        'id="decisionDistribution"',
        'id="alphaMetrics"',
        'id="promptMetadata"',
        'id="pairwiseGaps"',
    ]:
        assert marker in html
    for label in [
        "低置信场景",
        "Synthetic dataset",
        "Objective metrics",
        "Model version",
        "Availability",
        "Synthetic model",
        "Confidence",
        "Model score gap",
        "Prompt ID",
        "Prompt version",
        "Prompt hash",
        "Schema version",
        "Rendered variables",
        "Inference parameters",
        "Sample count",
        "Decision agreement",
        "Dimension MAE",
        "Pairwise model gap",
        "Human-model delta",
        "Fusion weight",
    ]:
        assert label in html
    for field in [
        "DATA.summary.decisions",
        "DATA.summary.low_confidence_count",
        "DATA.summary.synthetic",
        "selectedScene.objective_by_level",
        "m.model_version",
        "m.available",
        "m.synthetic",
        "m.confidence",
        "t.prompt_id",
        "t.prompt_version",
        "t.prompt_hash",
        "t.output_schema_version",
        "t.variables",
        "t.inference_parameters",
        "x.sample_count",
        "x.decision_agreement",
        "x.dimension_mae",
        "DATA.pairwise_mean_gap",
    ]:
        assert field in html


def test_reviewer_page_and_payload_are_free_of_model_and_system_evidence(tmp_path: Path) -> None:
    config, evaluations = _evaluations(tmp_path)
    queue = select_review_queue(evaluations, config.review)
    output = tmp_path / "dashboard"
    generate_dashboard(evaluations, output, title="Test Dashboard", review_queue=queue)

    html = (output / "review.html").read_text(encoding="utf-8")
    payload = json.loads((output / "data" / "reviewer_payload.json").read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)

    for label in ["人工盲审", "Review Queue", "Reviewer ID", "提交人工判断"]:
        assert label in html
    for corruption_marker in ["鍚", "鈫", "ºÏ", "鏁版", "螖"]:
        assert corruption_marker not in html
    for forbidden in [
        "Qwen3-VL-8B-Instruct",
        "model_evaluations",
        "model_id",
        "prompt_trace",
        "rendered_prompt",
        "raw_response",
        "parsed_response",
        "overall_score",
        "review_reasons",
        "stages",
        "system_decision",
    ]:
        assert forbidden not in html
        assert forbidden not in serialized
    assert "__EMBEDDED_REVIEWER_DATA__" not in html
    assert (output / "data" / "reveal_payload.json").exists()


def test_dashboard_copies_actual_scene_images_for_both_filmstrips(tmp_path: Path) -> None:
    _, evaluations = _evaluations(tmp_path)
    output = tmp_path / "dashboard"
    generate_dashboard(evaluations, output, title="Test", review_queue=evaluations)
    images = list((output / "assets" / "images").rglob("*.jpg"))
    assert len(images) >= 18
    engineering = (output / "index.html").read_text(encoding="utf-8")
    reviewer = (output / "review.html").read_text(encoding="utf-8")
    assert "alpha-filmstrip" in engineering
    assert "difference-map" in engineering
    assert 'id="reviewFilmstrip"' in reviewer
    assert 'id="reviewMainImage"' in reviewer


def test_blind_review_form_requires_human_values_but_no_model_values(tmp_path: Path) -> None:
    _, evaluations = _evaluations(tmp_path)
    output = tmp_path / "dashboard"
    generate_dashboard(evaluations, output, title="Test", review_queue=evaluations)
    html = (output / "review.html").read_text(encoding="utf-8")
    assert 'name="human_decision"' in html
    assert 'name="human_overall"' in html
    assert 'name="human_confidence"' in html
    assert 'name="human_rationale"' in html
    assert 'name="human_color"' in html
    assert 'name="human_control"' in html
    assert 'name="human_preference"' in html
    assert 'name="reviewer_id"' in html
    assert "reviewer_id:String(fd.get('reviewer_id')||'').trim()" in html
    assert "reviewer_id:'dashboard-human'" not in html
    form = html.split('id="blind-review-form"', 1)[1].split("</form>", 1)[0]
    assert "model_evaluations" not in form
    assert "localStorage" not in html


def test_blind_review_filmstrip_uses_its_own_selected_image(tmp_path: Path) -> None:
    _, evaluations = _evaluations(tmp_path)
    output = tmp_path / "dashboard"
    generate_dashboard(evaluations, output, title="Test Dashboard", review_queue=evaluations)

    html = (output / "review.html").read_text(encoding="utf-8")

    assert "reviewSelectedManifest" in html
    assert "selected&&selected.index===item.index" in html
    assert "renderReviewFilmstrip" in html
    assert "fetch(responseBody.reveal_url" in html


def test_engineering_dashboard_displays_calibration_sample_sufficiency(
    tmp_path: Path,
) -> None:
    _, evaluations = _evaluations(tmp_path)
    output = tmp_path / "dashboard"
    warning = "Insufficient real blind-review samples (3/20); not production-ready."
    generate_dashboard(
        evaluations,
        output,
        title="Test Dashboard",
        review_queue=evaluations,
        calibration_disclosure={
            "warning": warning,
            "production_eligible": False,
            "sample_sufficiency": {
                "policy_version": "tmqa.calibration.v1",
                "actual_sample_count": 3,
                "minimum_real_sample_count": 20,
                "sufficient": False,
            },
        },
    )

    html = (output / "index.html").read_text(encoding="utf-8")
    engineering = json.loads(
        (output / "data" / "engineering_evaluations.json").read_text(
            encoding="utf-8"
        )
    )
    assert warning in html
    assert engineering["calibration"]["production_eligible"] is False
    assert engineering["calibration"]["sample_sufficiency"]["sufficient"] is False
