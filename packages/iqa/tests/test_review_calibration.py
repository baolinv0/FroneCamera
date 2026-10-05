import json
from pathlib import Path

import pytest

from qwen_tmqa.cli import calibrate_command
from qwen_tmqa.config import ReviewConfig
from qwen_tmqa.domain import (
    HumanReview,
    ImageManifestItem,
    ModelEvaluation,
    ObjectiveMetrics,
    PromptTrace,
    SceneEvaluation,
    SequenceMetrics,
    StageResult,
)
from qwen_tmqa.review import (
    append_review,
    calibrate_models,
    load_reviews,
    select_review_queue,
)


def _trace() -> PromptTrace:
    return PromptTrace(
        prompt_id="tmqa.sequence",
        prompt_version="3.2",
        template="t",
        rendered_prompt="r",
        prompt_hash="h",
        input_manifest=[
            ImageManifestItem(
                index=1,
                role="baseline",
                alpha=0,
                level="a_000",
                path="x.png",
                sha256="f" * 64,
                width=16,
                height=16,
            )
        ],
    )


def _model(
    model_id: str,
    overall: float,
    decision: str,
    tone: float | None = None,
) -> ModelEvaluation:
    return ModelEvaluation(
        model_id=model_id,
        model_role="primary",
        model_version="v1",
        prompt_trace=_trace(),
        scores={"overall": overall, "tone": tone if tone is not None else overall},
        decision=decision,
        confidence=0.8,
    )


def _scene(
    scene_id: str,
    score: float,
    priority: float,
    reasons: list[str],
    gap: tuple[float, float],
) -> SceneEvaluation:
    objective = ObjectiveMetrics(
        alpha=0,
        level="a_000",
        mean_luminance=0.4,
        ev_mean=-1,
        p20=0.2,
        p50=0.4,
        p90=0.6,
        clipping_ratio=0,
        shadow_ratio=0,
        contrast=0.2,
        color_drift=0,
        edge_similarity=1,
    )
    sequence = SequenceMetrics(
        spearman_rho=1,
        violation_rate=0,
        smoothness_score=1,
        dead_zone_ratio=0,
        endpoint_range_ev=1,
        clipping_growth=0,
        control_score=1,
    )
    return SceneEvaluation(
        scene_id=scene_id,
        generator="g",
        decision="REVIEW" if reasons else "KEEP",
        overall_score=score,
        uncertainty=0.2,
        objective_by_level={"a_000": objective},
        sequence=sequence,
        stages=[StageResult(stage_id="x", label="x", status="PASS", score=1)],
        model_evaluations=[
            _model("m1", gap[0], "KEEP"),
            _model("m2", gap[1], "REVIEW"),
        ],
        review_priority=priority,
        review_reasons=reasons,
    )


def test_review_queue_prioritizes_disagreement_tail_and_high_risk() -> None:
    scenes = [
        _scene("clean", 0.9, 0.1, [], (0.88, 0.90)),
        _scene("disagree", 0.8, 1.5, ["model_disagreement"], (0.95, 0.55)),
        _scene("tail", 0.2, 1.0, [], (0.20, 0.25)),
        _scene("risk", 0.7, 2.0, ["high_risk"], (0.72, 0.70)),
    ]
    queue = select_review_queue(
        scenes,
        ReviewConfig(
            max_queue_size=3,
            tail_percentile=0.25,
            random_audit_fraction=0,
        ),
    )
    assert [item.scene_id for item in queue] == ["risk", "disagree", "tail"]
    assert "clean" not in {item.scene_id for item in queue}


def test_review_storage_is_append_only_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "reviews.jsonl"
    first = HumanReview(
        review_id="r1",
        scene_id="s1",
        reviewer_id="human-a",
        blind_review=True,
        decision="KEEP",
        scores={"overall": 80, "tone": 75},
        confidence=0.9,
    )
    second = first.model_copy(update={"review_id": "r2", "scene_id": "s2"})
    append_review(path, first)
    append_review(path, second)
    reviews = load_reviews(path)
    assert [item.review_id for item in reviews] == ["r1", "r2"]
    assert reviews[0].scores["overall"] == 0.8


def test_calibration_reports_human_error_and_inverse_error_weights() -> None:
    scenes = [
        _scene("s1", 0.8, 1, ["model_disagreement"], (0.90, 0.60)),
        _scene("s2", 0.7, 1, ["model_disagreement"], (0.75, 0.55)),
    ]
    reviews = [
        HumanReview(
            review_id="r1",
            scene_id="s1",
            reviewer_id="h",
            blind_review=True,
            decision="KEEP",
            scores={"overall": 0.85, "tone": 0.8},
            confidence=0.9,
        ),
        HumanReview(
            review_id="r2",
            scene_id="s2",
            reviewer_id="h",
            blind_review=True,
            decision="KEEP",
            scores={"overall": 0.70, "tone": 0.72},
            confidence=0.9,
        ),
    ]
    results = calibrate_models(scenes, reviews)
    by_id = {item.model_id: item for item in results}
    assert by_id["m1"].overall_mae < by_id["m2"].overall_mae
    assert by_id["m1"].fusion_weight > by_id["m2"].fusion_weight
    assert abs(sum(item.fusion_weight for item in results) - 1) < 1e-9
    assert "tone" in by_id["m1"].dimension_mae


def test_calibration_uses_every_independent_reviewer() -> None:
    scenes = [_scene("s1", 0.8, 1, ["model_disagreement"], (0.8, 0.6))]
    reviews = [
        HumanReview(
            review_id="r1",
            scene_id="s1",
            reviewer_id="h1",
            blind_review=True,
            decision="KEEP",
            scores={"overall": 0.8},
            confidence=0.9,
        ),
        HumanReview(
            review_id="r2",
            scene_id="s1",
            reviewer_id="h2",
            blind_review=True,
            decision="REVIEW",
            scores={"overall": 0.6},
            confidence=0.9,
        ),
    ]
    by_id = {
        item.model_id: item for item in calibrate_models(scenes, reviews)
    }
    assert by_id["m1"].sample_count == 2
    assert by_id["m1"].overall_mae == pytest.approx(0.1)
    assert by_id["m1"].decision_agreement == 0.5


def test_calibration_uses_only_latest_received_submission_from_same_reviewer() -> None:
    scenes = [_scene("s1", 0.8, 1, ["model_disagreement"], (0.8, 0.6))]
    reviews = [
        HumanReview(
            review_id="old",
            scene_id="s1",
            reviewer_id="h1",
            blind_review=True,
            decision="REJECT",
            scores={"overall": 0.0},
            confidence=0.9,
            received_at="2026-07-17T01:00:00+00:00",
        ),
        HumanReview(
            review_id="new",
            scene_id="s1",
            reviewer_id="h1",
            blind_review=True,
            decision="KEEP",
            scores={"overall": 0.8},
            confidence=0.9,
            received_at="2026-07-17T02:00:00+00:00",
        ),
    ]
    by_id = {
        item.model_id: item for item in calibrate_models(scenes, reviews)
    }
    assert by_id["m1"].sample_count == 1
    assert by_id["m1"].overall_mae == pytest.approx(0.0)
    assert by_id["m1"].decision_agreement == 1.0


def test_calibration_uses_received_at_instead_of_jsonl_order() -> None:
    scenes = [_scene("s1", 0.8, 1, ["model_disagreement"], (0.8, 0.6))]
    reviews = [
        HumanReview(
            review_id="new",
            scene_id="s1",
            reviewer_id="h1",
            blind_review=True,
            decision="KEEP",
            scores={"overall": 0.8},
            confidence=0.9,
            received_at="2026-07-17T02:00:00+00:00",
        ),
        HumanReview(
            review_id="old",
            scene_id="s1",
            reviewer_id="h1",
            blind_review=True,
            decision="REJECT",
            scores={"overall": 0.0},
            confidence=0.9,
            received_at="2026-07-17T01:00:00+00:00",
        ),
    ]
    by_id = {
        item.model_id: item for item in calibrate_models(scenes, reviews)
    }
    assert by_id["m1"].sample_count == 1
    assert by_id["m1"].overall_mae == pytest.approx(0.0)
    assert by_id["m1"].decision_agreement == 1.0


@pytest.mark.parametrize("sample_count", [1, 3])
def test_real_calibration_with_few_samples_is_marked_non_production(
    tmp_path: Path,
    sample_count: int,
) -> None:
    scenes = [
        _scene(f"s{index}", 0.8, 1, ["model_disagreement"], (0.8, 0.6))
        for index in range(sample_count)
    ]
    results_path = tmp_path / "evaluations.json"
    results_path.write_text(
        json.dumps([scene.model_dump(mode="json") for scene in scenes]),
        encoding="utf-8",
    )
    reviews_path = tmp_path / "reviews.jsonl"
    for index, scene in enumerate(scenes):
        append_review(
            reviews_path,
            HumanReview(
                review_id=f"r{index}",
                scene_id=scene.scene_id,
                reviewer_id="human",
                blind_review=True,
                decision="KEEP",
                scores={"overall": 0.8},
                confidence=0.9,
            ),
        )

    output = tmp_path / "calibration.json"
    calibrate_command(
        results_path,
        reviews_path,
        output,
        config_path=Path("configs/default.yaml"),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    sufficiency = payload["sample_sufficiency"]
    assert sufficiency["policy_version"] == "tmqa.calibration.v1"
    assert sufficiency["actual_sample_count"] == sample_count
    assert sufficiency["minimum_real_sample_count"] == 20
    assert sufficiency["sufficient"] is False
    assert payload["production_eligible"] is False
    assert "insufficient real" in payload["warning"].lower()
