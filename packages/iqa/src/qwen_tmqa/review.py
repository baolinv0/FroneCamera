from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from .config import ReviewConfig
from .domain import HumanReview, ReliabilityResult, SceneEvaluation

ReviewType = Literal["real", "synthetic"]


def select_review_queue(
    scenes: list[SceneEvaluation],
    config: ReviewConfig,
) -> list[SceneEvaluation]:
    if not scenes:
        return []
    scores = np.array([scene.overall_score for scene in scenes], dtype=np.float64)
    tail_cutoff = float(np.quantile(scores, config.tail_percentile))
    selected: list[SceneEvaluation] = []
    for scene in scenes:
        reasons = set(scene.review_reasons)
        if scene.overall_score <= tail_cutoff:
            reasons.add("low_tail")
        if reasons:
            if reasons != set(scene.review_reasons):
                scene = scene.model_copy(update={"review_reasons": sorted(reasons)})
            selected.append(scene)
    selected.sort(key=lambda item: (item.review_priority, -item.overall_score), reverse=True)
    return selected[: config.max_queue_size]


def append_review(path: Path, review: HumanReview) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(review.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_reviews(path: Path) -> list[HumanReview]:
    if not path.exists():
        return []
    reviews: list[HumanReview] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            reviews.append(HumanReview.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid review at line {line_number}: {exc}") from exc
    # Server-owned exposure snapshots override caller-declared blind status.
    snapshots = _load_exposure_events(path)
    for index, review in enumerate(reviews):
        event = snapshots.get((review.scene_id, review.reviewer_id))
        if event is not None:
            digest = hashlib.sha256(review.model_dump_json().encode()).hexdigest()
            eligible = event["gold_review_hashes"].get(review.review_id) == digest
            reviews[index] = review.model_copy(update={"gold_eligible": eligible})
    return reviews


def select_calibration_reviews(
    reviews: list[HumanReview],
    *,
    review_type: ReviewType | None = None,
    allow_synthetic: bool = False,
) -> tuple[list[HumanReview], int, str]:
    all_reviews = reviews
    reviews = [review for review in reviews if review.gold_eligible]
    present_types = {"synthetic" if review.synthetic else "real" for review in reviews}
    if len(present_types) > 1 and review_type is None:
        raise ValueError("mixed synthetic and real review evidence requires --review-type")

    if review_type is None:
        selected = list(reviews)
        selected_type = next(iter(present_types), "none")
    else:
        selected = [
            review
            for review in reviews
            if ("synthetic" if review.synthetic else "real") == review_type
        ]
        selected_type = review_type
        if reviews and not selected:
            raise ValueError(f"no {review_type} reviews available for calibration")

    if selected_type == "synthetic" and selected and not allow_synthetic:
        raise ValueError("synthetic reviews require explicit --allow-synthetic")
    return selected, len(all_reviews) - len(selected), selected_type


def calibrate_models(
    scenes: list[SceneEvaluation],
    reviews: list[HumanReview],
    *,
    review_type: ReviewType | None = None,
    allow_synthetic: bool = False,
) -> list[ReliabilityResult]:
    selected_reviews, _, _ = select_calibration_reviews(
        reviews,
        review_type=review_type,
        allow_synthetic=allow_synthetic,
    )
    scene_by_id: dict[str, SceneEvaluation] = {}
    for scene in scenes:
        if scene.scene_id in scene_by_id:
            raise ValueError(f"duplicate scene_id {scene.scene_id!r} in calibration results")
        model_ids: set[str] = set()
        for model in scene.model_evaluations:
            if model.model_id in model_ids:
                raise ValueError(
                    f"duplicate model_id {model.model_id!r} in scene {scene.scene_id!r}"
                )
            model_ids.add(model.model_id)
        scene_by_id[scene.scene_id] = scene
    latest_by_reviewer: dict[tuple[str, str], HumanReview] = {}
    review_ids: set[str] = set()
    for review in selected_reviews:
        if review.review_id in review_ids:
            raise ValueError(f"duplicate review_id {review.review_id!r} in calibration reviews")
        review_ids.add(review.review_id)
        key = (review.scene_id, review.reviewer_id)
        current = latest_by_reviewer.get(key)
        review_time = datetime.fromisoformat(review.received_at.replace("Z", "+00:00"))
        current_time = (
            datetime.fromisoformat(current.received_at.replace("Z", "+00:00"))
            if current is not None
            else None
        )
        if current_time is None or review_time >= current_time:
            latest_by_reviewer[key] = review

    absolute_errors: dict[str, list[float]] = defaultdict(list)
    decision_matches: dict[str, list[float]] = defaultdict(list)
    dimension_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    model_source_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for review in latest_by_reviewer.values():
        scene = scene_by_id.get(review.scene_id)
        if scene is None:
            continue
        human_overall = review.scores.get("overall")
        for model in scene.model_evaluations:
            if not model.available:
                continue
            source = "synthetic" if model.synthetic else "real"
            model_source_counts[model.model_id][source] += 1
            model_overall = model.scores.get("overall")
            if human_overall is not None and model_overall is not None:
                absolute_errors[model.model_id].append(abs(model_overall - human_overall))
            decision_matches[model.model_id].append(float(model.decision == review.decision))
            for dimension, human_score in review.scores.items():
                if dimension in model.scores:
                    dimension_errors[model.model_id][dimension].append(
                        abs(model.scores[dimension] - human_score)
                    )

    raw_weights: dict[str, float] = {}
    for model_id, errors in absolute_errors.items():
        mean_error = float(np.mean(errors)) if errors else 1.0
        raw_weights[model_id] = 1.0 / (mean_error + 1e-3)
    weight_sum = sum(raw_weights.values()) or 1.0

    results: list[ReliabilityResult] = []
    for model_id in sorted(set(absolute_errors) | set(decision_matches)):
        errors = absolute_errors.get(model_id, [])
        results.append(
            ReliabilityResult(
                model_id=model_id,
                sample_count=max(len(errors), len(decision_matches.get(model_id, []))),
                overall_sample_count=len(errors),
                model_sources=sorted(model_source_counts[model_id]),
                model_source_counts=dict(sorted(model_source_counts[model_id].items())),
                synthetic="synthetic" in model_source_counts[model_id],
                overall_mae=float(np.mean(errors)) if errors else 1.0,
                decision_agreement=float(np.mean(decision_matches[model_id]))
                if decision_matches.get(model_id)
                else 0.0,
                dimension_mae={
                    dimension: float(np.mean(values))
                    for dimension, values in sorted(dimension_errors[model_id].items())
                },
                fusion_weight=raw_weights.get(model_id, 0.0) / weight_sum,
            )
        )
    return results


def pairwise_mean_gaps(
    scenes: list[SceneEvaluation],
    dimension: str = "overall",
) -> dict[str, float]:
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for scene in scenes:
        available = [
            model
            for model in scene.model_evaluations
            if model.available and dimension in model.scores
        ]
        for left_index, left in enumerate(available):
            for right in available[left_index + 1 :]:
                first, second = sorted([left.model_id, right.model_id])
                pair_values[(first, second)].append(
                    abs(left.scores[dimension] - right.scores[dimension])
                )
    return {
        f"{left}__{right}": float(np.mean(values))
        for (left, right), values in sorted(pair_values.items())
    }


def write_calibration_report(path: Path, results: list[ReliabilityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": [item.model_dump(mode="json") for item in results],
        "note": "Metrics are valid only for the supplied human-review set.",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def exposure_path(reviews_path: Path) -> Path:
    return reviews_path.with_name(reviews_path.name + ".exposure.jsonl")


def _load_exposure_events(reviews_path: Path) -> dict[tuple[str, str], dict]:
    path = exposure_path(reviews_path)
    if not path.exists():
        return {}
    exposed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if (
            not isinstance(event, dict)
            or not all(
                isinstance(event.get(key), str) and event[key].strip()
                for key in ["scene_id", "reviewer_id", "review_id", "revealed_at"]
            )
            or not isinstance(event.get("gold_review_hashes"), dict)
            or not event["gold_review_hashes"]
        ):
            raise ValueError("invalid persisted exposure event")
        exposed.setdefault((event["scene_id"], event["reviewer_id"]), event)
    return exposed


def load_exposures(reviews_path: Path) -> set[tuple[str, str]]:
    return set(_load_exposure_events(reviews_path))


def append_exposure(reviews_path: Path, review: HumanReview) -> None:
    # Persist before returning model evidence. A storage failure fails closed.
    path = exposure_path(reviews_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = {
        item.review_id: hashlib.sha256(item.model_dump_json().encode()).hexdigest()
        for item in load_reviews(reviews_path)
        if item.scene_id == review.scene_id
        and item.reviewer_id == review.reviewer_id
        and item.gold_eligible
    }
    event = {
        "schema_version": "tmqa.exposure.v1",
        "gold_review_hashes": hashes,
        "scene_id": review.scene_id,
        "reviewer_id": review.reviewer_id,
        "review_id": review.review_id,
        "revealed_at": datetime.now().astimezone().isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
