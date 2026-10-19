"""Claim identity, proposition matching and narrowly applicable measurement evidence."""

from __future__ import annotations

import hashlib
import json
import math
from statistics import median
from typing import Any

from portrait_eval.models import ModelObservation


def claim_identity(
    device: str,
    scene: str,
    statement: str,
    region: str,
    attribute: str,
    direction: str,
    version: str,
    comparison: str = "unknown",
    conditions: list[str] | None = None,
) -> str:
    content = json.dumps(
        [
            device,
            scene,
            statement,
            region,
            attribute,
            direction,
            version,
            comparison,
            sorted(set(conditions or [])),
        ],
        ensure_ascii=False,
    )
    return "claim:" + hashlib.sha256(content.encode()).hexdigest()[:24]


def review_relation(primary: ModelObservation, review: ModelObservation) -> str:
    fields = ("device_id", "dimension", "region", "attribute", "comparison")
    if any(getattr(primary, key) != getattr(review, key) for key in fields) or set(
        primary.conditions
    ) != set(review.conditions):
        return "different_proposition"
    if (
        any(
            getattr(primary, key) == "unknown"
            for key in ("region", "attribute", "comparison", "direction")
        )
        or review.direction == "unknown"
    ):
        return "unknown"
    return "supports" if primary.direction == review.direction else "contradicts"


def numeric_metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get("whole", {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        return None
    return float(value)


def measure_observation(
    observation: ModelObservation, metrics: dict[str, Any], audit: dict[str, Any]
) -> tuple[float | None, list[dict[str, Any]]]:
    definitions = {
        "encoded_luma_mean": (
            "luma_mean",
            0.03,
            "mean(.2126 R prime + .7152 G prime + .0722 B prime)",
        ),
        "encoded_near_white_ratio": ("highlight_clip_ratio", 0.002, "mean(Y prime >= .98)"),
    }
    if (
        observation.attribute not in definitions
        or observation.region != "whole"
        or observation.comparison != "peer_median"
    ):
        return None, []
    canonical_dimension = {
        "encoded_luma_mean": "global_exposure",
        "encoded_near_white_ratio": "highlight_retention",
    }[observation.attribute]
    comparability = audit.get("attribute_comparability", {})
    states = [
        comparability[key]
        for key in {observation.attribute, observation.dimension, canonical_dimension}
        if key in comparability
    ]
    if any(state != "FULLY_COMPARABLE" for state in states) or (
        not states and audit.get("status") != "FULLY_COMPARABLE"
    ):
        return None, []
    legacy_key, threshold, formula = definitions[observation.attribute]
    values = {key: numeric_metric(payload, legacy_key) for key, payload in metrics.items()}
    if (
        observation.device_id not in values
        or len(values) < 2
        or any(value is None for value in values.values())
    ):
        return None, []
    current = values[observation.device_id]
    others = [value for value in values.values() if value is not None]
    if current is None:
        return None, []
    delta = current - median(others)
    actual = "higher" if delta > threshold else "lower" if delta < -threshold else "equal"
    record = {
        "attribute": observation.attribute,
        "formula": formula,
        "threshold": threshold,
        "comparison": "peer_median",
        "data": values,
        "delta": delta,
        "direction": actual,
        "applicability": "rendered_output_only",
    }
    return (1.0 if observation.direction == actual else 0.0), [record]


def related_external(
    finding: dict[str, Any], external: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not finding.get("claim_id"):
        return []
    return [
        item
        for item in external
        if all(
            item.get(key) == finding.get(key)
            for key in ("claim_id", "claim_version", "device_id", "attribute")
        )
    ]


def measured_statement(observation: ModelObservation) -> str:
    attribute = {
        "encoded_luma_mean": "encoded global luma",
        "encoded_near_white_ratio": "near-white pixel ratio",
    }[observation.attribute]
    return f"In this matched group, the {attribute} is {observation.direction} relative to the group median under the recorded threshold; this describes the SDR output, not image quality or sensor recovery."
