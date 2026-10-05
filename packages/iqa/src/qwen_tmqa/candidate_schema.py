from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from pydantic import BaseModel, ConfigDict, Field


class CandidatePreferenceEvidence(BaseModel):
    """Strict candidate-selection fields required by tmqa.sequence@3.4."""

    model_config = ConfigDict(extra="ignore", strict=True)

    preferred_level: str
    runner_up_level: str
    acceptable_levels: list[str]
    level_scores: dict[str, float]
    selection_confidence: float = Field(ge=0, le=1)
    baseline_improvement: float = Field(ge=-1, le=1)


def validate_candidate_preference_payload(
    payload: Mapping[str, object],
    *,
    expected_levels: AbstractSet[str],
    baseline_level: str,
) -> CandidatePreferenceEvidence:
    evidence = CandidatePreferenceEvidence.model_validate(payload)
    expected = set(expected_levels)
    actual = set(evidence.level_scores)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"level_scores keys must equal evaluated levels; missing={missing}, extra={extra}"
        )
    if baseline_level not in expected:
        raise ValueError(f"baseline level {baseline_level} is not evaluated")
    if evidence.preferred_level not in expected:
        raise ValueError("preferred_level must be an evaluated level")
    if evidence.runner_up_level not in expected:
        raise ValueError("runner_up_level must be an evaluated level")
    if evidence.runner_up_level == evidence.preferred_level and len(expected) > 1:
        raise ValueError("runner_up_level must differ from preferred_level")
    if not evidence.acceptable_levels:
        raise ValueError("acceptable_levels must not be empty")
    if not set(evidence.acceptable_levels).issubset(expected):
        raise ValueError("acceptable_levels must be evaluated levels")
    if evidence.preferred_level not in evidence.acceptable_levels:
        raise ValueError("preferred_level must be included in acceptable_levels")

    for level, score in evidence.level_scores.items():
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"level score for {level} must be finite and in [0,1]")
    if not math.isfinite(evidence.selection_confidence):
        raise ValueError("selection_confidence must be finite")
    if not math.isfinite(evidence.baseline_improvement):
        raise ValueError("baseline_improvement must be finite")

    preferred_score = evidence.level_scores[evidence.preferred_level]
    top_score = max(evidence.level_scores.values())
    if not math.isclose(preferred_score, top_score, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("preferred_level must be one of the highest-scoring levels")

    alternatives = {
        level: score
        for level, score in evidence.level_scores.items()
        if level != evidence.preferred_level
    }
    if alternatives:
        runner_score = evidence.level_scores[evidence.runner_up_level]
        expected_runner_score = max(alternatives.values())
        if not math.isclose(runner_score, expected_runner_score, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("runner_up_level must be a highest-scoring non-preferred level")

    expected_improvement = preferred_score - evidence.level_scores[baseline_level]
    if not math.isclose(
        evidence.baseline_improvement,
        expected_improvement,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("baseline_improvement must equal preferred score minus baseline score")
    return evidence
