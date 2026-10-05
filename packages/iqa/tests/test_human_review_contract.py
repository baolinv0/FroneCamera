from __future__ import annotations

import pytest

from qwen_tmqa.domain import HumanReview


def test_human_review_requires_overall_score() -> None:
    with pytest.raises(ValueError, match="overall"):
        HumanReview(
            review_id="missing-overall",
            scene_id="scene.png",
            reviewer_id="human",
            blind_review=True,
            decision="KEEP",
            scores={"tone": 0.8, "color": 0.8},
            confidence=0.9,
        )


def test_human_review_accepts_overall_with_optional_dimensions() -> None:
    review = HumanReview(
        review_id="with-overall",
        scene_id="scene.png",
        reviewer_id="human",
        blind_review=True,
        decision="KEEP",
        scores={"overall": 82, "tone": 78},
        confidence=0.9,
    )

    assert review.scores["overall"] == pytest.approx(0.82)
    assert review.scores["tone"] == pytest.approx(0.78)
