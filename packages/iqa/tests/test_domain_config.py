from pathlib import Path

import pytest

from qwen_tmqa.config import JudgeConfig, TMQAConfig, load_config
from qwen_tmqa.domain import (
    AlphaImage,
    HumanReview,
    ImageManifestItem,
    ModelEvaluation,
    PromptTrace,
    SceneSpec,
)

EXPECTED_LEVELS = [
    "a_m100",
    "a_m075",
    "a_m050",
    "a_m025",
    "a_000",
    "a_p025",
    "a_p050",
    "a_p075",
    "a_p100",
]


def test_scene_alpha_images_are_sorted() -> None:
    scene = SceneSpec(
        scene_id="s1",
        baseline_path=Path("a_000/s1.png"),
        alpha_images=[
            AlphaImage(level="a_p100", alpha=1.0, path=Path("a_p100/s1.png")),
            AlphaImage(level="a_m100", alpha=-1.0, path=Path("a_m100/s1.png")),
            AlphaImage(level="a_000", alpha=0.0, path=Path("a_000/s1.png")),
        ],
    )
    assert [item.alpha for item in scene.alpha_images] == [-1.0, 0.0, 1.0]


def test_prompt_trace_requires_explicit_manifest() -> None:
    with pytest.raises(ValueError, match="input_manifest"):
        PromptTrace(
            prompt_id="tmqa.sequence",
            prompt_version="3.2",
            template="x",
            rendered_prompt="x",
            prompt_hash="abc",
            input_manifest=[],
        )


def test_model_evaluation_normalizes_scores() -> None:
    trace = PromptTrace(
        prompt_id="tmqa.sequence",
        prompt_version="3.2",
        template="x",
        rendered_prompt="x",
        prompt_hash="abc",
        input_manifest=[
            ImageManifestItem(
                index=1,
                role="baseline",
                alpha=0.0,
                level="a_000",
                path="a_000/s.png",
                sha256="f" * 64,
                width=32,
                height=32,
            )
        ],
    )
    result = ModelEvaluation(
        model_id="m",
        model_role="primary",
        model_version="v1",
        prompt_trace=trace,
        scores={"tone": 82, "overall": 0.7},
        confidence=85,
        decision="KEEP",
    )
    assert result.scores == {"tone": 0.82, "overall": 0.7}
    assert result.confidence == 0.85


def test_human_review_rejects_nonblind_gold_label() -> None:
    with pytest.raises(ValueError, match="blind"):
        HumanReview(
            review_id="r1",
            scene_id="s1",
            reviewer_id="u1",
            blind_review=False,
            decision="KEEP",
            scores={"overall": 0.8},
            confidence=0.8,
        )


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.dataset.baseline_level == "a_000"
    assert config.dataset.expected_levels == EXPECTED_LEVELS
    assert len(config.judges) == 4
    assert {judge.prompt_version for judge in config.judges} == {"3.3"}
    assert config.review.max_queue_size == 200
    assert config.calibration.version == "tmqa.calibration.v1"
    assert config.calibration.minimum_real_sample_count == 20
    assert config.visualization.title == "Qwen-TMQA Visual Review"


def test_openai_compatible_judge_requires_a_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        JudgeConfig(
            id="remote",
            role="primary",
            adapter="openai_compatible",
            model="remote-model",
        )


def test_human_review_requires_a_non_empty_reviewer_id() -> None:
    with pytest.raises(ValueError, match="reviewer_id"):
        HumanReview(
            review_id="r-empty-reviewer",
            scene_id="scene.png",
            reviewer_id="",
            blind_review=True,
            decision="KEEP",
            scores={"overall": 0.8},
            confidence=0.9,
        )


def test_decision_policy_rejects_inverted_score_thresholds() -> None:
    with pytest.raises(ValueError, match="regenerate_score_threshold"):
        TMQAConfig.model_validate(
            {
                "judges": [],
                "decision_policy": {
                    "regenerate_score_threshold": 0.80,
                    "keep_score_threshold": 0.75,
                },
            }
        )


@pytest.mark.parametrize(
    ("config_path", "unknown_key", "expected_location"),
    [
        ((), "unknown_root", ("unknown_root",)),
        (("dataset",), "unknown_dataset", ("dataset", "unknown_dataset")),
        (("judges", 0), "unknown_judge", ("judges", 0, "unknown_judge")),
        (
            ("decision_policy",),
            "unknown_decision_policy",
            ("decision_policy", "unknown_decision_policy"),
        ),
        (("review",), "unknown_review", ("review", "unknown_review")),
        (
            ("calibration",),
            "unknown_calibration",
            ("calibration", "unknown_calibration"),
        ),
        (
            ("visualization",),
            "unknown_visualization",
            ("visualization", "unknown_visualization"),
        ),
    ],
)
def test_config_rejects_unknown_keys_at_every_supported_level(
    config_path: tuple[str | int, ...],
    unknown_key: str,
    expected_location: tuple[str | int, ...],
) -> None:
    payload: dict[str, object] = {
        "dataset": {},
        "judges": [
            {
                "id": "judge",
                "role": "primary",
                "model": "model",
            }
        ],
        "decision_policy": {},
        "review": {},
        "calibration": {},
        "visualization": {},
    }
    target: object = payload
    for component in config_path:
        target = target[component]  # type: ignore[index]
    target[unknown_key] = True  # type: ignore[index]

    with pytest.raises(ValueError) as error:
        TMQAConfig.model_validate(payload)

    assert expected_location in {item["loc"] for item in error.value.errors()}
