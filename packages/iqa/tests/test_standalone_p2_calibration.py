from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_review_calibration import _scene

from qwen_tmqa.domain import HumanReview, ReliabilityResult, SceneEvaluation
from qwen_tmqa.review import append_review, calibrate_models, write_calibration_report


def _evidence(sample_count: int = 20) -> tuple[list[SceneEvaluation], list[HumanReview]]:
    scenes = [
        _scene(f"s{index}", 0.8, 1, [], (0.6, 0.8))
        for index in range(sample_count)
    ]
    reviews = [
        HumanReview(
            review_id=f"r{index}",
            scene_id=scene.scene_id,
            reviewer_id="human",
            blind_review=True,
            decision="KEEP",
            scores={"overall": 0.8},
            confidence=0.9,
        )
        for index, scene in enumerate(scenes)
    ]
    return scenes, reviews


def _run_cli(
    tmp_path: Path,
    scenes: list[SceneEvaluation],
    reviews: list[HumanReview],
    *extra_args: str,
    expect_failure: str | None = None,
) -> dict:
    results_path = tmp_path / "evaluations.json"
    results_path.write_text(
        json.dumps([scene.model_dump(mode="json") for scene in scenes]),
        encoding="utf-8",
    )
    reviews_path = tmp_path / "reviews.jsonl"
    for review in reviews:
        append_review(reviews_path, review)
    output = tmp_path / "calibration.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "qwen_tmqa.cli",
            "calibrate",
            "--results",
            str(results_path),
            "--reviews",
            str(reviews_path),
            "--output",
            str(output),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if expect_failure is not None:
        assert completed.returncode != 0, "ambiguous evidence produced a calibration report"
        assert expect_failure in completed.stderr
        assert not output.exists()
        return {}
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_cli_sparse_high_weight_model_cannot_inherit_other_models_sample_gate(
    tmp_path: Path,
) -> None:
    scenes, reviews = _evidence()
    for scene in scenes[1:]:
        scene.model_evaluations[1].available = False

    payload = _run_cli(tmp_path, scenes, reviews)

    assert payload["production_eligible"] is False
    assert payload["sample_sufficiency"]["actual_sample_count"] == 1
    assert payload["sample_sufficiency"]["sufficient"] is False
    by_id = {item["model_id"]: item for item in payload["models"]}
    assert by_id["m2"]["fusion_weight"] > 0.99
    assert by_id["m2"]["sample_count"] == 1
    assert payload["fusion_weight_scope"] == "research"
    assert all(item["fusion_weight_scope"] == "research" for item in payload["models"])
    assert all(item["production_eligible"] is False for item in payload["models"])
    assert payload["sample_sufficiency"]["per_model"]["m1"]["sufficient"] is True
    assert payload["sample_sufficiency"]["per_model"]["m2"]["sufficient"] is False
    assert "m2" in payload["warning"]


def test_cli_real_reviews_of_synthetic_models_stay_experimental(tmp_path: Path) -> None:
    scenes, reviews = _evidence()
    for scene in scenes:
        for model in scene.model_evaluations:
            model.synthetic = True

    payload = _run_cli(tmp_path, scenes, reviews)

    assert payload["production_eligible"] is False
    assert payload["experimental"] is True
    assert payload["synthetic_reviews"] is False
    assert payload["selected_review_type"] == "real"
    assert payload["synthetic_models"] is True
    assert payload["fusion_weight_scope"] == "research"
    assert all(item["synthetic"] is True for item in payload["models"])
    assert all(item["model_sources"] == ["synthetic"] for item in payload["models"])
    assert "synthetic model" in payload["warning"].lower()


def test_cli_all_sufficient_real_models_enable_production_weights(tmp_path: Path) -> None:
    scenes, reviews = _evidence()

    payload = _run_cli(tmp_path, scenes, reviews)

    assert payload["production_eligible"] is True
    assert payload["experimental"] is False
    assert payload["synthetic_models"] is False
    assert payload["sample_sufficiency"]["actual_sample_count"] == 20
    assert payload["fusion_weight_scope"] == "production"
    assert all(item["production_eligible"] is True for item in payload["models"])
    assert all(item["fusion_weight_scope"] == "production" for item in payload["models"])
    assert all(item["overall_sample_count"] == 20 for item in payload["models"])
    assert all(item["model_sources"] == ["real"] for item in payload["models"])


def test_mixed_model_provenance_survives_calibration_and_report_serialization(
    tmp_path: Path,
) -> None:
    scenes, reviews = _evidence()
    scenes[-1].model_evaluations[0].synthetic = True

    payload = _run_cli(tmp_path, scenes, reviews)
    by_id = {item["model_id"]: item for item in payload["models"]}

    assert payload["experimental"] is True
    assert payload["production_eligible"] is False
    assert payload["mixed_model_sources"] is True
    assert by_id["m1"]["model_sources"] == ["real", "synthetic"]
    assert by_id["m1"]["model_source_counts"] == {"real": 19, "synthetic": 1}
    assert by_id["m1"]["synthetic"] is True
    assert by_id["m2"]["model_sources"] == ["real"]
    assert by_id["m2"]["synthetic"] is False
    assert sum(item["fusion_weight"] for item in payload["models"]) == pytest.approx(1)
    reliability = [ReliabilityResult.model_validate(item) for item in payload["models"]]
    report_path = tmp_path / "reliability.json"
    write_calibration_report(report_path, reliability)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["models"] == payload["models"]


@pytest.mark.parametrize("case", ["empty", "unmatched", "unavailable", "no_overall"])
def test_cli_no_weight_evidence_is_never_production_eligible(
    tmp_path: Path,
    case: str,
) -> None:
    scenes, reviews = _evidence()
    if case == "empty":
        reviews = []
    elif case == "unmatched":
        scenes = []
    elif case == "unavailable":
        for scene in scenes:
            for model in scene.model_evaluations:
                model.available = False
    elif case == "no_overall":
        for scene in scenes:
            for model in scene.model_evaluations:
                model.scores.pop("overall")

    payload = _run_cli(tmp_path, scenes, reviews)

    assert payload["production_eligible"] is False
    assert payload["sample_sufficiency"]["actual_sample_count"] == 0
    assert payload["sample_sufficiency"]["sufficient"] is False
    assert payload["fusion_weight_scope"] == "research"
    assert all(item["fusion_weight"] == 0 for item in payload["models"])


def test_weight_sample_gate_uses_paired_overall_scores_not_decision_only_samples(
    tmp_path: Path,
) -> None:
    scenes, reviews = _evidence()
    for scene in scenes[1:]:
        scene.model_evaluations[1].scores.pop("overall")

    payload = _run_cli(tmp_path, scenes, reviews)
    by_id = {item["model_id"]: item for item in payload["models"]}

    assert payload["production_eligible"] is False
    assert by_id["m2"]["sample_count"] == 20
    assert by_id["m2"]["overall_sample_count"] == 1
    assert by_id["m2"]["fusion_weight"] > 0.99
    assert payload["sample_sufficiency"]["actual_sample_count"] == 1


def test_unreviewed_and_unavailable_synthetic_models_do_not_taint_calibration(
    tmp_path: Path,
) -> None:
    scenes, reviews = _evidence()
    unreviewed, _ = _evidence(1)
    unreviewed[0].scene_id = "not-reviewed"
    for model in unreviewed[0].model_evaluations:
        model.synthetic = True
    scenes.extend(unreviewed)
    unavailable = scenes[0].model_evaluations[0].model_copy(
        update={"model_id": "unavailable-synthetic", "synthetic": True, "available": False}
    )
    scenes[0].model_evaluations.append(unavailable)

    payload = _run_cli(tmp_path, scenes, reviews)

    assert payload["production_eligible"] is True
    assert payload["experimental"] is False
    assert payload["synthetic_models"] is False
    assert {item["model_id"] for item in payload["models"]} == {"m1", "m2"}


def test_synthetic_reviews_keep_separate_review_and_model_provenance(tmp_path: Path) -> None:
    scenes, reviews = _evidence()
    reviews = [review.model_copy(update={"synthetic": True}) for review in reviews]

    payload = _run_cli(
        tmp_path, scenes, reviews, "--allow-synthetic", "--review-type", "synthetic"
    )

    assert payload["synthetic_reviews"] is True
    assert payload["synthetic_models"] is False
    assert payload["experimental"] is True
    assert payload["production_eligible"] is False
    assert all(item["model_sources"] == ["real"] for item in payload["models"])
    assert all(item["fusion_weight_scope"] == "research" for item in payload["models"])


def test_bare_calibration_metrics_default_to_research_and_legacy_results_still_load() -> None:
    scenes, reviews = _evidence()
    metrics = calibrate_models(scenes, reviews)

    assert all(item.fusion_weight_scope == "research" for item in metrics)
    assert all(item.production_eligible is False for item in metrics)
    legacy = ReliabilityResult.model_validate(
        {
            "model_id": "legacy",
            "sample_count": 20,
            "overall_mae": 0.1,
            "decision_agreement": 1.0,
            "dimension_mae": {"overall": 0.1},
            "fusion_weight": 1.0,
        }
    )
    assert legacy.fusion_weight_scope == "research"
    assert legacy.model_sources == []
    assert legacy.production_eligible is False


@pytest.mark.parametrize("duplicate_kind", ["identical", "conflicting", "unavailable"])
def test_cli_rejects_duplicate_model_ids_before_one_review_can_count_as_twenty(
    tmp_path: Path,
    duplicate_kind: str,
) -> None:
    scenes, reviews = _evidence(1)
    original = scenes[0].model_evaluations[0]
    duplicates = [original.model_copy(deep=True) for _ in range(19)]
    if duplicate_kind == "conflicting":
        duplicates[-1].scores["overall"] = 0.8
        duplicates[-1].synthetic = True
    elif duplicate_kind == "unavailable":
        for model in duplicates:
            model.available = False
    scenes[0].model_evaluations = [original, *duplicates]

    _run_cli(tmp_path, scenes, reviews, expect_failure="duplicate model_id")


@pytest.mark.parametrize("conflicting", [False, True])
def test_cli_rejects_ambiguous_duplicate_scene_ids(
    tmp_path: Path,
    conflicting: bool,
) -> None:
    scenes, reviews = _evidence()
    duplicate = scenes[0].model_copy(deep=True)
    if conflicting:
        duplicate.model_evaluations[0].scores["overall"] = 0.0
    scenes.append(duplicate)

    _run_cli(tmp_path, scenes, reviews, expect_failure="duplicate scene_id")


@pytest.mark.parametrize("independent_reviewer_ids", [False, True])
def test_cli_rejects_reused_review_id_in_imported_review_evidence(
    tmp_path: Path,
    independent_reviewer_ids: bool,
) -> None:
    scenes, reviews = _evidence(1)
    duplicates = [
        reviews[0].model_copy(
            update={"reviewer_id": f"human-{index}" if independent_reviewer_ids else "human"}
        )
        for index in range(20)
    ]

    _run_cli(tmp_path, scenes, duplicates, expect_failure="duplicate review_id")
