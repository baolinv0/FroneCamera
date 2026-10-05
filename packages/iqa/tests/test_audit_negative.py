from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import qwen_tmqa.audit as audit_module
from qwen_tmqa.audit import audit_verified_experiment
from qwen_tmqa.cli import (
    calibrate_command,
    evaluate_command,
    make_example,
    simulate_human,
    visualize_command,
)


def _audit(
    results: Path,
    dashboard: Path,
    reviews: Path,
    calibration: Path,
    evidence: Path,
    expected_head_sha: str = "a" * 40,
) -> dict[str, bool]:
    return audit_verified_experiment(
        results_path=results / "evaluations.json",
        dashboard_dir=dashboard,
        reviews_path=reviews,
        calibration_path=calibration,
        evidence_path=evidence,
        expected_head_sha=expected_head_sha,
    )


def _write_behavioral_evidence(path: Path, results_path: Path) -> None:
    checks = {
        key: {
            "command": f"pytest::{key}",
            "exit_code": 0,
            "output_sha256": key.encode().hex().ljust(64, "0")[:64],
        }
        for key in [
            "fatal_non_compensation",
            "decision_disagreement",
            "zero_judges",
            "all_unavailable",
            "dashboard_behavior",
            "blind_review_network",
            "route_isolation",
        ]
    }
    payload = {
        "schema_version": "tmqa.verification-evidence.v1",
        "artifact_head_sha": "a" * 40,
        "artifacts": {
            "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        },
        "checks": checks,
        "ci": {
            "provider": "github_actions",
            "tested_head_sha": "a" * 40,
            "jobs": {"python-3.10": "success", "python-3.12": "success"},
            "run_id": 123,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_audit_fails_closed_for_injected_contract_violations(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    reviews = tmp_path / "reviews.jsonl"
    calibration = tmp_path / "calibration.json"
    dashboard = tmp_path / "dashboard"
    config = Path("configs/default.yaml")

    make_example(dataset, 1)
    evaluate_command(dataset, results, config)
    simulate_human(results / "evaluations.json", reviews, config)
    calibrate_command(
        results / "evaluations.json",
        reviews,
        calibration,
        allow_synthetic=True,
        review_type="synthetic",
    )
    visualize_command(
        results / "evaluations.json",
        dashboard,
        config,
        reviews,
        calibration,
    )
    evidence = tmp_path / "verification_evidence.json"
    _write_behavioral_evidence(evidence, results / "evaluations.json")

    baseline = _audit(results, dashboard, reviews, calibration, evidence)
    assert len(baseline) == 19
    assert all(baseline.values()), baseline

    reviewer_path = dashboard / "data" / "reviewer_payload.json"
    reviewer_payload = json.loads(reviewer_path.read_text(encoding="utf-8"))
    reviewer_payload["scenes"][0]["model_id"] = "leaked-model"
    reviewer_path.write_text(
        json.dumps(reviewer_payload, indent=2),
        encoding="utf-8",
    )
    assert _audit(results, dashboard, reviews, calibration, evidence)["R-010"] is False
    reviewer_payload["scenes"][0].pop("model_id")
    reviewer_path.write_text(
        json.dumps(reviewer_payload, indent=2),
        encoding="utf-8",
    )

    evaluation_payload = json.loads(
        (results / "evaluations.json").read_text(encoding="utf-8")
    )
    evaluation_payload[0]["model_evaluations"][0]["prompt_trace"]["input_manifest"][0][
        "payload_sha256"
    ] = None
    (results / "evaluations.json").write_text(
        json.dumps(evaluation_payload, indent=2),
        encoding="utf-8",
    )
    assert _audit(results, dashboard, reviews, calibration, evidence)["R-006"] is False

    evaluate_command(dataset, results, config)
    _write_behavioral_evidence(evidence, results / "evaluations.json")
    review_payload = json.loads(reviews.read_text(encoding="utf-8").splitlines()[0])
    review_payload["scores"].pop("overall")
    reviews.write_text(json.dumps(review_payload) + "\n", encoding="utf-8")
    assert _audit(results, dashboard, reviews, calibration, evidence)["R-012"] is False

    simulate_human(results / "evaluations.json", reviews, config)
    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    calibration_payload["selected_review_type"] = "real"
    calibration.write_text(
        json.dumps(calibration_payload, indent=2),
        encoding="utf-8",
    )
    assert _audit(results, dashboard, reviews, calibration, evidence)["R-013"] is False


@pytest.mark.parametrize(
    ("check_name", "requirement_ids"),
    [
        ("fatal_non_compensation", ["R-008"]),
        ("decision_disagreement", ["R-008"]),
        ("zero_judges", ["R-003", "R-008"]),
        ("all_unavailable", ["R-008"]),
        ("dashboard_behavior", ["R-009"]),
        ("blind_review_network", ["R-010"]),
        ("route_isolation", ["R-010", "R-015"]),
    ],
)
def test_audit_fails_closed_when_behavioral_evidence_is_injected_failed(
    tmp_path: Path,
    check_name: str,
    requirement_ids: list[str],
) -> None:
    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    reviews = tmp_path / "reviews.jsonl"
    calibration = tmp_path / "calibration.json"
    dashboard = tmp_path / "dashboard"
    config = Path("configs/default.yaml")
    make_example(dataset, 1)
    evaluate_command(dataset, results, config)
    simulate_human(results / "evaluations.json", reviews, config)
    calibrate_command(
        results / "evaluations.json",
        reviews,
        calibration,
        allow_synthetic=True,
        review_type="synthetic",
        config_path=config,
    )
    visualize_command(
        results / "evaluations.json",
        dashboard,
        config,
        reviews,
        calibration,
    )
    evidence = tmp_path / "verification_evidence.json"
    _write_behavioral_evidence(evidence, results / "evaluations.json")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["checks"][check_name]["exit_code"] = 1
    evidence.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    checks = _audit(results, dashboard, reviews, calibration, evidence)

    assert all(checks[requirement_id] is False for requirement_id in requirement_ids)


def test_audit_fails_closed_for_stale_or_incomplete_exact_head_ci_evidence(
    tmp_path: Path,
) -> None:
    results = tmp_path / "evaluations.json"
    results.write_text("[]", encoding="utf-8")
    evidence = tmp_path / "verification_evidence.json"
    _write_behavioral_evidence(evidence, results)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["ci"]["tested_head_sha"] = "b" * 40
    payload["ci"]["jobs"]["python-3.10"] = "failure"
    evidence.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    checks = audit_verified_experiment(
        results_path=results,
        dashboard_dir=tmp_path / "missing-dashboard",
        reviews_path=tmp_path / "reviews.jsonl",
        calibration_path=tmp_path / "calibration.json",
        evidence_path=evidence,
        expected_head_sha="a" * 40,
    )

    assert checks["R-018"] is False


def test_requirement_audit_records_have_complete_traceability_fields() -> None:
    checks = {f"R-{index:03d}": True for index in range(1, 20)}
    records = audit_module.build_requirement_records(checks, actual_evidence={})

    assert set(records) == set(checks)
    for record in records.values():
        assert record["implementation_locations"]
        assert record["test_locations"]
        assert record["execution_command"]
        assert record["measurable_threshold"]
        assert record["actual_evidence"]
        assert record["final_status"] == "PASS"
