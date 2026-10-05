from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from qwen_tmqa.audit import audit_verified_experiment
from qwen_tmqa.server import create_server


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_review_server_stores_valid_blind_review_and_rejects_invalid(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    (dashboard / "index.html").write_text("ok", encoding="utf-8")
    (dashboard / "review.html").write_text("review", encoding="utf-8")
    (dashboard / "data").mkdir()
    (dashboard / "data" / "review_queue.json").write_text('["s1"]', encoding="utf-8")
    (dashboard / "data" / "reveal_payload.json").write_text(
        json.dumps({"s1": {"scene_id": "s1", "model_evaluations": []}}),
        encoding="utf-8",
    )
    (dashboard / "data" / "reviewer_payload.json").write_text(
        json.dumps({"title": "review", "scenes": [{"scene_id": "s1", "images": []}]}),
        encoding="utf-8",
    )
    reviews = tmp_path / "reviews.jsonl"
    server = create_server(dashboard, reviews, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        valid = {
            "review_id": "r1",
            "scene_id": "s1",
            "reviewer_id": "human",
            "blind_review": True,
            "decision": "KEEP",
            "scores": {"overall": 0.8},
            "confidence": 0.9,
            "created_at": "2099-01-01T00:00:00+00:00",
        }
        request = Request(
            f"http://127.0.0.1:{port}/api/reviews",
            data=json.dumps(valid).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert response.status == 201
        stored = json.loads(reviews.read_text().splitlines()[0])
        assert stored["blind_review"] is True
        assert stored["received_at"] != valid["created_at"]

        invalid = dict(valid, review_id="r2", blind_review=False)
        bad_request = Request(
            f"http://127.0.0.1:{port}/api/reviews",
            data=json.dumps(invalid).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(bad_request)
            raise AssertionError("invalid review was accepted")
        except HTTPError as error:
            assert error.code == 422

        outside_queue = dict(valid, review_id="r3", scene_id="s2")
        outside_request = Request(
            f"http://127.0.0.1:{port}/api/reviews",
            data=json.dumps(outside_queue).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(outside_request)
            raise AssertionError("review outside the focused queue was accepted")
        except HTTPError as error:
            assert error.code == 403
    finally:
        server.shutdown()
        thread.join(timeout=3)


def test_full_cli_experiment_generates_dashboard_reviews_and_calibration(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    dashboard = tmp_path / "dashboard"
    reviews = tmp_path / "reviews.jsonl"
    calibration = tmp_path / "calibration.json"
    commands = [
        [sys.executable, "-m", "qwen_tmqa.cli", "validate-config", "--config", "configs/default.yaml"],
        [sys.executable, "-m", "qwen_tmqa.cli", "make-example", "--output", str(dataset), "--scenes", "6"],
        [sys.executable, "-m", "qwen_tmqa.cli", "evaluate", "--root", str(dataset), "--output", str(results), "--config", "configs/default.yaml"],
        [sys.executable, "-m", "qwen_tmqa.cli", "simulate-human", "--results", str(results / "evaluations.json"), "--output", str(reviews), "--config", "configs/default.yaml"],
        [
            sys.executable,
            "-m",
            "qwen_tmqa.cli",
            "calibrate",
            "--results",
            str(results / "evaluations.json"),
            "--reviews",
            str(reviews),
            "--output",
            str(calibration),
            "--allow-synthetic",
            "--review-type",
            "synthetic",
        ],
        [sys.executable, "-m", "qwen_tmqa.cli", "visualize", "--results", str(results / "evaluations.json"), "--reviews", str(reviews), "--calibration", str(calibration), "--output", str(dashboard), "--config", "configs/default.yaml"],
    ]
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, env=_env(), check=False)
        assert completed.returncode == 0, f"{command}\nSTDOUT:{completed.stdout}\nSTDERR:{completed.stderr}"
    assert (dashboard / "index.html").exists()
    assert (dashboard / "review.html").exists()
    assert (dashboard / "data" / "engineering_evaluations.json").exists()
    assert (dashboard / "data" / "reviewer_payload.json").exists()
    assert (dashboard / "data" / "reveal_payload.json").exists()
    assert (results / "summary.json").exists()
    scene_data = json.loads((results / "evaluations.json").read_text())
    assert len(scene_data) == 6
    assert all(len(scene["model_evaluations"]) == 4 for scene in scene_data)
    assert reviews.read_text().strip()
    calibration_data = json.loads(calibration.read_text())
    assert calibration_data["models"]
    assert calibration_data["pairwise_mean_gap"]
    assert calibration_data["selected_review_type"] == "synthetic"
    assert calibration_data["experimental"] is True
    assert calibration_data["selected_review_count"] > 0
    assert calibration_data["production_eligible"] is False
    engineering_data = json.loads(
        (dashboard / "data" / "engineering_evaluations.json").read_text()
    )
    assert engineering_data["calibration"]["production_eligible"] is False
    assert engineering_data["calibration"]["warning"] == calibration_data["warning"]
    evidence = tmp_path / "verification_evidence.json"
    head_sha = "a" * 40
    evidence_command = [
        sys.executable,
        "scripts/generate_verification_evidence.py",
        "--results",
        str(results / "evaluations.json"),
        "--output",
        str(evidence),
        "--head-sha",
        head_sha,
        "--ci-run-id",
        "123",
        "--ci-tested-head-sha",
        head_sha,
        "--python-3-10-status",
        "success",
        "--python-3-12-status",
        "success",
    ]
    completed = subprocess.run(
        evidence_command,
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    audit = audit_verified_experiment(
        results_path=results / "evaluations.json",
        dashboard_dir=dashboard,
        reviews_path=reviews,
        calibration_path=calibration,
        evidence_path=evidence,
        expected_head_sha=head_sha,
    )
    assert set(audit) == {f"R-{index:03d}" for index in range(1, 20)}
    assert all(audit.values()), audit


def test_plain_verified_experiment_runs_v007_and_emits_deterministic_comparison(
    tmp_path: Path,
) -> None:
    output = tmp_path / "verified_experiment"
    env = _env()
    env.pop("TMQA_HEAD_SHA", None)
    env.pop("TMQA_CI_RUN_ID", None)

    completed = subprocess.run(
        ["bash", "scripts/run_verified_experiment.sh", str(output)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    comparison = json.loads(
        (output / "deterministic_rerun_comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["schema_version"] == "tmqa.deterministic-rerun-comparison.v1"
    assert comparison["matches"] is True
    assert comparison["comparisons"]["decisions"]["match"] is True
    assert comparison["comparisons"]["review_queue"]["match"] is True
    assert comparison["comparisons"]["stable_summary"]["match"] is True
    assert comparison["comparisons"]["stable_summary"]["fields"] == [
        "scene_count",
        "decisions",
        "mean_score",
        "synthetic_models",
    ]
    assert not (output / "requirements_audit.json").exists()
    assert not (output / "verification_evidence.json").exists()


def test_review_server_requires_a_focused_review_queue(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    (dashboard / "index.html").write_text("ok", encoding="utf-8")

    try:
        create_server(dashboard, tmp_path / "reviews.jsonl", port=0)
        raise AssertionError("server accepted a dashboard without a focused review queue")
    except FileNotFoundError as error:
        assert "review_queue.json" in str(error)
