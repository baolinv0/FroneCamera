#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROBES = {
    "fatal_non_compensation": (
        "tests/test_judges_evaluation.py::"
        "test_model_reported_fatal_issue_is_a_non_compensable_hard_failure"
    ),
    "decision_disagreement": (
        "tests/test_judges_evaluation.py::"
        "test_conflicting_model_decisions_force_human_review_even_when_scores_agree"
    ),
    "zero_judges": (
        "tests/test_zero_judge_visualization.py::"
        "test_zero_judge_pipeline_still_generates_engineering_and_reviewer_images"
    ),
    "all_unavailable": (
        "tests/test_judges_evaluation.py::"
        "test_all_unavailable_judges_cannot_silently_produce_keep"
    ),
    "dashboard_behavior": (
        "tests/test_visualization.py::"
        "test_engineering_dashboard_contains_required_debug_views_and_prompt_data"
    ),
    "blind_review_network": (
        "tests/test_p1_remediation.py::"
        "test_reviewer_client_is_data_blind_until_persisted_reveal"
    ),
    "route_isolation": (
        "tests/test_review_server_isolation.py::"
        "test_default_review_server_exposes_only_reviewer_surface"
    ),
}


def generate(
    *,
    results: Path,
    output: Path,
    head_sha: str,
    ci_run_id: int | None,
    ci_tested_head_sha: str | None,
    python_310_status: str,
    python_312_status: str,
) -> bool:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    checks: dict[str, dict[str, object]] = {}
    all_passed = True
    for name, node_id in PROBES.items():
        command = [sys.executable, "-m", "pytest", "-q", node_id]
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        probe_output = completed.stdout + completed.stderr
        checks[name] = {
            "command": " ".join(command),
            "exit_code": completed.returncode,
            "output_sha256": hashlib.sha256(probe_output.encode("utf-8")).hexdigest(),
            "output": probe_output,
        }
        all_passed = all_passed and completed.returncode == 0
    payload = {
        "schema_version": "tmqa.verification-evidence.v1",
        "artifact_head_sha": head_sha,
        "artifacts": {
            "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
        },
        "checks": checks,
        "ci": {
            "provider": "github_actions" if ci_run_id is not None else "not_run",
            "tested_head_sha": ci_tested_head_sha,
            "jobs": {
                "python-3.10": python_310_status,
                "python-3.12": python_312_status,
            },
            "run_id": ci_run_id,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--ci-run-id", type=int)
    parser.add_argument("--ci-tested-head-sha")
    parser.add_argument("--python-3-10-status", default="not_run")
    parser.add_argument("--python-3-12-status", default="not_run")
    args = parser.parse_args()
    passed = generate(
        results=args.results,
        output=args.output,
        head_sha=args.head_sha,
        ci_run_id=args.ci_run_id,
        ci_tested_head_sha=args.ci_tested_head_sha,
        python_310_status=args.python_3_10_status,
        python_312_status=args.python_3_12_status,
    )
    print(args.output)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
