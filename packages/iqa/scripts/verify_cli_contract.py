#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], env: dict[str, str], *, expected: int = 0) -> None:
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != expected:
        raise SystemExit(
            f"command returned {completed.returncode}, expected {expected}: {command}"
            f"\nSTDOUT:{completed.stdout}\nSTDERR:{completed.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.workdir
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "dataset"
    results = root / "results"
    reviews = root / "reviews.jsonl"
    calibration = root / "calibration.json"
    dashboard = root / "dashboard"
    audit = root / "requirements_audit.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cli = [sys.executable, "-m", "qwen_tmqa.cli"]
    commands = [
        cli + ["validate-config", "--config", str(args.config)],
        cli + ["make-example", "--output", str(dataset), "--scenes", "2"],
        cli + ["evaluate", "--root", str(dataset), "--output", str(results), "--config", str(args.config)],
        cli + ["simulate-human", "--results", str(results / "evaluations.json"), "--output", str(reviews), "--config", str(args.config)],
        cli + ["calibrate", "--results", str(results / "evaluations.json"), "--reviews", str(reviews), "--output", str(calibration), "--config", str(args.config), "--allow-synthetic", "--review-type", "synthetic"],
        cli + ["visualize", "--results", str(results / "evaluations.json"), "--reviews", str(reviews), "--calibration", str(calibration), "--output", str(dashboard), "--config", str(args.config)],
        cli + ["serve", "--help"],
    ]
    for command in commands:
        run(command, env)
    run(
        cli + ["audit", "--results", str(results / "evaluations.json"), "--dashboard", str(dashboard), "--reviews", str(reviews), "--calibration", str(calibration), "--output", str(audit)],
        env,
        expected=1,
    )
    print("CLI contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
