#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from qwen_tmqa.cli import audit_command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    audit_command(
        root / "results" / "evaluations.json",
        root / "dashboard",
        root / "reviews.jsonl",
        root / "calibration.json",
        args.output,
        args.evidence,
        args.expected_head_sha,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
