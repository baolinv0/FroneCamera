#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STABLE_SUMMARY_FIELDS = [
    "scene_count",
    "decisions",
    "mean_score",
    "synthetic_models",
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _decisions(root: Path) -> dict[str, str]:
    evaluations = _load(root / "results" / "evaluations.json")
    return {
        item["scene_id"]: item["decision"]
        for item in sorted(evaluations, key=lambda item: item["scene_id"])
    }


def _stable_summary(root: Path) -> dict[str, Any]:
    summary = _load(root / "results" / "summary.json")
    return {field: summary[field] for field in STABLE_SUMMARY_FIELDS}


def compare(first: Path, second: Path) -> dict[str, Any]:
    values = {
        "decisions": (
            _decisions(first),
            _decisions(second),
        ),
        "review_queue": (
            _load(first / "dashboard" / "data" / "review_queue.json"),
            _load(second / "dashboard" / "data" / "review_queue.json"),
        ),
        "stable_summary": (
            _stable_summary(first),
            _stable_summary(second),
        ),
    }
    comparisons = {
        name: {"match": left == right, "first": left, "second": right}
        for name, (left, right) in values.items()
    }
    comparisons["stable_summary"]["fields"] = STABLE_SUMMARY_FIELDS
    return {
        "schema_version": "tmqa.deterministic-rerun-comparison.v1",
        "matches": all(item["match"] for item in comparisons.values()),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
