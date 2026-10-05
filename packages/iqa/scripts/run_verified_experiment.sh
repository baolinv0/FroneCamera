#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-"$ROOT/examples/verified_experiment"}
RERUN="${OUT}.deterministic-rerun"
rm -rf "$OUT"
rm -rf "$RERUN"
trap 'rm -rf "$RERUN"' EXIT

run_v007() {
  local target=$1
  mkdir -p "$target"
  python -m qwen_tmqa.cli make-example --output "$target/dataset" --scenes 6
  python -m qwen_tmqa.cli evaluate --root "$target/dataset" --output "$target/results" --config "$ROOT/configs/default.yaml"
  python -m qwen_tmqa.cli simulate-human --results "$target/results/evaluations.json" --output "$target/reviews.jsonl" --config "$ROOT/configs/default.yaml"
  python -m qwen_tmqa.cli calibrate --results "$target/results/evaluations.json" --reviews "$target/reviews.jsonl" --output "$target/calibration.json" --config "$ROOT/configs/default.yaml" --allow-synthetic --review-type synthetic
  python -m qwen_tmqa.cli visualize --results "$target/results/evaluations.json" --reviews "$target/reviews.jsonl" --calibration "$target/calibration.json" --output "$target/dashboard" --config "$ROOT/configs/default.yaml"
}

python -m qwen_tmqa.cli validate-config --config "$ROOT/configs/default.yaml"
run_v007 "$OUT"
run_v007 "$RERUN"
python "$ROOT/scripts/compare_deterministic_runs.py" \
  --first "$OUT" \
  --second "$RERUN" \
  --output "$OUT/deterministic_rerun_comparison.json"
python "$ROOT/scripts/check_dashboard_javascript.py" "$OUT/dashboard/index.html"
python "$ROOT/scripts/check_dashboard_javascript.py" "$OUT/dashboard/review.html"
python "$ROOT/scripts/verify_no_secret_leakage.py" "$OUT"
echo "Verified experiment: $OUT"
