#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    wheels = sorted(args.dist.glob("qwen_tmqa-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, found {wheels}")
    shutil.rmtree(args.target, ignore_errors=True)
    args.target.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-deps", "--target",
            str(args.target), str(wheels[0]),
        ],
        check=True,
    )
    target_literal = repr(str(args.target.resolve()))
    code = (
        "import importlib.metadata, sys; "
        f"sys.path.insert(0, {target_literal}); "
        "import qwen_tmqa; print(qwen_tmqa.__file__); "
        "print(importlib.metadata.version('qwen-tmqa'))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr)
    lines = completed.stdout.strip().splitlines()
    if not lines or lines[-1] != args.expected_version:
        raise SystemExit(f"unexpected installed version: {completed.stdout}")
    if str(args.target.resolve()) not in str(Path(lines[0]).resolve()):
        raise SystemExit(f"module imported outside isolated target: {lines[0]}")
    print(completed.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
