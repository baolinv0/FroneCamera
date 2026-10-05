#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

_PATTERNS = {
    "authorization": re.compile(r"Authorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    "api_key": re.compile(r'"api_key"\s*:\s*"(?!EMPTY|REDACTED)[^"]+"', re.IGNORECASE),
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    findings: list[str] = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".html", ".json", ".jsonl", ".txt", ".log", ".csv"
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in _PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path}: {name}")
    if findings:
        raise SystemExit("secret leakage detected:\n" + "\n".join(findings))
    print(f"secret scan PASS: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
