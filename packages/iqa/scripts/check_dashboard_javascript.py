#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

_SCRIPT = re.compile(r"<script(?:\s[^>]*)?>(.*?)</script>", re.IGNORECASE | re.DOTALL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    scripts = _SCRIPT.findall(args.html.read_text(encoding="utf-8"))
    if not scripts:
        raise SystemExit("no inline JavaScript found")
    source = "\n".join(scripts)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", encoding="utf-8", delete=False
    ) as handle:
        handle.write(source)
        temporary = Path(handle.name)
    try:
        completed = subprocess.run(
            ["node", "--check", str(temporary)], text=True, capture_output=True, check=False
        )
    finally:
        temporary.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr)
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
