from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portrait_eval.dataset import propose_pairing, scan_folder  # noqa: E402
from portrait_eval.vlm import OpenAIResponsesVisionAdapter, resolve_openai_api_key  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a one-scene OpenAI vision smoke test on device folders."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--scene-index", type=int, default=1, help="One-based natural-sort index")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--max-image-edge", type=int, default=1536)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    device_folders = sorted(
        (path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name.casefold()
    )
    if len(device_folders) < 2:
        raise ValueError("Dataset must contain at least two device folders")
    if args.scene_index < 1:
        raise ValueError("--scene-index must be at least 1")

    selected: dict[str, Path] = {}
    devices: dict[str, object] = {}
    warnings: list[str] = []
    scans = [scan_folder(folder.name, folder) for folder in device_folders]
    pairing = propose_pairing(scans)
    if args.scene_index > len(pairing.groups):
        raise ValueError(f"--scene-index exceeds the {len(pairing.groups)} matched groups")
    matched_group = pairing.groups[args.scene_index - 1]
    summary: dict[str, object] = {
        "dataset_root": str(root),
        "scene_index": args.scene_index,
        "matching_confidence": matched_group.matching_confidence,
        "review_required": matched_group.review_required,
        "match_notes": matched_group.match_notes,
        "devices": devices,
        "warnings": warnings,
    }
    for index, (folder, scan) in enumerate(zip(device_folders, scans, strict=True)):
        code = chr(65 + index)
        chosen = matched_group.cells[folder.name]
        devices[code] = {
            "folder": folder.name,
            "supported_image_count": len(scan.files),
            "selected": chosen.name if chosen else None,
        }
        if chosen is not None:
            selected[code] = chosen

    if matched_group.review_required:
        warnings.append("Automatic matching marked this group for human review.")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if len(selected) < 2:
        raise ValueError("Selected position has fewer than two images")

    api_key = resolve_openai_api_key(os.getenv("PORTRAIT_EVAL_OPENAI_API_KEY"))
    if not api_key:
        raise ValueError(
            "Set PORTRAIT_EVAL_OPENAI_API_KEY or OPENAI_API_KEY, or configure "
            "OPENAI_API_KEY in ~/.codex/auth.json before a live smoke test"
        )
    adapter = OpenAIResponsesVisionAdapter(
        api_key,
        args.model,
        "smoke_test",
        max_image_edge=args.max_image_edge,
    )
    result = adapter.analyze(
        f"smoke-{args.scene_index:03d}",
        selected,
        {"pass": "visual", "dataset_root": root.name},
    )
    print(result.model_dump_json(indent=2, exclude={"raw"}))


if __name__ == "__main__":
    main()
