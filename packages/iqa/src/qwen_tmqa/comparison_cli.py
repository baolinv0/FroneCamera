"""Independent manifest evaluation, folder regressions, and representative audit CLI."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from .audit_sampling import representative_audit_sample
from .comparison import evaluate_comparison
from .comparison_models import ComparisonAsset, ComparisonRequest, ComparisonResult

_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".hdr",
    ".exr",
    ".npy",
    ".heic",
    ".heif",
    ".hif",
    ".avif",
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _csv_path(path: Path) -> Path:
    return path.with_suffix(".regression.csv" if path.suffix.lower() == ".csv" else ".csv")


def _save_csv(path: Path, results: list[ComparisonResult]):
    rows = []
    for result in results:
        for c in result.comparisons:
            rows.append(
                {
                    "scene_id": result.scene_id,
                    "mode": result.mode,
                    "baseline_id": c.baseline_id,
                    "candidate_id": c.candidate_id,
                    "status": c.status,
                    "same_source_status": c.same_source_status,
                    "fatal_reasons": "|".join(c.fatal_reasons),
                    "review_reasons": "|".join(c.review_reasons),
                    **c.relative_metrics,
                }
            )
    base_fields = [
        "scene_id",
        "mode",
        "baseline_id",
        "candidate_id",
        "status",
        "same_source_status",
        "fatal_reasons",
        "review_reasons",
    ]
    metric_fields = sorted({key for row in rows for key in row if key not in base_fields})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + metric_fields)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_request(path: Path) -> ComparisonRequest:
    data = _read(path)
    request = ComparisonRequest.model_validate(data)
    # JSON paths are relative to the manifest, not the caller's current directory.
    for asset in [
        request.baseline,
        *request.candidates,
        *([request.source] if request.source else []),
    ]:
        if not asset.path.is_absolute():
            asset.path = (path.resolve().parent / asset.path).resolve()
    return request


def _files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"image root is not a directory: {root}")
    return {
        p.relative_to(root).as_posix(): p.resolve()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    }


def compare_version_folders(
    baseline_root: Path,
    candidate_roots: list[Path],
    source_root: Path,
    *,
    encoding: Literal["srgb", "linear"] = "srgb",
    source_encoding: Literal["srgb", "linear"] = "srgb",
    split: Literal["train", "audit", "benchmark", "holdout"] = "audit",
) -> dict:
    """Exact relative-path correspondence is explicit; it establishes no input lineage.

    Repeated candidate roots allow arbitrary versions; missing counterparts are recorded
    separately, never dropped as if compared. No resizing or filename heuristics apply.
    """
    baseline_files = _files(baseline_root)
    source_files = _files(source_root)
    candidate_files = [_files(root) for root in candidate_roots]
    if not baseline_files:
        raise ValueError("baseline root has no supported images")
    results, missing_candidates, missing_sources = [], [], []
    for relative, path in baseline_files.items():
        candidates = []
        for index, (root, files) in enumerate(zip(candidate_roots, candidate_files)):
            if relative in files:
                candidates.append(
                    ComparisonAsset(
                        id=f"version-{index + 1}:{root.name}",
                        path=files[relative],
                        encoding=encoding,
                    )
                )
            else:
                missing_candidates.append(
                    {"relative_path": relative, "candidate_root": str(root.resolve())}
                )
        if not candidates:
            continue
        source_path = source_files.get(relative)
        if source_path is None:
            missing_sources.append(relative)
        request = ComparisonRequest(
            scene_id=relative,
            group_id=f"declared-folder-correspondence:{relative}",
            mode="algorithm",
            baseline=ComparisonAsset(id="baseline", path=path, encoding=encoding),
            candidates=candidates,
            source=ComparisonAsset(id="source", path=source_path, encoding=source_encoding)
            if source_path
            else None,
            split=split,
        )
        results.append(evaluate_comparison(request))
    unmatched = [
        {"candidate_root": str(root.resolve()), "relative_path": rel}
        for root, files in zip(candidate_roots, candidate_files)
        for rel in files
        if rel not in baseline_files
    ]
    return {
        "schema_version": "iqa-version-comparison/1",
        "baseline_root": str(baseline_root.resolve()),
        "candidate_roots": [str(p.resolve()) for p in candidate_roots],
        "source_root": str(source_root.resolve()),
        "scenes": [r.model_dump(mode="json") for r in results],
        "missing_candidates": missing_candidates,
        "missing_sources": missing_sources,
        "unmatched_candidates": unmatched,
        "warnings": [
            "relative_paths_are_correspondence_not_source_identity",
            "folder_group_ids_are_declared_correspondence_not_closed_canonical_metadata",
            "uncalibrated_proxies_require_review",
        ],
    }


def _parser():
    parser = argparse.ArgumentParser(prog="iqa-compare", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="compare declared B/C assets and per-person ROIs")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    versions = sub.add_parser(
        "compare-versions", help="compare corresponding relative paths across versions"
    )
    versions.add_argument("--baseline-root", type=Path, required=True)
    versions.add_argument(
        "--candidate-root",
        type=Path,
        action="append",
        required=True,
        help="repeat this option for multiple candidate versions",
    )
    versions.add_argument("--source-root", type=Path, required=True)
    versions.add_argument("--encoding", choices=["srgb", "linear"], default="srgb")
    versions.add_argument("--source-encoding", choices=["srgb", "linear"], default="srgb")
    versions.add_argument(
        "--split", choices=["train", "audit", "benchmark", "holdout"], default="audit"
    )
    versions.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser(
        "audit-sample", help="sample representative strata and retain the focused risk queue"
    )
    audit.add_argument(
        "--manifest", type=Path, required=True, help="JSON array or {records: [...]}"
    )
    audit.add_argument("--sample-size", type=int, default=20)
    audit.add_argument("--seed", default="iqa-audit-v1")
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate_comparison(_manifest_request(args.manifest))
            _save_json(args.output, result.model_dump(mode="json"))
            _save_csv(_csv_path(args.output), [result])
        elif args.command == "compare-versions":
            batch = compare_version_folders(
                args.baseline_root,
                args.candidate_root,
                args.source_root,
                encoding=args.encoding,
                source_encoding=args.source_encoding,
                split=args.split,
            )
            _save_json(args.output, batch)
            _save_csv(
                _csv_path(args.output),
                [ComparisonResult.model_validate(s) for s in batch["scenes"]],
            )
        else:
            data = _read(args.manifest)
            records = data.get("records") if isinstance(data, dict) else data
            if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
                raise ValueError("audit manifest requires a list of record objects")
            _save_json(
                args.output, representative_audit_sample(records, args.sample_size, args.seed)
            )
    except (OSError, ValueError, ValidationError, TypeError) as exc:
        print(f"iqa-compare: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
