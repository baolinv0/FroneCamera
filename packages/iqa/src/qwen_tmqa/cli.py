from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .audit import audit_verified_experiment, build_requirement_records
from .config import CalibrationPolicyConfig, load_config
from .dataset import discover_scenes
from .domain import HumanReview, ReliabilityResult, SceneEvaluation
from .evaluation import EvaluationPipeline
from .image_io import file_sha256
from .lineage import (
    SelectionLineage,
    build_evaluation_run_manifest,
    dataset_manifest_sha256,
    load_evaluation_run_manifest,
    load_split_assignments,
    require_matching_evaluation_config,
    write_evaluation_run_manifest,
)
from .pseudo_gt import select_scene_pseudo_gt
from .review import (
    append_review,
    calibrate_models,
    load_reviews,
    pairwise_mean_gaps,
    select_calibration_reviews,
    select_review_queue,
)
from .server import create_server
from .training_data import HumanConfirmation, TrainingCandidate, export_training_data
from .visualization import generate_dashboard

LEVELS = [
    ("a_m100", -1.0),
    ("a_m075", -0.75),
    ("a_m050", -0.5),
    ("a_m025", -0.25),
    ("a_000", 0.0),
    ("a_p025", 0.25),
    ("a_p050", 0.5),
    ("a_p075", 0.75),
    ("a_p100", 1.0),
]


def _base_scene(index: int, height: int = 160, width: int = 224) -> np.ndarray:
    x = np.linspace(0.04, 0.76, width, dtype=np.float32)
    y = np.linspace(0.85, 1.10, height, dtype=np.float32)[:, None]
    luminance = np.clip(y * x[None, :], 0, 1)
    image = np.stack([luminance, luminance * 0.96, luminance * 0.90], axis=2)
    image_u8 = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    cv2.rectangle(image_u8, (20, 45), (80, 130), (82, 140, 56), -1)
    cv2.circle(image_u8, (150, 70), 25, (199, 102, 56), -1)
    cv2.putText(
        image_u8,
        f"S{index}",
        (95, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (204, 204, 204),
        2,
    )
    return image_u8.astype(np.float32) / 255.0


def _render_example(base: np.ndarray, alpha: float, pattern: int) -> np.ndarray:
    effective_alpha = alpha
    if pattern == 4 and alpha >= 0.75:
        effective_alpha = 0.25 - (alpha - 0.75) * 0.2
    if pattern == 5 and alpha > 0:
        effective_alpha = 0.35
    gain = 2 ** (0.72 * effective_alpha)
    image = np.clip(base * gain, 0, 1)
    if pattern == 1 and alpha > 0.5:
        image = np.clip(image * 1.35, 0, 1)
    if pattern == 2 and alpha > 0.25:
        image = image.copy()
        image[..., 0] = np.clip(image[..., 0] * 1.20, 0, 1)
        image[..., 2] = np.clip(image[..., 2] * 0.80, 0, 1)
    if pattern == 3 and alpha > 0.5:
        image = image.copy()
        image[45:130, 20:80] = np.roll(image[45:130, 20:80], shift=10, axis=1)
    return image


def make_example(output: Path, scenes: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for scene_index in range(scenes):
        base = _base_scene(scene_index)
        pattern = scene_index % 6
        for level, alpha in LEVELS:
            image = _render_example(base, alpha, pattern)
            destination = output / level / f"scene_{scene_index:03d}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(image * 255).astype(np.uint8)).save(destination)


def _write_evaluations(path: Path, scenes: list[SceneEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [scene.model_dump(mode="json") for scene in scenes]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_evaluations(path: Path) -> list[SceneEvaluation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [SceneEvaluation.model_validate(item) for item in payload]


def _attach_sources(specs, source_root: Path | None):
    if source_root is None:
        return specs
    bound = []
    for spec in specs:
        source = source_root / spec.scene_id
        if not source.is_file():
            raise FileNotFoundError(f"source image is missing: {source}")
        bound.append(spec.model_copy(update={"source_path": source}))
    return bound


def evaluate_command(
    root: Path, output: Path, config_path: Path, source_root: Path | None = None
) -> None:
    config = load_config(config_path)
    specs = _attach_sources(discover_scenes(root, config.dataset), source_root)
    pipeline = EvaluationPipeline(config)
    scenes = [pipeline.evaluate_scene(spec) for spec in specs]
    output.mkdir(parents=True, exist_ok=True)
    _write_evaluations(output / "evaluations.json", scenes)
    write_evaluation_run_manifest(
        output / "evaluation_run_manifest.json",
        build_evaluation_run_manifest(specs, config_path, config),
    )
    decisions = {key: 0 for key in ["KEEP", "REGENERATE", "REVIEW", "REJECT"]}
    for scene in scenes:
        decisions[scene.decision] += 1
    summary = {
        "scene_count": len(scenes),
        "decisions": decisions,
        "mean_score": (float(np.mean([scene.overall_score for scene in scenes])) if scenes else 0),
        "synthetic_models": all(scene.synthetic_experiment for scene in scenes),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "errors.json").write_text("[]\n", encoding="utf-8")
    scene_rows = ["scene_id,decision,overall_score,uncertainty"]
    scene_rows.extend(
        f"{scene.scene_id},{scene.decision},{scene.overall_score:.8f},{scene.uncertainty:.8f}"
        for scene in scenes
    )
    (output / "scenes.csv").write_text("\n".join(scene_rows) + "\n", encoding="utf-8")


def simulate_human(results: Path, output: Path, config_path: Path) -> None:
    config = load_config(config_path)
    scenes = _load_evaluations(results)
    queue = select_review_queue(scenes, config.review)
    if output.exists():
        output.unlink()
    for index, scene in enumerate(queue):
        stage_scores = {stage.stage_id: stage.score for stage in scene.stages}
        tone = float(stage_scores.get("tone_color") or scene.overall_score)
        fidelity = float(stage_scores.get("fidelity") or scene.overall_score)
        control = scene.sequence.control_score
        overall = float(np.clip(0.35 * tone + 0.35 * fidelity + 0.30 * control, 0, 1))
        if "fatal_risk" in scene.review_reasons:
            decision = "REJECT"
        elif overall < 0.60:
            decision = "REGENERATE"
        elif overall < 0.78 or "model_disagreement" in scene.review_reasons:
            decision = "REVIEW"
        else:
            decision = "KEEP"
        append_review(
            output,
            HumanReview(
                review_id=f"synthetic-{index:04d}",
                scene_id=scene.scene_id,
                reviewer_id="synthetic-oracle",
                blind_review=True,
                decision=decision,
                scores={
                    "overall": overall,
                    "tone": tone,
                    "fidelity": fidelity,
                    "control": control,
                },
                issues=scene.review_reasons,
                rationale=(
                    "Synthetic experiment label derived from deterministic stages; "
                    "not human evidence."
                ),
                confidence=0.95,
                synthetic=True,
            ),
        )


def calibrate_command(
    results: Path,
    reviews_path: Path,
    output: Path,
    *,
    allow_synthetic: bool = False,
    review_type: str | None = None,
    config_path: Path | None = None,
) -> None:
    scenes = _load_evaluations(results)
    reviews = load_reviews(reviews_path)
    selected, ignored_count, selected_type = select_calibration_reviews(
        reviews,
        review_type=review_type,
        allow_synthetic=allow_synthetic,
    )
    reliability = calibrate_models(
        scenes,
        selected,
        review_type=(selected_type if selected_type in {"real", "synthetic"} else None),
        allow_synthetic=allow_synthetic,
    )
    calibration_policy = (
        load_config(config_path).calibration
        if config_path is not None
        else CalibrationPolicyConfig()
    )
    experimental = selected_type == "synthetic"
    actual_sample_count = max(
        (item.sample_count for item in reliability),
        default=0,
    )
    sufficient = (
        selected_type == "real"
        and actual_sample_count >= calibration_policy.minimum_real_sample_count
    )
    production_eligible = selected_type == "real" and sufficient
    if experimental:
        warning = "Synthetic calibration is experimental and not a production weight."
    elif not sufficient:
        warning = (
            "Insufficient real blind-review samples "
            f"({actual_sample_count}/{calibration_policy.minimum_real_sample_count}); "
            "calibration is not production-ready."
        )
    else:
        warning = "Human calibration applies only to the selected blind-review set."
    payload = {
        "models": [item.model_dump(mode="json") for item in reliability],
        "synthetic_reviews": experimental,
        "selected_review_type": selected_type,
        "selected_review_count": len(selected),
        "ignored_review_count": ignored_count,
        "review_count": len(reviews),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experimental": experimental,
        "production_eligible": production_eligible,
        "sample_sufficiency": {
            "policy_version": calibration_policy.version,
            "actual_sample_count": actual_sample_count,
            "minimum_real_sample_count": calibration_policy.minimum_real_sample_count,
            "sufficient": sufficient,
        },
        "warning": warning,
        "pairwise_mean_gap": pairwise_mean_gaps(scenes),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def visualize_command(
    results: Path,
    output: Path,
    config_path: Path,
    reviews_path: Path | None,
    calibration_path: Path | None,
) -> None:
    config = load_config(config_path)
    scenes = _load_evaluations(results)
    queue = select_review_queue(scenes, config.review)
    reliability: list[ReliabilityResult] = []
    pairwise_gaps: dict[str, float] = {}
    if calibration_path and calibration_path.exists():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        reliability = [ReliabilityResult.model_validate(item) for item in payload.get("models", [])]
        pairwise_gaps = {
            str(key): float(value) for key, value in payload.get("pairwise_mean_gap", {}).items()
        }
        calibration_disclosure = {
            key: payload[key]
            for key in [
                "selected_review_type",
                "selected_review_count",
                "experimental",
                "production_eligible",
                "sample_sufficiency",
                "warning",
            ]
            if key in payload
        }
    else:
        calibration_disclosure = {}
    generate_dashboard(
        scenes,
        output,
        title=config.visualization.title,
        review_queue=queue,
        reliability=reliability,
        pairwise_mean_gap=pairwise_gaps,
        calibration_disclosure=calibration_disclosure,
    )
    if reviews_path and reviews_path.exists():
        (output / "data" / "reviews.jsonl").write_text(
            reviews_path.read_text(encoding="utf-8"), encoding="utf-8"
        )


def audit_command(
    results: Path,
    dashboard: Path,
    reviews: Path,
    calibration: Path,
    output: Path,
    evidence: Path | None = None,
    expected_head_sha: str | None = None,
) -> None:
    checks = audit_verified_experiment(
        results_path=results,
        dashboard_dir=dashboard,
        reviews_path=reviews,
        calibration_path=calibration,
        evidence_path=evidence,
        expected_head_sha=expected_head_sha,
    )
    evidence_payload = (
        json.loads(evidence.read_text(encoding="utf-8"))
        if evidence is not None and evidence.exists()
        else {}
    )
    common_evidence = [
        f"artifact_head_sha={evidence_payload.get('artifact_head_sha', 'missing')}",
        (
            "results_sha256="
            f"{evidence_payload.get('artifacts', {}).get('results_sha256', 'missing')}"
        ),
    ]
    actual_evidence = {
        requirement_id: common_evidence
        + [
            f"{name}:exit={details.get('exit_code')}:sha256="
            f"{details.get('output_sha256', 'missing')}"
            for name, details in evidence_payload.get("checks", {}).items()
        ]
        for requirement_id in checks
    }
    records = build_requirement_records(checks, actual_evidence=actual_evidence)
    payload = {
        "schema_version": "tmqa.requirements-audit.v2",
        "expected_head_sha": expected_head_sha,
        "requirements": records,
        "passed": sum(checks.values()),
        "required": 19,
        "overall": ("PASS" if len(checks) == 19 and all(checks.values()) else "FAIL"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if payload["overall"] != "PASS":
        raise ValueError(f"requirements audit failed: {checks}")


def select_candidates_command(
    results: Path,
    root: Path,
    source_root: Path,
    config_path: Path,
    run_manifest_path: Path,
    splits_path: Path,
    output: Path,
) -> None:
    config = load_config(config_path)
    specs = _attach_sources(discover_scenes(root, config.dataset), source_root)
    scenes = _load_evaluations(results)
    spec_by_id = {spec.scene_id: spec for spec in specs}
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("duplicate evaluated scene identities")
    if {scene.scene_id for scene in scenes} != set(spec_by_id):
        raise ValueError("evaluations must cover the complete declared dataset")
    manifest = load_evaluation_run_manifest(run_manifest_path)
    require_matching_evaluation_config(manifest, config_path)
    if dataset_manifest_sha256(specs) != manifest.dataset_manifest_sha256:
        raise ValueError("dataset bytes do not match evaluation manifest")
    assignments = load_split_assignments(splits_path, expected_scene_ids=set(spec_by_id))
    if set(assignments) != set(spec_by_id):
        raise ValueError("split metadata must be closed over the complete dataset")
    candidates = []
    audit = []
    for scene in sorted(scenes, key=lambda item: item.scene_id):
        assignment = assignments[scene.scene_id]
        record = select_scene_pseudo_gt(
            scene,
            spec_by_id[scene.scene_id],
            config.pseudo_gt,
            lineage=SelectionLineage(
                evaluation=manifest,
                split=assignment.split,
                split_file_sha256=file_sha256(splits_path),
            ),
            baseline_level=config.dataset.baseline_level,
        )
        audit.append(record.model_dump(mode="json"))
        if not record.accepted:
            continue
        candidate = TrainingCandidate(
            scene_id=record.scene_id,
            candidate_id=record.selected_level,
            source_path=record.input_path,
            candidate_path=record.selected_gt_path,
            source_sha256=record.input_sha256,
            candidate_sha256=record.selected_gt_sha256,
            split=record.split,
            canonical_scene_id=assignment.canonical_scene_id,
            group_id=assignment.group_id,
            eligible=record.accepted,
            judge_ids=record.judge_ids,
            synthetic=record.synthetic,
            evidence_mode="algorithm",
            rejection_reasons=record.rejection_reasons,
            lineage={
                "dataset_scene_ids": sorted(spec_by_id),
                "evaluation": manifest.model_dump(mode="json"),
                "split_file_sha256": record.split_file_sha256,
                "selector_evidence": record.model_dump(mode="json"),
            },
        )
        candidates.append(candidate.model_dump(mode="json"))
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in [
        ("candidate_suggestions.jsonl", candidates),
        ("candidate_audit.jsonl", audit),
    ]:
        (output / filename).write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8"
        )
    (output / "candidate_summary.json").write_text(
        json.dumps(
            {"scene_count": len(scenes), "candidate_count": len(candidates), "training_count": 0}
        ),
        encoding="utf-8",
    )


def export_training_command(
    candidates_path: Path, confirmations_path: Path, splits_path: Path, output: Path
) -> None:
    records = [
        TrainingCandidate.model_validate_json(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    confirmations = [
        HumanConfirmation.model_validate_json(line)
        for line in confirmations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assignments = load_split_assignments(
        splits_path, expected_scene_ids={item.scene_id for item in records}
    )
    export_training_data(
        records, output, confirmations=confirmations, split_assignments=assignments
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-tmqa")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)

    example = sub.add_parser("make-example")
    example.add_argument("--output", type=Path, required=True)
    example.add_argument("--scenes", type=int, default=6)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--source-root", type=Path)

    simulate = sub.add_parser("simulate-human")
    simulate.add_argument("--results", type=Path, required=True)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--config", type=Path, required=True)

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--results", type=Path, required=True)
    calibrate.add_argument("--reviews", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--allow-synthetic", action="store_true")
    calibrate.add_argument("--review-type", choices=["real", "synthetic"])
    calibrate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )

    visualize = sub.add_parser("visualize")
    visualize.add_argument("--results", type=Path, required=True)
    visualize.add_argument("--output", type=Path, required=True)
    visualize.add_argument("--config", type=Path, required=True)
    visualize.add_argument("--reviews", type=Path)
    visualize.add_argument("--calibration", type=Path)

    serve = sub.add_parser("serve")
    serve.add_argument("--dashboard", type=Path, required=True)
    serve.add_argument("--reviews", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    audit = sub.add_parser("audit")
    audit.add_argument("--results", type=Path, required=True)
    audit.add_argument("--dashboard", type=Path, required=True)
    audit.add_argument("--reviews", type=Path, required=True)
    audit.add_argument("--calibration", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--evidence", type=Path)
    audit.add_argument("--expected-head-sha")
    select = sub.add_parser("select-candidates")
    for argument in [
        "results",
        "root",
        "source-root",
        "config",
        "run-manifest",
        "splits",
        "output",
    ]:
        select.add_argument("--" + argument, type=Path, required=True)
    training = sub.add_parser("export-training")
    for argument in ["candidates", "confirmations", "splits", "output"]:
        training.add_argument("--" + argument, type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        print(f"configuration valid: {len(config.judges)} judges")
    elif args.command == "make-example":
        make_example(args.output, args.scenes)
        print(f"created {args.scenes} scenes at {args.output}")
    elif args.command == "evaluate":
        evaluate_command(args.root, args.output, args.config, args.source_root)
        print(f"evaluation written to {args.output}")
    elif args.command == "simulate-human":
        simulate_human(args.results, args.output, args.config)
        print(f"synthetic reviews written to {args.output}")
    elif args.command == "calibrate":
        calibrate_command(
            args.results,
            args.reviews,
            args.output,
            allow_synthetic=args.allow_synthetic,
            review_type=args.review_type,
            config_path=args.config,
        )
        print(f"calibration written to {args.output}")
    elif args.command == "visualize":
        visualize_command(
            args.results,
            args.output,
            args.config,
            args.reviews,
            args.calibration,
        )
        print(f"dashboard written to {args.output / 'index.html'}")
    elif args.command == "serve":
        server = create_server(
            args.dashboard,
            args.reviews,
            host=args.host,
            port=args.port,
        )
        print(f"serving http://{args.host}:{server.server_address[1]}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "select-candidates":
        select_candidates_command(
            args.results,
            args.root,
            args.source_root,
            args.config,
            args.run_manifest,
            args.splits,
            args.output,
        )
        print(f"candidate suggestions written to {args.output}")
    elif args.command == "export-training":
        export_training_command(args.candidates, args.confirmations, args.splits, args.output)
        print(f"human-confirmed training labels written to {args.output}")
    elif args.command == "audit":
        audit_command(
            args.results,
            args.dashboard,
            args.reviews,
            args.calibration,
            args.output,
            args.evidence,
            args.expected_head_sha,
        )
        print(f"audit written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
