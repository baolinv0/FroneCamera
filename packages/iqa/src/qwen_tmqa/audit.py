from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_REQUIRED_LEVELS = {
    "a_m100",
    "a_m075",
    "a_m050",
    "a_m025",
    "a_000",
    "a_p025",
    "a_p050",
    "a_p075",
    "a_p100",
}
_REQUIRED_SCORES = {
    "tone",
    "color",
    "fidelity",
    "control",
    "preference",
    "overall",
}
_FORBIDDEN_REVIEW_TOKENS = {
    "model_evaluations",
    "model_id",
    "prompt_trace",
    "rendered_prompt",
    "raw_response",
    "parsed_response",
    "overall_score",
    "review_reasons",
    "stages",
    "system_decision",
}

_TRACEABILITY: dict[str, dict[str, object]] = {
    "R-001": {"implementation_locations": ["src/qwen_tmqa/dataset.py", "src/qwen_tmqa/config.py"], "test_locations": ["tests/test_dataset_metrics.py", "tests/test_p1_remediation.py"], "execution_command": "V-011", "measurable_threshold": "nine canonical levels and 100% strict completeness"},
    "R-002": {"implementation_locations": ["src/qwen_tmqa/metrics.py", "src/qwen_tmqa/evaluation.py"], "test_locations": ["tests/test_dataset_metrics.py"], "execution_command": "V-003;V-007", "measurable_threshold": "all objective, sequence, and six stage records present"},
    "R-003": {"implementation_locations": ["src/qwen_tmqa/evaluation.py", "src/qwen_tmqa/judges"], "test_locations": ["tests/test_judges_evaluation.py", "tests/test_zero_judge_visualization.py"], "execution_command": "V-012", "measurable_threshold": "0..N judges supported and failures isolated"},
    "R-004": {"implementation_locations": ["src/qwen_tmqa/domain.py", "src/qwen_tmqa/judges/openai_compatible.py"], "test_locations": ["tests/test_judges_evaluation.py"], "execution_command": "V-012", "measurable_threshold": "100% complete required model contract or unavailable"},
    "R-005": {"implementation_locations": ["src/qwen_tmqa/prompts.py", "src/qwen_tmqa/visualization.py"], "test_locations": ["tests/test_prompts.py", "tests/test_visualization.py"], "execution_command": "V-017", "measurable_threshold": "100% prompt trace fields present"},
    "R-006": {"implementation_locations": ["src/qwen_tmqa/image_io.py", "src/qwen_tmqa/prompts.py"], "test_locations": ["tests/test_p1_remediation.py"], "execution_command": "V-013", "measurable_threshold": "ordered manifest and exact sent-payload SHA-256"},
    "R-007": {"implementation_locations": ["src/qwen_tmqa/prompts.py"], "test_locations": ["tests/test_prompts.py", "tests/test_p1_remediation.py"], "execution_command": "V-014", "measurable_threshold": "immutable 3.2 and current 3.3/schema v4"},
    "R-008": {"implementation_locations": ["src/qwen_tmqa/evaluation.py", "src/qwen_tmqa/config.py"], "test_locations": ["tests/test_judges_evaluation.py", "tests/test_zero_judge_visualization.py"], "execution_command": "V-015", "measurable_threshold": "fatal/disagreement/zero/all-unavailable safety and configured provenance"},
    "R-009": {"implementation_locations": ["src/qwen_tmqa/visualization.py", "src/qwen_tmqa/assets/dashboard.html"], "test_locations": ["tests/test_visualization.py"], "execution_command": "V-017;V-008", "measurable_threshold": "dashboard behavioral probe and JavaScript syntax pass"},
    "R-010": {"implementation_locations": ["src/qwen_tmqa/server.py", "src/qwen_tmqa/assets/review.html"], "test_locations": ["tests/test_p1_remediation.py", "tests/test_review_server_isolation.py"], "execution_command": "V-016", "measurable_threshold": "network blind-review and route-isolation probes pass"},
    "R-011": {"implementation_locations": ["src/qwen_tmqa/review.py", "src/qwen_tmqa/server.py"], "test_locations": ["tests/test_review_calibration.py", "tests/test_server_cli.py"], "execution_command": "V-016", "measurable_threshold": "queue-only selection and server authorization"},
    "R-012": {"implementation_locations": ["src/qwen_tmqa/domain.py", "src/qwen_tmqa/review.py"], "test_locations": ["tests/test_human_review_contract.py", "tests/test_review_calibration.py"], "execution_command": "V-016", "measurable_threshold": "complete append-only reviews with authoritative received_at"},
    "R-013": {"implementation_locations": ["src/qwen_tmqa/review.py", "src/qwen_tmqa/judges/openai_compatible.py"], "test_locations": ["tests/test_p1_remediation.py", "tests/test_judges_evaluation.py"], "execution_command": "V-018", "measurable_threshold": "synthetic/real lineage and calibration isolation"},
    "R-014": {"implementation_locations": ["src/qwen_tmqa/review.py", "src/qwen_tmqa/cli.py"], "test_locations": ["tests/test_review_calibration.py"], "execution_command": "V-018", "measurable_threshold": "required reliability metrics, normalized weights, and sample sufficiency"},
    "R-015": {"implementation_locations": ["src/qwen_tmqa/server.py"], "test_locations": ["tests/test_review_server_isolation.py", "tests/test_server_cli.py"], "execution_command": "V-016;V-019", "measurable_threshold": "route isolation, validation, no-reveal failure, and no secrets"},
    "R-016": {"implementation_locations": ["src/qwen_tmqa/cli.py", "scripts/run_verified_experiment.sh"], "test_locations": ["tests/test_server_cli.py"], "execution_command": "V-006;V-007", "measurable_threshold": "all eight CLI surfaces and deterministic experiment pass"},
    "R-017": {"implementation_locations": ["src/qwen_tmqa/audit.py"], "test_locations": ["tests/test_audit_negative.py"], "execution_command": "V-009", "measurable_threshold": "19/19 complete records; any MUST failure makes overall FAIL"},
    "R-018": {"implementation_locations": [".github/workflows/ci.yml"], "test_locations": ["tests/test_audit_negative.py"], "execution_command": "V-001..V-009;N-003", "measurable_threshold": "Python 3.10/3.12 success on exact artifact-bound head"},
    "R-019": {"implementation_locations": ["README.md", "src/qwen_tmqa/cli.py", "src/qwen_tmqa/visualization.py"], "test_locations": ["tests/test_review_calibration.py", "tests/test_visualization.py"], "execution_command": "N-004;N-005", "measurable_threshold": "explicit synthetic/real and non-production boundaries"},
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _successful_behavior(evidence: dict[str, Any], name: str) -> bool:
    check = evidence.get("checks", {}).get(name, {})
    digest = check.get("output_sha256")
    return (
        isinstance(check.get("command"), str)
        and bool(check["command"])
        and check.get("exit_code") == 0
        and isinstance(digest, str)
        and len(digest) == 64
    )


def build_requirement_records(
    checks: dict[str, bool],
    *,
    actual_evidence: dict[str, list[str]],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for requirement_id in [f"R-{index:03d}" for index in range(1, 20)]:
        trace = _TRACEABILITY[requirement_id]
        passed = checks.get(requirement_id) is True
        records[requirement_id] = {
            **trace,
            "actual_evidence": actual_evidence.get(requirement_id)
            or [f"automated_check={passed}"],
            "final_status": "PASS" if passed else "FAIL",
        }
    return records


def _all_models(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        model
        for scene in scenes
        for model in scene.get("model_evaluations", [])
    ]


def audit_verified_experiment(
    *,
    results_path: Path,
    dashboard_dir: Path,
    reviews_path: Path,
    calibration_path: Path,
    evidence_path: Path | None = None,
    expected_head_sha: str | None = None,
) -> dict[str, bool]:
    scenes = _load_json(results_path) if results_path.exists() else []
    engineering_path = dashboard_dir / "data" / "engineering_evaluations.json"
    reviewer_path = dashboard_dir / "data" / "reviewer_payload.json"
    reveal_path = dashboard_dir / "data" / "reveal_payload.json"
    queue_path = dashboard_dir / "data" / "review_queue.json"
    index_path = dashboard_dir / "index.html"
    review_html_path = dashboard_dir / "review.html"
    engineering = _load_json(engineering_path) if engineering_path.exists() else {}
    reviewer = _load_json(reviewer_path) if reviewer_path.exists() else {}
    reveal = _load_json(reveal_path) if reveal_path.exists() else {}
    queue = _load_json(queue_path) if queue_path.exists() else []
    calibration = _load_json(calibration_path) if calibration_path.exists() else {}
    evidence = (
        _load_json(evidence_path)
        if evidence_path is not None and evidence_path.exists()
        else {}
    )
    review_lines = (
        [
            json.loads(line)
            for line in reviews_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if reviews_path.exists()
        else []
    )
    index_html = (
        index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    )
    review_html = (
        review_html_path.read_text(encoding="utf-8")
        if review_html_path.exists()
        else ""
    )
    models = _all_models(scenes)

    r1 = bool(scenes) and all(
        set(scene.get("objective_by_level", {})) == _REQUIRED_LEVELS
        for scene in scenes
    )
    metric_keys = {
        "mean_luminance",
        "ev_mean",
        "p20",
        "p50",
        "p90",
        "clipping_ratio",
        "shadow_ratio",
        "contrast",
        "color_drift",
        "edge_similarity",
    }
    sequence_keys = {
        "spearman_rho",
        "violation_rate",
        "smoothness_score",
        "dead_zone_ratio",
        "endpoint_range_ev",
        "clipping_growth",
        "control_score",
    }
    stage_ids = {
        "integrity",
        "hard_gate",
        "tone_color",
        "fidelity",
        "control",
        "model_judges",
    }
    r2 = bool(scenes) and all(
        all(
            metric_keys <= set(metrics)
            for metrics in scene.get("objective_by_level", {}).values()
        )
        and sequence_keys <= set(scene.get("sequence", {}))
        and stage_ids
        <= {stage.get("stage_id") for stage in scene.get("stages", [])}
        for scene in scenes
    )
    r3 = bool(models) and _successful_behavior(evidence, "zero_judges") and all(
        model.get("available") is not None
        and model.get("model_id")
        and model.get("model_role")
        and model.get("prompt_trace")
        for model in models
    )
    r4 = bool(models) and all(
        (not model.get("available"))
        or (
            _REQUIRED_SCORES <= set(model.get("scores", {}))
            and model.get("decision")
            in {"KEEP", "REGENERATE", "REVIEW", "REJECT"}
            and "confidence" in model
            and "issues" in model
        )
        for model in models
    )
    trace_keys = {
        "prompt_id",
        "prompt_version",
        "template",
        "variables",
        "rendered_prompt",
        "prompt_hash",
        "input_manifest",
        "output_schema_version",
        "inference_parameters",
    }
    r5 = bool(models) and all(
        trace_keys <= set(model.get("prompt_trace", {})) for model in models
    )
    manifest_items = [
        item
        for model in models
        for item in model.get("prompt_trace", {}).get("input_manifest", [])
    ]
    r6 = bool(manifest_items) and all(
        item.get("source_sha256")
        and item.get("payload_sha256")
        and item.get("payload_mime")
        and item.get("payload_encoding")
        and item.get("sent_width")
        and item.get("sent_height")
        for item in manifest_items
    )
    r7 = bool(models) and all(
        model.get("prompt_trace", {}).get("prompt_version") == "3.3"
        and model.get("prompt_trace", {}).get("output_schema_version")
        == "tmqa.sequence.v4"
        for model in models
    )
    r8 = all(
        _successful_behavior(evidence, check_name)
        for check_name in [
            "fatal_non_compensation",
            "decision_disagreement",
            "zero_judges",
            "all_unavailable",
        ]
    ) and all(
        scene.get("decision_provenance", {}).get("version")
        and scene.get("decision_provenance", {}).get("thresholds")
        for scene in scenes
    )
    for scene in scenes:
        available = [
            model
            for model in scene.get("model_evaluations", [])
            if model.get("available")
        ]
        decisions = {model.get("decision") for model in available}
        if len(decisions) == 1 and next(iter(decisions)) in {
            "REJECT",
            "REGENERATE",
        }:
            r8 = r8 and scene.get("decision") != "KEEP"
        if not available and scene.get("model_evaluations"):
            r8 = r8 and scene.get("decision") == "REVIEW"
    r9 = index_path.exists() and _successful_behavior(evidence, "dashboard_behavior")
    reviewer_serialized = json.dumps(reviewer, sort_keys=True)
    r10 = (
        reviewer_path.exists()
        and reveal_path.exists()
        and review_html_path.exists()
        and all(
            token not in reviewer_serialized
            for token in _FORBIDDEN_REVIEW_TOKENS
        )
        and _successful_behavior(evidence, "blind_review_network")
        and _successful_behavior(evidence, "route_isolation")
    )
    scene_by_id = {scene.get("scene_id"): scene for scene in scenes}
    r11 = isinstance(queue, list) and all(
        scene_id in scene_by_id
        and bool(scene_by_id[scene_id].get("review_reasons"))
        for scene_id in queue
    )
    review_keys = {
        "review_id",
        "scene_id",
        "reviewer_id",
        "blind_review",
        "decision",
        "scores",
        "confidence",
        "synthetic",
        "received_at",
    }
    r12 = bool(review_lines) and all(
        review_keys <= set(review)
        and review.get("reviewer_id")
        and review.get("blind_review") is True
        and "overall" in review.get("scores", {})
        for review in review_lines
    )
    selected_type = calibration.get("selected_review_type")
    selected_reviews = [
        review
        for review in review_lines
        if ("synthetic" if review.get("synthetic") else "real")
        == selected_type
    ]
    r13 = (
        selected_type in {"real", "synthetic"}
        and len(selected_reviews) == calibration.get("selected_review_count")
        and (
            selected_type != "synthetic"
            or calibration.get("experimental") is True
        )
    )
    weights = [
        float(model.get("fusion_weight", 0))
        for model in calibration.get("models", [])
    ]
    r14 = (
        bool(weights)
        and abs(sum(weights) - 1.0) <= 1e-9
        and all(
            "overall_mae" in model
            and "decision_agreement" in model
            and "dimension_mae" in model
            and "sample_count" in model
            for model in calibration.get("models", [])
        )
        and calibration.get("sample_sufficiency", {}).get("policy_version")
        == "tmqa.calibration.v1"
        and calibration.get("production_eligible") is False
    )
    artifact_text = "\n".join(
        [
            index_html,
            review_html,
            json.dumps(engineering),
            json.dumps(reviewer),
            json.dumps(reveal),
            json.dumps(calibration),
        ]
    )
    r15 = _successful_behavior(evidence, "route_isolation") and all(
        secret not in artifact_text
        for secret in ["Authorization: Bearer", '"api_key"', "OPENAI_API_KEY"]
    )
    r16 = (
        bool(scenes)
        and index_path.exists()
        and review_html_path.exists()
        and calibration_path.exists()
    )
    required_behavior_names = {
        "fatal_non_compensation",
        "decision_disagreement",
        "zero_judges",
        "all_unavailable",
        "dashboard_behavior",
        "blind_review_network",
        "route_isolation",
    }
    r17 = (
        engineering_path.exists()
        and reviewer_path.exists()
        and reveal_path.exists()
        and queue_path.exists()
        and evidence.get("schema_version") == "tmqa.verification-evidence.v1"
        and required_behavior_names <= set(evidence.get("checks", {}))
    )
    ci = evidence.get("ci", {})
    artifacts = evidence.get("artifacts", {})
    results_digest = (
        hashlib.sha256(results_path.read_bytes()).hexdigest()
        if results_path.exists()
        else ""
    )
    r18 = (
        bool(expected_head_sha)
        and evidence.get("artifact_head_sha") == expected_head_sha
        and artifacts.get("results_sha256") == results_digest
        and ci.get("provider") == "github_actions"
        and ci.get("tested_head_sha") == expected_head_sha
        and ci.get("jobs", {}).get("python-3.10") == "success"
        and ci.get("jobs", {}).get("python-3.12") == "success"
        and isinstance(ci.get("run_id"), int)
    )
    r19 = (
        engineering.get("summary", {}).get("synthetic") is True
        and calibration.get("experimental") is True
        and calibration.get("production_eligible") is False
        and "not a production" in calibration.get("warning", "")
    )

    values = [
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,
        r7,
        r8,
        r9,
        r10,
        r11,
        r12,
        r13,
        r14,
        r15,
        r16,
        r17,
        r18,
        r19,
    ]
    return {
        f"R-{index:03d}": bool(value)
        for index, value in enumerate(values, start=1)
    }
