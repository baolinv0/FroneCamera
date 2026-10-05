from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import httpx
import numpy as np

from .config import JudgeConfig, TMQAConfig
from .domain import ImageManifestItem, ModelEvaluation, SceneEvaluation, SceneSpec, StageResult
from .image_io import file_sha256, load_rgb
from .judges.mock import MockJudge
from .judges.openai_compatible import OpenAICompatibleJudge
from .metrics import compute_objective_metrics, compute_sequence_metrics
from .prompts import PromptRegistry, build_input_manifest, render_prompt


class _InputIntegrityError(ValueError):
    """The scene no longer has the inputs used to calculate its evidence."""


def _capture_input_hashes(scene: SceneSpec) -> dict[Path, str]:
    paths = [scene.baseline_path, *(item.path for item in scene.alpha_images)]
    if scene.source_path is not None:
        paths.append(scene.source_path)
    try:
        return {path: file_sha256(path) for path in dict.fromkeys(paths)}
    except OSError as exc:
        raise _InputIntegrityError(f"input integrity failure: {exc}") from exc


def _assert_inputs_unchanged(hashes: Mapping[Path, str]) -> None:
    for path, expected in hashes.items():
        try:
            actual = file_sha256(path)
        except OSError as exc:
            raise _InputIntegrityError(f"input integrity failure: {path}: {exc}") from exc
        if actual != expected:
            raise _InputIntegrityError(f"input integrity failure: source hash changed: {path}")


def _assert_manifest_bound(
    manifest: list[ImageManifestItem], hashes: Mapping[Path, str]
) -> None:
    for item in manifest:
        if item.source_sha256 != hashes.get(Path(item.path)):
            raise _InputIntegrityError(
                f"input integrity failure: manifest source hash mismatch: {item.path}"
            )


def model_score_gap(evaluations: list[ModelEvaluation], dimension: str) -> float:
    scores = [
        item.scores[dimension]
        for item in evaluations
        if item.available and dimension in item.scores
    ]
    return max(scores) - min(scores) if len(scores) >= 2 else 0.0


def _judge_from_config(
    config: JudgeConfig,
    transport: httpx.BaseTransport | None,
):
    if config.adapter == "mock":
        return MockJudge(config)
    return OpenAICompatibleJudge(config, transport=transport)


class EvaluationPipeline:
    def __init__(
        self,
        config: TMQAConfig,
        *,
        transports: Mapping[str, httpx.BaseTransport] | None = None,
    ):
        self.config = config
        self.transports = dict(transports or {})
        self.registry = PromptRegistry.default()

    def evaluate_scene(self, scene) -> SceneEvaluation:
        # Bind objective evidence and every judge trace to the same source bytes.
        # Rechecking also covers an explicit objective baseline outside the manifest.
        input_hashes = _capture_input_hashes(scene)
        try:
            input_manifest = build_input_manifest(scene)
            baseline = load_rgb(scene.baseline_path)
            objective = []
            for alpha_image in scene.alpha_images:
                image = load_rgb(alpha_image.path)
                objective.append(
                    compute_objective_metrics(
                        image,
                        alpha=alpha_image.alpha,
                        level=alpha_image.level,
                        baseline=baseline,
                    )
                )
        except OSError as exc:
            raise _InputIntegrityError(f"input integrity failure: {exc}") from exc
        _assert_inputs_unchanged(input_hashes)
        _assert_manifest_bound(input_manifest, input_hashes)
        sequence = compute_sequence_metrics(objective)
        evidence = {
            "max_clipping_ratio": max(item.clipping_ratio for item in objective),
            "max_shadow_ratio": max(item.shadow_ratio for item in objective),
            "max_color_drift": max(item.color_drift for item in objective),
            "min_edge_similarity": min(item.edge_similarity for item in objective),
            "control_score": sequence.control_score,
            "clipping_growth": sequence.clipping_growth,
            "endpoint_range_ev": sequence.endpoint_range_ev,
            "violation_rate": sequence.violation_rate,
        }
        prepared_judges = []
        for judge_config in self.config.judges:
            if not judge_config.enabled:
                continue
            try:
                trace = render_prompt(
                    registry=self.registry,
                    prompt_id=judge_config.prompt_id,
                    prompt_version=judge_config.prompt_version,
                    scene=scene,
                    model_role=judge_config.role,
                    objective_evidence=evidence,
                    inference_parameters={
                        "temperature": judge_config.temperature,
                        "max_tokens": judge_config.max_tokens,
                        "model": judge_config.model,
                    },
                )
            except OSError as exc:
                raise _InputIntegrityError(f"input integrity failure: {exc}") from exc
            _assert_inputs_unchanged(input_hashes)
            _assert_manifest_bound(trace.input_manifest, input_hashes)
            if trace.input_manifest != input_manifest:
                raise _InputIntegrityError("input integrity failure: judge manifest changed")
            prepared_judges.append((judge_config, trace))

        model_evaluations: list[ModelEvaluation] = []
        integrity_error = None
        for judge_config, trace in prepared_judges:
            if integrity_error is None:
                try:
                    _assert_inputs_unchanged(input_hashes)
                except _InputIntegrityError as exc:
                    integrity_error = str(exc)
            if integrity_error is not None:
                model_evaluations.append(
                    ModelEvaluation(
                        model_id=judge_config.id,
                        model_role=judge_config.role,
                        model_version=judge_config.version,
                        synthetic=judge_config.synthetic,
                        available=False,
                        prompt_trace=trace,
                        decision="REVIEW",
                        confidence=0,
                        error=integrity_error,
                    )
                )
                continue
            judge = _judge_from_config(judge_config, self.transports.get(judge_config.id))
            model_evaluations.append(judge.evaluate(trace))
            try:
                _assert_inputs_unchanged(input_hashes)
            except _InputIntegrityError as exc:
                integrity_error = str(exc)

        max_clip = evidence["max_clipping_ratio"]
        min_edge = evidence["min_edge_similarity"]
        policy = self.config.decision_policy
        objective_fatal = (
            max_clip > policy.objective_fatal_clipping_ratio
            or min_edge < policy.objective_fatal_min_edge_similarity
        )
        available_models = [item for item in model_evaluations if item.available]
        unavailable_models = [item for item in model_evaluations if not item.available]
        model_fatal_issues = [
            issue
            for model in available_models
            for issue in model.issues
            if issue.fatal
        ]
        fatal = objective_fatal or bool(model_fatal_issues) or integrity_error is not None
        tone_score = float(
            np.clip(1 - max_clip - 0.25 * evidence["max_shadow_ratio"], 0, 1)
        )
        fidelity_score = float(np.clip(min_edge - evidence["max_color_drift"], 0, 1))
        available_overall = [
            item.scores["overall"]
            for item in model_evaluations
            if item.available and "overall" in item.scores
        ]
        model_score = float(np.mean(available_overall)) if available_overall else (
            tone_score + fidelity_score + sequence.control_score
        ) / 3
        gap = model_score_gap(model_evaluations, "overall")
        available_decisions = {item.decision for item in available_models}
        decision_disagreement = len(available_decisions) > 1
        unanimous_decision = (
            next(iter(available_decisions)) if len(available_decisions) == 1 else None
        )

        if fatal:
            decision = "REJECT"
        elif not available_models or decision_disagreement:
            decision = "REVIEW"
        elif unanimous_decision == "REJECT":
            decision = "REJECT"
        elif unanimous_decision == "REGENERATE":
            decision = "REGENERATE"
        elif unanimous_decision == "REVIEW" or gap >= self.config.review.disagreement_threshold:
            decision = "REVIEW"
        elif model_score < policy.regenerate_score_threshold:
            decision = "REGENERATE"
        elif model_score < policy.keep_score_threshold:
            decision = "REVIEW"
        else:
            decision = "KEEP"

        unavailable_fraction = (
            len(unavailable_models) / len(model_evaluations)
            if available_models
            else 1.0
        )
        uncertainty = float(
            np.clip(
                policy.uncertainty_base
                + policy.uncertainty_gap_weight * gap
                + policy.uncertainty_control_weight * (1 - sequence.control_score)
                + policy.uncertainty_unavailable_weight * unavailable_fraction,
                0,
                1,
            )
        )
        risk = float(
            max_clip * policy.clipping_risk_multiplier
            + max(0, policy.edge_risk_reference - min_edge)
            + sequence.violation_rate
            + max((issue.severity for issue in model_fatal_issues), default=0.0)
        )
        reasons: list[str] = []
        if integrity_error is not None:
            reasons.append("input_integrity_failure")
        if gap >= self.config.review.disagreement_threshold:
            reasons.append("model_disagreement")
        if decision_disagreement:
            reasons.append("model_decision_disagreement")
        if not available_models or unavailable_models:
            reasons.append("judge_unavailable")
        if unanimous_decision in {"REJECT", "REGENERATE", "REVIEW"}:
            reasons.append(f"model_unanimous_{unanimous_decision.lower()}")
        if risk >= self.config.review.high_risk_threshold:
            reasons.append("high_risk")
        if model_score < policy.low_tail_score_threshold:
            reasons.append("low_tail_candidate")
        if fatal:
            reasons.append("fatal_risk")
        priority = 2.0 * gap + 0.5 * float(decision_disagreement) + risk + uncertainty
        decision_provenance = {
            "version": policy.version,
            "review_policy_version": self.config.review.version,
            "thresholds": {
                **policy.model_dump(exclude={"version"}),
                "disagreement_threshold": self.config.review.disagreement_threshold,
                "high_risk_threshold": self.config.review.high_risk_threshold,
            },
        }
        policy_evidence = f"decision_policy={policy.version}"
        stages = [
            StageResult(
                stage_id="integrity",
                label="Integrity",
                status="FAIL" if integrity_error is not None else "PASS",
                score=0.0 if integrity_error is not None else 1.0,
                evidence=[policy_evidence, *([integrity_error] if integrity_error else [])],
            ),
            StageResult(
                stage_id="hard_gate",
                label="Hard gate",
                status="FAIL" if fatal else "PASS",
                score=0.0 if fatal else 1.0,
                evidence=[
                    policy_evidence,
                    f"max_clipping={max_clip:.4f}",
                    f"min_edge={min_edge:.4f}",
                    f"model_fatal_issues={len(model_fatal_issues)}",
                ],
            ),
            StageResult(
                stage_id="tone_color",
                label="Tone & color",
                status=(
                    "PASS"
                    if tone_score >= policy.tone_stage_pass_threshold
                    else "WARN"
                ),
                score=tone_score,
                evidence=[policy_evidence],
            ),
            StageResult(
                stage_id="fidelity",
                label="Fidelity",
                status=(
                    "PASS"
                    if fidelity_score >= policy.fidelity_stage_pass_threshold
                    else "WARN"
                ),
                score=fidelity_score,
                evidence=[policy_evidence],
            ),
            StageResult(
                stage_id="control",
                label="Control",
                status=(
                    "PASS"
                    if sequence.control_score >= policy.control_stage_pass_threshold
                    else "WARN"
                ),
                score=sequence.control_score,
                evidence=[policy_evidence],
            ),
            StageResult(
                stage_id="model_judges",
                label="Model judges",
                status=(
                    "WARN"
                    if integrity_error is not None
                    or not available_models
                    or unavailable_models
                    or decision_disagreement
                    or unanimous_decision in {"REJECT", "REGENERATE", "REVIEW"}
                    or gap >= self.config.review.disagreement_threshold
                    else "PASS"
                ),
                score=model_score,
                evidence=[
                    policy_evidence,
                    f"overall_gap={gap:.4f}",
                    f"available={len(available_models)}",
                    f"unavailable={len(unavailable_models)}",
                    f"decision_disagreement={decision_disagreement}",
                    f"unanimous_decision={unanimous_decision or 'none'}",
                ],
            ),
        ]
        return SceneEvaluation(
            scene_id=scene.scene_id,
            generator=scene.generator,
            decision=decision,
            overall_score=float(np.clip(model_score, 0, 1)),
            uncertainty=uncertainty,
            objective_by_level={item.level: item for item in objective},
            sequence=sequence,
            stages=stages,
            input_manifest=input_manifest,
            model_evaluations=model_evaluations,
            review_priority=priority,
            review_reasons=reasons,
            decision_provenance=decision_provenance,
            synthetic_experiment=bool(model_evaluations)
            and all(item.synthetic for item in model_evaluations),
        )
