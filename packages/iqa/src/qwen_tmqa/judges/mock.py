from __future__ import annotations

import json
import time

import numpy as np

from ..config import JudgeConfig
from ..domain import ModelEvaluation, ModelIssue, PromptTrace


class MockJudge:
    """Deterministic synthetic judge for integration tests and demos only."""

    def __init__(self, config: JudgeConfig):
        self.config = config

    def evaluate(self, prompt_trace: PromptTrace) -> ModelEvaluation:
        started = time.perf_counter()
        evidence = json.loads(str(prompt_trace.variables.get("objective_evidence", "{}")))
        max_clip = float(evidence.get("max_clipping_ratio", 0))
        max_shadow = float(evidence.get("max_shadow_ratio", 0))
        max_color = float(evidence.get("max_color_drift", 0))
        min_edge = float(evidence.get("min_edge_similarity", 1))
        control = float(evidence.get("control_score", 0.8))
        clipping_growth = float(evidence.get("clipping_growth", 0))

        tone = float(np.clip(0.94 - 1.3 * max_clip - 0.25 * max_shadow, 0, 1))
        color = float(np.clip(0.96 - 2.2 * max_color, 0, 1))
        fidelity = float(np.clip(0.2 + 0.8 * min_edge, 0, 1))
        preference = float(np.clip(0.5 * tone + 0.3 * color + 0.2 * control, 0, 1))

        profile = self.config.bias_profile
        if profile == "aesthetic":
            tone = min(1.0, tone + 0.18)
            preference = min(1.0, preference + 0.18)
            fidelity = min(1.0, fidelity + 0.02)
            overall = (
                0.35 * tone + 0.15 * color + 0.10 * fidelity + 0.15 * control + 0.25 * preference
            )
            fatal_clip_threshold = 0.65
        elif profile == "fidelity":
            fidelity = max(0.0, fidelity - 0.35 * (1 - min_edge) - 0.12 * clipping_growth)
            preference = 0.25 * tone + 0.20 * color + 0.55 * fidelity
            overall = (
                0.10 * tone + 0.10 * color + 0.50 * fidelity + 0.20 * control + 0.10 * preference
            )
            fatal_clip_threshold = 0.40
        elif profile == "local":
            tone = max(0.0, tone - 0.70 * max_clip - 0.30 * clipping_growth)
            color = max(0.0, color - 0.40 * max_color)
            fidelity = max(0.0, fidelity - 0.15 * (1 - min_edge))
            preference = 0.50 * tone + 0.15 * color + 0.35 * fidelity
            overall = (
                0.45 * tone + 0.15 * color + 0.25 * fidelity + 0.10 * control + 0.05 * preference
            )
            fatal_clip_threshold = 0.35
        else:
            preference = 0.45 * tone + 0.25 * color + 0.30 * control
            overall = (
                0.30 * tone + 0.15 * color + 0.20 * fidelity + 0.25 * control + 0.10 * preference
            )
            fatal_clip_threshold = 0.50
        overall = float(np.clip(overall, 0, 1))
        issues: list[ModelIssue] = []
        if max_clip > 0.08:
            issues.append(
                ModelIssue(
                    dimension="highlight",
                    severity=min(1.0, max_clip * 3),
                    description="highlight clipping or compressed roll-off",
                    fatal=max_clip > fatal_clip_threshold,
                    region="highlight",
                )
            )
        if min_edge < 0.75:
            issues.append(
                ModelIssue(
                    dimension="fidelity",
                    severity=1 - min_edge,
                    description="possible structural or texture drift",
                    fatal=min_edge < 0.45,
                    region="high_difference",
                )
            )
        if control < 0.65:
            issues.append(
                ModelIssue(
                    dimension="control",
                    severity=1 - control,
                    description="non-smooth or ineffective alpha response",
                )
            )

        if any(issue.fatal for issue in issues):
            decision = "REJECT"
        elif overall < 0.60:
            decision = "REGENERATE"
        elif overall < 0.74 or issues:
            decision = "REVIEW"
        else:
            decision = "KEEP"
        response = {
            "scores": {
                "tone": tone,
                "color": color,
                "fidelity": fidelity,
                "control": control,
                "preference": preference,
                "overall": overall,
            },
            "decision": decision,
            "confidence": 0.78 if issues else 0.86,
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "rationale": f"Synthetic {profile} profile; not a real VLM result.",
        }
        return ModelEvaluation(
            model_id=self.config.id,
            model_role=self.config.role,
            model_version=self.config.version,
            synthetic=True,
            prompt_trace=prompt_trace,
            scores=response["scores"],
            decision=decision,
            confidence=response["confidence"],
            issues=issues,
            rationale=response["rationale"],
            raw_response=json.dumps(response, ensure_ascii=False),
            parsed_response=response,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
