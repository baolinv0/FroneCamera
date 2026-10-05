from __future__ import annotations

import math
from statistics import median
from typing import Any

from portrait_eval.adjudication import adjudicate_claim
from portrait_eval.models import AdjudicatedClaim, ClaimCandidate

HIGHEST_LUMINANCE_OBSERVATION = (
    "This output has the highest display-referred mean luminance in the matched group."
)


def _supporting_model_finding(finding: dict[str, Any], device_id: str, group_id: str) -> bool:
    """A matching field is insufficient: require a resolved, confident same-direction claim."""
    if finding.get("device_id") != device_id or finding.get("dimension") != "global_exposure":
        return False
    if finding.get("statement") != HIGHEST_LUMINANCE_OBSERVATION:
        return False
    if (
        finding.get("grade") not in {"A", "B"}
        or finding.get("status") not in {"SUPPORTED", "CONFIRMED"}
        or finding.get("provisional")
    ):
        return False
    if not finding.get("independent_models") or finding.get("contradicting_scene_ids"):
        return False
    for field in ("confidence", "model_agreement", "uncertainty"):
        value = finding.get(field, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            return False
    # Only observation references to this exact scene's measured luminance count.
    refs = {
        f"metric:{group_id}:{device_id}:luma_mean",
        f"metric:{group_id}:{device_id}:whole.luma_mean",
    }
    return bool(refs.intersection(finding.get("evidence_refs", [])))


def infer_cross_scene_claims(scene_results: list[dict[str, Any]]) -> list[AdjudicatedClaim]:
    """Infer conservative cross-scene output tendencies from matched-group metrics.

    The function only describes rendered-output tendencies. It does not infer a proprietary ISP
    mechanism. Claims need at least two supporting groups to escape the insufficient-coverage gate,
    and three groups for the strongest evidence grade.
    """

    device_ids: set[str] = set()
    for scene in scene_results:
        device_ids.update(scene.get("metrics", {}).keys())

    claims: list[AdjudicatedClaim] = []
    for device_id in sorted(device_ids):
        brighter_support: list[str] = []
        brighter_counter: list[str] = []
        lower_clip_support: list[str] = []
        lower_clip_counter: list[str] = []

        scene_validities: dict[str, float] = {}
        model_strengths: dict[str, float] = {}
        measured_scenes: set[str] = set()
        provisional_scenes: set[str] = set()
        model_source_claim_ids: dict[str, list[str]] = {}
        for scene in scene_results:
            metrics_by_device = scene.get("metrics", {})
            if device_id not in metrics_by_device or len(metrics_by_device) < 2:
                continue
            group_id = str(scene.get("group_id"))
            status = scene.get("audit", {}).get("status")
            if status == "NOT_COMPARABLE":
                continue

            # Missing values are unobserved, not zero-valued measurements.
            def values(metric: str, metrics_by_device=metrics_by_device) -> dict[str, float]:
                output = {}
                for current, payload in metrics_by_device.items():
                    value = payload.get("whole", {}).get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        output[current] = float(value)
                return output

            lumas = values("luma_mean")
            clips = values("highlight_clip_ratio")
            if len(lumas) != len(metrics_by_device):
                continue
            measured_scenes.add(group_id)
            scene_validities[group_id] = (
                1.0 if status == "FULLY_COMPARABLE" else 0.75 if status is None else 0.4
            )
            matching = [
                finding
                for finding in scene.get("findings", [])
                if _supporting_model_finding(finding, device_id, group_id)
            ]
            model_strengths[group_id] = (
                max(
                    (
                        min(
                            float(item["model_agreement"]),
                            float(item["confidence"]),
                            float(item["uncertainty"]),
                        )
                        for item in matching
                    ),
                    default=0.0,
                )
                if status == "FULLY_COMPARABLE"
                else 0.0
            )
            model_source_claim_ids[group_id] = [
                item["claim_id"] for item in matching if item.get("claim_id")
            ]
            if scene.get("provisional"):
                provisional_scenes.add(group_id)
            luma_mid = median(lumas.values())
            clip_mid = median(clips.values()) if clips else None
            if lumas[device_id] > luma_mid + 0.03:
                brighter_support.append(group_id)
            elif lumas[device_id] < luma_mid - 0.03:
                brighter_counter.append(group_id)
            if (
                clip_mid is not None
                and len(clips) == len(metrics_by_device)
                and clips[device_id] < clip_mid - 0.002
            ):
                lower_clip_support.append(group_id)
            elif (
                clip_mid is not None
                and len(clips) == len(metrics_by_device)
                and clips[device_id] > clip_mid + 0.002
            ):
                lower_clip_counter.append(group_id)

        if brighter_support:
            claims.append(
                adjudicate_claim(
                    ClaimCandidate(
                        device_id=device_id,
                        dimension="global_exposure",
                        source_claim_ids=[
                            claim_id
                            for group_id in brighter_support
                            for claim_id in model_source_claim_ids[group_id]
                        ],
                        evidence_refs=[
                            f"metric:{group_id}:{device_id}:whole.luma_mean"
                            for group_id in brighter_support
                        ],
                        provisional=bool(set(brighter_support) & provisional_scenes),
                        statement=(
                            "Across the matched groups listed as evidence, this device tends to render "
                            "a higher display-referred global luminance than the group median."
                        ),
                        claim_type="strategy",
                        supporting_scene_ids=brighter_support,
                        contradicting_scene_ids=brighter_counter,
                        model_agreement=sum(
                            model_strengths[group_id] for group_id in brighter_support
                        )
                        / len(brighter_support),
                        objective_support=len(set(brighter_support) & measured_scenes)
                        / len(set(brighter_support)),
                        scene_validity=sum(
                            scene_validities[group_id] for group_id in brighter_support
                        )
                        / len(brighter_support),
                        alternative_explanations=[
                            "capture exposure differences",
                            "scene mismatch or timing variation",
                            "global tone-mapping differences",
                        ],
                    )
                )
            )
        if lower_clip_support:
            claims.append(
                adjudicate_claim(
                    ClaimCandidate(
                        device_id=device_id,
                        dimension="highlight_integrity",
                        evidence_refs=[
                            f"metric:{group_id}:{device_id}:whole.highlight_clip_ratio"
                            for group_id in lower_clip_support
                        ],
                        provisional=bool(set(lower_clip_support) & provisional_scenes),
                        statement=(
                            "Across the matched groups listed as evidence, this device tends to show a "
                            "lower near-white clipping ratio than the group median."
                        ),
                        claim_type="strategy",
                        supporting_scene_ids=lower_clip_support,
                        contradicting_scene_ids=lower_clip_counter,
                        model_agreement=0.0,
                        objective_support=len(set(lower_clip_support) & measured_scenes)
                        / len(set(lower_clip_support)),
                        scene_validity=sum(
                            scene_validities[group_id] for group_id in lower_clip_support
                        )
                        / len(lower_clip_support),
                        alternative_explanations=[
                            "lower base exposure",
                            "different highlight area due to framing",
                            "highlight compression rather than detail preservation",
                        ],
                    )
                )
            )
    return claims
