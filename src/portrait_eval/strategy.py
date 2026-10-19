from __future__ import annotations

from statistics import median
from typing import Any

from portrait_eval.adjudication import adjudicate_claim
from portrait_eval.models import AdjudicatedClaim, ClaimCandidate


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

        for scene in scene_results:
            metrics_by_device = scene.get("metrics", {})
            if device_id not in metrics_by_device or len(metrics_by_device) < 2:
                continue
            group_id = str(scene.get("group_id"))
            lumas = {
                current: float(payload.get("whole", {}).get("luma_mean", 0.0))
                for current, payload in metrics_by_device.items()
            }
            clips = {
                current: float(payload.get("whole", {}).get("highlight_clip_ratio", 0.0))
                for current, payload in metrics_by_device.items()
            }
            luma_mid = median(lumas.values())
            clip_mid = median(clips.values())
            if lumas[device_id] > luma_mid + 0.03:
                brighter_support.append(group_id)
            elif lumas[device_id] < luma_mid - 0.03:
                brighter_counter.append(group_id)
            if clips[device_id] < clip_mid - 0.002:
                lower_clip_support.append(group_id)
            elif clips[device_id] > clip_mid + 0.002:
                lower_clip_counter.append(group_id)

        if brighter_support:
            claims.append(
                adjudicate_claim(
                    ClaimCandidate(
                        device_id=device_id,
                        statement=(
                            "Across the matched groups listed as evidence, this device tends to render "
                            "a higher display-referred global luminance than the group median."
                        ),
                        claim_type="strategy",
                        supporting_scene_ids=brighter_support,
                        contradicting_scene_ids=brighter_counter,
                        model_agreement=0.75,
                        objective_support=1.0,
                        scene_validity=0.85,
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
                        statement=(
                            "Across the matched groups listed as evidence, this device tends to show a "
                            "lower near-white clipping ratio than the group median."
                        ),
                        claim_type="strategy",
                        supporting_scene_ids=lower_clip_support,
                        contradicting_scene_ids=lower_clip_counter,
                        model_agreement=0.7,
                        objective_support=1.0,
                        scene_validity=0.85,
                        alternative_explanations=[
                            "lower base exposure",
                            "different highlight area due to framing",
                            "highlight compression rather than detail preservation",
                        ],
                    )
                )
            )
    return claims
