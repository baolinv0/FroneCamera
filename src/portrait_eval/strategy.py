from __future__ import annotations

from typing import Any, Literal

from portrait_eval.adjudication import adjudicate_claim
from portrait_eval.evidence import claim_identity, measure_observation
from portrait_eval.models import AdjudicatedClaim, ClaimCandidate, ModelObservation


def infer_cross_scene_claims(
    scene_results: list[dict[str, Any]], version: str = "0.2"
) -> list[AdjudicatedClaim]:
    """Summarize only measured output tendencies, retaining family conflicts and all outcomes."""
    devices = sorted({device for scene in scene_results for device in scene.get("metrics", {})})
    claims = []
    for device in devices:
        for attribute, direction, statement in (
            (
                "encoded_luma_mean",
                "higher",
                "In the listed comparable scene families, this output has higher encoded global luma than the group median.",
            ),
            (
                "encoded_near_white_ratio",
                "lower",
                "In the listed comparable scene families, this output has a lower near-white pixel ratio than the group median; this does not establish retained highlight detail.",
            ),
        ):
            buckets: dict[str, list[str]] = {
                key: [] for key in ("support", "counterexample", "unknown", "incomparable")
            }
            families = {}
            evidence: list[dict[str, Any]] = []
            typed_direction: Literal["higher", "lower", "equal", "unknown"] = (
                "higher" if direction == "higher" else "lower"
            )
            for scene in scene_results:
                group = str(scene["group_id"])
                family = scene.get("scene_family")
                if family:
                    families[group] = str(family)
                audit = scene.get("audit", {})
                if audit.get("status") not in {None, "FULLY_COMPARABLE"}:
                    buckets["incomparable"].append(group)
                    continue
                observation = ModelObservation(
                    device_id=device,
                    dimension=attribute,
                    statement=statement,
                    certainty=0,
                    region="whole",
                    attribute=attribute,
                    direction=typed_direction,
                    comparison="peer_median",
                )
                support, records = measure_observation(
                    observation,
                    scene.get("metrics", {}),
                    {**audit, "status": audit.get("status", "FULLY_COMPARABLE")},
                )
                evidence.extend(dict(record, scene_id=group) for record in records)
                if support is None or not records:
                    key = "unknown"
                elif support == 1:
                    key = "support"
                elif records[0]["direction"] == "equal":
                    key = "unknown"
                else:
                    key = "counterexample"
                buckets[key].append(group)
            if not buckets["support"]:
                continue
            # Same-family mixed outcomes are explicit counterexamples; never majority-vote them away.
            claims.append(
                adjudicate_claim(
                    ClaimCandidate(
                        device_id=device,
                        statement=statement,
                        claim_type="strategy",
                        claim_id=claim_identity(
                            device,
                            "cross_scene",
                            statement,
                            "whole",
                            attribute,
                            direction,
                            version,
                            comparison="peer_median",
                            conditions=[
                                "submitted captures",
                                "comparable framing",
                                "SDR sRGB analysis",
                            ],
                        ),
                        claim_version=version,
                        region="whole",
                        attribute=attribute,
                        direction=direction,
                        comparison="peer_median",
                        conditions=[
                            "submitted captures",
                            "comparable framing",
                            "SDR sRGB analysis",
                        ],
                        supporting_scene_ids=buckets["support"],
                        contradicting_scene_ids=buckets["counterexample"],
                        unknown_scene_ids=buckets["unknown"],
                        incomparable_scene_ids=buckets["incomparable"],
                        scene_family_by_id=families,
                        model_agreement=0,
                        objective_support=1,
                        scene_validity=1,
                        measurement_evidence=evidence,
                        alternative_explanations=[
                            "capture exposure differences",
                            "framing and timing differences",
                            "global tone and color rendering",
                        ],
                    )
                )
            )
    return claims
