"""Build challenge coverage and device summaries from the adjudicated claim ledger."""

from __future__ import annotations

from typing import Any

CHALLENGES = {
    "uniform_light": "均匀光",
    "high_contrast": "高反差 / 逆光",
    "mixed_light": "混合 / 有色光",
    "multi_person": "多人 / 景深",
    "low_light": "低光 / 夜景",
    "strong_light": "强光 / 反射",
    "unknown": "待确认",
}


def challenge_matrix(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "challenge": key,
            "label": label,
            "scene_ids": [
                scene["group_id"]
                for scene in scenes
                if scene.get("audit", {}).get("challenge", {}).get("primary", "unknown") == key
            ],
        }
        for key, label in CHALLENGES.items()
    ]


def device_profiles(
    devices: list[dict[str, Any]], claims: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    profiles = []
    for device in devices:
        related = [claim for claim in claims if claim.get("device_id") == device["id"]]
        supported = [
            claim
            for claim in related
            if claim.get("claim_type") == "strategy" and claim.get("status") == "SUPPORTED_SCOPED"
        ]
        profiles.append(
            {
                "device_id": device["id"],
                "device_name": device["name"],
                "label": "Evidence Profile",
                "summary": "；".join(claim["statement"] for claim in supported[:2])
                or "证据不足，暂不能形成跨场景画像。",
                "priority": "unknown: output tendency does not establish product intent",
                "strengths": [
                    {"claim_id": claim["claim_id"], "statement": claim["statement"]}
                    for claim in supported
                ],
                "tradeoffs": "unknown: assess quality and style separately before inferring a tradeoff",
                "counterexamples": sorted(
                    {
                        scene
                        for claim in related
                        for scene in claim.get("contradicting_scene_ids", [])
                    }
                ),
                "scope": sorted(
                    {
                        scene
                        for claim in supported
                        for scene in claim.get("supporting_scene_ids", [])
                    }
                ),
                "failure_conditions": "See listed counterexamples; untested conditions remain unknown",
            }
        )
    return profiles


def apply_review_decisions(payload, reviews: list[dict[str, Any]]):
    """Apply dispositions to a report copy while retaining the original observation ledger."""
    from copy import deepcopy

    result = payload.model_copy(deep=True)
    rejected = [
        {**row["payload"], "_category": row.get("category")}
        for row in reviews
        if row.get("status") in {"rejected", "insufficient_evidence"}
        and isinstance(row.get("payload"), dict)
    ]

    claim_rejections = [
        row for row in rejected if row.get("_category") != "external_corroboration_review"
    ]

    def matches(claim, review):
        if review.get("claim_id"):
            return all(
                claim.get(key) == review.get(key)
                for key in ("claim_id", "claim_version", "device_id")
            )
        return (
            result.schema_version == "0.1"
            and review.get("statement")
            and claim.get("statement") == review.get("statement")
        )

    def marked(claim):
        copy = deepcopy(claim)
        if any(matches(claim, row) for row in claim_rejections):
            copy.update(status="HUMAN_REJECTED", grade="C", human_disposition="rejected")
        return copy

    result.all_claims = [marked(claim) for claim in result.all_claims]
    result.findings = [
        claim
        for claim in result.findings
        if not any(matches(claim, row) for row in claim_rejections)
    ]
    result.attributions = [
        claim
        for claim in result.attributions
        if not any(matches(claim, row) for row in claim_rejections)
    ]
    for scene in result.scene_results:
        scene["findings"] = [marked(claim) for claim in scene.get("findings", [])]
    result.external_validation = [
        item
        for item in result.external_validation
        if not any(matches(item, row) and item.get("url") == row.get("url") for row in rejected)
    ]
    if result.schema_version == "0.2":
        devices = [
            {"id": profile["device_id"], "name": profile["device_name"]}
            for profile in result.device_profiles
        ]
        result.device_profiles = device_profiles(devices, result.all_claims)
    result.review_provenance["disposition_count"] = len(reviews)
    result.review_provenance["calibration"] = "not_performed"
    return result
