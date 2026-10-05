"""Use the independently installable IQA core on actual declared scene assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_tmqa.comparison import evaluate_comparison
from qwen_tmqa.comparison_models import ComparisonAsset, ComparisonRequest, ComparisonROI


def evaluate_scene(
    scene_id: str, image_paths: dict[str, Path], metrics: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if len(image_paths) < 2:
        return {
            "scene_id": scene_id,
            "mode": "device",
            "status": "missing_assets",
            "warnings": ["At least two declared assets are required for comparison"],
        }
    ordered = sorted(image_paths)
    assets = [
        ComparisonAsset(
            id=device_id,
            path=str(image_paths[device_id]),
            encoding="srgb",
            color_policy="embedded_to_srgb",
        )
        for device_id in ordered
    ]
    rois = []
    for device_id in ordered:
        payload = metrics.get(device_id, {})
        bboxes = payload.get("face_bboxes")
        if bboxes is None:
            bboxes = [payload["face_bbox"]] if payload.get("face_bbox") else []
        for index, bbox in enumerate(bboxes):
            rois.append(
                ComparisonROI(
                    id=f"{device_id}:face:{index + 1}",
                    kind="face",
                    bbox=tuple(bbox),
                    person_id=f"{device_id}:unverified-person-{index + 1}",
                    asset_id=device_id,
                )
            )
    request = ComparisonRequest(
        scene_id=scene_id,
        mode="device",
        baseline=assets[0],
        candidates=assets[1:],
        rois=rois,
        split="audit",
        group_id=scene_id,
    )
    result = evaluate_comparison(request).model_dump(mode="json")
    result["bridge_provenance"] = {
        "entrypoint": "qwen_tmqa.comparison.evaluate_comparison",
        "label_scope": "submitted_device_captures_only",
        "person_correspondence": "unverified" if rois else "missing",
        "warning": "Detected face regions do not establish person identity correspondence or absolute portrait quality.",
    }
    return result


def scene_evidence_audit(evidence: dict[str, Any], device_ids: list[str]) -> dict[str, Any]:
    assets = evidence.get("assets", [])
    assessed = {asset.get("id"): asset for asset in assets}
    warnings = []
    if set(assessed) != set(device_ids):
        warnings.append("iqa_declared_asset_coverage_missing")
    invalid_ids = [
        device_id for device_id in device_ids if assessed.get(device_id, {}).get("state") != "valid"
    ]
    if invalid_ids:
        warnings.append("iqa_invalid_asset_evidence")
    fatal_reasons = [
        reason
        for comparison in evidence.get("comparisons", [])
        for reason in comparison.get("fatal_reasons", [])
    ]
    if fatal_reasons:
        warnings.append("iqa_fatal_comparison_evidence")
    color_warnings = list(
        dict.fromkeys(
            warning
            for asset in assets
            for warning in asset.get("trace", {}).get("warnings", [])
            if _is_color_warning(warning)
        )
    )
    return {
        "valid": len(device_ids) >= 2 and not warnings,
        "invalid_asset_ids": invalid_ids,
        "fatal_reasons": fatal_reasons,
        "warnings": warnings,
        "color_warnings": color_warnings,
    }


def dimension_evidence(evidence: dict[str, Any], device_id: str, dimension: str) -> dict[str, Any]:
    asset = next(
        (asset for asset in evidence.get("assets", []) if asset.get("id") == device_id), None
    )
    if asset is None or asset.get("state") != "valid":
        return {"state": "invalid", "missing_evidence": ["valid_declared_asset"]}
    if dimension == "global_exposure":
        objective = asset.get("objective", {})
        return {
            "state": "measured_proxy"
            if objective.get("mean_luminance") is not None
            else "unobservable",
            "facts": {"mean_luminance": objective.get("mean_luminance")},
            "uncertainty": ["global_luminance_is_not_absolute_portrait_quality"],
        }
    canonical = "highlight_integrity" if dimension == "highlight_retention" else dimension
    return asset.get("dimensions", {}).get(
        canonical, {"state": "unobservable", "missing_evidence": ["dimension_evidence"]}
    )


def _is_color_warning(warning: str) -> bool:
    return warning.startswith(("embedded_", "unprofiled_color_"))


def model_evidence_context(evidence: dict[str, Any], *, measurements: bool) -> dict[str, Any]:
    """Expose core applicability to blind passes; measured facts to validation passes."""
    assets = []
    for asset in evidence.get("assets", []):
        dimensions = {}
        for dimension, observation in asset.get("dimensions", {}).items():
            dimensions[dimension] = {
                "state": observation.get("state"),
                "missing_evidence": observation.get("missing_evidence", []),
                "uncertainty": observation.get("uncertainty", []),
            }
            if measurements:
                dimensions[dimension]["facts"] = observation.get("facts", {})
        payload = {
            "id": asset.get("id"),
            "state": asset.get("state"),
            "dimensions": dimensions,
            "reasons": asset.get("reasons", []),
            "color_warnings": [
                warning
                for warning in asset.get("trace", {}).get("warnings", [])
                if _is_color_warning(warning)
            ],
        }
        if measurements:
            payload["objective"] = asset.get("objective", {})
        assets.append(payload)
    return {
        "mode": evidence.get("mode"),
        "assets": assets,
        "warnings": list(
            dict.fromkeys(
                [
                    *evidence.get("warnings", []),
                    *(warning for asset in assets for warning in asset["color_warnings"]),
                ]
            )
        ),
        "scope": "measured_photometric_proxies_and_explicit_missing_evidence",
    }
