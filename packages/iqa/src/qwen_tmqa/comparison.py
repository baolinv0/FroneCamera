"""Read-only B/C evaluation, without HTTP, database, model votes, or quality calibration."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

import cv2
import numpy as np

from .asset_io import AssetDecodeError, LoadedComparisonAsset, load_comparison_asset
from .comparison_models import (
    DIMENSION_IDS,
    AlignmentEvidence,
    AssetAssessment,
    CandidateComparison,
    ComparisonAsset,
    ComparisonRequest,
    ComparisonResult,
    DimensionEvidence,
    EvidenceState,
    ROIAssessment,
)


def _linear(image: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "linear":
        return image
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def _luma(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722])


def _edge(y: np.ndarray) -> np.ndarray:
    # Float64 avoids quantizing narrow 16-bit or floating-point differences.
    return np.hypot(cv2.Sobel(y, cv2.CV_64F, 1, 0), cv2.Sobel(y, cv2.CV_64F, 0, 1))


def _objective(image: np.ndarray) -> dict[str, float | int | None]:
    y = _luma(image)
    p05, p50, p95 = np.percentile(y, [5, 50, 95])
    mean = float(np.mean(y))
    return {
        "pixel_count": int(y.size),
        "mean_luminance": mean,
        "ev_mean": math.log2(mean) if mean > 0 else None,
        "p05_luminance": float(p05),
        "p50_luminance": float(p50),
        "p95_luminance": float(p95),
        "contrast_std": float(np.std(y)),
        "highlight_fraction": float(np.mean(np.max(image, axis=2) >= 0.99)),
        "shadow_fraction": float(np.mean(y <= 0.02)),
        "edge_mean": float(np.mean(_edge(y))),
        "red_mean": float(np.mean(image[..., 0])),
        "green_mean": float(np.mean(image[..., 1])),
        "blue_mean": float(np.mean(image[..., 2])),
    }


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    a = left.reshape(-1)
    b = right.reshape(-1)
    if min(np.std(a), np.std(b)) <= 1e-10:
        return None
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _dimensions(assessment: AssetAssessment) -> dict[str, DimensionEvidence]:
    rois = assessment.rois
    by_kind: dict[str, list[ROIAssessment]] = {}
    for r in rois.values():
        if r.state == "measured_proxy":
            by_kind.setdefault(r.kind, []).append(r)
    base_uncertainty = ["uncalibrated_photometric_proxy", "no_reference_chart_or_human_preference"]
    required = {
        "face_exposure_readability": ("face",),
        "highlight_integrity": ("highlight",),
        "shadow_black_rendering": ("background", "shadow"),
        "skin_awb": ("skin",),
        "lighting_causality": ("face_highlight", "face_shadow"),
        "face_background_relation": ("face", "face_ring", "background"),
        "local_face_lift_naturalness": ("face", "face_ring", "background"),
        "artifact_texture_control": ("face", "edge"),
    }
    out = {}
    for dimension in DIMENSION_IDS:
        if dimension == "scene_adaptability":
            out[dimension] = DimensionEvidence(
                state="not_applicable", missing_evidence=["multiple_comparable_scenes"]
            )
            continue
        if assessment.state == "invalid":
            out[dimension] = DimensionEvidence(state="invalid", missing_evidence=["valid_image"])
            continue
        if dimension == "multi_face_consistency":
            faces = by_kind.get("face", [])
            people = {r.person_id for r in faces if r.person_id is not None}
            usable = [r for r in faces if r.person_id in people]
            if len(people) >= 2:
                means = [float(r.objective["mean_luminance"] or 0.0) for r in usable]
                out[dimension] = DimensionEvidence(
                    state="measured_proxy",
                    facts={
                        "person_count": len(people),
                        "face_luminance_spread": max(means) - min(means),
                    },
                    evidence_refs=[f"roi:{r.id}" for r in usable],
                    uncertainty=base_uncertainty
                    + ["different_skin_reflectance_and_lighting_confound_consistency"],
                )
            else:
                out[dimension] = DimensionEvidence(
                    state="not_applicable", missing_evidence=["at_least_two_valid_person_faces"]
                )
            continue
        kinds = required[dimension]
        missing = [f"valid_{k}_region" for k in kinds if k not in by_kind]
        refs = [r for k in kinds for r in by_kind.get(k, [])]
        facts = {f"{r.id}.{key}": value for r in refs for key, value in r.objective.items()}
        uncertainty = list(base_uncertainty)
        if dimension == "skin_awb":
            # RGB means are observations; absent illumination target they cannot establish AWB accuracy.
            missing.append("illumination_or_color_reference")
            uncertainty.append("skin_rgb_is_not_white_balance_accuracy")
        if dimension == "lighting_causality":
            uncertainty.append("internal_contrast_does_not_establish_physical_lighting_causation")
        if dimension == "local_face_lift_naturalness":
            uncertainty.append("boundary_contrast_is_not_human_naturalness")
        if dimension in ("face_background_relation", "local_face_lift_naturalness") and not missing:
            faces = by_kind["face"]
            for face in faces:
                rings = [r for r in by_kind["face_ring"] if r.person_id == face.person_id]
                if not rings:
                    missing.append(f"person_matched_face_ring:{face.person_id or face.id}")
                else:
                    facts[f"{face.id}.face_ring_mean_delta"] = _number(
                        face.objective, "mean_luminance", 0
                    ) - _number(rings[0].objective, "mean_luminance", 0)
        out[dimension] = DimensionEvidence(
            state="unobservable" if missing else "measured_proxy",
            facts=facts,
            evidence_refs=[f"roi:{r.id}" for r in refs],
            missing_evidence=missing,
            uncertainty=uncertainty,
        )
    return out


def _assessment(
    asset: ComparisonAsset,
    loaded: LoadedComparisonAsset | None,
    request: ComparisonRequest,
    error: str | None,
) -> AssetAssessment:
    result = AssetAssessment(
        id=asset.id,
        state="valid" if loaded else "invalid",
        dimensions={},
        reasons=[error] if error else [],
    )
    if loaded:
        result.trace = loaded.trace
        image = _linear(loaded.pixels, asset.encoding)
        result.objective = _objective(image)
        for roi in request.rois:
            if request.mode == "device" and roi.asset_id not in (None, asset.id):
                continue
            if request.mode == "algorithm" and request.source and asset.id == request.source.id:
                continue
            state: EvidenceState = "measured_proxy"
            reasons = []
            if request.mode == "device" and roi.asset_id is None:
                state = "unobservable"
                reasons.append("device_roi_requires_explicit_asset_correspondence")
            x, y, w, h = roi.bbox
            if x + w > image.shape[1] or y + h > image.shape[0]:
                state = "invalid"
                reasons.append("roi_outside_oriented_image")
            metrics = _objective(image[y : y + h, x : x + w]) if state == "measured_proxy" else {}
            result.rois[roi.id] = ROIAssessment(
                id=roi.id,
                kind=roi.kind,
                bbox=roi.bbox,
                person_id=roi.person_id,
                state=state,
                objective=metrics,
                reasons=reasons,
            )
            if roi.person_id and state == "measured_proxy":
                result.persons.setdefault(roi.person_id, {})[roi.id] = metrics
    else:
        result.trace = {"asset_id": asset.id, "path": str(asset.path), "error": error}
    result.dimensions = _dimensions(result)
    return result


def _alignment(base: np.ndarray, cand: np.ndarray):
    a, b = _luma(base), _luma(cand)
    if min(a.shape) < 16 or a.size < 256:
        return AlignmentEvidence(
            state="unobservable", reasons=["insufficient_spatial_support"]
        ), None
    if min(np.std(a), np.std(b)) <= 1e-8:
        return AlignmentEvidence(
            state="unobservable", reasons=["constant_or_degenerate_image"]
        ), None
    ea, eb = _edge(a), _edge(b)
    # A few isolated bright pixels are not usable spatial texture.
    if (
        min(
            np.mean(ea > max(float(ea.max()) * 0.01, 1e-10)),
            np.mean(eb > max(float(eb.max()) * 0.01, 1e-10)),
        )
        < 0.02
    ):
        return AlignmentEvidence(state="unobservable", reasons=["sparse_spatial_support"]), None
    # Scale separately for numerical stability over ordinary linear HDR ranges.
    norm_a = (a - np.mean(a)) / np.std(a)
    norm_b = (b - np.mean(b)) / np.std(b)
    (sx, sy), response = cv2.phaseCorrelate(norm_a.astype(np.float64), norm_b.astype(np.float64))
    if not all(math.isfinite(v) for v in (sx, sy, response)):
        return AlignmentEvidence(state="invalid", reasons=["nonfinite_translation_estimate"]), None
    h, w = a.shape
    dx, dy = round(sx), round(sy)
    overlap = max(0, w - abs(dx)) * max(0, h - abs(dy)) / (h * w)
    reasons = []
    if response < 0.2:
        reasons.append("weak_translation_response")
    if abs(sx) > w * 0.15 or abs(sy) > h * 0.15:
        reasons.append("translation_exceeds_guard")
    if overlap < 0.8:
        reasons.append("insufficient_overlap")
    evidence = AlignmentEvidence(
        state="unobservable" if reasons else "measured_proxy",
        shift_x=sx,
        shift_y=sy,
        response=response,
        overlap_fraction=overlap,
        valid_pixels=int(overlap * h * w),
        reasons=reasons,
    )
    if reasons:
        return evidence, None
    ax, ay = max(0, -dx), max(0, -dy)
    bx, by = max(0, dx), max(0, dy)
    ww, hh = w - abs(dx), h - abs(dy)
    return evidence, (base[ay : ay + hh, ax : ax + ww], cand[by : by + hh, bx : bx + ww])


def _number(metrics: dict[str, float | int | None], key: str, default: float) -> float:
    value = metrics.get(key)
    return default if value is None else float(value)


def _delta(base: dict, cand: dict):
    return {
        f"{key}_delta": float(cand[key]) - float(value)
        if value is not None and cand.get(key) is not None
        else None
        for key, value in base.items()
        if key != "pixel_count"
    }


def evaluate_comparison(request: ComparisonRequest) -> ComparisonResult:
    """Compare declared assets. Invalid content returns evidence rather than NaN or a score.

    The source describes common input intent; even matching source hashes remain a user
    declaration. Translation is an integer-overlap diagnostic, never physical HDR truth.
    """
    if not isinstance(request, ComparisonRequest):
        raise TypeError("request must be a ComparisonRequest")
    ordered = [request.baseline, *request.candidates]
    if request.source:
        ordered.append(request.source)
    loaded, assessments = {}, []
    for asset in ordered:
        error = None
        invalid_trace = None
        try:
            loaded[asset.id] = load_comparison_asset(asset)
        except (OSError, ValueError, cv2.error, OverflowError) as exc:
            error = f"{type(exc).__name__}:{exc}"
            if isinstance(exc, AssetDecodeError):
                invalid_trace = exc.trace
        assessed = _assessment(asset, loaded.get(asset.id), request, error)
        if invalid_trace:
            assessed.trace.update(invalid_trace)
        assessments.append(assessed)
    lookup = {a.id: a for a in assessments}
    baseline = lookup[request.baseline.id]
    comparisons = []
    warnings = [
        "measurements_are_uncalibrated_proxies_not_absolute_quality",
        "single_comparison_does_not_measure_controllability",
        "highlight_threshold_occupancy_is_not_proof_of_sensor_clipping",
    ]
    if request.mode == "device":
        warnings.extend(
            [
                "different_captures_cannot_produce_same_input_training_labels",
                "capture_illumination_geometry_and_hardware_are_confounders",
            ]
        )
    for candidate in request.candidates:
        assessed = lookup[candidate.id]
        fatal, review = [], ["uncalibrated_quality_requires_review"]
        relative: dict[str, float | None] = {
            "pixel_mae": None,
            "aligned_pixel_mae": None,
            "edge_correlation": None,
            "aligned_edge_correlation": None,
        }
        alignment = AlignmentEvidence(state="unobservable", reasons=["not_computed"])
        same_source: Literal[
            "declared_unverified", "hash_bound_declaration", "missing", "mismatch", "not_comparable"
        ] = "not_comparable" if request.mode == "device" else "missing"
        if baseline.state == "invalid":
            fatal.append("invalid_baseline")
        if assessed.state == "invalid":
            fatal.append("invalid_candidate")
        if request.mode == "algorithm":
            if request.source:
                source = loaded.get(request.source.id)
                if source is None:
                    fatal.append("invalid_source")
                else:
                    source_hash = source.trace["source_bytes_sha256"]
                    claims = [request.baseline.source_sha256, candidate.source_sha256]
                    if any(claim is not None and claim != source_hash for claim in claims):
                        same_source = "mismatch"
                        fatal.append("source_lineage_hash_mismatch")
                    elif all(claim is not None for claim in claims):
                        same_source = "hash_bound_declaration"
                        review.append("declared_lineage_not_independently_verified")
                    else:
                        same_source = "declared_unverified"
                        review.append("same_source_lineage_unverified")
            else:
                review.append("common_source_evidence_missing")
                if request.baseline.source_sha256 or candidate.source_sha256:
                    review.append("source_hash_claim_cannot_be_checked_without_source")
        relative_rois = {}
        if baseline.state == assessed.state == "valid":
            relative.update(_delta(baseline.objective, assessed.objective))
            base = _linear(loaded[baseline.id].pixels, request.baseline.encoding)
            cand = _linear(loaded[candidate.id].pixels, candidate.encoding)
            if request.mode == "algorithm":
                if base.shape != cand.shape:
                    fatal.append("geometry_mismatch")
                else:
                    relative["pixel_mae"] = float(np.mean(np.abs(cand - base)))
                    if np.std(_luma(base)) > 0.02 and np.std(_luma(cand)) < 1e-10:
                        fatal.append("complete_content_collapse")
                    alignment, aligned = _alignment(base, cand)
                    if alignment.state == "measured_proxy":
                        relative["edge_correlation"] = _corr(_edge(_luma(base)), _edge(_luma(cand)))
                        if aligned:
                            ab, ac = aligned
                            relative["aligned_pixel_mae"] = float(np.mean(np.abs(ac - ab)))
                            relative["aligned_edge_correlation"] = _corr(
                                _edge(_luma(ab)), _edge(_luma(ac))
                            )
                    else:
                        review.extend(alignment.reasons)
            else:
                alignment = AlignmentEvidence(
                    state="not_applicable", reasons=["cross_capture_pixel_alignment_not_validated"]
                )
            for roi_id, roi in baseline.rois.items():
                cr = assessed.rois.get(roi_id)
                if cr is not None and roi.state == cr.state == "measured_proxy":
                    if roi.kind != cr.kind or roi.person_id != cr.person_id:
                        review.append(f"roi_correspondence_mismatch:{roi_id}")
                    else:
                        relative_rois[roi_id] = _delta(roi.objective, cr.objective)
                        # Detect a single destroyed face even when scene averages barely
                        # move. These thresholds identify loss of spatial information,
                        # not calibrated human portrait quality. Cross-capture ROIs are
                        # never used to infer a same-input fatal regression.
                        bm, cm = roi.objective, cr.objective
                        if (
                            roi.kind in ("face", "skin")
                            and _number(bm, "pixel_count", 0) >= 16
                            and _number(bm, "contrast_std", 0) > 0.02
                        ):
                            collapse = _number(cm, "contrast_std", 1) < 1e-10
                            severe_highlight = (
                                _number(bm, "highlight_fraction", 1) < 0.1
                                and _number(cm, "highlight_fraction", 0) > 0.95
                                and _number(cm, "contrast_std", 1) < 0.01
                            )
                            if collapse or severe_highlight:
                                correspondence = (
                                    request.mode == "algorithm"
                                    and base.shape == cand.shape
                                    and request.source is not None
                                    and request.source.id in loaded
                                    and same_source != "mismatch"
                                )
                                if correspondence:
                                    if collapse:
                                        fatal.append(f"local_content_collapse:{roi_id}")
                                    if severe_highlight:
                                        fatal.append(
                                            f"local_destructive_highlight_occupancy:{roi_id}"
                                        )
                                else:
                                    if collapse:
                                        review.append(f"possible_local_content_collapse:{roi_id}")
                                    if severe_highlight:
                                        review.append(
                                            f"possible_local_destructive_highlight_occupancy:{roi_id}"
                                        )
            if any(r.state == "invalid" for r in assessed.rois.values()):
                review.append("invalid_roi_evidence")
        comparisons.append(
            CandidateComparison(
                candidate_id=candidate.id,
                baseline_id=baseline.id,
                status="REJECT" if fatal else "REVIEW",
                relative_metrics=relative,
                relative_rois=relative_rois,
                fatal_reasons=fatal,
                review_reasons=review,
                alignment=alignment,
                same_source_status=same_source,
                uncertainty=[
                    "no_calibrated_absolute_score",
                    "no_semantic_content_or_style_judge",
                    "translation_diagnostic_uses_rounded_shift_and_overlap_only",
                ],
            )
        )
    serialized = request.model_dump(mode="json")
    request_hash = hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding = {
        "request_sha256": request_hash,
        "ordered_asset_hashes": [a.trace.get("source_bytes_sha256") for a in assessments],
    }
    input_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return ComparisonResult(
        scene_id=request.scene_id,
        mode=request.mode,
        baseline_id=baseline.id,
        assets=assessments,
        comparisons=comparisons,
        warnings=warnings,
        input_trace={
            "request_sha256": request_hash,
            "input_sha256": input_hash,
            "luminance_domain": "linear_declared_encoding",
            "group_id": request.group_id,
            "split": request.split,
            "ordered_assets": [a.trace for a in assessments],
            "roi_coordinate_convention": "oriented_baseline_pixels"
            if request.mode == "algorithm"
            else "explicit_oriented_asset_pixels",
        },
    )
