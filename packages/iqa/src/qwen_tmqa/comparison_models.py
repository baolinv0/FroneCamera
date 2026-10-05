"""Standalone comparison contracts; measurements are never calibrated quality scores."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EvidenceState = Literal["measured_proxy", "unobservable", "not_applicable", "invalid"]
DIMENSION_IDS = (
    "face_exposure_readability",
    "highlight_integrity",
    "shadow_black_rendering",
    "skin_awb",
    "lighting_causality",
    "face_background_relation",
    "local_face_lift_naturalness",
    "multi_face_consistency",
    "scene_adaptability",
    "artifact_texture_control",
)


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite numbers are prohibited")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def finite(self):
        _finite(self.model_dump())
        return self


class ComparisonAsset(StrictModel):
    id: str = Field(min_length=1)
    path: Path
    encoding: Literal["srgb", "linear"] = "srgb"
    # User assertion bound to bytes, not independent proof of a processing pipeline.
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must be nonempty")
        return value


class ComparisonROI(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal[
        "face",
        "skin",
        "highlight",
        "shadow",
        "background",
        "face_ring",
        "face_highlight",
        "face_shadow",
        "edge",
    ]
    bbox: tuple[int, int, int, int]
    person_id: str | None = None
    asset_id: str | None = None

    @field_validator("id", "person_id", "asset_id")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("ROI ids must be nonempty")
        return value

    @field_validator("bbox", mode="before")
    @classmethod
    def integer_coordinates(cls, value):
        if not isinstance(value, (list, tuple)) or any(type(x) is not int for x in value):
            raise ValueError("bbox requires integer pixel coordinates")
        return value

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value):
        x, y, w, h = value
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError("bbox requires nonnegative origin and positive area")
        return value


class ComparisonRequest(StrictModel):
    scene_id: str = Field(min_length=1)
    mode: Literal["algorithm", "device"]
    baseline: ComparisonAsset
    candidates: list[ComparisonAsset] = Field(min_length=1)
    source: ComparisonAsset | None = None
    rois: list[ComparisonROI] = Field(default_factory=list)
    split: Literal["train", "audit", "benchmark", "holdout"] = "audit"
    group_id: str = Field(min_length=1)

    @field_validator("scene_id", "group_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scene/group ids must be nonempty")
        return value

    @model_validator(mode="after")
    def identifiers(self):
        ids = [self.baseline.id, *(a.id for a in self.candidates)]
        if self.source:
            ids.append(self.source.id)
        if len(set(ids)) != len(ids):
            raise ValueError("all asset ids must be unique")
        effective_roi_ids: set[tuple[str, str]] = set()
        for roi in self.rois:
            if roi.asset_id is not None and roi.asset_id not in ids:
                raise ValueError("ROI references an unknown asset")
            if self.mode == "algorithm" and roi.asset_id not in (None, self.baseline.id):
                raise ValueError("algorithm ROIs must use baseline coordinates")
            # Validate the actual assessment dictionary scopes, not just declarations.
            # Algorithm None/B scopes are identical. A device None ROI is currently
            # unobservable in every asset, so it also cannot shadow an explicit ROI.
            scopes = (
                [self.baseline.id]
                if self.mode == "algorithm"
                else [roi.asset_id]
                if roi.asset_id is not None
                else ids
            )
            for scope in scopes:
                key = (scope, roi.id)
                if key in effective_roi_ids:
                    raise ValueError("ROI ids must be unique within each effective asset scope")
                effective_roi_ids.add(key)
        return self


class DimensionEvidence(StrictModel):
    state: EvidenceState
    facts: dict[str, float | int | None] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    absolute_quality_score: None = None


class ROIAssessment(StrictModel):
    id: str
    kind: str
    bbox: tuple[int, int, int, int]
    person_id: str | None
    state: EvidenceState
    objective: dict[str, float | int | None] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class AssetAssessment(StrictModel):
    id: str
    state: Literal["valid", "invalid"]
    objective: dict[str, float | int | None] = Field(default_factory=dict)
    dimensions: dict[str, DimensionEvidence]
    rois: dict[str, ROIAssessment] = Field(default_factory=dict)
    persons: dict[str, dict[str, dict[str, float | int | None]]] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class AlignmentEvidence(StrictModel):
    state: EvidenceState
    shift_x: float | None = None
    shift_y: float | None = None
    response: float | None = None
    overlap_fraction: float | None = None
    valid_pixels: int = 0
    reasons: list[str] = Field(default_factory=list)


class CandidateComparison(StrictModel):
    candidate_id: str
    baseline_id: str
    status: Literal["REVIEW", "REJECT"]
    relative_metrics: dict[str, float | None] = Field(default_factory=dict)
    relative_rois: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    fatal_reasons: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    alignment: AlignmentEvidence
    same_source_status: Literal[
        "declared_unverified", "hash_bound_declaration", "missing", "mismatch", "not_comparable"
    ]
    uncertainty: list[str] = Field(default_factory=list)
    absolute_quality_score: None = None
    controllability: Literal["unobservable"] = "unobservable"
    style_preference: Literal["requires_human_review"] = "requires_human_review"


class ComparisonResult(StrictModel):
    schema_version: Literal["iqa-comparison/1"] = "iqa-comparison/1"
    scene_id: str
    mode: Literal["algorithm", "device"]
    baseline_id: str
    assets: list[AssetAssessment]
    comparisons: list[CandidateComparison]
    warnings: list[str] = Field(default_factory=list)
    input_trace: dict[str, Any]
