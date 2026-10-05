from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from portrait_eval.core.models_v2 import DimensionId


class ProjectStatus(StrEnum):
    CREATED = "CREATED"
    PAIRING_REQUIRED = "PAIRING_REQUIRED"
    PAIRING_CONFIRMED = "PAIRING_CONFIRMED"
    ANALYZING = "ANALYZING"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REPORT_DRAFT_READY = "REPORT_DRAFT_READY"
    REPORT_FINALIZED = "REPORT_FINALIZED"
    FAILED = "FAILED"


class EvidenceGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class DeviceScan(BaseModel):
    device_id: str
    files: list[Path]
    sequence_slots: list[Path | None] | None = None


class PairingGroup(BaseModel):
    group_id: str
    label: str | None = None
    cells: dict[str, Path | None]
    matching_confidence: float | None = Field(default=None, ge=0, le=1)
    review_required: bool = False
    match_notes: list[str] = Field(default_factory=list)

    @property
    def analyzable(self) -> bool:
        return sum(value is not None for value in self.cells.values()) >= 2


class PairingDraft(BaseModel):
    version: int = 1
    confirmed: bool = False
    groups: list[PairingGroup]


class ImageMetrics(BaseModel):
    image_id: str
    values: dict[str, float]
    face_bbox: list[int] | None = None
    warnings: list[str] = Field(default_factory=list)


KNOWN_DIMENSIONS = {item.value for item in DimensionId} | {"global_exposure", "highlight_retention"}


class ModelObservation(BaseModel):
    claim_id: str = ""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    device_id: str
    dimension: str
    statement: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    certainty: float = Field(ge=0, le=1)

    @field_validator("dimension")
    @classmethod
    def known_dimension(cls, value: str) -> str:
        if value not in KNOWN_DIMENSIONS:
            raise ValueError("Unknown evaluation dimension")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def nonempty_refs(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Evidence references must be nonempty strings")
        return values


class ModelHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    statement: str = Field(min_length=1)
    evidence_refs: list[str]
    alternatives: list[str]
    confidence: float = Field(ge=0, le=1)


class ModelEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    scene_id: str
    role: str
    observations: list[ModelObservation]
    scores: dict[str, StrictInt] = Field(default_factory=dict)
    hypotheses: list[ModelHypothesis] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    raw: dict[str, Any] = Field(default_factory=dict)
    input_trace: dict[str, Any] = Field(default_factory=dict)
    provisional: bool = False

    @model_validator(mode="after")
    def finite_payload(self) -> ModelEvaluationResult:
        def check(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite model payload values are prohibited")
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)

        check(self.model_dump())
        return self

    @field_validator("scores")
    @classmethod
    def valid_scores(cls, values: dict[str, int]) -> dict[str, int]:
        if any(
            key not in KNOWN_DIMENSIONS or not 0 <= value <= 100 for key, value in values.items()
        ):
            raise ValueError("Scores require known dimensions and integers from 0 to 100")
        return values


class ClaimCandidate(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    claim_id: str = ""
    dimension: str = ""
    provisional: bool = False
    uncertainty: float = Field(default=1.0, ge=0, le=1)
    source_claim_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    device_id: str
    statement: str
    claim_type: str
    supporting_scene_ids: list[str]
    contradicting_scene_ids: list[str]
    model_agreement: float = Field(ge=0, le=1)
    objective_support: float = Field(ge=0, le=1)
    scene_validity: float = Field(ge=0, le=1)
    alternative_explanations: list[str] = Field(default_factory=list)


class AdjudicatedClaim(BaseModel):
    claim_id: str = ""
    dimension: str = ""
    claim_type: str = "observation"
    provisional: bool = False
    evidence_refs: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    device_id: str
    statement: str
    status: str
    grade: EvidenceGrade
    confidence: float
    supporting_scene_ids: list[str]
    contradicting_scene_ids: list[str]
    alternative_explanations: list[str]


class ExternalEvidence(BaseModel):
    title: str
    url: str
    source_domain: str
    snippet: str = ""
    relevance: str = "unknown"
    supports_claim: bool | None = None
