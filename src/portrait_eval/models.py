from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


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


class ModelObservation(BaseModel):
    device_id: str
    dimension: str
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    certainty: float = Field(ge=0, le=1)
    region: str = "unknown"
    attribute: str = "unknown"
    direction: Literal["higher", "lower", "equal", "unknown"] = "unknown"
    comparison: str = "unknown"
    strength: str = "unknown"
    conditions: list[str] = Field(default_factory=list)


class ModelEvaluationResult(BaseModel):
    scene_id: str
    role: str
    observations: list[ModelObservation]
    scores: dict[str, int] = Field(default_factory=dict)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    raw: dict[str, Any] = Field(default_factory=dict)
    model_id: str = "unknown"
    model_family: str = "unknown"
    viewed_evidence: list[dict[str, Any]] = Field(default_factory=list)


class ClaimCandidate(BaseModel):
    device_id: str
    statement: str
    claim_type: str
    supporting_scene_ids: list[str]
    contradicting_scene_ids: list[str]
    model_agreement: float = Field(ge=0, le=1)
    objective_support: float | None = Field(default=None, ge=0, le=1)
    scene_validity: float = Field(ge=0, le=1)
    alternative_explanations: list[str] = Field(default_factory=list)

    claim_id: str = ""
    claim_version: str = "0.2"
    region: str = "unknown"
    attribute: str = "unknown"
    direction: str = "unknown"
    comparison: str = "unknown"
    conditions: list[str] = Field(default_factory=list)
    scene_family_by_id: dict[str, str] = Field(default_factory=dict)
    unknown_scene_ids: list[str] = Field(default_factory=list)
    incomparable_scene_ids: list[str] = Field(default_factory=list)
    measurement_evidence: list[dict[str, Any]] = Field(default_factory=list)
    review_relations: list[str] = Field(default_factory=list)
    reviewer_dependency: str = "unknown"


class AdjudicatedClaim(BaseModel):
    device_id: str
    statement: str
    status: str
    grade: EvidenceGrade
    confidence: float
    supporting_scene_ids: list[str]
    contradicting_scene_ids: list[str]
    alternative_explanations: list[str]
    claim_id: str = ""
    claim_version: str = "0.2"
    region: str = "unknown"
    attribute: str = "unknown"
    direction: str = "unknown"
    comparison: str = "unknown"
    conditions: list[str] = Field(default_factory=list)
    objective_support: float | None = None
    score_kind: str = "uncalibrated_rule_score"
    independent_family_count: int = 0
    scene_family_by_id: dict[str, str] = Field(default_factory=dict)
    unknown_scene_ids: list[str] = Field(default_factory=list)
    incomparable_scene_ids: list[str] = Field(default_factory=list)
    evidence_counts: dict[str, int] = Field(default_factory=dict)
    unassigned_scene_ids: list[str] = Field(default_factory=list)
    measurement_evidence: list[dict[str, Any]] = Field(default_factory=list)
    review_relations: list[str] = Field(default_factory=list)
    reviewer_dependency: str = "unknown"


class ExternalEvidence(BaseModel):
    title: str
    url: str
    source_domain: str
    snippet: str = ""
    relevance: str = "unknown"
    supports_claim: bool | None = None
