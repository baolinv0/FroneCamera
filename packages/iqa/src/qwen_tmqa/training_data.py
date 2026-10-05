"""Human-confirmed training admission, separate from model candidate suggestions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .image_io import file_sha256
from .lineage import SplitAssignment, SplitName

_SHA = r"^[0-9a-f]{64}$"
LabelScope = Literal["tone_mapping_preference", "global_rendering_preference"]


class TrainingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    record_kind: Literal["candidate_suggestion"] = "candidate_suggestion"
    training_weight: Literal[0.0] = 0.0
    scene_id: str
    candidate_id: str
    source_path: str
    candidate_path: str
    source_sha256: str = Field(pattern=_SHA)
    candidate_sha256: str = Field(pattern=_SHA)
    split: SplitName
    canonical_scene_id: str
    group_id: str
    eligible: bool
    judge_ids: list[str]
    synthetic: bool = False
    evidence_mode: Literal["algorithm", "device"]
    rejection_reasons: list[str] = Field(default_factory=list)
    lineage: dict[str, object]

    @field_validator(
        "scene_id",
        "candidate_id",
        "canonical_scene_id",
        "group_id",
        "source_path",
        "candidate_path",
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("explicit identity/path metadata must not be empty")
        return value.strip()


class HumanConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    confirmation_id: str
    reviewer_id: str
    scene_id: str
    candidate_id: str
    canonical_scene_id: str
    group_id: str
    source_sha256: str = Field(pattern=_SHA)
    candidate_sha256: str = Field(pattern=_SHA)
    confirmed: bool
    confirmation_kind: Literal["human"]
    synthetic: bool
    label_scope: LabelScope
    # Training adjudication is distinct from blind gold used for calibration.
    review_stage: Literal["blind", "adjudication"] = "adjudication"
    confirmed_at: str

    @field_validator(
        "confirmation_id",
        "reviewer_id",
        "scene_id",
        "candidate_id",
        "canonical_scene_id",
        "group_id",
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("human confirmation identities must not be empty")
        return value.strip()

    @field_validator("confirmed_at")
    @classmethod
    def aware_timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("confirmed_at must include timezone")
        return value


def export_training_data(
    records: list[TrainingCandidate],
    output_dir: Path,
    *,
    confirmations: list[HumanConfirmation],
    split_assignments: dict[str, SplitAssignment],
) -> dict[str, object]:
    """Validate the complete batch before creating any formal label artifact.

    Confirmation JSON is an explicit human attestation supplied by the operator;
    model suggestions and synthetic review simulations cannot generate it here.
    """
    records = [TrainingCandidate.model_validate(item.model_dump()) for item in records]
    confirmations = [HumanConfirmation.model_validate(item.model_dump()) for item in confirmations]
    assignments = {
        key: SplitAssignment.model_validate(item.model_dump())
        for key, item in split_assignments.items()
    }
    if not assignments:
        raise ValueError("explicit canonical/group metadata is required")
    canonical_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    for key, assignment in assignments.items():
        if key != assignment.scene_id:
            raise ValueError("split metadata key mismatch")
        for identity, seen in [
            (assignment.canonical_scene_id, canonical_splits),
            (assignment.group_id, group_splits),
        ]:
            if identity in seen and seen[identity] != assignment.split:
                raise ValueError("canonical/group metadata crosses splits")
            seen[identity] = assignment.split
    confirmation_map: dict[tuple[str, str], HumanConfirmation] = {}
    confirmation_ids: set[str] = set()
    for confirmation in confirmations:
        key = (confirmation.scene_id, confirmation.candidate_id)
        if key in confirmation_map or confirmation.confirmation_id in confirmation_ids:
            raise ValueError("human confirmation identities must be unique")
        confirmation_map[key] = confirmation
        confirmation_ids.add(confirmation.confirmation_id)
    exported: list[dict[str, object]] = []
    keys: set[tuple[str, str]] = set()
    for record in sorted(records, key=lambda item: (item.scene_id, item.candidate_id)):
        key = (record.scene_id, record.candidate_id)
        if key in keys:
            raise ValueError("training candidate identities must be unique")
        keys.add(key)
        if record.split != "train":
            raise ValueError("formal training export requires train split")
        if record.synthetic:
            raise ValueError("synthetic evidence cannot become formal training labels")
        if record.evidence_mode != "algorithm":
            raise ValueError("formal training requires same-source algorithm evidence")
        if not record.eligible or record.rejection_reasons:
            raise ValueError("candidate is not eligible for training")
        ids = [identity.strip() for identity in record.judge_ids]
        if len(ids) < 2 or not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("at least two unique Judge identities are required")
        dataset_ids = record.lineage.get("dataset_scene_ids")
        if (
            not isinstance(dataset_ids, list)
            or not dataset_ids
            or not all(isinstance(item, str) and item for item in dataset_ids)
            or len(set(dataset_ids)) != len(dataset_ids)
            or set(dataset_ids) != set(assignments)
        ):
            raise ValueError("dataset split metadata must be closed over all scene IDs")
        assignment = assignments.get(record.scene_id)
        if assignment is None or (
            assignment.split,
            assignment.canonical_scene_id,
            assignment.group_id,
        ) != (record.split, record.canonical_scene_id, record.group_id):
            raise ValueError("candidate canonical/group metadata does not match split metadata")
        confirmation = confirmation_map.get(key)
        if confirmation is None:
            raise ValueError("real human confirmation required")
        if not confirmation.confirmed or confirmation.synthetic:
            raise ValueError("real confirmed human evidence required")
        for field in [
            "scene_id",
            "candidate_id",
            "canonical_scene_id",
            "group_id",
            "source_sha256",
            "candidate_sha256",
        ]:
            if getattr(confirmation, field) != getattr(record, field):
                raise ValueError(f"stale human confirmation: {field} mismatch")
        for field, hash_field in [
            ("source_path", "source_sha256"),
            ("candidate_path", "candidate_sha256"),
        ]:
            if file_sha256(Path(getattr(record, field))) != getattr(record, hash_field):
                raise ValueError(f"confirmed {field} bytes changed")
        exported.append(
            {
                **record.model_dump(mode="json"),
                "record_kind": "human_confirmed_training_label",
                "label_scope": confirmation.label_scope,
                "training_weight": 1.0,
                "confirmation": confirmation.model_dump(mode="json"),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_manifest.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in exported),
        encoding="utf-8",
    )
    summary = {"schema_version": "tmqa.training_export.v1", "training_count": len(exported)}
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
