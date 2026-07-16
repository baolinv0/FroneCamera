from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from portrait_eval.database import Base


def _stored_at() -> datetime:
    return datetime.now(UTC)


class EvaluationBatchRowV2(Base):
    __tablename__ = "evaluation_batches_v2"

    batch_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    schema_version: Mapped[str] = mapped_column(String(20))
    dataset_version: Mapped[str] = mapped_column(String(100))
    dimension_policy_version: Mapped[str] = mapped_column(String(100))
    scoring_policy_version: Mapped[str] = mapped_column(String(100))
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MatchedSceneGroupRowV2(Base):
    __tablename__ = "matched_scene_groups_v2"

    batch_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_batches_v2.batch_id", ondelete="CASCADE"),
        primary_key=True,
    )
    scene_id: Mapped[str] = mapped_column(String, primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(20))
    match_status: Mapped[str] = mapped_column(String(50), index=True)
    comparability_status: Mapped[str] = mapped_column(String(50), index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    stored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_stored_at, index=True
    )


class ObjectiveEvidenceRowV2(Base):
    __tablename__ = "objective_evidence_v2"

    batch_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_batches_v2.batch_id", ondelete="CASCADE"),
        primary_key=True,
    )
    image_id: Mapped[str] = mapped_column(String, primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(20))
    metric_version: Mapped[str] = mapped_column(String(100), index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    stored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_stored_at, index=True
    )


class DimensionApplicabilityRowV2(Base):
    __tablename__ = "dimension_applicability_v2"

    batch_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_batches_v2.batch_id", ondelete="CASCADE"),
        primary_key=True,
    )
    scene_id: Mapped[str] = mapped_column(String, primary_key=True)
    dimension_id: Mapped[str] = mapped_column(String, primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(50), index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    stored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_stored_at, index=True
    )


class RoughRankingRowV2(Base):
    __tablename__ = "rough_rankings_v2"

    batch_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_batches_v2.batch_id", ondelete="CASCADE"),
        primary_key=True,
    )
    scene_id: Mapped[str] = mapped_column(String, primary_key=True)
    dimension_id: Mapped[str] = mapped_column(String, primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(20))
    prompt_version: Mapped[str] = mapped_column(String(100), index=True)
    model_version: Mapped[str] = mapped_column(String(100), index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    stored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_stored_at, index=True
    )
