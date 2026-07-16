from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from portrait_eval.database import Base


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
