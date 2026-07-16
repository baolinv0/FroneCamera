from __future__ import annotations

from sqlalchemy.orm import Session

from portrait_eval.core.models_v2 import EvaluationBatchV2
from portrait_eval.database import ProjectRow
from portrait_eval.persistence_v2 import EvaluationBatchRowV2


class V2Repository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save_evaluation_batch(self, batch: EvaluationBatchV2) -> EvaluationBatchV2:
        if self.session.get(ProjectRow, batch.project_id) is None:
            raise KeyError(batch.project_id)
        if self.session.get(EvaluationBatchRowV2, batch.batch_id) is not None:
            raise ValueError("batch already exists")

        self.session.add(
            EvaluationBatchRowV2(
                batch_id=batch.batch_id,
                project_id=batch.project_id,
                schema_version=batch.schema_version,
                dataset_version=batch.dataset_version,
                dimension_policy_version=batch.dimension_policy_version,
                scoring_policy_version=batch.scoring_policy_version,
                payload_json=batch.model_dump_json(),
                created_at=batch.created_at,
            )
        )
        self.session.commit()
        return batch

    def get_evaluation_batch(self, batch_id: str) -> EvaluationBatchV2:
        row = self.session.get(EvaluationBatchRowV2, batch_id)
        if row is None:
            raise KeyError(batch_id)
        return EvaluationBatchV2.model_validate_json(row.payload_json)
