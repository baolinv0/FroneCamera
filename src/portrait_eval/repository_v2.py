from __future__ import annotations

from sqlalchemy.orm import Session

from portrait_eval.core.models_v2 import (
    DimensionApplicabilityV2,
    DimensionId,
    EvaluationBatchV2,
    MatchedSceneGroupV2,
    ObjectiveEvidenceV2,
    RoughRankingV2,
)
from portrait_eval.database import ProjectRow
from portrait_eval.persistence_v2 import (
    DimensionApplicabilityRowV2,
    EvaluationBatchRowV2,
    MatchedSceneGroupRowV2,
    ObjectiveEvidenceRowV2,
    RoughRankingRowV2,
)


class V2Repository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _require_batch(self, batch_id: str) -> None:
        if self.session.get(EvaluationBatchRowV2, batch_id) is None:
            raise KeyError(batch_id)

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

    def save_matched_scene_group(
        self, group: MatchedSceneGroupV2
    ) -> MatchedSceneGroupV2:
        self._require_batch(group.batch_id)
        key = (group.batch_id, group.scene_id)
        if self.session.get(MatchedSceneGroupRowV2, key) is not None:
            raise ValueError("matched scene group already exists")

        self.session.add(
            MatchedSceneGroupRowV2(
                batch_id=group.batch_id,
                scene_id=group.scene_id,
                schema_version=group.schema_version,
                match_status=group.match_status.value,
                comparability_status=group.comparability_status.value,
                payload_json=group.model_dump_json(),
            )
        )
        self.session.commit()
        return group

    def get_matched_scene_group(
        self, batch_id: str, scene_id: str
    ) -> MatchedSceneGroupV2:
        row = self.session.get(MatchedSceneGroupRowV2, (batch_id, scene_id))
        if row is None:
            raise KeyError((batch_id, scene_id))
        return MatchedSceneGroupV2.model_validate_json(row.payload_json)

    def save_objective_evidence(
        self, batch_id: str, evidence: ObjectiveEvidenceV2
    ) -> ObjectiveEvidenceV2:
        self._require_batch(batch_id)
        key = (batch_id, evidence.image_id)
        if self.session.get(ObjectiveEvidenceRowV2, key) is not None:
            raise ValueError("objective evidence already exists")

        self.session.add(
            ObjectiveEvidenceRowV2(
                batch_id=batch_id,
                image_id=evidence.image_id,
                schema_version=evidence.schema_version,
                metric_version=evidence.metric_version,
                payload_json=evidence.model_dump_json(),
            )
        )
        self.session.commit()
        return evidence

    def get_objective_evidence(
        self, batch_id: str, image_id: str
    ) -> ObjectiveEvidenceV2:
        row = self.session.get(ObjectiveEvidenceRowV2, (batch_id, image_id))
        if row is None:
            raise KeyError((batch_id, image_id))
        return ObjectiveEvidenceV2.model_validate_json(row.payload_json)

    def save_dimension_applicability(
        self,
        batch_id: str,
        applicability: DimensionApplicabilityV2,
    ) -> DimensionApplicabilityV2:
        self._require_batch(batch_id)
        key = (batch_id, applicability.scene_id, applicability.dimension_id.value)
        if self.session.get(DimensionApplicabilityRowV2, key) is not None:
            raise ValueError("dimension applicability already exists")

        self.session.add(
            DimensionApplicabilityRowV2(
                batch_id=batch_id,
                scene_id=applicability.scene_id,
                dimension_id=applicability.dimension_id.value,
                schema_version=applicability.schema_version,
                status=applicability.status.value,
                payload_json=applicability.model_dump_json(),
            )
        )
        self.session.commit()
        return applicability

    def get_dimension_applicability(
        self,
        batch_id: str,
        scene_id: str,
        dimension_id: DimensionId,
    ) -> DimensionApplicabilityV2:
        row = self.session.get(
            DimensionApplicabilityRowV2,
            (batch_id, scene_id, dimension_id.value),
        )
        if row is None:
            raise KeyError((batch_id, scene_id, dimension_id.value))
        return DimensionApplicabilityV2.model_validate_json(row.payload_json)

    def save_rough_ranking(
        self, batch_id: str, ranking: RoughRankingV2
    ) -> RoughRankingV2:
        self._require_batch(batch_id)
        key = (batch_id, ranking.scene_id, ranking.dimension_id.value)
        if self.session.get(RoughRankingRowV2, key) is not None:
            raise ValueError("rough ranking already exists")

        self.session.add(
            RoughRankingRowV2(
                batch_id=batch_id,
                scene_id=ranking.scene_id,
                dimension_id=ranking.dimension_id.value,
                schema_version=ranking.schema_version,
                prompt_version=ranking.prompt_version,
                model_version=ranking.model_version,
                payload_json=ranking.model_dump_json(),
            )
        )
        self.session.commit()
        return ranking

    def get_rough_ranking(
        self,
        batch_id: str,
        scene_id: str,
        dimension_id: DimensionId,
    ) -> RoughRankingV2:
        row = self.session.get(
            RoughRankingRowV2,
            (batch_id, scene_id, dimension_id.value),
        )
        if row is None:
            raise KeyError((batch_id, scene_id, dimension_id.value))
        return RoughRankingV2.model_validate_json(row.payload_json)
