from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from portrait_eval.core.models_v2 import (
    DeviceDimensionScoreV2,
    DeviceOverallScoreV2,
    DimensionApplicabilityV2,
    DimensionId,
    EvaluationBatchV2,
    FactCheckResultV2,
    JudgeDecisionType,
    JudgeDecisionV2,
    MatchedSceneGroupV2,
    MechanismInterpretationV2,
    ObjectiveEvidenceV2,
    PairwiseComparisonV2,
    ReportEvidencePackageV2,
    RoughRankingV2,
    SceneDimensionScoreV2,
    SceneScoreStatus,
)
from portrait_eval.database import ProjectRow
from portrait_eval.persistence_v2 import (
    DeviceDimensionScoreRowV2,
    DeviceOverallScoreRowV2,
    DimensionApplicabilityRowV2,
    EvaluationBatchRowV2,
    FactCheckAttemptRowV2,
    JudgeDecisionAttemptRowV2,
    MatchedSceneGroupRowV2,
    MechanismInterpretationRowV2,
    ObjectiveEvidenceRowV2,
    PairwiseComparisonAttemptRowV2,
    ReportEvidencePackageRowV2,
    RoughRankingRowV2,
    SceneDimensionScoreRowV2,
)


@dataclass(frozen=True, slots=True)
class StoredAttempt[AttemptPayloadT]:
    attempt_id: str
    batch_id: str
    comparison_id: str
    attempt_number: int
    parent_attempt_id: str | None
    source_attempt_ids: tuple[str, ...]
    payload: AttemptPayloadT
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredSceneDimensionScore:
    batch_id: str
    score: SceneDimensionScoreV2
    accepted_judge_attempt_ids: tuple[str, ...]


def _decode_string_tuple(value: str) -> tuple[str, ...]:
    raw: object = json.loads(value)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("stored string-list payload is invalid")
    return tuple(raw)


class V2Repository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _require_batch(self, batch_id: str) -> None:
        if self.session.get(EvaluationBatchRowV2, batch_id) is None:
            raise KeyError(batch_id)

    @staticmethod
    def _next_attempt_number(
        latest_attempt_id: str | None,
        latest_attempt_number: int | None,
        parent_attempt_id: str | None,
    ) -> int:
        if latest_attempt_id is None:
            if parent_attempt_id is not None:
                raise ValueError("first attempt cannot have parent")
            return 1
        if parent_attempt_id is None:
            raise ValueError("retry requires parent attempt")
        if parent_attempt_id != latest_attempt_id:
            raise ValueError("parent attempt is not latest")
        if latest_attempt_number is None:
            raise RuntimeError("latest attempt number is missing")
        return latest_attempt_number + 1

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

    def save_matched_scene_group(self, group: MatchedSceneGroupV2) -> MatchedSceneGroupV2:
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

    def get_matched_scene_group(self, batch_id: str, scene_id: str) -> MatchedSceneGroupV2:
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

    def get_objective_evidence(self, batch_id: str, image_id: str) -> ObjectiveEvidenceV2:
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

    def save_rough_ranking(self, batch_id: str, ranking: RoughRankingV2) -> RoughRankingV2:
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

    def _latest_pairwise_row(
        self, batch_id: str, comparison_id: str
    ) -> PairwiseComparisonAttemptRowV2 | None:
        statement = (
            select(PairwiseComparisonAttemptRowV2)
            .where(
                PairwiseComparisonAttemptRowV2.batch_id == batch_id,
                PairwiseComparisonAttemptRowV2.comparison_id == comparison_id,
            )
            .order_by(PairwiseComparisonAttemptRowV2.attempt_number.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    @staticmethod
    def _stored_pairwise(
        row: PairwiseComparisonAttemptRowV2,
    ) -> StoredAttempt[PairwiseComparisonV2]:
        return StoredAttempt(
            attempt_id=row.attempt_id,
            batch_id=row.batch_id,
            comparison_id=row.comparison_id,
            attempt_number=row.attempt_number,
            parent_attempt_id=row.parent_attempt_id,
            source_attempt_ids=(),
            payload=PairwiseComparisonV2.model_validate_json(row.payload_json),
            created_at=row.created_at,
        )

    def append_pairwise_attempt(
        self,
        batch_id: str,
        comparison: PairwiseComparisonV2,
        *,
        parent_attempt_id: str | None = None,
    ) -> StoredAttempt[PairwiseComparisonV2]:
        self._require_batch(batch_id)
        latest = self._latest_pairwise_row(batch_id, comparison.comparison_id)
        attempt_number = self._next_attempt_number(
            latest.attempt_id if latest is not None else None,
            latest.attempt_number if latest is not None else None,
            parent_attempt_id,
        )
        row = PairwiseComparisonAttemptRowV2(
            attempt_id=str(uuid4()),
            batch_id=batch_id,
            comparison_id=comparison.comparison_id,
            attempt_number=attempt_number,
            parent_attempt_id=parent_attempt_id,
            schema_version=comparison.schema_version,
            primary_model_version=comparison.primary_model_version,
            prompt_version=comparison.prompt_version,
            payload_json=comparison.model_dump_json(),
        )
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return self._stored_pairwise(row)

    def get_pairwise_attempt(self, attempt_id: str) -> StoredAttempt[PairwiseComparisonV2]:
        row = self.session.get(PairwiseComparisonAttemptRowV2, attempt_id)
        if row is None:
            raise KeyError(attempt_id)
        return self._stored_pairwise(row)

    def list_pairwise_attempts(
        self, batch_id: str, comparison_id: str
    ) -> list[StoredAttempt[PairwiseComparisonV2]]:
        statement = (
            select(PairwiseComparisonAttemptRowV2)
            .where(
                PairwiseComparisonAttemptRowV2.batch_id == batch_id,
                PairwiseComparisonAttemptRowV2.comparison_id == comparison_id,
            )
            .order_by(PairwiseComparisonAttemptRowV2.attempt_number)
        )
        return [self._stored_pairwise(row) for row in self.session.scalars(statement).all()]

    def get_latest_pairwise_attempt(
        self, batch_id: str, comparison_id: str
    ) -> StoredAttempt[PairwiseComparisonV2]:
        row = self._latest_pairwise_row(batch_id, comparison_id)
        if row is None:
            raise KeyError((batch_id, comparison_id))
        return self._stored_pairwise(row)

    def _latest_fact_check_row(
        self, batch_id: str, comparison_id: str
    ) -> FactCheckAttemptRowV2 | None:
        statement = (
            select(FactCheckAttemptRowV2)
            .where(
                FactCheckAttemptRowV2.batch_id == batch_id,
                FactCheckAttemptRowV2.comparison_id == comparison_id,
            )
            .order_by(FactCheckAttemptRowV2.attempt_number.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    @staticmethod
    def _stored_fact_check(row: FactCheckAttemptRowV2) -> StoredAttempt[FactCheckResultV2]:
        return StoredAttempt(
            attempt_id=row.attempt_id,
            batch_id=row.batch_id,
            comparison_id=row.comparison_id,
            attempt_number=row.attempt_number,
            parent_attempt_id=row.parent_attempt_id,
            source_attempt_ids=(row.source_pairwise_attempt_id,),
            payload=FactCheckResultV2.model_validate_json(row.payload_json),
            created_at=row.created_at,
        )

    def append_fact_check_attempt(
        self,
        batch_id: str,
        result: FactCheckResultV2,
        *,
        source_pairwise_attempt_id: str,
        parent_attempt_id: str | None = None,
    ) -> StoredAttempt[FactCheckResultV2]:
        self._require_batch(batch_id)
        source = self.session.get(PairwiseComparisonAttemptRowV2, source_pairwise_attempt_id)
        if source is None:
            raise KeyError(source_pairwise_attempt_id)
        if source.batch_id != batch_id or source.comparison_id != result.comparison_id:
            raise ValueError("source pairwise attempt does not match comparison")
        latest = self._latest_fact_check_row(batch_id, result.comparison_id)
        attempt_number = self._next_attempt_number(
            latest.attempt_id if latest is not None else None,
            latest.attempt_number if latest is not None else None,
            parent_attempt_id,
        )
        row = FactCheckAttemptRowV2(
            attempt_id=str(uuid4()),
            batch_id=batch_id,
            comparison_id=result.comparison_id,
            attempt_number=attempt_number,
            parent_attempt_id=parent_attempt_id,
            source_pairwise_attempt_id=source_pairwise_attempt_id,
            schema_version=result.schema_version,
            rule_version=result.rule_version,
            payload_json=result.model_dump_json(),
        )
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return self._stored_fact_check(row)

    def get_fact_check_attempt(self, attempt_id: str) -> StoredAttempt[FactCheckResultV2]:
        row = self.session.get(FactCheckAttemptRowV2, attempt_id)
        if row is None:
            raise KeyError(attempt_id)
        return self._stored_fact_check(row)

    def list_fact_check_attempts(
        self, batch_id: str, comparison_id: str
    ) -> list[StoredAttempt[FactCheckResultV2]]:
        statement = (
            select(FactCheckAttemptRowV2)
            .where(
                FactCheckAttemptRowV2.batch_id == batch_id,
                FactCheckAttemptRowV2.comparison_id == comparison_id,
            )
            .order_by(FactCheckAttemptRowV2.attempt_number)
        )
        return [self._stored_fact_check(row) for row in self.session.scalars(statement).all()]

    def get_latest_fact_check_attempt(
        self, batch_id: str, comparison_id: str
    ) -> StoredAttempt[FactCheckResultV2]:
        row = self._latest_fact_check_row(batch_id, comparison_id)
        if row is None:
            raise KeyError((batch_id, comparison_id))
        return self._stored_fact_check(row)

    def _latest_judge_row(
        self, batch_id: str, comparison_id: str
    ) -> JudgeDecisionAttemptRowV2 | None:
        statement = (
            select(JudgeDecisionAttemptRowV2)
            .where(
                JudgeDecisionAttemptRowV2.batch_id == batch_id,
                JudgeDecisionAttemptRowV2.comparison_id == comparison_id,
            )
            .order_by(JudgeDecisionAttemptRowV2.attempt_number.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    @staticmethod
    def _stored_judge(row: JudgeDecisionAttemptRowV2) -> StoredAttempt[JudgeDecisionV2]:
        return StoredAttempt(
            attempt_id=row.attempt_id,
            batch_id=row.batch_id,
            comparison_id=row.comparison_id,
            attempt_number=row.attempt_number,
            parent_attempt_id=row.parent_attempt_id,
            source_attempt_ids=(
                row.source_pairwise_attempt_id,
                row.source_fact_check_attempt_id,
            ),
            payload=JudgeDecisionV2.model_validate_json(row.payload_json),
            created_at=row.created_at,
        )

    def append_judge_attempt(
        self,
        batch_id: str,
        decision: JudgeDecisionV2,
        *,
        source_pairwise_attempt_id: str,
        source_fact_check_attempt_id: str,
        parent_attempt_id: str | None = None,
    ) -> StoredAttempt[JudgeDecisionV2]:
        self._require_batch(batch_id)
        pairwise = self.session.get(PairwiseComparisonAttemptRowV2, source_pairwise_attempt_id)
        if pairwise is None:
            raise KeyError(source_pairwise_attempt_id)
        if pairwise.batch_id != batch_id or pairwise.comparison_id != decision.comparison_id:
            raise ValueError("source pairwise attempt does not match comparison")
        fact_check = self.session.get(FactCheckAttemptRowV2, source_fact_check_attempt_id)
        if fact_check is None:
            raise KeyError(source_fact_check_attempt_id)
        if fact_check.batch_id != batch_id or fact_check.comparison_id != decision.comparison_id:
            raise ValueError("source fact-check attempt does not match comparison")
        if fact_check.source_pairwise_attempt_id != source_pairwise_attempt_id:
            raise ValueError("fact-check lineage does not match pairwise attempt")
        latest = self._latest_judge_row(batch_id, decision.comparison_id)
        attempt_number = self._next_attempt_number(
            latest.attempt_id if latest is not None else None,
            latest.attempt_number if latest is not None else None,
            parent_attempt_id,
        )
        row = JudgeDecisionAttemptRowV2(
            attempt_id=str(uuid4()),
            batch_id=batch_id,
            comparison_id=decision.comparison_id,
            attempt_number=attempt_number,
            parent_attempt_id=parent_attempt_id,
            source_pairwise_attempt_id=source_pairwise_attempt_id,
            source_fact_check_attempt_id=source_fact_check_attempt_id,
            schema_version=decision.schema_version,
            judge_model_version=decision.judge_model_version,
            prompt_version=decision.prompt_version,
            payload_json=decision.model_dump_json(),
        )
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return self._stored_judge(row)

    def get_judge_attempt(self, attempt_id: str) -> StoredAttempt[JudgeDecisionV2]:
        row = self.session.get(JudgeDecisionAttemptRowV2, attempt_id)
        if row is None:
            raise KeyError(attempt_id)
        return self._stored_judge(row)

    def list_judge_attempts(
        self, batch_id: str, comparison_id: str
    ) -> list[StoredAttempt[JudgeDecisionV2]]:
        statement = (
            select(JudgeDecisionAttemptRowV2)
            .where(
                JudgeDecisionAttemptRowV2.batch_id == batch_id,
                JudgeDecisionAttemptRowV2.comparison_id == comparison_id,
            )
            .order_by(JudgeDecisionAttemptRowV2.attempt_number)
        )
        return [self._stored_judge(row) for row in self.session.scalars(statement).all()]

    def get_latest_judge_attempt(
        self, batch_id: str, comparison_id: str
    ) -> StoredAttempt[JudgeDecisionV2]:
        row = self._latest_judge_row(batch_id, comparison_id)
        if row is None:
            raise KeyError((batch_id, comparison_id))
        return self._stored_judge(row)

    def save_scene_dimension_score(
        self,
        batch_id: str,
        score: SceneDimensionScoreV2,
        *,
        accepted_judge_attempt_ids: list[str] | tuple[str, ...] = (),
    ) -> StoredSceneDimensionScore:
        self._require_batch(batch_id)
        judge_ids = tuple(accepted_judge_attempt_ids)
        if len(set(judge_ids)) != len(judge_ids):
            raise ValueError("duplicate judge attempt ID")
        if score.status is SceneScoreStatus.NOT_APPLICABLE and judge_ids:
            raise ValueError("NOT_APPLICABLE cannot reference judge attempts")
        contribution_ids = set(score.contributing_comparison_ids)
        accepted_comparison_ids: list[str] = []
        for attempt_id in judge_ids:
            row = self.session.get(JudgeDecisionAttemptRowV2, attempt_id)
            if row is None:
                raise KeyError(attempt_id)
            if row.batch_id != batch_id:
                raise ValueError("judge attempt does not match batch")
            decision = JudgeDecisionV2.model_validate_json(row.payload_json)
            if decision.decision not in {
                JudgeDecisionType.ACCEPT,
                JudgeDecisionType.REVISE,
            }:
                raise ValueError("judge attempt is not accepted")
            if row.comparison_id not in contribution_ids:
                raise ValueError("judge attempt does not match contributing comparison")
            accepted_comparison_ids.append(row.comparison_id)
        if set(accepted_comparison_ids) != contribution_ids:
            raise ValueError("accepted judge attempts must cover contributing comparisons")
        if len(accepted_comparison_ids) != len(set(accepted_comparison_ids)):
            raise ValueError("multiple judge attempts reference one contributing comparison")
        key = (batch_id, score.scene_id, score.dimension_id.value, score.device_id)
        if self.session.get(SceneDimensionScoreRowV2, key) is not None:
            raise ValueError("scene dimension score already exists")
        self.session.add(
            SceneDimensionScoreRowV2(
                batch_id=batch_id,
                scene_id=score.scene_id,
                dimension_id=score.dimension_id.value,
                device_id=score.device_id,
                schema_version=score.schema_version,
                status=score.status.value,
                accepted_judge_attempt_ids_json=json.dumps(judge_ids),
                payload_json=score.model_dump_json(),
            )
        )
        self.session.commit()
        return StoredSceneDimensionScore(
            batch_id=batch_id,
            score=score,
            accepted_judge_attempt_ids=judge_ids,
        )

    def get_scene_dimension_score(
        self,
        batch_id: str,
        scene_id: str,
        dimension_id: DimensionId,
        device_id: str,
    ) -> StoredSceneDimensionScore:
        row = self.session.get(
            SceneDimensionScoreRowV2,
            (batch_id, scene_id, dimension_id.value, device_id),
        )
        if row is None:
            raise KeyError((batch_id, scene_id, dimension_id.value, device_id))
        return StoredSceneDimensionScore(
            batch_id=batch_id,
            score=SceneDimensionScoreV2.model_validate_json(row.payload_json),
            accepted_judge_attempt_ids=_decode_string_tuple(
                row.accepted_judge_attempt_ids_json
            ),
        )

    def save_device_dimension_score(
        self, score: DeviceDimensionScoreV2
    ) -> DeviceDimensionScoreV2:
        self._require_batch(score.batch_id)
        key = (score.batch_id, score.device_id, score.dimension_id.value)
        if self.session.get(DeviceDimensionScoreRowV2, key) is not None:
            raise ValueError("device dimension score already exists")
        self.session.add(
            DeviceDimensionScoreRowV2(
                batch_id=score.batch_id,
                device_id=score.device_id,
                dimension_id=score.dimension_id.value,
                schema_version=score.schema_version,
                payload_json=score.model_dump_json(),
            )
        )
        self.session.commit()
        return score

    def get_device_dimension_score(
        self,
        batch_id: str,
        device_id: str,
        dimension_id: DimensionId,
    ) -> DeviceDimensionScoreV2:
        row = self.session.get(
            DeviceDimensionScoreRowV2,
            (batch_id, device_id, dimension_id.value),
        )
        if row is None:
            raise KeyError((batch_id, device_id, dimension_id.value))
        return DeviceDimensionScoreV2.model_validate_json(row.payload_json)

    def save_device_overall_score(
        self, score: DeviceOverallScoreV2
    ) -> DeviceOverallScoreV2:
        self._require_batch(score.batch_id)
        key = (score.batch_id, score.device_id)
        if self.session.get(DeviceOverallScoreRowV2, key) is not None:
            raise ValueError("device overall score already exists")
        self.session.add(
            DeviceOverallScoreRowV2(
                batch_id=score.batch_id,
                device_id=score.device_id,
                schema_version=score.schema_version,
                payload_json=score.model_dump_json(),
            )
        )
        self.session.commit()
        return score

    def get_device_overall_score(
        self, batch_id: str, device_id: str
    ) -> DeviceOverallScoreV2:
        row = self.session.get(DeviceOverallScoreRowV2, (batch_id, device_id))
        if row is None:
            raise KeyError((batch_id, device_id))
        return DeviceOverallScoreV2.model_validate_json(row.payload_json)

    def save_mechanism_interpretation(
        self,
        batch_id: str,
        interpretation: MechanismInterpretationV2,
    ) -> MechanismInterpretationV2:
        self._require_batch(batch_id)
        key = (batch_id, interpretation.interpretation_id)
        if self.session.get(MechanismInterpretationRowV2, key) is not None:
            raise ValueError("mechanism interpretation already exists")
        self.session.add(
            MechanismInterpretationRowV2(
                batch_id=batch_id,
                interpretation_id=interpretation.interpretation_id,
                device_id=interpretation.device_id,
                schema_version=interpretation.schema_version,
                attribution=interpretation.attribution.value,
                payload_json=interpretation.model_dump_json(),
            )
        )
        self.session.commit()
        return interpretation

    def get_mechanism_interpretation(
        self, batch_id: str, interpretation_id: str
    ) -> MechanismInterpretationV2:
        row = self.session.get(
            MechanismInterpretationRowV2,
            (batch_id, interpretation_id),
        )
        if row is None:
            raise KeyError((batch_id, interpretation_id))
        return MechanismInterpretationV2.model_validate_json(row.payload_json)

    def save_report_evidence_package(
        self,
        package_id: str,
        package: ReportEvidencePackageV2,
    ) -> ReportEvidencePackageV2:
        if self.session.get(ReportEvidencePackageRowV2, package_id) is not None:
            raise ValueError("report evidence package already exists")
        batch_id = package.batch.batch_id
        self._require_batch(batch_id)
        if self.get_evaluation_batch(batch_id) != package.batch:
            raise ValueError("report package batch does not match persisted batch")
        for expected in package.device_dimension_scores:
            key = (batch_id, expected.device_id, expected.dimension_id.value)
            row = self.session.get(DeviceDimensionScoreRowV2, key)
            if row is None or DeviceDimensionScoreV2.model_validate_json(
                row.payload_json
            ) != expected:
                raise ValueError(
                    "report package references unpersisted device dimension score"
                )
        for expected in package.device_overall_scores:
            row = self.session.get(
                DeviceOverallScoreRowV2,
                (batch_id, expected.device_id),
            )
            if row is None or DeviceOverallScoreV2.model_validate_json(
                row.payload_json
            ) != expected:
                raise ValueError(
                    "report package references unpersisted device overall score"
                )
        for expected in package.mechanism_interpretations:
            row = self.session.get(
                MechanismInterpretationRowV2,
                (batch_id, expected.interpretation_id),
            )
            if row is None or MechanismInterpretationV2.model_validate_json(
                row.payload_json
            ) != expected:
                raise ValueError(
                    "report package references unpersisted mechanism interpretation"
                )
        self.session.add(
            ReportEvidencePackageRowV2(
                package_id=package_id,
                batch_id=batch_id,
                schema_version=package.schema_version,
                payload_json=package.model_dump_json(),
            )
        )
        self.session.commit()
        return package

    def get_report_evidence_package(self, package_id: str) -> ReportEvidencePackageV2:
        row = self.session.get(ReportEvidencePackageRowV2, package_id)
        if row is None:
            raise KeyError(package_id)
        return ReportEvidencePackageV2.model_validate_json(row.payload_json)
