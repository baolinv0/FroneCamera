from __future__ import annotations

from sqlalchemy import select

from portrait_eval.core.models_v2 import (
    DeviceDimensionScoreV2,
    DeviceOverallScoreV2,
    DimensionApplicabilityV2,
    EvaluationBatchV2,
    MatchedSceneGroupV2,
    MechanismInterpretationV2,
    ObjectiveEvidenceV2,
    PairwiseComparisonV2,
    RoughRankingV2,
    SceneDimensionScoreV2,
)
from portrait_eval.database import DeviceRow, ImageRow
from portrait_eval.persistence_v2 import (
    FactCheckAttemptRowV2,
    JudgeDecisionAttemptRowV2,
    MatchedSceneGroupRowV2,
    PairwiseComparisonAttemptRowV2,
)
from portrait_eval.repository_v2 import (
    StoredAttempt,
    StoredSceneDimensionScore,
    V2Repository as BaseV2Repository,
)


class V2Repository(BaseV2Repository):
    """V2 persistence facade with batch-graph and lineage integrity gates."""

    def _batch_contract(self, batch_id: str) -> EvaluationBatchV2:
        return self.get_evaluation_batch(batch_id)

    def _batch_device_ids(self, batch_id: str) -> set[str]:
        return set(self._batch_contract(batch_id).device_ids)

    def _require_batch_device(
        self,
        batch_id: str,
        device_id: str,
        *,
        message: str = "device does not belong to batch",
    ) -> DeviceRow:
        batch = self._batch_contract(batch_id)
        device = self.session.get(DeviceRow, device_id)
        if (
            device is None
            or device_id not in batch.device_ids
            or device.project_id != batch.project_id
        ):
            raise ValueError(message)
        return device

    def _require_batch_image(
        self,
        batch_id: str,
        image_id: str,
        *,
        expected_device_id: str | None = None,
    ) -> ImageRow:
        image = self.session.get(ImageRow, image_id)
        if image is None:
            raise ValueError("image does not belong to batch")
        if expected_device_id is not None and image.device_id != expected_device_id:
            raise ValueError("image does not belong to scene device")
        self._require_batch_device(batch_id, image.device_id, message="image does not belong to batch")
        return image

    def _require_scene(self, batch_id: str, scene_id: str) -> MatchedSceneGroupRowV2:
        row = self.session.get(MatchedSceneGroupRowV2, (batch_id, scene_id))
        if row is None:
            raise KeyError((batch_id, scene_id))
        return row

    def save_evaluation_batch(self, batch: EvaluationBatchV2) -> EvaluationBatchV2:
        devices = list(
            self.session.scalars(select(DeviceRow).where(DeviceRow.id.in_(batch.device_ids)))
        )
        if (
            {device.id for device in devices} != set(batch.device_ids)
            or any(device.project_id != batch.project_id for device in devices)
        ):
            raise ValueError("batch devices must belong to project")
        return super().save_evaluation_batch(batch)

    def save_matched_scene_group(
        self, group: MatchedSceneGroupV2
    ) -> MatchedSceneGroupV2:
        batch_devices = self._batch_device_ids(group.batch_id)
        if set(group.cells) != batch_devices:
            raise ValueError("matched scene devices must equal batch devices")
        for device_id, image_id in group.cells.items():
            self._require_batch_device(group.batch_id, device_id)
            if image_id is not None:
                self._require_batch_image(
                    group.batch_id,
                    image_id,
                    expected_device_id=device_id,
                )
        return super().save_matched_scene_group(group)

    def save_objective_evidence(
        self, batch_id: str, evidence: ObjectiveEvidenceV2
    ) -> ObjectiveEvidenceV2:
        self._require_batch_image(batch_id, evidence.image_id)
        return super().save_objective_evidence(batch_id, evidence)

    def save_dimension_applicability(
        self,
        batch_id: str,
        applicability: DimensionApplicabilityV2,
    ) -> DimensionApplicabilityV2:
        self._require_scene(batch_id, applicability.scene_id)
        return super().save_dimension_applicability(batch_id, applicability)

    def save_rough_ranking(
        self, batch_id: str, ranking: RoughRankingV2
    ) -> RoughRankingV2:
        self._require_scene(batch_id, ranking.scene_id)
        batch_devices = self._batch_device_ids(batch_id)
        ordered_devices = ranking.ordered_device_ids
        if len(ordered_devices) != len(set(ordered_devices)) or set(ordered_devices) != batch_devices:
            raise ValueError("rough ranking devices must equal batch devices")
        tie_devices = [device_id for group in ranking.tie_groups for device_id in group]
        if len(tie_devices) != len(set(tie_devices)) or not set(tie_devices).issubset(
            batch_devices
        ):
            raise ValueError("rough ranking tie groups are invalid")
        return super().save_rough_ranking(batch_id, ranking)

    def append_pairwise_attempt(
        self,
        batch_id: str,
        comparison: PairwiseComparisonV2,
        *,
        parent_attempt_id: str | None = None,
    ) -> StoredAttempt[PairwiseComparisonV2]:
        self._require_scene(batch_id, comparison.scene_id)
        self._require_batch_device(
            batch_id,
            comparison.device_a_id,
            message="pairwise device does not belong to batch",
        )
        self._require_batch_device(
            batch_id,
            comparison.device_b_id,
            message="pairwise device does not belong to batch",
        )
        return super().append_pairwise_attempt(
            batch_id,
            comparison,
            parent_attempt_id=parent_attempt_id,
        )

    def _validate_score_judge_lineage(
        self,
        batch_id: str,
        score: SceneDimensionScoreV2,
        judge_attempt_id: str,
    ) -> None:
        judge_row = self.session.get(JudgeDecisionAttemptRowV2, judge_attempt_id)
        if judge_row is None:
            raise KeyError(judge_attempt_id)
        latest_judge = self._latest_judge_row(batch_id, judge_row.comparison_id)
        if latest_judge is None or latest_judge.attempt_id != judge_attempt_id:
            raise ValueError("judge attempt is not latest")

        pairwise_row = self.session.get(
            PairwiseComparisonAttemptRowV2,
            judge_row.source_pairwise_attempt_id,
        )
        fact_row = self.session.get(
            FactCheckAttemptRowV2,
            judge_row.source_fact_check_attempt_id,
        )
        if pairwise_row is None or fact_row is None:
            raise ValueError("judge lineage is incomplete")
        latest_pairwise = self._latest_pairwise_row(batch_id, judge_row.comparison_id)
        latest_fact = self._latest_fact_check_row(batch_id, judge_row.comparison_id)
        if (
            latest_pairwise is None
            or latest_pairwise.attempt_id != pairwise_row.attempt_id
            or latest_fact is None
            or latest_fact.attempt_id != fact_row.attempt_id
        ):
            raise ValueError("judge lineage is not current")

        comparison = PairwiseComparisonV2.model_validate_json(pairwise_row.payload_json)
        if (
            comparison.scene_id != score.scene_id
            or comparison.dimension_id != score.dimension_id
            or score.device_id
            not in {comparison.device_a_id, comparison.device_b_id}
        ):
            raise ValueError("judge attempt does not match score context")

    def save_scene_dimension_score(
        self,
        batch_id: str,
        score: SceneDimensionScoreV2,
        *,
        accepted_judge_attempt_ids: list[str] | tuple[str, ...] = (),
    ) -> StoredSceneDimensionScore:
        self._require_scene(batch_id, score.scene_id)
        self._require_batch_device(batch_id, score.device_id)
        for judge_attempt_id in accepted_judge_attempt_ids:
            self._validate_score_judge_lineage(batch_id, score, judge_attempt_id)
        return super().save_scene_dimension_score(
            batch_id,
            score,
            accepted_judge_attempt_ids=accepted_judge_attempt_ids,
        )

    def save_device_dimension_score(
        self, score: DeviceDimensionScoreV2
    ) -> DeviceDimensionScoreV2:
        self._require_batch_device(score.batch_id, score.device_id)
        for scene_id in [*score.supporting_scene_ids, *score.counterexample_scene_ids]:
            self._require_scene(score.batch_id, scene_id)
        return super().save_device_dimension_score(score)

    def save_device_overall_score(
        self, score: DeviceOverallScoreV2
    ) -> DeviceOverallScoreV2:
        self._require_batch_device(score.batch_id, score.device_id)
        return super().save_device_overall_score(score)

    def save_mechanism_interpretation(
        self,
        batch_id: str,
        interpretation: MechanismInterpretationV2,
    ) -> MechanismInterpretationV2:
        self._require_batch_device(batch_id, interpretation.device_id)
        for scene_id in interpretation.scene_ids:
            self._require_scene(batch_id, scene_id)
        return super().save_mechanism_interpretation(batch_id, interpretation)


__all__ = ["StoredAttempt", "StoredSceneDimensionScore", "V2Repository"]
