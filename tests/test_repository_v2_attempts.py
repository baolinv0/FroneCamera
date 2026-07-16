from __future__ import annotations

from datetime import UTC, datetime

import pytest

from portrait_eval.core.models_v2 import (
    ApplicabilityStatus,
    DifferenceType,
    DimensionId,
    EvaluationBatchV2,
    FactCheckResultV2,
    JudgeDecisionType,
    JudgeDecisionV2,
    PairwiseComparisonV2,
    PairwisePreference,
)
from portrait_eval.database import Database
from portrait_eval.repository import Repository
from portrait_eval.repository_v2 import V2Repository


def _batch(project_id: str) -> EvaluationBatchV2:
    return EvaluationBatchV2(
        batch_id="batch-attempts",
        project_id=project_id,
        device_ids=["device-a", "device-b"],
        dataset_version="dataset-2026-07",
        dimension_policy_version="dimension-policy-v1",
        scoring_policy_version="scoring-policy-v1",
        primary_model_version="primary-v1",
        judge_model_version="judge-v1",
        report_model_version=None,
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
    )


def _comparison(
    comparison_id: str,
    observation: str,
    preference: PairwisePreference = PairwisePreference.A_SLIGHTLY_BETTER,
) -> PairwiseComparisonV2:
    strength = {
        PairwisePreference.A_STRONGLY_BETTER: 2,
        PairwisePreference.A_SLIGHTLY_BETTER: 1,
        PairwisePreference.EQUIVALENT: 0,
        PairwisePreference.B_SLIGHTLY_BETTER: -1,
        PairwisePreference.B_STRONGLY_BETTER: -2,
    }[preference]
    return PairwiseComparisonV2(
        comparison_id=comparison_id,
        scene_id="scene-001",
        dimension_id=DimensionId.FACE_EXPOSURE_READABILITY,
        device_a_id="device-a",
        device_b_id="device-b",
        applicability=ApplicabilityStatus.APPLICABLE,
        preference=preference,
        strength=strength,
        difference_type=DifferenceType.QUALITY,
        confidence=0.86,
        observations=[observation],
        evidence_refs=["objective:image-a", "objective:image-b"],
        style_axes={},
        primary_model_version="primary-v1",
        prompt_version="pairwise-prompt-v1",
    )


def _fact_check(comparison_id: str) -> FactCheckResultV2:
    return FactCheckResultV2(
        comparison_id=comparison_id,
        rule_version="fact-rules-v1",
        passed_rules=["face luma direction agrees"],
        failed_rules=[],
        warnings=[],
        critical_conflict=False,
        objective_support=0.91,
    )


def _judge(comparison_id: str) -> JudgeDecisionV2:
    return JudgeDecisionV2(
        comparison_id=comparison_id,
        decision=JudgeDecisionType.ACCEPT,
        accepted_preference=PairwisePreference.A_SLIGHTLY_BETTER,
        accepted_strength=1,
        accepted_difference_type=DifferenceType.QUALITY,
        revised_observations=["device A keeps the primary face more readable"],
        fact_conflicts=[],
        unsupported_attributions=[],
        confidence=0.89,
        judge_model_version="judge-v1",
        prompt_version="judge-prompt-v1",
    )


def _repository() -> tuple[Database, V2Repository]:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()
    session = database.session_factory()
    project = Repository(session).create_project("Immutable v2 attempts")
    repository = V2Repository(session)
    repository.save_evaluation_batch(_batch(project.id))
    return database, repository


def test_pairwise_retries_are_append_only_and_require_latest_parent() -> None:
    database, repository = _repository()
    try:
        first_payload = _comparison("comparison-001", "first observation")
        first = repository.append_pairwise_attempt("batch-attempts", first_payload)

        assert first.attempt_number == 1
        assert first.parent_attempt_id is None
        assert first.source_attempt_ids == ()
        assert first.payload == first_payload

        with pytest.raises(ValueError, match="retry requires parent attempt"):
            repository.append_pairwise_attempt(
                "batch-attempts",
                _comparison(
                    "comparison-001",
                    "revised without lineage",
                    PairwisePreference.EQUIVALENT,
                ),
            )

        second_payload = _comparison(
            "comparison-001",
            "revised with lineage",
            PairwisePreference.EQUIVALENT,
        )
        second = repository.append_pairwise_attempt(
            "batch-attempts",
            second_payload,
            parent_attempt_id=first.attempt_id,
        )

        assert second.attempt_number == 2
        assert second.parent_attempt_id == first.attempt_id
        assert repository.get_latest_pairwise_attempt(
            "batch-attempts", "comparison-001"
        ) == second
        assert repository.list_pairwise_attempts(
            "batch-attempts", "comparison-001"
        ) == [first, second]
        assert repository.get_pairwise_attempt(first.attempt_id).payload == first_payload

        with pytest.raises(ValueError, match="parent attempt is not latest"):
            repository.append_pairwise_attempt(
                "batch-attempts",
                _comparison("comparison-001", "stale retry"),
                parent_attempt_id=first.attempt_id,
            )
    finally:
        repository.session.close()
        database.engine.dispose()


def test_fact_check_and_judge_attempts_preserve_cross_stage_lineage() -> None:
    database, repository = _repository()
    try:
        pairwise = repository.append_pairwise_attempt(
            "batch-attempts",
            _comparison("comparison-001", "primary comparison"),
        )
        fact_payload = _fact_check("comparison-001")
        fact = repository.append_fact_check_attempt(
            "batch-attempts",
            fact_payload,
            source_pairwise_attempt_id=pairwise.attempt_id,
        )
        judge_payload = _judge("comparison-001")
        judge = repository.append_judge_attempt(
            "batch-attempts",
            judge_payload,
            source_pairwise_attempt_id=pairwise.attempt_id,
            source_fact_check_attempt_id=fact.attempt_id,
        )

        assert fact.source_attempt_ids == (pairwise.attempt_id,)
        assert fact.payload == fact_payload
        assert judge.source_attempt_ids == (pairwise.attempt_id, fact.attempt_id)
        assert judge.payload == judge_payload
        assert repository.get_fact_check_attempt(fact.attempt_id) == fact
        assert repository.get_judge_attempt(judge.attempt_id) == judge

        other_pairwise = repository.append_pairwise_attempt(
            "batch-attempts",
            _comparison("comparison-002", "different comparison"),
        )
        with pytest.raises(
            ValueError,
            match="source pairwise attempt does not match comparison",
        ):
            repository.append_fact_check_attempt(
                "batch-attempts",
                fact_payload,
                source_pairwise_attempt_id=other_pairwise.attempt_id,
            )
    finally:
        repository.session.close()
        database.engine.dispose()
