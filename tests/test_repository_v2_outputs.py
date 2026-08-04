from __future__ import annotations

from datetime import UTC, datetime

import pytest

from portrait_eval.core.models_v2 import (
    ApplicabilityStatus,
    DeviceDimensionScoreV2,
    DeviceOverallScoreV2,
    DifferenceType,
    DimensionId,
    EvaluationBatchV2,
    EvidenceGradeV2,
    FactCheckResultV2,
    JudgeDecisionType,
    JudgeDecisionV2,
    MechanismAttribution,
    MechanismInterpretationV2,
    PairwiseComparisonV2,
    PairwisePreference,
    ReportAllowedLevel,
    ReportEvidencePackageV2,
    SceneDimensionScoreV2,
    SceneScoreStatus,
)
from portrait_eval.database import Database
from portrait_eval.repository import Repository
from portrait_eval.repository_v2 import V2Repository


def _batch(project_id: str) -> EvaluationBatchV2:
    return EvaluationBatchV2(
        batch_id="batch-outputs",
        project_id=project_id,
        device_ids=["device-a", "device-b"],
        dataset_version="dataset-2026-07",
        dimension_policy_version="dimension-policy-v1",
        scoring_policy_version="scoring-policy-v1",
        primary_model_version="primary-v1",
        judge_model_version="judge-v1",
        report_model_version="report-v1",
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
    )


def _repository() -> tuple[Database, V2Repository, EvaluationBatchV2]:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()
    session = database.session_factory()
    project = Repository(session).create_project("V2 output persistence")
    batch = _batch(project.id)
    repository = V2Repository(session)
    repository.save_evaluation_batch(batch)
    return database, repository, batch


def _accepted_judge(repository: V2Repository, comparison_id: str):
    comparison = PairwiseComparisonV2(
        comparison_id=comparison_id,
        scene_id="scene-001",
        dimension_id=DimensionId.FACE_EXPOSURE_READABILITY,
        device_a_id="device-a",
        device_b_id="device-b",
        applicability=ApplicabilityStatus.APPLICABLE,
        preference=PairwisePreference.A_SLIGHTLY_BETTER,
        strength=1,
        difference_type=DifferenceType.QUALITY,
        confidence=0.86,
        observations=["device A keeps the primary face more readable"],
        evidence_refs=["objective:image-a", "objective:image-b"],
        style_axes={},
        primary_model_version="primary-v1",
        prompt_version="pairwise-prompt-v1",
    )
    pairwise = repository.append_pairwise_attempt("batch-outputs", comparison)
    fact = repository.append_fact_check_attempt(
        "batch-outputs",
        FactCheckResultV2(
            comparison_id=comparison_id,
            rule_version="fact-rules-v1",
            passed_rules=["face luma direction agrees"],
            failed_rules=[],
            warnings=[],
            critical_conflict=False,
            objective_support=0.91,
        ),
        source_pairwise_attempt_id=pairwise.attempt_id,
    )
    return repository.append_judge_attempt(
        "batch-outputs",
        JudgeDecisionV2(
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
        ),
        source_pairwise_attempt_id=pairwise.attempt_id,
        source_fact_check_attempt_id=fact.attempt_id,
    )


def test_scene_score_round_trips_with_accepted_judge_lineage() -> None:
    database, repository, _batch_value = _repository()
    try:
        judge = _accepted_judge(repository, "comparison-001")
        score = SceneDimensionScoreV2(
            scene_id="scene-001",
            dimension_id=DimensionId.FACE_EXPOSURE_READABILITY,
            device_id="device-a",
            latent_quality=0.72,
            normalized_score=78.0,
            confidence=0.88,
            diagnostic_weight=0.90,
            objective_adjustment=2.0,
            final_score=80.0,
            status=SceneScoreStatus.AUTO_PASS,
            contributing_comparison_ids=["comparison-001"],
        )

        stored = repository.save_scene_dimension_score(
            "batch-outputs",
            score,
            accepted_judge_attempt_ids=[judge.attempt_id],
        )

        assert stored.score == score
        assert stored.accepted_judge_attempt_ids == (judge.attempt_id,)
        assert (
            repository.get_scene_dimension_score(
                "batch-outputs",
                "scene-001",
                DimensionId.FACE_EXPOSURE_READABILITY,
                "device-a",
            )
            == stored
        )

        unrelated_judge = _accepted_judge(repository, "comparison-002")
        with pytest.raises(
            ValueError,
            match="judge attempt does not match contributing comparison",
        ):
            repository.save_scene_dimension_score(
                "batch-outputs",
                score.model_copy(update={"device_id": "device-b"}),
                accepted_judge_attempt_ids=[unrelated_judge.attempt_id],
            )
    finally:
        repository.session.close()
        database.engine.dispose()


def test_not_applicable_score_preserves_nulls_and_rejects_attempt_lineage() -> None:
    database, repository, _batch_value = _repository()
    try:
        score = SceneDimensionScoreV2(
            scene_id="scene-002",
            dimension_id=DimensionId.MULTI_FACE_CONSISTENCY,
            device_id="device-a",
            latent_quality=None,
            normalized_score=None,
            confidence=0.97,
            diagnostic_weight=0.0,
            objective_adjustment=None,
            final_score=None,
            status=SceneScoreStatus.NOT_APPLICABLE,
            contributing_comparison_ids=[],
        )
        stored = repository.save_scene_dimension_score("batch-outputs", score)

        assert stored.score.final_score is None
        assert stored.accepted_judge_attempt_ids == ()

        with pytest.raises(ValueError, match="NOT_APPLICABLE cannot reference judge attempts"):
            repository.save_scene_dimension_score(
                "batch-outputs",
                score.model_copy(update={"device_id": "device-b"}),
                accepted_judge_attempt_ids=["judge-attempt-not-allowed"],
            )
    finally:
        repository.session.close()
        database.engine.dispose()


def test_aggregate_mechanism_and_report_package_round_trip() -> None:
    database, repository, batch = _repository()
    try:
        dimension_score = DeviceDimensionScoreV2(
            batch_id=batch.batch_id,
            device_id="device-a",
            dimension_id=DimensionId.FACE_EXPOSURE_READABILITY,
            aggregate_score=80.0,
            rank=1,
            rank_group=1,
            confidence=0.87,
            applicable_scene_count=6,
            style_position={"face_target": "medium_high"},
            supporting_scene_ids=["scene-001"],
            counterexample_scene_ids=["scene-004"],
        )
        overall_score = DeviceOverallScoreV2(
            batch_id=batch.batch_id,
            device_id="device-a",
            relative_score=81.0,
            rank=1,
            confidence=0.85,
            dimension_scores={DimensionId.FACE_EXPOSURE_READABILITY: 80.0},
        )
        mechanism = MechanismInterpretationV2(
            interpretation_id="mechanism-001",
            device_id="device-a",
            scene_ids=["scene-001", "scene-004"],
            observable_effect="the face is selectively lifted while the background remains dark",
            strategy_inference="subject and environment tone budgets are partially decoupled",
            mechanism_hypothesis=None,
            attribution=MechanismAttribution.ALGORITHM_DEFINED,
            confidence=0.82,
            evidence_grade=EvidenceGradeV2.B,
            supporting_evidence_refs=["scene-score:scene-001"],
            alternative_explanations=["capture exposure may differ"],
            report_allowed_level=ReportAllowedLevel.STRATEGY_INFERENCE,
        )

        repository.save_device_dimension_score(dimension_score)
        repository.save_device_overall_score(overall_score)
        repository.save_mechanism_interpretation(batch.batch_id, mechanism)

        package = ReportEvidencePackageV2(
            batch=batch,
            scene_summaries=[{"scene_id": "scene-001", "summary": "selective face lift"}],
            device_dimension_scores=[dimension_score],
            device_overall_scores=[overall_score],
            device_profiles=[{"device_id": "device-a", "profile": "subject-priority"}],
            mechanism_interpretations=[mechanism],
            selected_visual_assets=[{"asset_id": "montage-001"}],
            review_notes=[],
            limitations=["batch-relative evaluation"],
            prohibited_claims=["not a comprehensive camera ranking"],
        )
        repository.save_report_evidence_package("report-package-001", package)

        assert (
            repository.get_device_dimension_score(
                batch.batch_id,
                "device-a",
                DimensionId.FACE_EXPOSURE_READABILITY,
            )
            == dimension_score
        )
        assert repository.get_device_overall_score(batch.batch_id, "device-a") == overall_score
        assert repository.get_mechanism_interpretation(batch.batch_id, "mechanism-001") == mechanism
        assert repository.get_report_evidence_package("report-package-001") == package

        with pytest.raises(ValueError, match="report evidence package already exists"):
            repository.save_report_evidence_package("report-package-001", package)
    finally:
        repository.session.close()
        database.engine.dispose()
