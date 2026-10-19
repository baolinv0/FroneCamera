from __future__ import annotations

from datetime import UTC, datetime

import pytest

from portrait_eval.core.models_v2 import (
    ApplicabilityStatus,
    ComparabilityStatus,
    DifferenceType,
    DimensionApplicabilityV2,
    DimensionId,
    EvaluationBatchV2,
    EvidenceGradeV2,
    FactCheckResultV2,
    JudgeDecisionType,
    JudgeDecisionV2,
    MatchedSceneGroupV2,
    MatchEvidenceV2,
    MatchStatus,
    MechanismAttribution,
    MechanismInterpretationV2,
    ObjectiveEvidenceV2,
    PairwiseComparisonV2,
    PairwisePreference,
    ReportAllowedLevel,
    SceneDimensionScoreV2,
    SceneScoreStatus,
)
from portrait_eval.database import Database
from portrait_eval.repository_v2_integrity import V2Repository
from v2_test_support import PersistedV2Graph, create_persisted_v2_graph


def _batch(graph: PersistedV2Graph, batch_id: str = "batch-integrity") -> EvaluationBatchV2:
    return EvaluationBatchV2(
        batch_id=batch_id,
        project_id=graph.project.id,
        device_ids=[graph.device_a.id, graph.device_b.id],
        dataset_version="dataset-2026-07",
        dimension_policy_version="dimension-policy-v1",
        scoring_policy_version="scoring-policy-v1",
        primary_model_version="primary-v1",
        judge_model_version="judge-v1",
        report_model_version=None,
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
    )


def _scene(
    graph: PersistedV2Graph,
    *,
    scene_id: str = "scene-001",
    batch_id: str = "batch-integrity",
) -> MatchedSceneGroupV2:
    return MatchedSceneGroupV2(
        scene_id=scene_id,
        batch_id=batch_id,
        label=scene_id,
        cells={
            graph.device_a.id: graph.image_a.id,
            graph.device_b.id: graph.image_b.id,
        },
        match_status=MatchStatus.CONFIRMED_MANIFEST,
        match_confidence=0.98,
        match_evidence=MatchEvidenceV2(
            exif_time_score=0.95,
            perceptual_hash_score=0.91,
            scene_embedding_score=0.97,
            face_geometry_score=0.90,
            composition_score=0.94,
            aggregate_score=0.95,
        ),
        comparability_status=ComparabilityStatus.FULLY_COMPARABLE,
        confounders=[],
        confirmed_by="manifest-v1",
    )


def _comparison(
    graph: PersistedV2Graph,
    *,
    comparison_id: str,
    scene_id: str = "scene-001",
    dimension_id: DimensionId = DimensionId.FACE_EXPOSURE_READABILITY,
) -> PairwiseComparisonV2:
    return PairwiseComparisonV2(
        comparison_id=comparison_id,
        scene_id=scene_id,
        dimension_id=dimension_id,
        device_a_id=graph.device_a.id,
        device_b_id=graph.device_b.id,
        applicability=ApplicabilityStatus.APPLICABLE,
        preference=PairwisePreference.A_SLIGHTLY_BETTER,
        strength=1,
        difference_type=DifferenceType.QUALITY,
        confidence=0.86,
        observations=["device A is more readable"],
        evidence_refs=[f"objective:{graph.image_a.id}", f"objective:{graph.image_b.id}"],
        style_axes={},
        primary_model_version="primary-v1",
        prompt_version="pairwise-prompt-v1",
    )


def _accepted_judge(
    repository: V2Repository,
    graph: PersistedV2Graph,
    *,
    comparison_id: str,
    scene_id: str = "scene-001",
    dimension_id: DimensionId = DimensionId.FACE_EXPOSURE_READABILITY,
    parent_pairwise_attempt_id: str | None = None,
    parent_fact_attempt_id: str | None = None,
    parent_judge_attempt_id: str | None = None,
    decision: JudgeDecisionType = JudgeDecisionType.ACCEPT,
):
    pairwise = repository.append_pairwise_attempt(
        "batch-integrity",
        _comparison(
            graph,
            comparison_id=comparison_id,
            scene_id=scene_id,
            dimension_id=dimension_id,
        ),
        parent_attempt_id=parent_pairwise_attempt_id,
    )
    fact = repository.append_fact_check_attempt(
        "batch-integrity",
        FactCheckResultV2(
            comparison_id=comparison_id,
            rule_version="fact-rules-v1",
            passed_rules=["direction agrees"],
            failed_rules=[],
            warnings=[],
            critical_conflict=False,
            objective_support=0.91,
        ),
        source_pairwise_attempt_id=pairwise.attempt_id,
        parent_attempt_id=parent_fact_attempt_id,
    )
    accepted = decision in {JudgeDecisionType.ACCEPT, JudgeDecisionType.REVISE}
    judge = repository.append_judge_attempt(
        "batch-integrity",
        JudgeDecisionV2(
            comparison_id=comparison_id,
            decision=decision,
            accepted_preference=(PairwisePreference.A_SLIGHTLY_BETTER if accepted else None),
            accepted_strength=1 if accepted else None,
            accepted_difference_type=DifferenceType.QUALITY if accepted else None,
            revised_observations=["accepted observation"] if accepted else [],
            fact_conflicts=[],
            unsupported_attributions=[],
            confidence=0.89,
            judge_model_version="judge-v1",
            prompt_version="judge-prompt-v1",
        ),
        source_pairwise_attempt_id=pairwise.attempt_id,
        source_fact_check_attempt_id=fact.attempt_id,
        parent_attempt_id=parent_judge_attempt_id,
    )
    return pairwise, fact, judge


def test_batch_rejects_devices_outside_project() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        primary = create_persisted_v2_graph(session)
        foreign = create_persisted_v2_graph(session, prefix="foreign-")
        repository = V2Repository(session)
        invalid = _batch(primary).model_copy(
            update={"device_ids": [primary.device_a.id, foreign.device_b.id]}
        )

        with pytest.raises(ValueError, match="batch devices must belong to project"):
            repository.save_evaluation_batch(invalid)


def test_matched_scene_rejects_image_assigned_to_wrong_device() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        graph = create_persisted_v2_graph(session)
        repository = V2Repository(session)
        repository.save_evaluation_batch(_batch(graph))
        invalid = _scene(graph).model_copy(
            update={
                "cells": {
                    graph.device_a.id: graph.image_b.id,
                    graph.device_b.id: graph.image_a.id,
                }
            }
        )

        with pytest.raises(ValueError, match="image does not belong to scene device"):
            repository.save_matched_scene_group(invalid)


def test_objective_evidence_rejects_image_outside_batch() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        primary = create_persisted_v2_graph(session)
        foreign = create_persisted_v2_graph(session, prefix="foreign-")
        repository = V2Repository(session)
        repository.save_evaluation_batch(_batch(primary))

        with pytest.raises(ValueError, match="image does not belong to batch"):
            repository.save_objective_evidence(
                "batch-integrity",
                ObjectiveEvidenceV2(
                    image_id=foreign.image_a.id,
                    metric_version="objective-v1",
                ),
            )


def test_applicability_requires_persisted_scene() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        graph = create_persisted_v2_graph(session)
        repository = V2Repository(session)
        repository.save_evaluation_batch(_batch(graph))

        with pytest.raises(KeyError):
            repository.save_dimension_applicability(
                "batch-integrity",
                DimensionApplicabilityV2(
                    scene_id="missing-scene",
                    dimension_id=DimensionId.MULTI_FACE_CONSISTENCY,
                    status=ApplicabilityStatus.NOT_APPLICABLE,
                    score=0.0,
                    reasons=["scene missing"],
                    required_evidence_present=False,
                ),
            )


def test_pairwise_requires_existing_scene_and_batch_devices() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        graph = create_persisted_v2_graph(session)
        foreign = create_persisted_v2_graph(session, prefix="foreign-")
        repository = V2Repository(session)
        repository.save_evaluation_batch(_batch(graph))
        repository.save_matched_scene_group(_scene(graph))
        invalid = _comparison(graph, comparison_id="comparison-foreign").model_copy(
            update={"device_b_id": foreign.device_b.id}
        )

        with pytest.raises(ValueError, match="pairwise device does not belong to batch"):
            repository.append_pairwise_attempt("batch-integrity", invalid)


def test_scene_score_rejects_judge_from_different_scene() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        graph = create_persisted_v2_graph(session)
        repository = V2Repository(session)
        repository.save_evaluation_batch(_batch(graph))
        repository.save_matched_scene_group(_scene(graph, scene_id="scene-001"))
        repository.save_matched_scene_group(_scene(graph, scene_id="scene-002"))
        _pairwise, _fact, judge = _accepted_judge(
            repository,
            graph,
            comparison_id="comparison-001",
            scene_id="scene-002",
        )
        score = SceneDimensionScoreV2(
            scene_id="scene-001",
            dimension_id=DimensionId.FACE_EXPOSURE_READABILITY,
            device_id=graph.device_a.id,
            latent_quality=0.72,
            normalized_score=78.0,
            confidence=0.88,
            diagnostic_weight=0.90,
            objective_adjustment=2.0,
            final_score=80.0,
            status=SceneScoreStatus.AUTO_PASS,
            contributing_comparison_ids=["comparison-001"],
        )

        with pytest.raises(ValueError, match="judge attempt does not match score context"):
            repository.save_scene_dimension_score(
                "batch-integrity",
                score,
                accepted_judge_attempt_ids=[judge.attempt_id],
            )


def test_scene_score_rejects_stale_judge_attempt() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        graph = create_persisted_v2_graph(session)
        repository = V2Repository(session)
        repository.save_evaluation_batch(_batch(graph))
        repository.save_matched_scene_group(_scene(graph))
        pairwise_1, fact_1, judge_1 = _accepted_judge(
            repository,
            graph,
            comparison_id="comparison-001",
        )
        _accepted_judge(
            repository,
            graph,
            comparison_id="comparison-001",
            parent_pairwise_attempt_id=pairwise_1.attempt_id,
            parent_fact_attempt_id=fact_1.attempt_id,
            parent_judge_attempt_id=judge_1.attempt_id,
            decision=JudgeDecisionType.MANUAL_REVIEW,
        )
        score = SceneDimensionScoreV2(
            scene_id="scene-001",
            dimension_id=DimensionId.FACE_EXPOSURE_READABILITY,
            device_id=graph.device_a.id,
            latent_quality=0.72,
            normalized_score=78.0,
            confidence=0.88,
            diagnostic_weight=0.90,
            objective_adjustment=2.0,
            final_score=80.0,
            status=SceneScoreStatus.AUTO_PASS,
            contributing_comparison_ids=["comparison-001"],
        )

        with pytest.raises(ValueError, match="judge attempt is not latest"):
            repository.save_scene_dimension_score(
                "batch-integrity",
                score,
                accepted_judge_attempt_ids=[judge_1.attempt_id],
            )


def test_mechanism_requires_batch_device_and_persisted_scenes() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        graph = create_persisted_v2_graph(session)
        foreign = create_persisted_v2_graph(session, prefix="foreign-")
        repository = V2Repository(session)
        repository.save_evaluation_batch(_batch(graph))
        invalid = MechanismInterpretationV2(
            interpretation_id="mechanism-invalid",
            device_id=foreign.device_a.id,
            scene_ids=["missing-scene"],
            observable_effect="invalid",
            strategy_inference="invalid",
            mechanism_hypothesis=None,
            attribution=MechanismAttribution.INDETERMINATE,
            confidence=0.2,
            evidence_grade=EvidenceGradeV2.C,
            supporting_evidence_refs=[],
            alternative_explanations=[],
            report_allowed_level=ReportAllowedLevel.OBSERVABLE,
        )

        with pytest.raises(ValueError, match="device does not belong to batch"):
            repository.save_mechanism_interpretation("batch-integrity", invalid)
