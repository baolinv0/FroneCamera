from __future__ import annotations

from datetime import UTC, datetime

import pytest

from portrait_eval.core.models_v2 import (
    ApplicabilityStatus,
    ComparabilityStatus,
    DimensionApplicabilityV2,
    DimensionId,
    EvaluationBatchV2,
    FaceRegionV2,
    FaceRole,
    MatchedSceneGroupV2,
    MatchEvidenceV2,
    MatchStatus,
    ObjectiveEvidenceV2,
    RoughRankingV2,
)
from portrait_eval.database import Database
from portrait_eval.repository import Repository
from portrait_eval.repository_v2 import V2Repository


def _batch(project_id: str, batch_id: str) -> EvaluationBatchV2:
    return EvaluationBatchV2(
        batch_id=batch_id,
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


def _scene(batch_id: str, label: str) -> MatchedSceneGroupV2:
    return MatchedSceneGroupV2(
        scene_id="scene-001",
        batch_id=batch_id,
        label=label,
        cells={"device-a": "image-a", "device-b": "image-b"},
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


def _objective_evidence(face_luma: float) -> ObjectiveEvidenceV2:
    return ObjectiveEvidenceV2(
        image_id="image-a",
        faces=[
            FaceRegionV2(
                face_id="face-primary",
                bbox_xywh=(100, 120, 220, 260),
                detection_confidence=0.99,
                area_ratio=0.18,
                center_xy_normalized=(0.38, 0.46),
                landmarks_ref="artifacts/face-primary-landmarks.json",
                masks={"skin": "artifacts/face-primary-skin.png"},
                role=FaceRole.PRIMARY,
            ),
            FaceRegionV2(
                face_id="face-secondary",
                bbox_xywh=(410, 150, 120, 150),
                detection_confidence=0.96,
                area_ratio=0.07,
                center_xy_normalized=(0.72, 0.49),
                landmarks_ref=None,
                masks={"skin": "artifacts/face-secondary-skin.png"},
                role=FaceRole.SECONDARY,
            ),
        ],
        whole_image_metrics={"display_luma_median": 102.5},
        region_metrics={"highlight": {"clip_ratio": 0.018}},
        per_face_metrics={
            "face-primary": {"display_luma_median": face_luma},
            "face-secondary": {"display_luma_median": 88.0},
        },
        relation_metrics={"primary_secondary_luma_gap": face_luma - 88.0},
        artifact_metrics={"halo_score": 0.04},
        warnings=["secondary face is small"],
        metric_version="objective-metrics-v1",
    )


def _applicability(score: float) -> DimensionApplicabilityV2:
    return DimensionApplicabilityV2(
        scene_id="scene-001",
        dimension_id=DimensionId.MULTI_FACE_CONSISTENCY,
        status=ApplicabilityStatus.APPLICABLE,
        score=score,
        reasons=["two independently detected faces"],
        required_evidence_present=True,
    )


def _ranking(first_device: str) -> RoughRankingV2:
    second_device = "device-b" if first_device == "device-a" else "device-a"
    return RoughRankingV2(
        scene_id="scene-001",
        dimension_id=DimensionId.MULTI_FACE_CONSISTENCY,
        ordered_device_ids=[first_device, second_device],
        tie_groups=[],
        confidence=0.84,
        evidence_refs=["objective:image-a", "objective:image-b"],
        prompt_version="rough-ranking-prompt-v1",
        model_version="primary-v1",
    )


def test_batch_scoped_entities_round_trip_without_cross_batch_leakage() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        project = Repository(session).create_project("V2 entity persistence")
        repository = V2Repository(session)
        batch_a = _batch(project.id, "batch-a")
        batch_b = _batch(project.id, "batch-b")
        repository.save_evaluation_batch(batch_a)
        repository.save_evaluation_batch(batch_b)

        scene_a = _scene(batch_a.batch_id, "scene in batch A")
        scene_b = _scene(batch_b.batch_id, "scene in batch B")
        evidence_a = _objective_evidence(126.0)
        evidence_b = _objective_evidence(104.0)
        applicability_a = _applicability(0.95)
        applicability_b = _applicability(0.72)
        ranking_a = _ranking("device-a")
        ranking_b = _ranking("device-b")

        repository.save_matched_scene_group(scene_a)
        repository.save_matched_scene_group(scene_b)
        repository.save_objective_evidence(batch_a.batch_id, evidence_a)
        repository.save_objective_evidence(batch_b.batch_id, evidence_b)
        repository.save_dimension_applicability(batch_a.batch_id, applicability_a)
        repository.save_dimension_applicability(batch_b.batch_id, applicability_b)
        repository.save_rough_ranking(batch_a.batch_id, ranking_a)
        repository.save_rough_ranking(batch_b.batch_id, ranking_b)

        assert repository.get_matched_scene_group(batch_a.batch_id, "scene-001") == scene_a
        assert repository.get_matched_scene_group(batch_b.batch_id, "scene-001") == scene_b
        assert repository.get_objective_evidence(batch_a.batch_id, "image-a") == evidence_a
        assert repository.get_objective_evidence(batch_b.batch_id, "image-a") == evidence_b
        assert (
            repository.get_dimension_applicability(
                batch_a.batch_id,
                "scene-001",
                DimensionId.MULTI_FACE_CONSISTENCY,
            )
            == applicability_a
        )
        assert (
            repository.get_dimension_applicability(
                batch_b.batch_id,
                "scene-001",
                DimensionId.MULTI_FACE_CONSISTENCY,
            )
            == applicability_b
        )
        assert (
            repository.get_rough_ranking(
                batch_a.batch_id,
                "scene-001",
                DimensionId.MULTI_FACE_CONSISTENCY,
            )
            == ranking_a
        )
        assert (
            repository.get_rough_ranking(
                batch_b.batch_id,
                "scene-001",
                DimensionId.MULTI_FACE_CONSISTENCY,
            )
            == ranking_b
        )


def test_batch_scoped_entities_are_insert_only() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        project = Repository(session).create_project("V2 entity immutability")
        repository = V2Repository(session)
        batch = _batch(project.id, "batch-a")
        repository.save_evaluation_batch(batch)
        scene = _scene(batch.batch_id, "original")
        repository.save_matched_scene_group(scene)

        with pytest.raises(ValueError, match="matched scene group already exists"):
            repository.save_matched_scene_group(scene)
