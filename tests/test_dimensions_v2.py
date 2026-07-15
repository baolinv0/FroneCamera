import pytest
from pydantic import ValidationError

from portrait_eval.core.dimensions import (
    DIMENSION_BY_ID,
    DIMENSIONS,
    TOTAL_DIMENSION_WEIGHT,
    DimensionDefinition,
    DimensionScope,
    get_dimension,
)
from portrait_eval.core.models_v2 import DimensionId


def test_dimension_registry_is_complete_and_ordered() -> None:
    assert [item.id for item in DIMENSIONS] == list(DimensionId)
    assert set(DIMENSION_BY_ID) == set(DimensionId)
    assert len(DIMENSIONS) == len(DIMENSION_BY_ID) == 10


def test_frozen_dimension_weights_sum_to_one() -> None:
    expected = {
        DimensionId.FACE_EXPOSURE_READABILITY: 0.15,
        DimensionId.HIGHLIGHT_INTEGRITY: 0.12,
        DimensionId.SHADOW_BLACK_RENDERING: 0.08,
        DimensionId.SKIN_AWB: 0.12,
        DimensionId.LIGHTING_CAUSALITY: 0.10,
        DimensionId.FACE_BACKGROUND_RELATION: 0.10,
        DimensionId.LOCAL_FACE_LIFT_NATURALNESS: 0.10,
        DimensionId.MULTI_FACE_CONSISTENCY: 0.08,
        DimensionId.SCENE_ADAPTABILITY: 0.10,
        DimensionId.ARTIFACT_TEXTURE_CONTROL: 0.05,
    }
    assert {item.id: item.weight for item in DIMENSIONS} == expected
    assert abs(TOTAL_DIMENSION_WEIGHT - 1.0) <= 1e-12


def test_scene_adaptability_is_cross_scene_only() -> None:
    definition = get_dimension(DimensionId.SCENE_ADAPTABILITY)
    assert definition.scope is DimensionScope.CROSS_SCENE
    assert definition.single_scene_applicable is False
    assert definition.required_evidence == ("multiple_comparable_scenes",)


def test_multi_face_dimension_requires_multiple_valid_faces() -> None:
    definition = get_dimension(DimensionId.MULTI_FACE_CONSISTENCY)
    assert definition.scope is DimensionScope.SCENE
    assert definition.single_scene_applicable is True
    assert "at_least_two_valid_faces_in_at_least_two_devices" in definition.required_evidence


def test_scene_dimensions_are_single_scene_applicable() -> None:
    for definition in DIMENSIONS:
        if definition.id is DimensionId.SCENE_ADAPTABILITY:
            continue
        assert definition.scope is DimensionScope.SCENE
        assert definition.single_scene_applicable is True
        assert definition.required_evidence


def test_dimension_definitions_are_immutable() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        DIMENSIONS[0].weight = 0.20


def test_dimension_definitions_reject_unknown_fields() -> None:
    payload = DIMENSIONS[0].model_dump()
    payload["unknown"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DimensionDefinition.model_validate(payload)
