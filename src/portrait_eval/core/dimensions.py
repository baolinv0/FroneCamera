from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from portrait_eval.core.models_v2 import DimensionId


class DimensionScope(StrEnum):
    SCENE = "SCENE"
    CROSS_SCENE = "CROSS_SCENE"


class DimensionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: DimensionId
    display_name: str
    weight: float = Field(gt=0, le=1)
    evaluation_unit: str
    scope: DimensionScope
    single_scene_applicable: bool
    required_evidence: tuple[str, ...]


DIMENSIONS: tuple[DimensionDefinition, ...] = (
    DimensionDefinition(
        id=DimensionId.FACE_EXPOSURE_READABILITY,
        display_name="Face exposure and readability",
        weight=0.15,
        evaluation_unit="per_face_and_scene",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=("valid_face_region", "face_luminance_metrics"),
    ),
    DimensionDefinition(
        id=DimensionId.HIGHLIGHT_INTEGRITY,
        display_name="Highlight integrity",
        weight=0.12,
        evaluation_unit="semantic_highlight_regions",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=("comparable_semantic_highlight_region",),
    ),
    DimensionDefinition(
        id=DimensionId.SHADOW_BLACK_RENDERING,
        display_name="Shadow and black rendering",
        weight=0.08,
        evaluation_unit="scene_and_background",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=("background_region", "shadow_metrics"),
    ),
    DimensionDefinition(
        id=DimensionId.SKIN_AWB,
        display_name="Skin tone and white balance",
        weight=0.12,
        evaluation_unit="skin_regions_and_environment_relation",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=("valid_skin_region", "illumination_or_color_evidence"),
    ),
    DimensionDefinition(
        id=DimensionId.LIGHTING_CAUSALITY,
        display_name="Lighting causality and dimensionality",
        weight=0.10,
        evaluation_unit="face_internal_light_structure",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=(
            "face_highlight_and_shadow_regions",
            "face_internal_contrast_metrics",
        ),
    ),
    DimensionDefinition(
        id=DimensionId.FACE_BACKGROUND_RELATION,
        display_name="Face/background relation and separation",
        weight=0.10,
        evaluation_unit="face_face_ring_and_background",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=("face_region", "face_ring_region", "background_region"),
    ),
    DimensionDefinition(
        id=DimensionId.LOCAL_FACE_LIFT_NATURALNESS,
        display_name="Local face-lift naturalness",
        weight=0.10,
        evaluation_unit="face_gain_and_transition_behavior",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=("face_background_relation_metrics", "face_edge_transition_metrics"),
    ),
    DimensionDefinition(
        id=DimensionId.MULTI_FACE_CONSISTENCY,
        display_name="Multi-face consistency",
        weight=0.08,
        evaluation_unit="two_or_more_valid_faces",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=("at_least_two_valid_faces_in_at_least_two_devices",),
    ),
    DimensionDefinition(
        id=DimensionId.SCENE_ADAPTABILITY,
        display_name="Scene adaptability and stability",
        weight=0.10,
        evaluation_unit="cross_scene_device_level",
        scope=DimensionScope.CROSS_SCENE,
        single_scene_applicable=False,
        required_evidence=("multiple_comparable_scenes",),
    ),
    DimensionDefinition(
        id=DimensionId.ARTIFACT_TEXTURE_CONTROL,
        display_name="Artifact and texture control",
        weight=0.05,
        evaluation_unit="semantic_edges_and_face_texture",
        scope=DimensionScope.SCENE,
        single_scene_applicable=True,
        required_evidence=("semantic_edge_metrics", "face_texture_metrics"),
    ),
)

_dimension_ids = tuple(item.id for item in DIMENSIONS)
if len(set(_dimension_ids)) != len(_dimension_ids):
    raise RuntimeError("Evaluation Core v2 dimension IDs must be unique")
if set(_dimension_ids) != set(DimensionId):
    raise RuntimeError("Evaluation Core v2 dimension registry must cover every frozen ID")

DIMENSION_BY_ID: dict[DimensionId, DimensionDefinition] = {
    item.id: item for item in DIMENSIONS
}
TOTAL_DIMENSION_WEIGHT = sum(item.weight for item in DIMENSIONS)
if abs(TOTAL_DIMENSION_WEIGHT - 1.0) > 1e-12:
    raise RuntimeError("Evaluation Core v2 dimension weights must sum to 1.0")


def get_dimension(dimension_id: DimensionId) -> DimensionDefinition:
    return DIMENSION_BY_ID[dimension_id]
