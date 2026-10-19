"""Append-only capture/ROI annotations bound to a confirmed pairing revision."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from portrait_eval.color import load_analysis_rgb
from portrait_eval.repository import Repository

Challenge = Literal[
    "uniform_light",
    "high_contrast",
    "mixed_light",
    "multi_person",
    "low_light",
    "strong_light",
    "unknown",
]


class FaceAnnotation(BaseModel):
    bbox_xywh: tuple[int, int, int, int]
    role: Literal["PRIMARY", "SECONDARY", "UNASSIGNED"] = "UNASSIGNED"
    position: Literal["foreground", "rear", "edge", "unknown"] = "unknown"

    @model_validator(mode="after")
    def valid_box(self):
        x, y, w, h = self.bbox_xywh
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError("Face box must have nonnegative origin and positive area")
        return self


class SceneAnnotation(BaseModel):
    expected_pairing_version: int
    expected_revision: int = Field(default=0, ge=0)
    author: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    scene_family: str | None = Field(default=None, min_length=1)
    challenge: Challenge = "unknown"
    tags: list[str] = Field(default_factory=list)
    faces: dict[str, list[FaceAnnotation]] = Field(default_factory=dict)
    attribute_comparability: dict[
        str,
        Literal[
            "FULLY_COMPARABLE",
            "COMPARABLE_WITH_CONFOUNDERS",
            "NOT_COMPARABLE",
            "INSUFFICIENT_EVIDENCE",
        ],
    ] = Field(default_factory=dict)


def latest_scene_annotation(repo: Repository, project_id: str, group_key: str) -> dict:
    rows = [
        row["payload"]
        for row in repo.list_analysis(project_id, "scene_annotation")
        if row["payload"]["group_id"] == group_key
    ]
    return max(rows, key=lambda row: row["revision"]) if rows else {}


def save_scene_annotation(
    repo: Repository, project_id: str, group_key: str, body: SceneAnnotation
) -> dict:
    pairing = repo.get_pairing(project_id)
    if pairing["version"] != body.expected_pairing_version:
        raise RuntimeError("version_conflict")
    if not pairing["confirmed"]:
        raise ValueError("Pairing must be confirmed before annotation")
    group = next((g for g in pairing["groups"] if g["group_id"] == group_key), None)
    if group is None:
        raise ValueError("Unknown scene group")
    previous = latest_scene_annotation(repo, project_id, group_key)
    if previous.get("revision", 0) != body.expected_revision:
        raise RuntimeError("revision_conflict")
    assets = {}
    for device_id, faces in body.faces.items():
        cell = group["cells"].get(device_id)
        if not cell:
            raise ValueError("Face annotation device has no paired asset")
        pixels, rendering = load_analysis_rgb(Path(cell["path"]))
        for face in faces:
            x, y, w, h = face.bbox_xywh
            if x + w > pixels.shape[1] or y + h > pixels.shape[0]:
                raise ValueError("Face box lies outside oriented pixels")
        assets[device_id] = {"image_id": cell["image_id"], "sha256": rendering["input_sha256"]}
    payload = {
        **body.model_dump(mode="json"),
        "group_id": group_key,
        "revision": body.expected_revision + 1,
        "assets": assets,
        "schema_version": "0.2",
    }
    repo.save_analysis(project_id, "scene_annotation", payload, scene_group_id=group["id"])
    return payload
