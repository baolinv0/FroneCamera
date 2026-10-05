from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import TMQAConfig
from .domain import SceneEvaluation, SceneSpec
from .image_io import file_sha256
from .prompts import build_input_manifest

SplitName = Literal["train", "validation", "audit", "benchmark", "holdout"]
SceneEvaluationDigest = Annotated[
    str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
]


class EvaluationRunManifest(BaseModel):
    schema_version: str = "tmqa.evaluation_run.v1"
    evaluation_run_id: str
    evaluation_config_sha256: str = Field(min_length=64, max_length=64)
    dataset_manifest_sha256: str = Field(min_length=64, max_length=64)
    dataset_version: str
    prompt_versions: list[str]
    # Historical manifests remain readable, but cannot authorize candidate selection.
    scene_evaluation_sha256: dict[str, SceneEvaluationDigest] | None = None


class SelectionLineage(BaseModel):
    evaluation: EvaluationRunManifest
    split: SplitName
    split_file_sha256: str = Field(min_length=64, max_length=64)


class SplitAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_id: str
    canonical_scene_id: str
    group_id: str
    split: SplitName

    @field_validator("scene_id", "canonical_scene_id", "group_id")
    @classmethod
    def explicit_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("explicit canonical/group metadata must not be empty")
        return value.strip()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest_signature(spec: SceneSpec) -> dict[str, object]:
    images = [
        {
            "role": item.role,
            "level": item.level,
            "sha256": item.sha256,
            "width": item.width,
            "height": item.height,
        }
        for item in build_input_manifest(spec)
    ]
    return {"scene_id": spec.scene_id, "images": images}


def dataset_manifest_sha256(specs: Iterable[SceneSpec]) -> str:
    payload = [_manifest_signature(spec) for spec in sorted(specs, key=lambda item: item.scene_id)]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def scene_evaluation_sha256(scene: SceneEvaluation) -> str:
    """Bind all evaluated policy, scores, stages and judge evidence, after typed normalization."""
    return hashlib.sha256(_canonical_json(scene.model_dump(mode="json"))).hexdigest()


def _scene_evaluation_digests(evaluations: Iterable[SceneEvaluation]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for scene in sorted(evaluations, key=lambda item: item.scene_id):
        if scene.scene_id in digests:
            raise ValueError(f"duplicate evaluated scene identity: {scene.scene_id}")
        digests[scene.scene_id] = scene_evaluation_sha256(scene)
    return digests


def build_evaluation_run_manifest(
    specs: Iterable[SceneSpec],
    config_path: Path,
    config: TMQAConfig,
    *,
    evaluations: Iterable[SceneEvaluation] | None = None,
    config_sha256: str | None = None,
) -> EvaluationRunManifest:
    specs_list = list(specs)
    scene_digests = None
    if evaluations is not None:
        scene_ids = {spec.scene_id for spec in specs_list}
        if len(scene_ids) != len(specs_list):
            raise ValueError("duplicate declared scene identities")
        scene_digests = _scene_evaluation_digests(evaluations)
        if set(scene_digests) != scene_ids:
            raise ValueError("evaluation scene coverage must match the complete declared dataset")
    config_sha = config_sha256 if config_sha256 is not None else file_sha256(config_path)
    dataset_sha = dataset_manifest_sha256(specs_list)
    prompt_versions = sorted(
        {f"{judge.prompt_id}@{judge.prompt_version}" for judge in config.judges if judge.enabled}
    )
    identity = {
        "evaluation_config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "dataset_version": getattr(config.dataset, "version", "unversioned"),
        "prompt_versions": prompt_versions,
    }
    run_id = hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
    return EvaluationRunManifest(
        evaluation_run_id=run_id,
        evaluation_config_sha256=config_sha,
        dataset_manifest_sha256=dataset_sha,
        dataset_version=getattr(config.dataset, "version", "unversioned"),
        prompt_versions=prompt_versions,
        scene_evaluation_sha256=scene_digests,
    )


def write_evaluation_run_manifest(path: Path, manifest: EvaluationRunManifest) -> None:
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_evaluation_run_manifest(path: Path) -> EvaluationRunManifest:
    if not path.exists():
        raise FileNotFoundError(path)
    return EvaluationRunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def require_matching_evaluation_config(
    manifest: EvaluationRunManifest,
    config_path: Path,
    *,
    config_sha256: str | None = None,
) -> None:
    current = config_sha256 if config_sha256 is not None else file_sha256(config_path)
    if current != manifest.evaluation_config_sha256:
        raise ValueError(
            "evaluation config hash mismatch: rerun evaluation before selecting pseudo-GT"
        )


def require_matching_scene_evaluations(
    manifest: EvaluationRunManifest,
    evaluations: Iterable[SceneEvaluation],
) -> None:
    """Validate the same in-memory scenes that selection will consume, never a reread file."""
    if manifest.scene_evaluation_sha256 is None:
        raise ValueError(
            "evaluation manifest has no scene result binding: "
            "rerun evaluation before selecting pseudo-GT"
        )
    actual = _scene_evaluation_digests(evaluations)
    if set(actual) != set(manifest.scene_evaluation_sha256):
        raise ValueError(
            "evaluation scene binding coverage mismatch: "
            "rerun evaluation before selecting pseudo-GT"
        )
    if actual != manifest.scene_evaluation_sha256:
        raise ValueError(
            "evaluation scene content mismatch: rerun evaluation before selecting pseudo-GT"
        )


def _trace_signature(scene: SceneEvaluation) -> list[list[dict[str, object]]]:
    traces: list[list[dict[str, object]]] = []
    for evaluation in scene.model_evaluations:
        traces.append(
            [
                {
                    "role": item.role,
                    "level": item.level,
                    "sha256": item.sha256,
                    "width": item.width,
                    "height": item.height,
                }
                for item in evaluation.prompt_trace.input_manifest
            ]
        )
    return traces


def verify_scene_evidence(
    scene: SceneEvaluation,
    spec: SceneSpec,
    evaluation_manifest: EvaluationRunManifest,
) -> list[str]:
    reasons: list[str] = []
    if evaluation_manifest.scene_evaluation_sha256 is None:
        reasons.append("EVIDENCE_RUN_BINDING_MISSING")
    elif evaluation_manifest.scene_evaluation_sha256.get(scene.scene_id) != scene_evaluation_sha256(
        scene
    ):
        reasons.append("EVIDENCE_RUN_BINDING_MISMATCH")
    expected = _manifest_signature(spec)["images"]
    if not scene.model_evaluations:
        return reasons + ["EVIDENCE_IMAGE_MISMATCH"]
    for trace in _trace_signature(scene):
        if trace != expected:
            return reasons + ["EVIDENCE_IMAGE_MISMATCH"]
    allowed_prompts = set(evaluation_manifest.prompt_versions)
    for evaluation in scene.model_evaluations:
        prompt = f"{evaluation.prompt_trace.prompt_id}@{evaluation.prompt_trace.prompt_version}"
        if prompt not in allowed_prompts:
            return reasons + ["EVIDENCE_PROMPT_MISMATCH"]
    return reasons


def load_split_assignments(
    path: Path,
    *,
    expected_scene_ids: set[str],
) -> dict[str, SplitAssignment]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scene_id", "canonical_scene_id", "group_id", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "split file must contain explicit scene_id, canonical_scene_id, group_id and split columns"
            )
        assignments: dict[str, SplitAssignment] = {}
        canonical_splits: dict[str, str] = {}
        group_splits: dict[str, str] = {}
        for row in reader:
            scene_id = str(row.get("scene_id", "")).strip()
            split = str(row.get("split", "")).strip()
            canonical = str(row.get("canonical_scene_id", "")).strip()
            group = str(row.get("group_id", "")).strip()
            if not scene_id:
                raise ValueError("split file contains empty scene_id")
            if scene_id in assignments:
                raise ValueError(f"duplicate scene_id in split file: {scene_id}")
            try:
                assignment = SplitAssignment(
                    scene_id=scene_id,
                    canonical_scene_id=canonical,
                    group_id=group,
                    split=split,
                )
            except Exception as exc:
                raise ValueError(f"invalid split assignment for {scene_id}: {exc}") from exc
            previous = canonical_splits.get(canonical)
            if previous is not None and previous != assignment.split:
                raise ValueError(
                    f"canonical scene {canonical} appears in multiple splits: "
                    f"{previous}, {assignment.split}"
                )
            previous_group = group_splits.get(group)
            if previous_group is not None and previous_group != assignment.split:
                raise ValueError(f"group {group} appears in multiple splits")
            group_splits[group] = assignment.split
            canonical_splits[canonical] = assignment.split
            assignments[scene_id] = assignment

    missing = sorted(expected_scene_ids - set(assignments))
    if missing:
        raise ValueError(f"unassigned scenes in split file: {missing[:10]}")
    return assignments
