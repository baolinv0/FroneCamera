from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Decision = Literal["KEEP", "REGENERATE", "REVIEW", "REJECT"]
ModelRole = Literal["prescreener", "primary", "arbiter", "local_inspector", "custom"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_aware_timestamp(value: str, field_name: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


class AlphaImage(BaseModel):
    level: str
    alpha: float
    path: Path


class SceneSpec(BaseModel):
    scene_id: str
    baseline_path: Path
    alpha_images: list[AlphaImage]
    source_path: Path | None = None
    generator: str = "unknown"
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)

    @field_validator("alpha_images")
    @classmethod
    def sort_and_validate_alphas(cls, value: list[AlphaImage]) -> list[AlphaImage]:
        alphas = [item.alpha for item in value]
        if len(set(alphas)) != len(alphas):
            raise ValueError("duplicate alpha values are not allowed")
        return sorted(value, key=lambda item: item.alpha)


class ImageManifestItem(BaseModel):
    index: int = Field(ge=1)
    role: Literal["source", "baseline", "candidate", "crop", "difference"]
    alpha: float | None = None
    level: str
    path: str
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    source_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    payload_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    payload_mime: str | None = None
    payload_encoding: dict[str, object] = Field(default_factory=dict)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sent_width: int | None = Field(default=None, gt=0)
    sent_height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def synchronize_source_hash(self) -> ImageManifestItem:
        source_hash = self.source_sha256 or self.sha256
        if source_hash is None:
            raise ValueError("source_sha256 is required")
        self.source_sha256 = source_hash
        self.sha256 = source_hash
        return self


class PromptTrace(BaseModel):
    prompt_id: str
    prompt_version: str
    template: str
    variables: dict[str, object] = Field(default_factory=dict)
    rendered_prompt: str
    prompt_hash: str
    input_manifest: list[ImageManifestItem]
    output_schema_version: str = "tmqa.model.v1"
    inference_parameters: dict[str, object] = Field(default_factory=dict)

    @field_validator("input_manifest")
    @classmethod
    def require_manifest(cls, value: list[ImageManifestItem]) -> list[ImageManifestItem]:
        if not value:
            raise ValueError("input_manifest must not be empty")
        indexes = [item.index for item in value]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("input_manifest indexes must be contiguous and ordered")
        return value


class ModelIssue(BaseModel):
    dimension: str
    severity: float = Field(ge=0, le=1)
    description: str
    fatal: bool = False
    region: str | None = None
    bbox: tuple[float, float, float, float] | None = None


class ModelEvaluation(BaseModel):
    model_id: str
    model_role: ModelRole
    model_version: str
    synthetic: bool = False
    available: bool = True
    prompt_trace: PromptTrace
    scores: dict[str, float] = Field(default_factory=dict)
    decision: Decision
    confidence: float = Field(ge=0, le=1)
    issues: list[ModelIssue] = Field(default_factory=list)
    rationale: str = ""
    raw_response: str = ""
    parsed_response: dict[str, object] = Field(default_factory=dict)
    latency_ms: float = Field(default=0, ge=0)
    error: str | None = None

    @field_validator("scores")
    @classmethod
    def normalize_scores(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, score in value.items():
            converted = float(score)
            if converted > 1:
                converted /= 100.0
            if not 0 <= converted <= 1:
                raise ValueError(f"score {key} must be in [0,1] or [0,100]")
            normalized[key] = converted
        return normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: float) -> float:
        converted = float(value)
        return converted / 100.0 if converted > 1 else converted


class ObjectiveMetrics(BaseModel):
    alpha: float
    level: str
    mean_luminance: float
    ev_mean: float
    p20: float
    p50: float
    p90: float
    clipping_ratio: float
    shadow_ratio: float
    contrast: float
    color_drift: float
    edge_similarity: float


class SequenceMetrics(BaseModel):
    spearman_rho: float
    violation_rate: float
    smoothness_score: float
    dead_zone_ratio: float
    endpoint_range_ev: float
    clipping_growth: float
    control_score: float


class StageResult(BaseModel):
    stage_id: str
    label: str
    status: Literal["PASS", "WARN", "FAIL", "SKIP"]
    score: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class SceneEvaluation(BaseModel):
    scene_id: str
    generator: str
    decision: Decision
    overall_score: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    objective_by_level: dict[str, ObjectiveMetrics]
    sequence: SequenceMetrics
    stages: list[StageResult]
    input_manifest: list[ImageManifestItem] = Field(default_factory=list)
    model_evaluations: list[ModelEvaluation]
    review_priority: float = Field(ge=0)
    review_reasons: list[str] = Field(default_factory=list)
    decision_provenance: dict[str, object] = Field(default_factory=dict)
    synthetic_experiment: bool = False


class HumanReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    scene_id: str
    reviewer_id: str
    blind_review: bool
    decision: Decision
    scores: dict[str, float]
    issues: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(ge=0, le=1)
    synthetic: bool = False
    created_at: str = Field(default_factory=_utc_now)
    received_at: str = Field(default_factory=_utc_now)
    gold_eligible: bool = True

    @field_validator("review_id", "scene_id", "reviewer_id")
    @classmethod
    def require_reviewer_id(cls, value: str, info) -> str:
        reviewer_id = value.strip()
        if not reviewer_id:
            raise ValueError(f"{info.field_name} must not be empty")
        return reviewer_id

    @field_validator("created_at", "received_at")
    @classmethod
    def require_aware_timestamps(cls, value: str, info) -> str:
        return _require_aware_timestamp(value, info.field_name)

    @model_validator(mode="after")
    def require_blind_gold_label(self) -> HumanReview:
        if not self.blind_review and not self.synthetic:
            raise ValueError("gold-label human review must be blind")
        return self

    @field_validator("scores")
    @classmethod
    def normalize_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if "overall" not in value:
            raise ValueError("human review scores must include overall")
        output: dict[str, float] = {}
        for key, score in value.items():
            converted = float(score)
            if converted > 1:
                converted /= 100.0
            if not 0 <= converted <= 1:
                raise ValueError(f"human score {key} outside [0,1]")
            output[key] = converted
        return output


class ReliabilityResult(BaseModel):
    model_id: str
    sample_count: int
    overall_sample_count: int = Field(default=0, ge=0)
    model_sources: list[Literal["real", "synthetic"]] = Field(default_factory=list)
    model_source_counts: dict[str, int] = Field(default_factory=dict)
    synthetic: bool = False
    overall_mae: float
    decision_agreement: float
    dimension_mae: dict[str, float]
    fusion_weight: float
    fusion_weight_scope: Literal["research", "production"] = "research"
    production_eligible: bool = False
