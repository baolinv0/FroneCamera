from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_EXPECTED_LEVELS: tuple[str, ...] = (
    "a_m100",
    "a_m075",
    "a_m050",
    "a_m025",
    "a_000",
    "a_p025",
    "a_p050",
    "a_p075",
    "a_p100",
)
_LEVEL_PATTERN = re.compile(r"^a_(?:000|[mp]\d{3})$")


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_level: str = "a_000"
    expected_levels: list[str] = Field(default_factory=lambda: list(DEFAULT_EXPECTED_LEVELS))
    strict_complete: bool = True
    recursive: bool = True
    generator: str = "unknown"

    @model_validator(mode="after")
    def validate_levels(self) -> DatasetConfig:
        if len(set(self.expected_levels)) != len(self.expected_levels):
            raise ValueError("duplicate expected alpha levels are not allowed")
        invalid = [level for level in self.expected_levels if not _LEVEL_PATTERN.fullmatch(level)]
        if invalid:
            raise ValueError(f"invalid expected alpha levels: {invalid}")
        if self.baseline_level not in self.expected_levels:
            raise ValueError("baseline level must be included in expected_levels")
        return self


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: Literal["prescreener", "primary", "arbiter", "local_inspector", "custom"]
    adapter: Literal["mock", "openai_compatible"] = "mock"
    model: str
    version: str = "unknown"
    enabled: bool = True
    synthetic: bool = False
    base_url: str | None = None
    api_key: str = "EMPTY"
    prompt_id: str = "tmqa.sequence"
    prompt_version: str = "3.3"
    timeout_seconds: float = 120
    temperature: float = 0
    max_tokens: int = 1400
    bias_profile: str = "balanced"

    @field_validator("id")
    @classmethod
    def require_judge_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Judge id must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def require_remote_endpoint(self) -> JudgeConfig:
        if self.adapter == "openai_compatible" and not self.base_url:
            raise ValueError("base_url is required for openai_compatible judges")
        return self


class DecisionPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="tmqa.decision.v1", min_length=1)
    objective_fatal_clipping_ratio: float = Field(default=0.45, ge=0, le=1)
    objective_fatal_min_edge_similarity: float = Field(default=0.45, ge=0, le=1)
    regenerate_score_threshold: float = Field(default=0.60, ge=0, le=1)
    keep_score_threshold: float = Field(default=0.75, ge=0, le=1)
    low_tail_score_threshold: float = Field(default=0.70, ge=0, le=1)
    tone_stage_pass_threshold: float = Field(default=0.65, ge=0, le=1)
    fidelity_stage_pass_threshold: float = Field(default=0.75, ge=0, le=1)
    control_stage_pass_threshold: float = Field(default=0.75, ge=0, le=1)
    edge_risk_reference: float = Field(default=0.75, ge=0, le=1)
    clipping_risk_multiplier: float = Field(default=1.50, ge=0)
    uncertainty_base: float = Field(default=0.15, ge=0, le=1)
    uncertainty_gap_weight: float = Field(default=0.80, ge=0)
    uncertainty_control_weight: float = Field(default=0.20, ge=0)
    uncertainty_unavailable_weight: float = Field(default=0.35, ge=0)

    @model_validator(mode="after")
    def validate_score_threshold_order(self) -> DecisionPolicyConfig:
        if self.regenerate_score_threshold >= self.keep_score_threshold:
            raise ValueError("regenerate_score_threshold must be below keep_score_threshold")
        return self


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="tmqa.review.v1", min_length=1)
    max_queue_size: int = Field(default=200, gt=0)
    disagreement_threshold: float = Field(default=0.15, ge=0, le=1)
    tail_percentile: float = Field(default=0.10, gt=0, le=0.5)
    high_risk_threshold: float = Field(default=0.60, ge=0)
    random_audit_fraction: float = Field(default=0.0, ge=0, le=1)


class CalibrationPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="tmqa.calibration.v1", min_length=1)
    minimum_real_sample_count: int = Field(default=20, gt=0)


class PseudoGTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_judges: int = Field(default=2, ge=1)
    min_vote_ratio: float = Field(default=0.66, gt=0, le=1)
    min_selection_confidence: float = Field(default=0.75, ge=0, le=1)
    min_improvement: float = Field(default=0.08, ge=0, le=1)
    min_score_margin: float = Field(default=0.03, ge=0, le=1)
    max_clipping_ratio: float = Field(default=0.05, ge=0, le=1)
    max_clipping_increase: float = Field(default=0.02, ge=0, le=1)
    max_shadow_ratio: float = Field(default=0.60, ge=0, le=1)
    max_shadow_increase: float = Field(default=0.10, ge=0, le=1)
    max_color_drift: float = Field(default=0.08, ge=0, le=1)
    min_edge_similarity: float = Field(default=0.85, ge=0, le=1)
    fixed_shift_std_threshold_ev: float = Field(default=0.05, ge=0)
    fixed_shift_tolerance_ev: float = Field(default=0.05, ge=0)
    fixed_shift_fraction_threshold: float = Field(default=0.80, ge=0, le=1)


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = "Qwen-TMQA Visual Review"
    include_raw_response: bool = True
    include_prompt_diff: bool = True
    default_blind_review: bool = True


class TMQAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    judges: list[JudgeConfig]
    decision_policy: DecisionPolicyConfig = Field(default_factory=DecisionPolicyConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    calibration: CalibrationPolicyConfig = Field(default_factory=CalibrationPolicyConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    pseudo_gt: PseudoGTConfig = Field(default_factory=PseudoGTConfig)

    @field_validator("judges")
    @classmethod
    def unique_judges(cls, value: list[JudgeConfig]) -> list[JudgeConfig]:
        if len({item.id for item in value}) != len(value):
            raise ValueError("Judge identities must be unique")
        return value


def load_config(path: Path) -> TMQAConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TMQAConfig.model_validate(data)
