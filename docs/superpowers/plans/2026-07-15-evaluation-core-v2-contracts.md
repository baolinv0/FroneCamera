# Evaluation Core v2 Contract Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved v2 Pydantic contracts and frozen ten-dimension registry without changing runtime behavior.

**Architecture:** Add an isolated `portrait_eval.core` package beside the v0.1 modules. `models_v2.py` owns versioned enums, validation and persistent payload schemas. `dimensions.py` owns the fixed dimension policy. Existing database, pipeline, VLM, imaging, API, UI and report modules remain untouched.

**Tech Stack:** Python 3.12, Pydantic 2.x, pytest, Ruff, MyPy.

## Global Constraints

- `schema_version` is exactly `"2.0"` on every persistent v2 payload.
- Scoring scope is exactly `batch_relative`; no anchors or absolute industry score.
- Unknown fields are rejected with `ConfigDict(extra="forbid")`.
- Confidence/applicability fields are in `[0,1]`; scores in `[0,100]`; objective adjustment in `[-5,5]`.
- Pairwise preference and strength must agree: `+2,+1,0,-1,-2` from device A's perspective.
- `NOT_APPLICABLE` is explicit and is never converted to score zero.
- Quality and style remain separate.
- The exact fields and semantics are normative from `docs/superpowers/specs/2026-07-15-evaluation-core-v2-design.md`, sections 4–6.
- No migration, persistence, pipeline, prompt, vision backend or renderer change is permitted in this slice.
- All v0.1 tests must remain green.

---

## File Map

- Create `src/portrait_eval/core/__init__.py` — public exports.
- Create `src/portrait_eval/core/models_v2.py` — enums and all contracts from design sections 4 and 5.
- Create `src/portrait_eval/core/dimensions.py` — dimension definitions, weights and applicability metadata.
- Create `tests/test_models_v2_enums.py` — exact enum values.
- Create `tests/test_models_v2_contracts.py` — validation and round-trip tests.
- Create `tests/test_dimensions_v2.py` — policy invariants.

---

### Task 1: Frozen enum vocabulary and base contract

**Files:**
- Create: `src/portrait_eval/core/__init__.py`
- Create: `src/portrait_eval/core/models_v2.py`
- Create: `tests/test_models_v2_enums.py`

**Produces:** `SCHEMA_VERSION`, `V2Contract`, `DimensionId`, `MatchStatus`, `ComparabilityStatus`, `ApplicabilityStatus`, `DifferenceType`, `PairwisePreference`, `JudgeDecisionType`, `MechanismAttribution`, `ReportState`, `FaceRole`, `SceneScoreStatus`, `EvidenceGradeV2`, `ReportAllowedLevel`.

- [ ] **Step 1: Write the failing enum test**

```python
from portrait_eval.core.models_v2 import DimensionId, MatchStatus, SCHEMA_VERSION


def test_frozen_v2_vocabulary() -> None:
    assert SCHEMA_VERSION == "2.0"
    assert [item.value for item in DimensionId] == [
        "face_exposure_readability",
        "highlight_integrity",
        "shadow_black_rendering",
        "skin_awb",
        "lighting_causality",
        "face_background_relation",
        "local_face_lift_naturalness",
        "multi_face_consistency",
        "scene_adaptability",
        "artifact_texture_control",
    ]
    assert [item.value for item in MatchStatus] == [
        "CONFIRMED_MANIFEST",
        "CONFIRMED_MANUAL",
        "AUTO_HIGH_CONFIDENCE",
        "PENDING_REVIEW",
        "UNMATCHED",
        "INVALID",
    ]
```

Add equivalent assertions for every enum listed under **Produces**, using the exact values in design section 4.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_models_v2_enums.py -q`

Expected: import fails because `portrait_eval.core` does not exist.

- [ ] **Step 3: Implement base and enums**

```python
from enum import StrEnum
from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "2.0"


class V2Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DimensionId(StrEnum):
    FACE_EXPOSURE_READABILITY = "face_exposure_readability"
    HIGHLIGHT_INTEGRITY = "highlight_integrity"
    SHADOW_BLACK_RENDERING = "shadow_black_rendering"
    SKIN_AWB = "skin_awb"
    LIGHTING_CAUSALITY = "lighting_causality"
    FACE_BACKGROUND_RELATION = "face_background_relation"
    LOCAL_FACE_LIFT_NATURALNESS = "local_face_lift_naturalness"
    MULTI_FACE_CONSISTENCY = "multi_face_consistency"
    SCENE_ADAPTABILITY = "scene_adaptability"
    ARTIFACT_TEXTURE_CONTROL = "artifact_texture_control"
```

Implement the remaining enums with the exact names and values from design section 4. Export `SCHEMA_VERSION` and `DimensionId` from `core/__init__.py`.

- [ ] **Step 4: Verify pass and static quality**

Run:

```bash
pytest tests/test_models_v2_enums.py -q
ruff check src/portrait_eval/core tests/test_models_v2_enums.py
mypy src/portrait_eval/core
```

Expected: tests pass and both static checks exit `0`.

- [ ] **Step 5: Commit**

```bash
git add src/portrait_eval/core tests/test_models_v2_enums.py
git commit -m "feat: add frozen Evaluation Core v2 enums"
```

---

### Task 2: Persistent v2 contracts and semantic validators

**Files:**
- Modify: `src/portrait_eval/core/models_v2.py`
- Create: `tests/test_models_v2_contracts.py`

**Produces:**

`EvaluationBatchV2`, `MatchEvidenceV2`, `MatchedSceneGroupV2`, `FaceRegionV2`, `ObjectiveEvidenceV2`, `DimensionApplicabilityV2`, `RoughRankingV2`, `PairwiseComparisonV2`, `FactCheckResultV2`, `JudgeDecisionV2`, `SceneDimensionScoreV2`, `DeviceDimensionScoreV2`, `DeviceOverallScoreV2`, `MechanismInterpretationV2`, `ReportEvidencePackageV2`.

Each class uses the exact field names and meanings from design sections 5.1–5.12. Add `faces: list[FaceRegionV2]` to `ObjectiveEvidenceV2`, because multi-face evidence must be persisted rather than discarded.

- [ ] **Step 1: Write failing construction and round-trip tests**

```python
from datetime import UTC, datetime
import pytest
from pydantic import ValidationError

from portrait_eval.core.models_v2 import EvaluationBatchV2


def make_batch() -> EvaluationBatchV2:
    return EvaluationBatchV2(
        batch_id="batch-001",
        project_id="project-001",
        device_ids=["a", "b"],
        dataset_version="dataset-v1",
        dimension_policy_version="dimensions-v2.0",
        scoring_policy_version="batch-relative-v1",
        primary_model_version="primary-v1",
        judge_model_version="judge-v1",
        report_model_version=None,
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def test_batch_defaults_and_round_trip() -> None:
    batch = make_batch()
    assert batch.schema_version == "2.0"
    assert batch.task_scope == "front_camera_portrait_tone_semantic_rendering"
    assert batch.scoring_scope == "batch_relative"
    assert EvaluationBatchV2.model_validate_json(batch.model_dump_json()) == batch


def test_unknown_fields_are_rejected() -> None:
    payload = make_batch().model_dump()
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        EvaluationBatchV2.model_validate(payload)
```

Add focused tests for:

- two persisted faces in `ObjectiveEvidenceV2`;
- all confidence bounds;
- score and adjustment bounds;
- report package JSON round trip;
- every persistent class declaring `schema_version`.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_models_v2_contracts.py -q`

Expected: import errors for missing classes.

- [ ] **Step 3: Implement bounded fields**

Use these exact Pydantic patterns throughout:

```python
schema_version: Literal["2.0"] = SCHEMA_VERSION
confidence: float = Field(ge=0, le=1)
normalized_score: float = Field(ge=0, le=100)
objective_adjustment: float = Field(ge=-5, le=5)
score_scope: Literal["batch_relative"] = "batch_relative"
```

Use `Field(default_factory=list)` and `Field(default_factory=dict)` for mutable defaults. Do not reuse or reinterpret legacy v0.1 models.

- [ ] **Step 4: Implement pairwise consistency validator**

```python
PAIRWISE_STRENGTH = {
    PairwisePreference.A_STRONGLY_BETTER: 2,
    PairwisePreference.A_SLIGHTLY_BETTER: 1,
    PairwisePreference.EQUIVALENT: 0,
    PairwisePreference.B_SLIGHTLY_BETTER: -1,
    PairwisePreference.B_STRONGLY_BETTER: -2,
}


@model_validator(mode="after")
def validate_preference_strength(self) -> "PairwiseComparisonV2":
    if self.strength != PAIRWISE_STRENGTH[self.preference]:
        raise ValueError("pairwise preference and strength do not agree")
    if self.device_a_id == self.device_b_id:
        raise ValueError("pairwise devices must be distinct")
    return self
```

Add a test that `A_SLIGHTLY_BETTER` with strength `-1` raises `ValidationError`.

- [ ] **Step 5: Implement judge semantic validator**

```python
@model_validator(mode="after")
def validate_accepted_semantics(self) -> "JudgeDecisionV2":
    accepted = (
        self.accepted_preference,
        self.accepted_strength,
        self.accepted_difference_type,
    )
    requires_result = self.decision in {JudgeDecisionType.ACCEPT, JudgeDecisionType.REVISE}
    if requires_result and any(value is None for value in accepted):
        raise ValueError("accept/revise requires accepted semantic fields")
    if not requires_result and any(value is not None for value in accepted):
        raise ValueError("reject/escalate cannot carry accepted semantic fields")
    if self.accepted_preference is not None:
        if self.accepted_strength != PAIRWISE_STRENGTH[self.accepted_preference]:
            raise ValueError("accepted preference and strength do not agree")
    return self
```

Add tests for both invalid directions.

- [ ] **Step 6: Verify all model contracts**

Run:

```bash
pytest tests/test_models_v2_enums.py tests/test_models_v2_contracts.py -q
ruff check src/portrait_eval/core tests/test_models_v2_enums.py tests/test_models_v2_contracts.py
mypy src/portrait_eval/core
```

Expected: all commands exit `0`.

- [ ] **Step 7: Commit**

```bash
git add src/portrait_eval/core/models_v2.py tests/test_models_v2_contracts.py
git commit -m "feat: add Evaluation Core v2 data contracts"
```

---

### Task 3: Frozen ten-dimension policy registry

**Files:**
- Create: `src/portrait_eval/core/dimensions.py`
- Modify: `src/portrait_eval/core/__init__.py`
- Create: `tests/test_dimensions_v2.py`

**Produces:** `DimensionScope`, `DimensionDefinition`, `DIMENSIONS`, `DIMENSION_BY_ID`, `TOTAL_DIMENSION_WEIGHT`, `get_dimension()`.

- [ ] **Step 1: Write failing registry tests**

```python
from math import isclose
from portrait_eval.core.dimensions import DIMENSIONS, TOTAL_DIMENSION_WEIGHT, get_dimension
from portrait_eval.core.models_v2 import DimensionId

EXPECTED = [
    (DimensionId.FACE_EXPOSURE_READABILITY, 0.15),
    (DimensionId.HIGHLIGHT_INTEGRITY, 0.12),
    (DimensionId.SHADOW_BLACK_RENDERING, 0.08),
    (DimensionId.SKIN_AWB, 0.12),
    (DimensionId.LIGHTING_CAUSALITY, 0.10),
    (DimensionId.FACE_BACKGROUND_RELATION, 0.10),
    (DimensionId.LOCAL_FACE_LIFT_NATURALNESS, 0.10),
    (DimensionId.MULTI_FACE_CONSISTENCY, 0.08),
    (DimensionId.SCENE_ADAPTABILITY, 0.10),
    (DimensionId.ARTIFACT_TEXTURE_CONTROL, 0.05),
]


def test_policy_is_exact() -> None:
    assert [(item.id, item.weight) for item in DIMENSIONS] == EXPECTED
    assert tuple(item.id for item in DIMENSIONS) == tuple(DimensionId)
    assert isclose(TOTAL_DIMENSION_WEIGHT, 1.0, abs_tol=1e-12)


def test_scene_adaptability_is_cross_scene() -> None:
    item = get_dimension(DimensionId.SCENE_ADAPTABILITY)
    assert item.scope.value == "CROSS_SCENE"
    assert item.single_scene_applicable is False
```

Add a test requiring `at_least_two_valid_faces_in_at_least_two_devices` for `multi_face_consistency`.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_dimensions_v2.py -q`

Expected: import fails because `dimensions.py` does not exist.

- [ ] **Step 3: Implement registry types and invariants**

```python
class DimensionScope(StrEnum):
    SCENE = "SCENE"
    CROSS_SCENE = "CROSS_SCENE"


class DimensionDefinition(V2Contract):
    id: DimensionId
    display_name: str
    weight: float = Field(gt=0, le=1)
    evaluation_unit: str
    scope: DimensionScope
    single_scene_applicable: bool
    required_evidence: tuple[str, ...]
```

Create the ten definitions in the exact design order with the approved names, weights and evaluation units. Add explicit required-evidence tokens for each dimension. Build:

```python
DIMENSION_BY_ID = {item.id: item for item in DIMENSIONS}
TOTAL_DIMENSION_WEIGHT = sum(item.weight for item in DIMENSIONS)
```

Raise `RuntimeError` at import time when IDs are missing/duplicated or the total weight differs from `1.0` by more than `1e-12`.

- [ ] **Step 4: Verify registry and all focused tests**

Run:

```bash
pytest tests/test_models_v2_enums.py tests/test_models_v2_contracts.py tests/test_dimensions_v2.py -q
ruff check src/portrait_eval/core tests/test_models_v2_enums.py tests/test_models_v2_contracts.py tests/test_dimensions_v2.py
mypy src/portrait_eval/core
```

Expected: all commands exit `0`.

- [ ] **Step 5: Commit**

```bash
git add src/portrait_eval/core tests/test_dimensions_v2.py
git commit -m "feat: add frozen ten-dimension evaluation policy"
```

---

### Task 4: Full compatibility and quality gate

**Files:**
- Modify: `tests/test_models_v2_contracts.py`
- Modify: `tests/test_dimensions_v2.py`

- [ ] **Step 1: Add regression assertions**

Assert:

```python
PERSISTENT_MODELS = [
    EvaluationBatchV2,
    MatchedSceneGroupV2,
    ObjectiveEvidenceV2,
    DimensionApplicabilityV2,
    RoughRankingV2,
    PairwiseComparisonV2,
    FactCheckResultV2,
    JudgeDecisionV2,
    SceneDimensionScoreV2,
    DeviceDimensionScoreV2,
    DeviceOverallScoreV2,
    MechanismInterpretationV2,
    ReportEvidencePackageV2,
]

for model in PERSISTENT_MODELS:
    assert model.model_fields["schema_version"].default == "2.0"
```

Also assert the frozen pairwise and judge field names are present, preventing silent renames.

- [ ] **Step 2: Run focused and full tests**

```bash
pytest tests/test_models_v2_enums.py tests/test_models_v2_contracts.py tests/test_dimensions_v2.py -q
pytest -q
```

Expected: all v2 and existing v0.1 tests pass.

- [ ] **Step 3: Run repository quality gates**

```bash
ruff check src tests scripts
ruff format --check src scripts
mypy src/portrait_eval
```

Expected: every command exits `0`.

- [ ] **Step 4: Confirm runtime isolation**

Run:

```bash
git diff --name-only feature/full-system...HEAD
```

Production changes must be limited to:

```text
src/portrait_eval/core/__init__.py
src/portrait_eval/core/models_v2.py
src/portrait_eval/core/dimensions.py
```

No database, migration, pipeline, API, UI, VLM, imaging, legacy model or report file may change.

- [ ] **Step 5: Commit**

```bash
git add tests/test_models_v2_contracts.py tests/test_dimensions_v2.py
git commit -m "test: freeze Evaluation Core v2 contract invariants"
```

---

## Completion Gate

This slice is complete only when:

1. all frozen enums and ten dimensions are exact;
2. every persistent contract round-trips through JSON;
3. unknown fields and out-of-range values are rejected;
4. pairwise and judge semantic contradictions are rejected;
5. multi-face evidence is representable;
6. every persistent payload carries schema version `2.0`;
7. the full v0.1 suite remains green;
8. Ruff, formatting and MyPy pass;
9. no runtime behavior outside `portrait_eval.core` changes.
