# Evaluation Core v2 — Frozen Data Contract and Architecture Design

**Status:** FROZEN FOR REVIEW  
**Branch:** `feature/evaluation-core-v2`  
**Base:** `feature/full-system`  
**Scope:** local ISP competitor evaluation for front-camera portrait tone rendering and semantic rendering  
**Score semantics:** batch-relative only; not an absolute industry score

## 1. Product boundary

Evaluation Core v2 evaluates matched front-camera portrait outputs from `N >= 2` devices. It supports two data-entry paths:

1. standardized capture groups with explicit scene IDs;
2. historical device folders with automatic matching proposals and confidence-based review.

The v2 core evaluates only static front-camera portrait rendering. It does not evaluate rear cameras, video, RAW sensor performance, AF latency, laboratory MTF, or universal camera quality.

The v2 pipeline is:

```text
registered images
→ matching and comparability
→ semantic regions and objective evidence
→ dimension applicability
→ rough ranking
→ adaptive pairwise comparison
→ deterministic fact validation
→ independent judge adjudication
→ latent-quality estimation
→ batch-relative normalization
→ cross-scene device profile
→ report evidence package
→ HTML / DOCX / PPTX renderers
```

## 2. Non-negotiable invariants

1. A VLM may propose observations and comparisons, but it may not overwrite objective measurements.
2. A report generator may only consume adjudicated evidence packages; it may not independently score source images.
3. Quality and style are separate outputs.
4. A missing or inapplicable dimension is `NOT_APPLICABLE`, never zero.
5. Batch-relative scores are scoped by `batch_id`, dataset version, model versions, prompt versions, and scoring-policy version.
6. JPEG/HEIC output evidence cannot prove a proprietary ISP implementation.
7. Mechanism statements are hypotheses with evidence level and confidence.
8. Unresolved critical matching, fact, or judge conflicts cannot enter a reviewed report.
9. The legacy v0.1 pipeline remains runnable during migration.
10. All persistent v2 payloads include `schema_version = "2.0"`.

## 3. Frozen evaluation dimensions

| ID | Name | Weight | Unit of evaluation |
|---|---|---:|---|
| `face_exposure_readability` | Face exposure and readability | 0.15 | per face and scene |
| `highlight_integrity` | Highlight integrity | 0.12 | semantic highlight regions |
| `shadow_black_rendering` | Shadow and black rendering | 0.08 | scene and background |
| `skin_awb` | Skin tone and white balance | 0.12 | skin regions and environment relation |
| `lighting_causality` | Lighting causality and dimensionality | 0.10 | face internal light structure |
| `face_background_relation` | Face/background relation and separation | 0.10 | face, face ring, background |
| `local_face_lift_naturalness` | Local face-lift naturalness | 0.10 | face gain and transition behavior |
| `multi_face_consistency` | Multi-face consistency | 0.08 | two or more valid faces |
| `scene_adaptability` | Scene adaptability and stability | 0.10 | cross-scene device level |
| `artifact_texture_control` | Artifact and texture control | 0.05 | semantic edges and face texture |

Weights sum to `1.00`. They are configuration-controlled but versioned. Changing them creates a new scoring-policy version.

## 4. Frozen enums

### 4.1 Match status

```text
CONFIRMED_MANIFEST
CONFIRMED_MANUAL
AUTO_HIGH_CONFIDENCE
PENDING_REVIEW
UNMATCHED
INVALID
```

Only the first three statuses can enter formal scoring.

### 4.2 Comparability status

```text
FULLY_COMPARABLE
COMPARABLE_WITH_CONFOUNDERS
NOT_COMPARABLE
```

### 4.3 Applicability status

```text
APPLICABLE
WEAKLY_APPLICABLE
NOT_APPLICABLE
INSUFFICIENT_EVIDENCE
```

### 4.4 Difference type

```text
QUALITY
STYLE_PREFERENCE
MIXED
INSUFFICIENT_EVIDENCE
NOT_APPLICABLE
```

### 4.5 Pairwise preference

```text
A_STRONGLY_BETTER
A_SLIGHTLY_BETTER
EQUIVALENT
B_SLIGHTLY_BETTER
B_STRONGLY_BETTER
```

The numeric strength mapping is `+2, +1, 0, -1, -2` from A's perspective.

### 4.6 Judge decision

```text
ACCEPT
REVISE
REJECT
MANUAL_REVIEW
INVALID_SAMPLE
```

### 4.7 Mechanism attribution

```text
HARDWARE_ENABLED
ALGORITHM_DEFINED
PREFERENCE_DRIVEN
CAPTURE_LIMITED
RECONSTRUCTION_LIMITED
STRATEGY_LIMITED
MIXED_CAUSE
INDETERMINATE
```

### 4.8 Report state

```text
AUTO_REPORT
REVIEWED_REPORT
```

`AUTO_REPORT` may contain non-critical notes. `REVIEWED_REPORT` requires all critical conflicts to be resolved.

## 5. Persistent data contracts

The following contracts are normative. Implementations may add derived fields, but may not rename or change the meaning of frozen fields without a schema-version increment.

### 5.1 Evaluation batch

```python
class EvaluationBatchV2:
    schema_version: Literal["2.0"]
    batch_id: str
    project_id: str
    task_scope: Literal["front_camera_portrait_tone_semantic_rendering"]
    device_ids: list[str]
    scoring_scope: Literal["batch_relative"]
    dataset_version: str
    dimension_policy_version: str
    scoring_policy_version: str
    primary_model_version: str
    judge_model_version: str
    report_model_version: str | None
    created_at: datetime
```

### 5.2 Match candidate and confirmed scene group

```python
class MatchEvidenceV2:
    exif_time_score: float
    perceptual_hash_score: float
    scene_embedding_score: float
    face_geometry_score: float | None
    composition_score: float
    aggregate_score: float

class MatchedSceneGroupV2:
    schema_version: Literal["2.0"]
    scene_id: str
    batch_id: str
    label: str | None
    cells: dict[str, str | None]  # device_id -> image_id
    match_status: MatchStatus
    match_confidence: float
    match_evidence: MatchEvidenceV2
    comparability_status: ComparabilityStatus
    confounders: list[str]
    confirmed_by: str | None
```

### 5.3 Semantic image analysis

```python
class FaceRegionV2:
    face_id: str
    bbox_xywh: tuple[int, int, int, int]
    detection_confidence: float
    area_ratio: float
    center_xy_normalized: tuple[float, float]
    landmarks_ref: str | None
    masks: dict[str, str]  # region name -> artifact/asset ID
    role: Literal["PRIMARY", "SECONDARY", "UNASSIGNED"]

class ObjectiveEvidenceV2:
    image_id: str
    whole_image_metrics: dict[str, float]
    region_metrics: dict[str, dict[str, float]]
    per_face_metrics: dict[str, dict[str, float]]
    relation_metrics: dict[str, float]
    artifact_metrics: dict[str, float]
    warnings: list[str]
    metric_version: str
```

Required region names when available:

```text
face
skin
forehead
left_cheek
right_cheek
eye_region
mouth_region
face_shadow
face_highlight
hair
face_edge
glasses
face_ring
background
semantic_highlight
semantic_shadow
```

Required metric families:

- display-referred luminance quantiles and clipping ratios;
- Lab/HSV skin statistics and skin/environment chroma relation;
- face/background and face/ring luminance ratios;
- per-face brightness, color and area differences;
- face internal light ratio and local contrast;
- edge transition, halo, blur, oversharpening and texture/noise proxies;
- semantic highlight texture and boundary proxies.

### 5.4 Dimension applicability

```python
class DimensionApplicabilityV2:
    scene_id: str
    dimension_id: DimensionId
    status: ApplicabilityStatus
    score: float
    reasons: list[str]
    required_evidence_present: bool
```

Examples:

- `multi_face_consistency` requires at least two valid faces in at least two devices;
- `highlight_integrity` requires a comparable semantic highlight region;
- `skin_awb` requires a valid skin region and meaningful illumination/color evidence;
- `scene_adaptability` is not evaluated at single-scene level.

### 5.5 VLM rough ranking

```python
class RoughRankingV2:
    scene_id: str
    dimension_id: DimensionId
    ordered_device_ids: list[str]
    tie_groups: list[list[str]]
    confidence: float
    evidence_refs: list[str]
    prompt_version: str
    model_version: str
```

Rough ranking selects pairwise comparisons. It never directly determines the final score.

### 5.6 Pairwise comparison

```python
class PairwiseComparisonV2:
    comparison_id: str
    scene_id: str
    dimension_id: DimensionId
    device_a_id: str
    device_b_id: str
    applicability: ApplicabilityStatus
    preference: PairwisePreference
    strength: int  # -2..2 from A perspective
    difference_type: DifferenceType
    confidence: float
    observations: list[str]
    evidence_refs: list[str]
    style_axes: dict[str, str | float]
    primary_model_version: str
    prompt_version: str
```

A quality score difference is produced only when `difference_type` is `QUALITY` or `MIXED`. Pure style preference comparisons are retained for profiles but are score-neutral or strongly damped.

### 5.7 Deterministic fact validation

```python
class FactCheckResultV2:
    comparison_id: str
    rule_version: str
    passed_rules: list[str]
    failed_rules: list[str]
    warnings: list[str]
    critical_conflict: bool
    objective_support: float
```

Rules are explicit predicates. Examples:

- a claim that A has a brighter face requires a sufficient positive face-luminance delta;
- a claim that A has better highlight integrity cannot rely only on a lower global exposure;
- a claim of local face lift requires face gain to exceed background gain;
- a claim of better multi-face consistency requires lower unexplained inter-face dispersion;
- color claims require valid skin masks and cannot be inferred from global RGB alone.

### 5.8 Independent judge decision

```python
class JudgeDecisionV2:
    comparison_id: str
    decision: JudgeDecisionType
    accepted_preference: PairwisePreference | None
    accepted_strength: int | None
    accepted_difference_type: DifferenceType | None
    revised_observations: list[str]
    fact_conflicts: list[str]
    unsupported_attributions: list[str]
    confidence: float
    judge_model_version: str
    prompt_version: str
```

The judge must compare semantic direction, not only device and dimension labels.

### 5.9 Scene-dimension score

```python
class SceneDimensionScoreV2:
    scene_id: str
    dimension_id: DimensionId
    device_id: str
    latent_quality: float
    normalized_score: float  # 0..100, batch-relative
    confidence: float
    diagnostic_weight: float
    objective_adjustment: float  # bounded to [-5, 5]
    final_score: float
    status: Literal["AUTO_PASS", "PASS_WITH_NOTE", "MANUAL_REVIEW", "NOT_APPLICABLE"]
    contributing_comparison_ids: list[str]
```

### 5.10 Device-dimension score and overall score

```python
class DeviceDimensionScoreV2:
    batch_id: str
    device_id: str
    dimension_id: DimensionId
    aggregate_score: float
    rank: int | None
    rank_group: int | None
    confidence: float
    applicable_scene_count: int
    style_position: dict[str, str | float]
    supporting_scene_ids: list[str]
    counterexample_scene_ids: list[str]

class DeviceOverallScoreV2:
    batch_id: str
    device_id: str
    relative_score: float
    rank: int
    confidence: float
    dimension_scores: dict[DimensionId, float]
    score_scope: Literal["batch_relative"]
```

### 5.11 Mechanism interpretation

```python
class MechanismInterpretationV2:
    interpretation_id: str
    device_id: str
    scene_ids: list[str]
    observable_effect: str
    strategy_inference: str
    mechanism_hypothesis: str | None
    attribution: MechanismAttribution
    confidence: float
    evidence_grade: Literal["A", "B", "C"]
    supporting_evidence_refs: list[str]
    alternative_explanations: list[str]
    report_allowed_level: Literal["OBSERVABLE", "STRATEGY_INFERENCE", "INTERNAL_HYPOTHESIS"]
```

Default report output is restricted to `OBSERVABLE` and `STRATEGY_INFERENCE`.

### 5.12 Report evidence package

```python
class ReportEvidencePackageV2:
    schema_version: Literal["2.0"]
    batch: EvaluationBatchV2
    scene_summaries: list[dict]
    device_dimension_scores: list[DeviceDimensionScoreV2]
    device_overall_scores: list[DeviceOverallScoreV2]
    device_profiles: list[dict]
    mechanism_interpretations: list[MechanismInterpretationV2]
    selected_visual_assets: list[dict]
    review_notes: list[dict]
    limitations: list[str]
    prohibited_claims: list[str]
```

The report language model receives only this package.

## 6. Scoring contract

### 6.1 Pairwise graph

For each applicable scene and dimension:

1. obtain a rough ordering;
2. compare adjacent devices;
3. add comparisons automatically when confidence is low, a fact conflict exists, the judge revises a result, or the graph contains a cycle;
4. fit a Bradley–Terry latent-quality model from accepted quality-bearing comparisons;
5. preserve style-only comparisons separately.

### 6.2 Batch-relative normalization

Min-max scaling is prohibited. The normalized score is:

```text
score = 50 + alpha * tanh((q - median(q)) / tau)
```

where:

```text
alpha = 35 by default
tau = max(MAD(q), tau_min)
```

A low-dispersion dimension compresses score differences and may produce tied rank groups.

### 6.3 Objective adjustment

Objective evidence may adjust an accepted perceptual score by at most `±5`. A critical contradiction triggers judge review instead of a numerical correction.

### 6.4 Aggregation

Scene scores are aggregated with:

```text
diagnostic_weight × match_confidence × comparability_confidence
× primary_confidence × judge_confidence × rule_confidence
```

The denominator includes only applicable scenes. Missing dimensions are not treated as zero.

## 7. Model-role separation

### Primary VLM

- scene interpretation;
- rough ranking;
- pairwise comparison;
- visible evidence description;
- quality/style classification.

### Deterministic rule engine

- metric predicates;
- direction and threshold checks;
- applicability checks;
- cycle and consistency detection.

### Independent judge VLM

- accept, revise, reject or escalate;
- compare conclusion direction and evidence;
- downgrade unsupported mechanism attribution;
- distinguish quality failure from style preference.

### Report LLM

- receives no source images;
- receives only adjudicated evidence package;
- writes constrained scene summaries, device profiles, generation comparisons and limitations;
- cannot create new scores or evidence.

## 8. Error and review policy

Critical errors:

- scene mismatch or match confidence below formal threshold;
- inconsistent face count that invalidates the target dimension;
- missing semantic region required by the dimension;
- primary/judge direction conflict after retry;
- critical objective contradiction;
- disconnected or cyclic pairwise graph that cannot be resolved;
- malformed persistent schema.

Critical errors create review items and block `REVIEWED_REPORT`. Non-critical uncertainty produces `PASS_WITH_NOTE` and conservative wording.

## 9. Compatibility and migration rules

1. Existing v0.1 tables and payloads remain readable.
2. v2 analyses use new analysis types or new tables; they do not reinterpret v0.1 JSON in place.
3. The current `EvaluationPipeline` remains available as `legacy_v01` until cutover.
4. New API routes are versioned under `/api/v2/`.
5. v2 report files are stored separately from v0.1 report bundles.
6. Legacy heuristic VLM output is allowed only for infrastructure smoke tests and cannot produce reviewed scores.
7. Database migrations are additive until v2 reaches cutover criteria.

## 10. Acceptance criteria for the design

The implementation is conformant only when:

1. all ten dimensions use the frozen IDs and applicability contract;
2. two or more faces are persisted and evaluated independently;
3. pairwise comparisons separate quality from style;
4. deterministic rules can detect at least brightness, highlight, local-lift and multi-face direction conflicts;
5. the judge compares semantic direction and can revise results;
6. scores are computed from accepted pairwise evidence and normalized within batch;
7. repeated runs with frozen models and prompts produce stable ordering within defined tolerance;
8. every report statement is traceable to evidence references;
9. the system can generate HTML, DOCX and PPTX from the same evidence package;
10. v0.1 remains runnable during migration.

## 11. Explicit non-goals for v2 MVP

- absolute industry-calibrated IQA scores;
- human-maintained anchor images;
- cloud model APIs;
- RAW reconstruction or sensor dynamic-range measurement;
- rear-camera and video evaluation;
- automated capture control;
- model training or fine-tuning;
- multi-user cloud deployment;
- public claims about proprietary ISP implementation.
