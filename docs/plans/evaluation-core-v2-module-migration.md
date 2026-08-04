# Evaluation Core v2 — Frozen Module Migration Plan

**Status:** FROZEN FOR REVIEW
**Target branch:** `feature/evaluation-core-v2`
**Source branch:** `feature/full-system`
**Design contract:** `docs/superpowers/specs/2026-07-15-evaluation-core-v2-design.md`

## 1. Migration strategy

The repository will not be rewritten. The migration preserves the operational shell and replaces the evaluation science in isolated v2 modules.

```text
retain project/database/task/report lifecycle
+ add versioned v2 contracts and APIs
+ implement new semantic and scoring core
+ run v0.1 and v2 in parallel
+ cut over only after evidence-based gates pass
```

All database changes are additive until cutover. The legacy pipeline remains available for smoke tests and regression comparison.

## 2. Keep, extend, replace

| Existing area | Decision | v2 action |
|---|---|---|
| `database.py` | Keep | add additive migrations and v2 persistence |
| `repository.py` | Extend | add typed v2 save/load methods; do not overload generic analysis JSON indefinitely |
| `tasking.py` | Keep | add v2 stage task types and dependencies |
| `worker.py` | Extend | dispatch stage-specific v2 workers |
| `api.py`, `product_api.py` | Extend | add `/api/v2/` routes; keep v0.1 routes |
| React/Vite shell | Keep | add v2 match, evidence, score and report views |
| pairing snapshots | Keep | add matching confidence and evidence |
| review items | Keep | add typed critical/non-critical v2 conflicts |
| `dataset.py` | Replace internally | move legacy natural-sort code under compatibility layer; create matching package |
| `imaging.py` | Replace internally | split into semantic vision modules; retain legacy facade |
| `models.py` | Keep legacy | add `core/models_v2.py`; no in-place semantic reinterpretation |
| `vlm.py` | Keep adapter | add role-specific protocols and prompts |
| `adjudication.py` | Replace for v2 | implement explicit rules and judge decisions |
| `strategy.py` | Replace for v2 | implement cross-scene style/profile engine |
| `reporting.py` | Split | evidence package, narrative and renderers |
| external research | Defer | keep operational, remove from v2 critical path |
| hardware attribution | Defer/extend later | consume v2 interpretations after scoring is stable |

## 3. Target package structure

```text
src/portrait_eval/
├── core/
│   ├── __init__.py
│   ├── models_v2.py
│   ├── dimensions.py
│   ├── evidence.py
│   └── versions.py
├── matching/
│   ├── __init__.py
│   ├── standardized.py
│   ├── automatic.py
│   ├── features.py
│   └── confidence.py
├── vision/
│   ├── __init__.py
│   ├── loader.py
│   ├── face_detection.py
│   ├── face_parsing.py
│   ├── regions.py
│   ├── photometry.py
│   ├── color_metrics.py
│   ├── artifact_metrics.py
│   └── comparability.py
├── evaluation/
│   ├── __init__.py
│   ├── applicability.py
│   ├── rough_ranking.py
│   ├── pairwise.py
│   ├── rules.py
│   ├── judge.py
│   ├── bradley_terry.py
│   ├── normalization.py
│   └── aggregation.py
├── profiling/
│   ├── __init__.py
│   ├── cross_scene.py
│   ├── style.py
│   └── mechanism.py
├── reporting_v2/
│   ├── __init__.py
│   ├── evidence_package.py
│   ├── narrative.py
│   ├── assets.py
│   ├── html_renderer.py
│   ├── docx_renderer.py
│   └── pptx_renderer.py
├── pipeline_v2.py
└── legacy/
    └── v01_compat.py
```

The initial implementation may use fewer files, but dependencies must follow these boundaries.

## 4. Database migration plan

### 4.1 Additive entities

Create additive tables or equivalent strongly typed records for:

- evaluation batches;
- v2 matched scene groups and match evidence;
- semantic faces and region artifacts;
- objective evidence;
- dimension applicability;
- rough rankings;
- pairwise comparisons;
- fact-check results;
- judge decisions;
- scene-dimension scores;
- device-dimension scores;
- device overall scores;
- mechanism interpretations;
- report evidence packages.

### 4.2 Persistence rules

- Existing v0.1 records are never rewritten as v2 records.
- Each v2 record includes batch, schema and policy versions.
- Large masks and visual assets are stored as registered artifacts, not database blobs.
- Pairwise and judge outputs are immutable; retries create new attempts with lineage.
- Aggregated scores reference the accepted comparison attempt IDs.

### 4.3 Migration ordering

1. create v2 tables and indexes;
2. add repository methods;
3. add read-only inspection APIs;
4. add write pipeline stages;
5. add report package persistence;
6. add cleanup/retention policy after cutover.

## 5. Pipeline stage migration

### Stage 0 — Batch creation

**Input:** project, device set, dataset version, model/policy versions.
**Output:** `EvaluationBatchV2`.

### Stage 1 — Matching

- standardized scene IDs bypass automatic matching;
- historical folders use time, perceptual hash, scene embedding, face geometry and composition;
- Hungarian assignment proposes groups;
- high-confidence groups may auto-confirm;
- ambiguous groups create review items.

### Stage 2 — Semantic analysis

- detect all valid faces;
- assign primary/secondary roles without discarding other faces;
- generate semantic masks and diagnostic crops;
- compute objective evidence and relation metrics;
- persist warnings and model/backend versions.

### Stage 3 — Comparability and applicability

- audit scene-level confounders;
- calculate each dimension's applicability;
- exclude invalid or inapplicable evidence before VLM scoring.

### Stage 4 — Primary evaluation

- rough ranking per applicable dimension;
- adjacent pairwise comparisons;
- retain quality/style classification and evidence refs.

### Stage 5 — Rules and judge

- run explicit deterministic predicates;
- submit conflicts and low-confidence comparisons to independent judge;
- expand pairwise graph on cycles or unresolved adjacency;
- produce accepted comparison set.

### Stage 6 — Scoring

- fit Bradley–Terry latent qualities;
- apply robust batch-relative normalization;
- apply bounded objective adjustments;
- aggregate scene and device dimension scores;
- create tied rank groups when dispersion is insufficient.

### Stage 7 — Device profiling

- aggregate style axes and repeated patterns;
- retain counterexamples;
- produce observable effects, strategy inferences and mechanism hypotheses;
- restrict report-allowed claim level.

### Stage 8 — Reporting

- build a versioned report evidence package;
- generate constrained narrative JSON;
- render HTML, DOCX and PPTX from the same package;
- store `AUTO_REPORT` and `REVIEWED_REPORT` as distinct states.

## 6. API migration

Add versioned endpoints without breaking current clients.

```text
POST   /api/v2/projects/{project_id}/batches
POST   /api/v2/batches/{batch_id}/matching/run
GET    /api/v2/batches/{batch_id}/matching
POST   /api/v2/batches/{batch_id}/semantic-analysis/run
GET    /api/v2/batches/{batch_id}/evidence
POST   /api/v2/batches/{batch_id}/evaluation/run
GET    /api/v2/batches/{batch_id}/pairwise
GET    /api/v2/batches/{batch_id}/scores
GET    /api/v2/batches/{batch_id}/profiles
POST   /api/v2/batches/{batch_id}/reports
GET    /api/v2/reports/{report_id}
```

Mutation endpoints use optimistic version checks where users can change matching or review decisions.

## 7. UI migration

The existing four-step product flow remains, but v2 adds explicit evidence views.

### Step 1 — Create test data

- select standardized or historical-folder mode;
- display automatic match confidence and unresolved cells;
- allow manual corrections only where needed.

### Step 2 — Run evaluation

- stage progress by matching, semantic analysis, pairwise, judge and scoring;
- show invalid or blocked scenes;
- do not expose arbitrary raw model text as the primary interface.

### Step 3 — Review results

- ten-dimension scorecard;
- pairwise evidence and accepted judge decision;
- whole image plus face/highlight/background crops;
- quality/style separation;
- mechanism interpretation with confidence and alternatives.

### Step 4 — Generate report

- choose HTML, DOCX and PPTX;
- distinguish auto and reviewed reports;
- show batch-relative scope and frozen model/policy versions.

## 8. Test migration

### 8.1 Contract tests

- enum values and frozen dimension IDs;
- schema serialization and validation;
- prohibited renames and missing version fields.

### 8.2 Unit tests

- matching feature aggregation and confidence;
- dimension applicability;
- claim-to-metric predicates;
- pairwise graph expansion and cycle detection;
- Bradley–Terry fitting;
- MAD/tanh normalization;
- quality/style score damping;
- report claim-level restrictions.

### 8.3 Integration tests

- standardized dataset → v2 report package;
- historical folders → match proposal → confirmed groups;
- multi-face scene → per-face evidence → multi-face score;
- fact conflict → judge revise/reject;
- low dispersion → compressed score differences and tie groups;
- v0.1 and v2 pipelines runnable in the same repository.

### 8.4 Golden evaluation tests

Use the existing six-device dataset as a regression corpus. The test does not require exact numeric equality initially. It requires:

- expected directional findings for selected scenes;
- no known scene mismatches;
- traceable evidence refs;
- stable ordering under repeated frozen inference;
- no unsupported proprietary-mechanism claims.

## 9. Phased implementation and gates

### Phase A — Contract and persistence foundation

Deliver:

- `core/models_v2.py`;
- `core/dimensions.py`;
- schema tests;
- additive database migration;
- repository methods.

**Gate A:** all contracts round-trip and v0.1 tests remain green.

### Phase B — Semantic ROI and objective evidence

Deliver:

- multi-face backend interface;
- face parsing interface;
- per-face and relation metrics;
- diagnostic assets;
- applicability engine.

**Gate B:** representative single- and multi-face scenes produce valid evidence without manual ROI editing.

### Phase C — Pairwise, rules and judge

Deliver:

- role-specific VLM prompts;
- rough ranking;
- adaptive pairwise graph;
- deterministic rules;
- judge decision schema and retry policy.

**Gate C:** known direction conflicts are detected and resolved or escalated.

### Phase D — Batch-relative scoring

Deliver:

- Bradley–Terry fitting;
- robust normalization;
- confidence-weighted aggregation;
- quality/style separation;
- device scores and tie groups.

**Gate D:** repeated frozen runs preserve major dimension ordering within tolerance.

### Phase E — Profiles and reports

Deliver:

- cross-scene style/profile engine;
- report evidence package;
- constrained narrative;
- HTML, DOCX and PPTX renderers.

**Gate E:** generated report reproduces the required evidence-first structure and every claim is traceable.

### Phase F — Cutover

Deliver:

- v2 UI workflow;
- operational documentation;
- performance and failure recovery validation;
- comparison against v0.1 on the six-device corpus.

**Cutover gate:** v2 is the default only after all critical regression cases pass. v0.1 remains available for one release as a rollback path.

## 10. Deferred work

The following are deliberately outside the critical path:

- external professional-review corroboration;
- automatic hardware web research;
- absolute-score anchors;
- cloud APIs;
- model fine-tuning;
- RAW and video evaluation;
- rear-camera benchmark expansion.

Existing implementations may remain, but they must not delay Evaluation Core v2.

## 11. First implementation slice after approval

The first code change is deliberately narrow:

1. add `core/models_v2.py` with frozen enums and Pydantic contracts;
2. add `core/dimensions.py` with ten dimensions, weights and applicability metadata;
3. add contract tests;
4. add no database or pipeline behavior until the contracts pass.

This slice establishes a stable interface before semantic vision, VLM prompts or scoring algorithms are modified.
