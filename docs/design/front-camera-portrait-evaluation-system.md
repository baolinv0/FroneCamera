# Front-Camera Portrait Evaluation System

## Status

Approved product and architecture specification for a local, single-user front-camera portrait comparison system.

## 1. Goal

Evaluate portrait photos captured by multiple phones in matched real-world scenes, reduce bias caused by capture variation, distinguish observable image behavior from mechanism hypotheses, and generate traceable HTML/PDF reports.

The system must not assume that the input follows a fixed 20-scene protocol. Existing data may contain only several matched scenes from different brands, with unknown or incomplete scene labels.

## 2. Input Contract

- Accept `N >= 2` device folders.
- Filenames may differ across devices.
- Natural ordering proposes the initial correspondence, but the user must confirm or correct pairing.
- A scene group may contain missing devices; at least two valid images are required for cross-device comparison.
- Repeated captures are optional and must be explicitly grouped. Consecutive files are not automatically treated as repeats.
- Original images are read-only.

The core entity is a matched scene group:

```text
G001: device A image ↔ device B image ↔ device C image
G002: device A image ↔ device B image ↔ device C image
```

Scene groups may remain `UNCLASSIFIED`.

## 3. Optional Reference Scene Library

A 20-scene front-camera portrait template is retained only as an optional future capture protocol. It is not a mandatory classification scheme and must not block analysis of existing images.

Examples include open shade, strong sunlight, side light, backlight, mixed light, neon, low light, close-up, edge framing, multi-face scenes, and portrait-mode segmentation scenes.

## 4. Content-Adaptive Analysis

Every scene receives baseline analysis:

- pairing and comparability audit;
- face/person detection;
- face and background luminance;
- highlight and shadow statistics;
- skin color consistency;
- apparent detail, noise, smoothing, and artifacts;
- anonymous cross-device visual review.

Specialized analysis is activated from observed content:

- illuminated signs/screens → emissive-highlight analysis;
- multiple faces → multi-face exposure and skin consistency;
- strong blur → portrait segmentation analysis;
- mixed illuminants → local color consistency and AWB behavior;
- side light → left/right facial light-ratio analysis;
- face near frame edge → geometry and stretching analysis.

Automatic scene tags are suggestions, not facts. Each tag records its source: user input, imported metadata, model suggestion, or human confirmation.

## 5. Evaluation Pipeline

```text
Create project
→ register devices and folders
→ scan and propose matched scene groups
→ human pairing confirmation
→ image decode, EXIF, checksum, and comparability audit
→ ROI extraction and objective metrics
→ blind primary VLM analysis
→ independent heterogeneous reviewer analysis
→ deterministic evidence adjudication
→ scene-level conclusions
→ scene-coverage gate
→ optional cross-scene strategy inference
→ freeze internal results
→ retrieve professional external reviews
→ corroboration and capture-bias assessment
→ hardware/capture/reconstruction/rendering attribution
→ human review
→ versioned HTML/PDF report
```

## 6. Image Facts and ROI

The system separates:

1. original encoded image;
2. display-referred sRGB image;
3. approximately linearized display RGB for relative luminance analysis.

Linearized JPEG values are display-space measurements, not recovered scene radiance.

ROI hierarchy:

- whole image;
- person/subject;
- face and facial skin;
- forehead, cheeks, nose, chin, eye regions, neck;
- left/right facial regions;
- near-face background ring;
- bright, dark, emissive, sky, window, and scene-specific regions.

Invalid or low-quality ROI must not silently produce trusted metrics.

## 7. Objective Metrics

Core metric families:

- face and background luminance percentiles;
- face-background relative EV;
- highlight clipping and near-highlight ratios;
- shadow and deep-black ratios;
- facial left/right light ratio;
- skin lightness, hue, chroma, and face-neck discontinuity;
- apparent detail, edge overshoot, noise proxy, smoothing, and blockiness;
- geometry, edge stretching, flare, ghosting, and segmentation artifacts.

Metrics describe output behavior. They are not automatically converted into “higher is better” scores.

Every metric stores:

```text
value
ROI reference
algorithm version
validity
confidence
warnings
```

## 8. Dual-Model Review

### Primary analyst

Performs two passes:

1. visual-only observation;
2. metric-assisted correction.

### Independent reviewer

Performs:

1. independent blind observation;
2. challenge pass after seeing primary claims, metrics, and audit warnings.

Device order is randomized independently for each scene. Brand, model, folder name, and public reviews are hidden during blind analysis.

Model output must be schema-valid and separate:

- observable fact;
- degree judgment;
- subjective preference;
- strategy hypothesis;
- mechanism attribution.

## 9. Evidence Adjudication

The final result is not a model vote. A deterministic adjudicator combines:

- scene comparability;
- ROI quality;
- objective evidence;
- primary/reviewer agreement;
- repeatability when available;
- confounders and counterexamples.

Results include:

- confirmed observation;
- disputed observation;
- subjective preference;
- insufficient evidence;
- rejected mechanism inference;
- human review required.

## 10. Cross-Scene Strategy Gate

Sparse or unclassified scenes must not be over-generalized.

The system may always report scene-specific differences. Device-level strategy claims require sufficient coverage across multiple relevant and distinct conditions.

If coverage is inadequate, create:

```text
INSUFFICIENT_SCENE_COVERAGE
```

and stop at scene-level conclusions.

Strategy claims use A/B/C evidence grades and must include supporting scenes, contradictory scenes, scope, and alternative explanations. C-grade hypotheses do not enter the executive summary.

## 11. External Professional Corroboration

External search occurs only after internal image analysis is frozen.

The system searches professional labs and review media using claim-specific queries, such as front-camera backlight HDR, selfie low-light detail, skin tone, autofocus, flare, or portrait segmentation.

Source priority:

1. independent labs with clear protocol and original samples;
2. professional camera/technology reviewers;
3. general media with relevant front-camera samples;
4. forums/social posts only as issue-discovery signals.

Manufacturer pages are hardware/function sources, not independent image-quality corroboration.

Each external comparison is classified as:

```text
CORROBORATED
PARTIALLY_CORROBORATED
CONTRADICTED
CONSENSUS_SPLIT
INCOMPARABLE
NO_RELEVANT_EVIDENCE
```

External reviews do not change the score of the current samples. They affect generalizability, evidence grade, and capture-bias risk.

## 12. Capture-Bias Assessment

Create `POSSIBLE_CAPTURE_BIAS` when the internal result conflicts with multiple relevant professional sources and the current capture has confounders, only one sample, unknown modes, uncertain fill light, missing EXIF, or abnormal outlier behavior.

Possible actions:

- retain the current-sample result but prohibit device-level generalization;
- reduce evidence grade;
- narrow scope;
- request human review;
- generate a controlled reshoot protocol;
- mark `RESHOOT_RECOMMENDED` or `RESHOOT_REQUIRED_FOR_GENERALIZATION`.

Reshoots create a new run and never overwrite the original evidence.

## 13. Attribution

Attribution uses five layers:

```text
Hardware
Capture
Reconstruction
Rendering
Beautification / Portrait Processing
```

Allowed top-level results:

- hardware-dominant;
- software-dominant;
- hardware-software joint;
- indeterminate.

The system must not infer a named proprietary algorithm from rendered JPEGs unless supported by explicit public evidence.

## 14. Architecture

```text
React/Vite review console
        ↓
FastAPI
        ↓
Dramatiq + Redis workers
        ↓
Python evaluation core
        ↓
SQLite + local artifact store
```

Deployment:

- Linux GPU server;
- Windows browser client;
- SSH tunnel and configurable LAN mode;
- native `uv` development;
- Docker Compose deployment;
- source images mounted read-only.

The first release uses SQLite behind repository abstractions so PostgreSQL remains a future migration path.

## 15. Review Console

Core pages:

- projects;
- project overview;
- pairing review;
- dataset audit;
- scene analysis;
- claims and attribution;
- hardware research;
- report export.

Human operations include accept, edit, reject, lower confidence, mark preference, mark insufficient evidence, correct ROI, rerun selected tasks, and invalidate downstream results.

## 16. Reports

MVP outputs:

- HTML;
- PDF;
- JSON;
- CSV.

Every important statement maps to an evidence claim and links to images, ROI, metrics, model outputs, review decisions, or external sources.

Draft reports may contain unresolved items with explicit warnings. Final reports require all consequential review items to be resolved and must preserve immutable versions.

## 17. Failure and Recovery

- Redis is not the source of truth; SQLite reconstructs pending/interrupted tasks.
- Single-scene failures do not block unrelated scenes.
- Network research failure does not block visual evaluation.
- HTML remains available if PDF rendering fails.
- ROI, prompt, model, pairing, and hardware changes invalidate only dependent downstream nodes.
- GPU OOM degradation must be explicit and reviewable; resolution reduction is never silent.

## 18. MVP Milestones

1. repository, schemas, configuration, SQLite/Alembic, and state machine;
2. folder scan, natural pairing, edit/freeze pairing, basic audit and asset serving;
3. ROI and objective metrics;
4. primary VLM structured evaluation;
5. reviewer VLM, conflicts, adjudication, and review queue;
6. cross-scene claims, external corroboration, hardware attribution, reports, Docker, and recovery.

## 19. Core Invariants

```text
Rendered-image observation ≠ internal algorithm fact
Cross-scene regularity ≠ unique mechanism identification
Hardware capability ≠ achieved final image quality
VLM agreement ≠ ground truth
Objective metric ≠ aesthetic preference
External media consensus ≠ replacement for current samples
```

The preferred conclusion is the narrowest statement that survives image evidence, metrics, independent review, counterexamples, external corroboration, and human inspection.
