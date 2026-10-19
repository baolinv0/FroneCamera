# Front-Camera Portrait Evaluation System — Current Design Specification

**Status:** approved and implemented as v0.1 MVP
**Deployment:** local Linux server, Windows browser client
**Data:** original photographs remain local and read-only

## 1. Problem statement

The system compares front-camera portrait outputs from `N >= 2` phones in matched real-world scenes. The current data may contain only several scenes, scene labels may be unknown, filenames may differ, and one or more devices may be missing a capture.

The system therefore must not force the data into a fixed 20-scene taxonomy. The primary entity is a generic matched group:

```text
G001: device A image ↔ device B image ↔ device C image
G002: device A image ↔ MISSING        ↔ device C image
```

The 20-scene list remains an optional future capture protocol, not an input requirement.

## 2. Evidence hierarchy

Every conclusion is separated into four levels:

1. **Observable output:** visible and measurable behavior in a submitted image.
2. **Scene-level trade-off:** relative behavior within one matched group.
3. **Cross-scene tendency:** a repeated output pattern supported by multiple comparable groups.
4. **Mechanism attribution:** a conservative hypothesis across hardware, capture, reconstruction, rendering, or beautification.

A single image cannot establish a stable device strategy. Mechanism attribution is never written as a confirmed proprietary implementation unless public evidence directly supports it.

## 3. Input and pairing contract

- Accept arbitrary device folders and image counts.
- Natural sort proposes initial correspondence.
- The user may replace any pairing cell with another image from the same device or mark it missing.
- Missing cells remain explicit and never shift later groups silently.
- At least two valid devices are required for a group to be comparatively analyzable.
- Repeats are explicit-only; adjacent files are not automatically treated as repeated captures.
- Pairing confirmation uses an optimistic version check and freezes the current snapshot.

## 4. Image fact layer

For every valid image, the system records:

- checksum, dimensions, orientation, EXIF summary, and original path;
- display-referred luminance quantiles;
- near-white clipping and deep-shadow proportions;
- saturation, apparent detail, and noise proxies;
- primary-face detection;
- face, background, face-ring, highlight, and shadow regions;
- diagnostic overlays and validity warnings.

The metrics describe rendered output. They do not reconstruct sensor radiance and do not equate higher values with better quality.

## 5. Adaptive scene audit

Scene groups are classified as:

- `FULLY_COMPARABLE`
- `COMPARABLE_WITH_CONFOUNDERS`
- `NOT_COMPARABLE`

The system may suggest content tags such as `low_light`, `high_dynamic_range`, or `strong_highlight`, but these tags are not required and are not treated as ground truth without review.

## 6. Dual-model protocol

- Devices are anonymized independently for every group.
- The primary model performs the main structured analysis.
- The reviewer first performs an independent pass.
- The reviewer then performs a challenge pass with the primary output and objective evidence.
- Model output is validated against a structured schema.
- The deterministic metric/adjudication layer, not majority voting, decides claim status.
- OpenAI-compatible endpoints allow local Qwen-VL and InternVL-family deployment.
- A deterministic heuristic adapter is available only for infrastructure tests.

## 7. Evidence adjudication

Claims contain:

- statement and device;
- supporting and contradicting groups;
- objective support;
- model agreement;
- scene validity;
- alternative explanations;
- confidence and evidence grade.

Grades:

- **A:** repeated, comparable, objective and model support, limited counterevidence;
- **B:** meaningful support with one material uncertainty;
- **C:** single-scene, conflicted, or insufficiently controlled.

C-grade strategy claims do not enter the executive findings of the report.

## 8. Cross-scene inference

The current engine computes conservative output tendencies such as:

- higher display-referred global luminance relative to the matched-group median;
- lower near-white clipping ratio relative to the matched-group median.

Claims retain counterexamples and alternative explanations. The system does not equate low clipping with actual highlight detail, nor global brightness with local face relighting.

## 9. External professional corroboration

The order is mandatory:

```text
internal blind analysis
→ internal result snapshot frozen
→ hardware and professional-review search
→ corroboration and conflict review
→ capture-bias assessment
```

External sources:

- cannot overwrite the submitted-image result;
- cannot directly modify image-quality scores;
- can support or weaken generalization from the submitted samples to a device-level tendency;
- can trigger `POSSIBLE_CAPTURE_BIAS` or a controlled reshoot recommendation.

Sources are deduplicated by URL and graded by source type. Back-camera results or other models are not treated as direct front-camera evidence.

## 10. Capture-bias handling

Potential capture bias is raised when:

- a claim relies on one scene or one capture;
- group comparability warnings are present;
- independent professional evidence contradicts the attempted generalization;
- shooting mode, fill light, beautification, firmware, or EXIF is unknown.

A recommended reshoot requires alternating device order, at least three valid repetitions, fixed shooting modes, recorded firmware and fill-light states, consistent subject geometry, and original uncompressed files.

## 11. Hardware and mechanism attribution

Hardware research and attribution are separated from the blind image review. Candidate causes are organized across:

- hardware;
- capture;
- reconstruction;
- rendering;
- beautification/portrait processing.

The default result is `indeterminate` when EXIF, public specifications, or discriminating scenes are insufficient. Hardware parameters do not directly contribute to the image-quality score.

## 12. Review and reporting

The browser console supports:

- project and device registration;
- pairing review and correction;
- task submission and status;
- diagnostic and analysis retrieval;
- review-item resolution;
- draft report access and finalization.

Reports are generated from structured evidence, not from an unconstrained final language-model response. Outputs include HTML, JSON, CSV, optional PDF, and a privacy-preserving project export ZIP.

## 13. Architecture

```text
Windows browser
      ↓ SSH tunnel / LAN
FastAPI modular application
      ↓
SQLite persistent source of truth
      ↓
Persistent task queue + Linux worker
      ↓
CV metrics / local VLMs / web research / reporting
      ↓
Local artifact workspace
```

SQLite uses foreign keys, WAL mode, and a busy timeout. Original image paths are never accepted directly by asset APIs; registered image IDs are used.

## 14. Final-report gate

A final report cannot be generated while open review items remain. Final versions are immutable files and do not overwrite earlier report versions.

## 15. Non-goals of v0.1

- camera capture automation;
- video or RAW evaluation;
- model training/fine-tuning;
- multi-user cloud deployment;
- multi-GPU distributed inference;
- a universal rear-camera benchmark.
