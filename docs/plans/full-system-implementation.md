# Full-System Implementation Plan

## Goal

Deliver a runnable local front-camera portrait evaluation system from generic matched folders through evidence-backed reports and professional-review corroboration.

## Completed vertical slices

1. **Foundation and persistence** — Python package, SQLite schema, Alembic baseline, project/device API, access controls, read-only source paths.
2. **Pairing** — natural-sort proposals, explicit missing cells, manual corrections, optimistic versions, immutable confirmation snapshots.
3. **Image fact layer** — EXIF/dimensions/checksum, display-referred metrics, face/background/highlight/shadow regions, diagnostics, comparability audit.
4. **Model protocol** — per-scene anonymization, primary visual pass, primary metric-validation pass, independent reviewer, challenge pass, structured JSON repair and validation.
5. **Evidence reasoning** — deterministic adjudication, A/B/C grades, conflict review items, cross-scene coverage gates and counterexamples.
6. **External evidence** — internal-result freeze, SearxNG hardware/professional queries, source tiers, reviewer-model corroboration classification, capture-bias and reshoot decisions.
7. **Attribution** — hardware/capture/reconstruction/rendering candidate causes with explicit EXIF and source-evidence summaries.
8. **Operations** — persistent atomic task claiming, bounded retries, API/worker split, Docker Compose, native deployment, project export.
9. **Review and reporting** — React review console, diagnostic access, review resolution, HTML/JSON/CSV/PDF outputs, review-aware immutable final versions.
10. **Verification** — backend tests, type checking, lint/format checks, frontend tests/build, dependency audit, and end-to-end smoke execution.

## Formal-use calibration gates

The codebase is complete as a software workflow. A formal device study still requires target-server model deployment, prompt calibration, ROI validation on diverse faces, metric-threshold calibration, an approved external-source list, and controlled repeated-capture validation.
