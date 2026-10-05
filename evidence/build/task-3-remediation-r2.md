# Task 3 remediation, round 2

Builder remediation for both final findings in `evidence/review/task-3-r2.md`. Previous R1 opposition, lineage, privacy, HEIC and invalid-core regressions remain intact. No historical tests were changed or weakened; no publication modules, commits or pushes were made. Independent closure remains pending.

## T3-R2-01 — P2 — INT-003/013/014

Fail-before reproduction used actual uint16 PNG fixtures at 10000/50000. Grayscale Front measurements and JPEGs both clipped to white; RGB measurements lost original precision. Both cases failed fixed source-range measurement assertions before changes. The independent actual SQLite probe additionally established incorrect darker-capture highest-luminance B=.933/objective1.

A shared `portrait_eval.pixel_io` now uses the standalone core's precision/orientation decoder. Front measurements consume normalized float pixels at the fixed source dtype range (uint16 denominator 65535), without per-image normalization or contrast stretch. Detector, diagnostic and model JPEG images use the same pixels with fixed unit-range-to-8-bit rounding; model downsizing is applied afterward. Content pairing features also consume the same decoded pixels. Measurement input trace retains source bit depth, normalization denominator, orientation, decoded pixel hash and original source byte hash.

Additive tests verify:

- Both grayscale/RGB16 measured means track 10000/65535 and 50000/65535; model JPEG means preserve the original ordering and avoid white clipping, within explicit 8-bit quantization tolerance.
- Actual SQLite pipeline with distinct configured adapters deliberately selecting the darker source no longer gives it objective support or a B claim; core linear facts and display-referred measurements retain the original ordering and sRGB relation.
- Real uint16 PNG EXIF orientation controls in both grayscale/RGB match the core's exact oriented pixels and model image shape/order.
- Both actual HTTP transports preserve these encoded image differences, and stored JPEG/prompt hashes match the actual captured request bytes.

## T3-R2-02 — P2 — INT-004/008, R1-04 context residual

Fail-before helper reproduction demonstrated `artifact_texture_control` removal (10→9 dimensions) and `high`/`low` alias corruption of highlight dimension/metric keys.

Location filtering now recognizes exact/scoped location fields instead of artifact-prefix substrings. Canonical dimension, metric and schema keys are preserved even when a device display name exactly equals one. Identity substitutions use complete tokens; structured evidence references substitute the device identity component while preserving namespaces, scene IDs and scientific identifiers. Canonical protocol values remain protocol values. Serialized inter-stage JSON is recursively sanitized as structured data, preserving schema and removing paths. Already-anonymous A/B image codes cannot be swapped by colliding user display names. Embedded UNC/path anonymization and actual encoded-payload traces remain intact.

Additive tests verify all ten dimension IDs and metric names survive actual pipeline contexts for devices named `high`/`low`; actual prompts in both transports contain all ten IDs and valid metric vocabulary; exact `color`, `face`, `whole`, `luma_mean` and `artifact_texture_control` aliases preserve structural fields, references, dimension values and serialized JSON; A/B display-name collisions preserve the established image-to-metric mapping; privacy redaction retains useful numeric facts.

## Verification

- Fail-before: **3 failed** new precision/schema regressions before implementation.
- New `tests/test_task3_remediation_r2.py`: 15 additive tests; final result recorded below.
- Prior R1/runtime/R2 regression subset: **67 passed**, 1 existing warning, before the final additional schema controls.
- Full Front suite: **184 passed**, 5 existing deprecation warnings, 26.46 s before the final single enum-value control; final current result recorded below.
- Owned **12 source modules** pass mypy; root also reported global mypy clean.
- Ruff on owned sources and all three additive Task 3 regression modules: clean.

Final current checks: new R2 regressions **15 passed** (1.38 s); full Front suite **185 passed**, 5 existing deprecation warnings (25.41 s). Owned-source mypy and Ruff checks are clean; root reported global **36-source mypy clean**. No code changes followed these checks.

Interpreter: `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python`.

## Limits and handoff

Only local software behavior is established: actual image files, SQLite, model encoders, core facts and captured HTTP payloads with mocked responses. The JPEG representation necessarily has 8-bit/downsampling/JPEG quantization; measurement/source precision remains separately retained. No paid-model accuracy, physical capture/hardware causation, human Gold or training benefit is claimed. The final Critic record contains the two IDs above and requires fresh independent closure before acceptance.
