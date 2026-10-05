# Task 1 builder report — standalone algorithm comparison

## Scope and implementation

Implemented only these new owned modules and their new tests:

- `packages/iqa/src/qwen_tmqa/comparison_models.py`: strict, extra-field-rejecting request/result/evidence models, unique nonblank asset IDs, validated ROI coordinates/correspondence, finite outputs, all ten Front dimension IDs. Optional `ComparisonAsset.source_sha256` is an explicit source-byte-bound user declaration; it is never described as independently verified generation lineage.
- `asset_io.py`: precision-preserving unchanged decoding of uint8/uint16 and floating-point data; safe single-array NPY; RGB channel order; all eight EXIF orientations applied once; source byte/decoded pixel hashes and source precision/orientation/encoding trace. Invalid existing assets retain exact byte provenance. Floating linear HDR is preserved, not quantized or silently clipped.
- `comparison.py`: pure declared-image-reading `evaluate_comparison(ComparisonRequest) -> ComparisonResult`, arbitrary candidates without alpha sequence, separated objective/relative/ROI/person evidence, explicit missing/unobservable/applicability states, conservative content/geometry rejection and uncertainty. Device comparisons never compute same-input pixel fidelity or create algorithm training labels.
- `comparison_cli.py`: independent `evaluate`, repeated-candidate-root `compare-versions`, and `audit-sample` commands; reproducible sorted finite JSON and candidate regression CSV; relative manifest paths resolve against manifest directory. Exact relative folder paths are only correspondence, not proof of common source identity. Missing and unmatched counterparts are recorded.
- `audit_sampling.py`: input-order-independent SHA-ranked balanced strata selection, clean/risk coverage when sample size allows, design/population/selection metadata, and a separate complete focused risk queue (including unknown-risk records).
- Tests: `test_comparison_core.py`, `test_comparison_cli.py`, `test_asset_io_precision.py`, `test_audit_sampling.py`.

No legacy TMQA modules/tests, Front files, packaging, CI, commit or push were changed by this builder. Package script registration and isolated install verification remain the integration owner's scope.

## Requirement coverage

| Requirement | Builder evidence |
|---|---|
| INT-001 | Core/CLI import no Front, HTTP or database module. Independent module CLI smoke succeeds. Integration owner performs isolated wheel installation. |
| INT-002 | Multiple arbitrary candidates; same/changed pixels; multiple version roots; missing/unmatched folder images; JSON/CSV CLI fixtures. |
| INT-003 | uint16 gray/RGB distinctions, float64 linear >1 values, all eight EXIF orientations (including 6), identical/constant/small/sparse/invalid/NaN/Inf/negative/NPZ-disguised-as-NPY assets, geometry mismatch, fixed per-person ROIs, guarded overlap-only translation, device size differences. |
| INT-004 | All ten frozen Front IDs always present. No face/skin/reference evidence is explicit. Per-person luminance/ROI metrics remain separate. Scene adaptability is not applicable to one scene. |
| INT-007 | Clean+risk representative sample deterministic across input order; independent full focused risk queue; design metadata prevents unsupported population accuracy claims. |
| INT-014 | Full IQA package regression suite and owned-file lint/format/type checks pass; no replacement of TMQA 0.2 behavior. |

## Decisions and conservative boundaries

All otherwise valid comparisons remain `REVIEW` because these are uncalibrated observations. No absolute quality score, model vote, winner, controllability score or style preference is fabricated. `REJECT` is used for invalid declared image inputs, algorithm geometry mismatch, mismatching declared source hashes, complete loss of spatial content relative to a textured baseline, or severe local information destruction with declared algorithm correspondence.

A per-person regression cannot hide inside global averages: the tested 8×8 destroyed face in a 128×128 mostly unchanged image has global pixel MAE below .004, yet emits `local_content_collapse:person-one`; white collapse also emits `local_destructive_highlight_occupancy:person-one`. Another person's unchanged ROI retains zero relative change. A missing valid declared source/cross-capture mode produces `possible_local_*` REVIEW reasons instead of a same-input local fatal. These local guards require at least 16 ROI pixels and baseline linear luminance contrast >.02; complete collapse requires candidate contrast <1e-10. The severe highlight guard requires baseline threshold occupancy <.1, candidate occupancy >.95 and candidate contrast <.01. These are explicit diagnostic guards, not calibrated portrait quality thresholds or proof of sensor clipping.

Translation estimates require matching geometry, spatial support ≥16 pixels per axis and ≥256 total pixels, nonconstant images, sufficient nonsparse edges, phase response ≥.2, shifts ≤15% of each axis, and overlap ≥80%. Diagnostics use rounded translation and only the valid overlap, retaining the original fixed baseline ROI. Constant/small/sparse inputs cannot acquire a perfect correlation; invalid evidence uses null instead of NaN/Inf.

Ordinary float linear HDR ranges are retained; negative values and >1e12 linear values fail explicitly. sRGB floats outside [0,1] fail explicitly. Nonopaque alpha requires explicit compositing. Embedded ICC profiles are recorded as unapplied; the declared encoding controls luminance conversion. Unsupported codecs fail explicitly; HEIC/raw demosaic and physical HDR fidelity are not claimed. No chart, calibrated illuminant, human naturalness label, semantic detection or person-identity verification is fabricated from ROI names. Skin RGB facts cannot establish AWB accuracy without an illumination/color reference. Adaptability needs multiple comparable scenes, outside this entrypoint.

## Red/green sequence

1. First 14 boundary tests failed before modules existed (explicit missing-module assertion/import failures). Implemented models/loader/core/audit and observed 14 passing.
2. Added CLI and complete black content-collapse boundary tests: 5 failed, 12 passed. Added CLI/content guard; 21 passed.
3. Added retained invalid-byte trace/input binding and `.csv` output-name collision tests: 2 failed, 20 passed. Added invalid decode trace and an input fingerprint binding the request plus ordered asset byte hashes; CSV collisions use `.regression.csv`. 24 passed.
4. Added per-person black/white destroyed-face, full-white collapse and disguised NPZ boundary tests: 4 failed, 17 passed in the selected core/asset run. Added conservative ROI/content/NPY guards; final owned suite 28 passed.

## Exact verification commands and outcomes

Interpreter: `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python`.

From repository root:

```bash
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH PYTHONPATH=packages/iqa/src /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest packages/iqa/tests/test_comparison_core.py packages/iqa/tests/test_comparison_cli.py packages/iqa/tests/test_asset_io_precision.py packages/iqa/tests/test_audit_sampling.py -q
```

Final result: **28 passed in 0.53s**, exit 0.

From `packages/iqa` (the required working directory for legacy relative config/test resources):

```bash
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest -q > ../../evidence/build/task-1-iqa-suite.log 2>&1
```

Final result: **184 passed in 34.31s**, exit 0. The count includes concurrently landed tests from the other IQA owner. Full log: `evidence/build/task-1-iqa-suite.log`.

Initial full-suite invocation from repository root was incorrect for existing relative paths: **49 failed, 121 passed**, predominantly `configs/default.yaml` not found in legacy audit/domain/judge/review/server/visualization tests. That invocation issue was corrected by running from `packages/iqa`; the complete suite then passed (first 170, final 184). No test was skipped or legacy source modified to mask those failures.

From repository root:

```bash
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m ruff check packages/iqa/src/qwen_tmqa/comparison_models.py packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/src/qwen_tmqa/comparison.py packages/iqa/src/qwen_tmqa/comparison_cli.py packages/iqa/src/qwen_tmqa/audit_sampling.py packages/iqa/tests/test_comparison_core.py packages/iqa/tests/test_comparison_cli.py packages/iqa/tests/test_asset_io_precision.py packages/iqa/tests/test_audit_sampling.py
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m ruff format --check packages/iqa/src/qwen_tmqa/comparison_models.py packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/src/qwen_tmqa/comparison.py packages/iqa/src/qwen_tmqa/comparison_cli.py packages/iqa/src/qwen_tmqa/audit_sampling.py packages/iqa/tests/test_comparison_core.py packages/iqa/tests/test_comparison_cli.py packages/iqa/tests/test_asset_io_precision.py packages/iqa/tests/test_audit_sampling.py
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m mypy packages/iqa/src/qwen_tmqa/comparison_models.py packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/src/qwen_tmqa/comparison.py packages/iqa/src/qwen_tmqa/comparison_cli.py packages/iqa/src/qwen_tmqa/audit_sampling.py --follow-imports=silent --ignore-missing-imports
```

Results: **All checks passed**, **9 files already formatted**, **Success: no issues found in 5 source files**, exit 0 each. Initial lint/type findings were corrected solely in owned files.

CLI mechanical evidence:

```bash
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m qwen_tmqa.comparison_cli --help
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH PYTHONPATH=packages/iqa/src /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m qwen_tmqa.comparison_cli evaluate --manifest evidence/build/task-1-example/manifest.json --output evidence/build/task-1-example/result.json
```

Both exit 0. `task-1-example` contains encoded uint8/uint16 PNG assets, a float64 linear NPY source, the relative-path manifest, finite result JSON and regression CSV. They are deterministic generated array fixtures demonstrating software behavior, **not real mobile capture, calibrated quality, real-model, gold or training-accuracy evidence**. The source hash is bound to the actual NPY bytes and remains a declaration, not independent proof of the generator.

## Remaining integration responsibilities and concerns

Root owns packaging/entrypoint registration, isolated installation, real Front invocation persistence, independent review/remediation, final evaluator and GitHub branch uploads. Task 2 owns real-human/Judge/split/canonical-group training gates. The new comparison output includes declared paths/hashes, split/group metadata, rejection/review reasons and explicit uncertainty for those consumers; it never creates eligible training records itself.

Folder-derived group IDs are explicitly declared correspondence, not closed canonical grouping. An archive's relative filenames establish no algorithm lineage. Device comparisons remain descriptive/confounded. Representative audit sampling is balanced across declared strata; it is not population-weighted and cannot produce a statistical accuracy estimate without real human labels and appropriate weights. Same-environment deterministic software fixtures do not establish cross-codec/platform numerical bit equivalence or perceptual model validity.
