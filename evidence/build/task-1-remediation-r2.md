# Task 1 remediation — independent review round 2

Builder status: IQA-R2-01 implemented and locally verified; ready for fresh independent closure. Read the persisted review `evidence/review/task-1-r2.md`. The reviewer independently closed IQA-R1-01/02/03 and verified native HEIC decoding before discovering this new resource-guard exception. That independent closure remains the reviewer's evidence; this report records the subsequent builder change and verification.

## IQA-R2-01 — P2 — INT-003

Actual PNG bytes with a valid 20,000×20,000 IHDR and tiny/truncated compressed payload cause Pillow to raise `Image.DecompressionBombError`. That exception derives directly from Exception and previously escaped the expected image-error normalization, aborting library and CLI evaluation without the declared asset's byte trace.

The **sole production change** is adding `Image.DecompressionBombError` to `load_comparison_asset`'s existing explicit normalization tuple in `asset_io.py`. It becomes `AssetDecodeError`, retaining the exact source-byte SHA/count and the original resource-guard exception as its cause. The evaluator then reports invalid candidate/finite REJECT; installed CLI writes result JSON and CSV and continues to evaluate valid candidates. No general Exception catch was added. Pillow `MAX_IMAGE_PIXELS` was not changed, disabled or bypassed, and the image is not decoded into a giant array.

New owned regression file: `packages/iqa/tests/test_asset_io_resource_guard.py`. No historical test, Front source, packaging, CI or other decoder behavior changed. No commits or pushes.

## Actual-header fail-before probes

The crafted file uses the PNG signature, CRC-valid 20,000×20,000 RGB8 IHDR, tiny zlib IDAT and IEND. Direct Pillow Image.open verifies that the resource guard raises before any pixel decompression. This is an actual encoded header probe, not a mocked exception.

All **four tests failed before the one-line correction**, saved in `evidence/build/task-1-remediation-r2-red.log`:

1. Loader must normalize the actual resource guard, retain exact SHA/count, preserve its cause, leave MAX_IMAGE_PIXELS unchanged, and demonstrate that direct Pillow still raises afterward.
2. Evaluator must report invalid/REJECT for the guarded candidate while a valid sibling remains valid/REVIEW with pixel MAE 0. JSON is finite.
3. **Installed `iqa-compare`**, run as a subprocess using the verification environment's bin directory, must exit 0 without traceback and produce REJECT/REVIEW JSON plus CSV and the exact invalid candidate SHA. Before correction it exited 1 with traceback.
4. Version-folder batch must retain both the guarded scene's REJECT and an unaffected valid scene's REVIEW, rather than aborting the batch.

The exact probe plan was sent to the round-2 Critic before remediation. Their round-2 report states no additional material Task 1 finding; fresh independent closure of IQA-R2-01 remains pending.

## Fresh final verification

From repository root, fail-before command:

```bash
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH PYTHONPATH=packages/iqa/src /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest packages/iqa/tests/test_asset_io_resource_guard.py -q > evidence/build/task-1-remediation-r2-red.log 2>&1
```

Observed **4 failed in 1.05s**, pytest exit 1. Full log retains direct Pillow error and installed-CLI traceback.

From repository root, final scoped regression command:

```bash
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH PYTHONPATH=packages/iqa/src /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest packages/iqa/tests/test_asset_io_resource_guard.py packages/iqa/tests/test_asset_io_remediation_r1.py packages/iqa/tests/test_asset_io_heif.py packages/iqa/tests/test_asset_io_precision.py packages/iqa/tests/test_comparison_core.py packages/iqa/tests/test_comparison_cli.py packages/iqa/tests/test_audit_sampling.py -q > evidence/build/task-1-remediation-r2-green.log 2>&1
```

Result: **82 passed in 2.15s**, exit 0.

From `packages/iqa`, complete legacy and new IQA suite:

```bash
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest -q > ../../evidence/build/task-1-remediation-r2-suite.log 2>&1
```

Result: **254 passed in 40.50s**, exit 0. No tests skipped or edited to mask the failure.

From repository root:

```bash
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m ruff check packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/tests/test_asset_io_resource_guard.py
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m ruff format --check packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/tests/test_asset_io_resource_guard.py
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m mypy packages/iqa/src/qwen_tmqa/comparison_models.py packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/src/qwen_tmqa/comparison.py packages/iqa/src/qwen_tmqa/comparison_cli.py packages/iqa/src/qwen_tmqa/audit_sampling.py --follow-imports=silent --ignore-missing-imports
```

Results: **All checks passed**, **2 files already formatted**, **Success: no issues found in 5 source files**, each exit 0.

No unresolved builder test/static-check failure remains. This change makes the existing resource guard reportable; it does not raise the pixel limit, provide general oversized-image processing, calibrate image quality, establish physical HDR fidelity, or prove real-model/training utility. Independent Critic/evaluator/GitHub delivery remain root-owned.
