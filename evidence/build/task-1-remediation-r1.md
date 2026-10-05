# Task 1 remediation — independent review round 1

Builder status: **all three Task 1 R1 findings and the coordinated HEIC decoder part of T3-R1-04 are implemented and locally verified; ready for fresh non-editing independent closure.** This report does not replace the Critic's independent verdict.

Read the persisted findings in `evidence/review/task-1-r1.md` and the coordinated HEIC finding in `evidence/review/task-3-r1.md`. Changes are restricted to Task 1 ownership: existing new modules `asset_io.py`, `comparison_models.py`, `comparison_cli.py`; two newly added tests `test_asset_io_remediation_r1.py`, `test_asset_io_heif.py`; build evidence. Historical TMQA tests, Front modules, packaging, CI, commits and pushes were not changed by this builder. Root owns the required `pillow-heif>=0.21,<2` dependency registration (now present in IQA pyproject).

## Finding dispositions and behavior

| Finding | Requirement | Remediation and evidence |
|---|---|---|
| IQA-R1-01, P1 | INT-002/003 | OpenCV TIFF decoding applies TIFF orientation even under IMREAD_UNCHANGED. Normalize only the first-IFD orientation field to 1 in an **in-memory decoder copy**, then apply the original orientation once in `_orient`. Original bytes/hash are unchanged; raw uint8/uint16/float decoding remains unchanged. Decoder trace records the metadata normalization. Sixteen actual TIFF fixtures cover all eight orientations in uint8 and uint16, adjacent 16-bit values, shape and exact pixels; eight uint16 TIFF-versus-equivalent-oriented-PNG end-to-end comparisons retain pixel MAE 0, no geometry rejection. |
| IQA-R1-02, P1 | INT-003/004 | Validate ROI IDs in their **effective assessment scopes**, before any image evaluation. Algorithm None and explicit baseline-ID scopes are identical. Device implicit scope reaches every asset and therefore conflicts with same-ID explicit ROIs there. Reject either insertion order; same-ID explicit ROIs on distinct devices remain supported. Eight order/mode/baseline-or-candidate combinations and two actual destroyed-face/intact-decoy exploits verify rejection before evidence can be overwritten. |
| IQA-R1-03, P2 | INT-003 | Normalize NumPy EOFError to AssetDecodeError inside the byte-traced loader boundary. Empty and truncated NPY probes cover direct loader, evaluator and CLI. Existing file byte SHA/count survive failure. Evaluator yields finite invalid/REJECT evidence; CLI writes finite JSON plus CSV rather than crashing. |
| T3-R1-04, decoder portion, P1 | INT-013/014 | Actual HEIC/HEIF now uses native `pillow_heif.open_heif(convert_hdr_to_8bit=False, hdr_to_16bit=True)` plus its NumPy view. Preserve 10/12-bit content in uint16, record source/container bit depth separately from decoded storage precision, and follow native HEIF display/container orientation without re-applying EXIF. Eight encoded 8-bit HEIF orientation fixtures match Front's registered Pillow opener; a real encoded 10-bit HEIF retains >256 levels and sub-8-bit steps; uppercase HEIC folder matching and a truncated container return valid/invalid evidence explicitly. Task 3 owner retains pipeline validity/context/claim propagation responsibilities. |

The TIFF helper recognizes TIFF/BigTIFF headers and byte order, checks IFD bounds, and rejects malformed/duplicate orientation tags. It changes no file, source byte trace, sample data, compression or pixel encoding. Normalizing the decoder metadata instead of reversing a decoder's output avoids quantization and decoder-dependent double application.

HEIF trace records `source_bit_depth`, actual storage `bit_depth`/dtype, decoder and native orientation convention, original EXIF metadata, and source byte hash. Embedded HEIF ICC/NCLX profiles are disclosed as unapplied: caller-declared sRGB/linear encoding controls measurements. Native display/container transforms are authoritative for HEIF, matching Front/Pillow behavior; EXIF is recorded rather than applied a second time. These codec probes establish software compatibility/precision, not physical HDR or calibrated portrait quality. Native decoder EOF/Syntax/Runtime errors are normalized to invalid byte-traced evidence.

## Fail-before evidence

Before remediation, new R1 probes returned **23 failed, 7 passed**:

- 14 failed actual TIFF tests (orientations 2–8, both uint8 and uint16).
- Six failed effective ROI collision probes (both algorithm baseline insertion orders and all four device combinations); the two algorithm candidate-ID cases already rejected unknown algorithm scopes, as intended.
- Three failed empty NPY direct-loader/evaluator/CLI tests with uncaught EOFError. Truncated counterparts already produced explicit invalid evidence and were retained as regression controls.

Saved full failure output: `evidence/build/task-1-remediation-r1-red.log`. These failures were observed before the decoder/validator corrections. After corrections, the original 30 probes passed. Added eight end-to-end TIFF/Png comparison and two actual destroyed-face-decoy rejection checks bring the R1 module to 40 passing tests.

Before HEIF implementation, all five initial actual encoded HEIF probes failed with `AssetDecodeError:image_decode_failed`; full output is `evidence/build/task-1-heif-red.log`. Native decoding made them green. Added the remaining four EXIF orientations and actual folder/corrupt-container checks; the HEIF module now has 10 passing tests.

## Final verification

From repository root:

```bash
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH PYTHONPATH=packages/iqa/src /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest packages/iqa/tests/test_asset_io_remediation_r1.py packages/iqa/tests/test_asset_io_heif.py packages/iqa/tests/test_asset_io_precision.py packages/iqa/tests/test_comparison_core.py packages/iqa/tests/test_comparison_cli.py packages/iqa/tests/test_audit_sampling.py -q > evidence/build/task-1-remediation-r1-green.log 2>&1
```

Result: **78 passed in 1.05s**, exit 0. No historical tests were edited or excluded.

From `packages/iqa`:

```bash
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest -q > ../../evidence/build/task-1-remediation-r1-suite.log 2>&1
```

Result: **250 passed in 41.74s**, exit 0. This full package run includes concurrently landed tests from Task 2. Full saved log is `evidence/build/task-1-remediation-r1-suite.log`.

From repository root:

```bash
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m ruff check packages/iqa/src/qwen_tmqa/comparison_models.py packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/src/qwen_tmqa/comparison.py packages/iqa/src/qwen_tmqa/comparison_cli.py packages/iqa/src/qwen_tmqa/audit_sampling.py packages/iqa/tests/test_comparison_core.py packages/iqa/tests/test_comparison_cli.py packages/iqa/tests/test_asset_io_precision.py packages/iqa/tests/test_audit_sampling.py packages/iqa/tests/test_asset_io_remediation_r1.py packages/iqa/tests/test_asset_io_heif.py
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m ruff format --check packages/iqa/src/qwen_tmqa/comparison_models.py packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/src/qwen_tmqa/comparison.py packages/iqa/src/qwen_tmqa/comparison_cli.py packages/iqa/src/qwen_tmqa/audit_sampling.py packages/iqa/tests/test_comparison_core.py packages/iqa/tests/test_comparison_cli.py packages/iqa/tests/test_asset_io_precision.py packages/iqa/tests/test_audit_sampling.py packages/iqa/tests/test_asset_io_remediation_r1.py packages/iqa/tests/test_asset_io_heif.py
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m mypy packages/iqa/src/qwen_tmqa/comparison_models.py packages/iqa/src/qwen_tmqa/asset_io.py packages/iqa/src/qwen_tmqa/comparison.py packages/iqa/src/qwen_tmqa/comparison_cli.py packages/iqa/src/qwen_tmqa/audit_sampling.py --follow-imports=silent --ignore-missing-imports
```

Results: **All checks passed**, **11 files already formatted**, **Success: no issues found in 5 source files**; each exit 0. No unresolved builder test/lint/type failure remains.

## Review boundary

Fresh non-editing Critic owns independent closure. Root owns isolated package install/build and final evaluator/GitHub delivery. Task 3 owns propagation of valid/invalid/missing shared-core evidence through Front audit/model context/adjudication and its actual HEIC pipeline test. HEIF files here are generated encoded fixtures, not real camera, real-model or training-accuracy evidence. Skin/AWB/absolute portrait quality remain uncalibrated/missing where their evidence is absent; this remediation does not relax those limits.
