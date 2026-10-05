Read DESIGN.md for the shared contract.

### Task 1: Standalone algorithm comparison core

Owner: new files in `packages/iqa/src/qwen_tmqa/{comparison_models.py,comparison.py,comparison_cli.py,asset_io.py,audit_sampling.py}` and corresponding new `test_comparison*.py`, `test_asset_io*.py`, `test_audit_sampling.py`. May split these NEW files into focused modules with unique names.

Produce the exact DESIGN interfaces. Implement arbitrary B/C candidates, ten-dimension availability/facts, fixed and per-person ROI, bit depth and EXIF, guarded translation fidelity, objective/relative separation. Add `iqa-compare evaluate --manifest PATH --output PATH`, `compare-versions --baseline-root PATH --candidate-root PATH --source-root PATH --output PATH`, and representative `audit-sample` command. Output finite reproducible JSON and comparison/regression CSV. Uncalibrated quality remains REVIEW; fatal content/geometry defects are explicit. No same-input inference from similar filenames.

Write and run failure tests first. Verify source precision, orientation=6, constant/small/invalid images, same and changed input, multiple candidates/persons, differing device sizes and unavailable dimensions. Test audit sampling selects clean and risk strata deterministically without replacing the focused queue. Report evidence to `evidence/build/task-1.md`.


Verification interpreter: /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python
