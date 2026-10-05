# TMQA V1 Usability Closure Design

## Goal

Ship a usable first version without architectural redesign. Preserve the existing pipeline:

`dataset -> objective metrics -> judges -> decision -> review queue -> blind review -> calibration -> dashboard`

## Scope

Only three user-visible gaps are in scope:

1. Expose the already-produced engineering evidence in the Dashboard.
2. Reject unknown configuration keys instead of silently falling back to defaults.
3. Make the documented V-007 command run successfully without CI-only environment variables and emit deterministic rerun evidence.

The previously deferred CRIT-012, CRIT-015, and CRIT-016 findings are not part of this V1 closure. No production release claim is added.

## Design

### Dashboard

Keep the current single-file, self-contained Dashboard. Extend the existing render functions rather than redesigning the UI. Display:

- overview decision distribution and low-confidence/synthetic summary;
- per-alpha objective metrics next to the filmstrip;
- model version, role, availability, synthetic flag, confidence, decision, six scores, and model score gaps;
- prompt ID, version, hash, schema version, rendered variables, and inference parameters above the existing prompt tabs;
- calibration sample count, decision agreement, per-dimension MAE, pairwise gaps, fusion weight, and available human-model deltas.

Do not alter `review.html` or reviewer-safe payloads except for regression fixes required to preserve blind review.

### Configuration

Apply Pydantic `extra="forbid"` to the root config and every nested config model. Unknown keys at any supported nesting level must fail validation with a path-specific error. Existing valid configuration remains unchanged.

### V-007 and deterministic rerun

Make `scripts/run_verified_experiment.sh runs/verified_experiment` perform the V-007 pipeline and exit successfully without CI metadata. Keep formal exact-head auditing in the separate aggregate CI job. Run the deterministic evaluation path twice and write a comparison artifact covering decisions, review queue, and stable summary fields. Update README and CI to use the exact documented command.

## Acceptance

- Existing architecture and blind-review boundary remain intact.
- New focused tests demonstrate RED before implementation and GREEN after it.
- Full pytest, Ruff, compile, build, isolated install, CLI checks, JavaScript checks, secret scan, and standard V-007 command pass.
- Exact-head Python 3.10/3.12 CI passes on the new branch head.
- Stop after these conditions; record remaining limitations for use-driven follow-up.
