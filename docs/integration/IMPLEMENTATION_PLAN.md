# IQA and FroneCamera Implementation Plan

> **For agentic workers:** use independent parallel Builders with disjoint ownership, independent non-editing Critic agents, Builder remediation, fresh Critic and fresh Evaluator. The human explicitly selected parallel execution and GitHub upload.

**Goal:** Run IQA independently for algorithm/data development and make it FroneCamera's real shared evaluation core while fixing the confirmed defects.

**Architecture:** A nested installable IQA package based on the repaired P1 source; a Front bridge calling a new comparison interface; version/snapshot-bound workflow and unified publication projection. Preserve the existing nine-alpha interface and immutable prompts.

**Tech Stack:** Python 3.12, Pydantic, NumPy/OpenCV/Pillow, FastAPI/SQLite/SQLAlchemy, httpx, python-docx, React/Vite.

**Spec:** `docs/integration/DESIGN.md`

## Global constraints

- Never modify old frozen Project Sources or fabricate human approval. Current scope is authorized by the live instruction and the preceding integration recommendation.
- Preserve old tests and prompt bytes. Add regression tests, rather than weakening requirements to pass.
- Independent IQA install must not import `portrait_eval`, FastAPI, or SQLAlchemy.
- Candidate training labels require real confirmation, valid train split and traceable scope; heuristic/synthetic output never becomes production evidence.
- Builders edit only their assigned paths and do not commit shared files, upload, adjudicate their work, or spawn reviewers. Root commits coherent reviewed changes.
- Critics/Evaluator do not edit. Fix all actionable P0/P1/P2; disclose missing real-world experiments.

## Review focus

Different statement wording is not necessarily opposition or support; defer ambiguous comparisons. Device captures differ in geometry and cannot reuse a baseline face ROI without explicit correspondence. Exposure state must persist across server restart and prevent later gold replacement. Rejected identical text on two devices must be filtered by claim identity. Install/package/CI must exercise nested IQA and bridge without relying on editable paths.

### Task 1: Standalone algorithm comparison core

Owner: new files in `packages/iqa/src/qwen_tmqa/{comparison_models.py,comparison.py,comparison_cli.py,asset_io.py,audit_sampling.py}` and corresponding new `test_comparison*.py`, `test_asset_io*.py`, `test_audit_sampling.py`. May split these NEW files into focused modules with unique names.

Produce the exact DESIGN interfaces. Implement arbitrary B/C candidates, ten-dimension availability/facts, fixed and per-person ROI, bit depth and EXIF, guarded translation fidelity, objective/relative separation. Add `iqa-compare evaluate --manifest PATH --output PATH`, `compare-versions --baseline-root PATH --candidate-root PATH --source-root PATH --output PATH`, and representative `audit-sample` command. Output finite reproducible JSON and comparison/regression CSV. Uncalibrated quality remains REVIEW; fatal content/geometry defects are explicit. No same-input inference from similar filenames.

Write and run failure tests first. Verify source precision, orientation=6, constant/small/invalid images, same and changed input, multiple candidates/persons, differing device sizes and unavailable dimensions. Test audit sampling selects clean and risk strata deterministically without replacing the focused queue. Report evidence to `evidence/build/task-1.md`.

### Task 2: IQA blind review and PGT repair

Owner: existing `packages/iqa/src/qwen_tmqa/{config.py,domain.py,review.py,server.py,cli.py,prompts.py,judges/*}` plus newly ported `{candidate_schema.py,lineage.py,pseudo_gt.py,training_data.py}`; tests for these paths. Do not edit Task 1 files, existing `evaluation.py`/visualization unless coordinated with root. Root owns packaging.

Port Canary selection/lineage selectively from `repo_review/IQA/61cbaed3` preserving P1 payload trace, decoder, review and real/synthetic isolation. Historical prompts 3.2 and 3.3 remain immutable; if candidate schema requires changed contract, add a new version explicitly. Reject duplicate Judge IDs and count unique evidence. Enforce train export in library and CLI; require explicit canonical metadata and real human confirmation bound to source/candidate bytes and a supported label scope. Keep candidate suggestions separately exportable; never call unconfirmed suggestions training labels.

Track reviewer/scene exposure server-side persistently. Before reveal, latest blind revisions are valid. After reveal, allow an explicitly non-gold adjudication path or reject blind resubmission; calibration must not replace the pre-reveal gold. Preserve append-only history and authorized queue behavior. Add meaningful HTTP/storage failure and restart tests. Keep production weights limited to selected review evidence; no unverified automatic fusion activation. Report `evidence/build/task-2.md`.

### Task 3: Front pipeline, decoder, pairing and IQA bridge

Owner: `src/portrait_eval/{pipeline.py,models.py,vlm.py,strategy.py,adjudication.py,dataset.py,repository.py,database.py,worker.py,imaging.py,iqa_bridge.py}` and new/extended tests for these modules. Do not edit reporting/API/workflow or Task 1/2 files. May create focused helper modules for run/trace/claim validation.

Call Task 1 core through `iqa_bridge` in the actual pipeline and persist `iqa_evaluation`. Bind to confirmed snapshot version and scanned bytes, including retry attempts. Remove equal-count high-confidence shortcut when ordinal/content evidence conflicts; present unresolved pairing as review-required. Make failed model run retryable with confirmed snapshot intact. Add strict local/provider response validation for actual known devices, dimensions, scores and evidence; capture exact prompt and ordered encoded JPEG trace. Remove diagnostic paths from prompt context.

Conservative semantic adjudication: support/opposition/unknown, never field-only agreement, certainty/confidence honored, objective claims require actual facts. Cross-scene counterexample ratio and audit comparability affect grading; heuristic/synthetic results are provisional. Stable claim IDs travel in findings and review payloads for Task 4. Add tests reproducing every previously reported failure using actual pipeline/database/provider transport where appropriate. Report `evidence/build/task-3.md`.

### Task 4: Unified report projection, privacy and UI

Owner: `src/portrait_eval/{api.py,reporting.py,docx_reporting.py,exporting.py,workflow.py,product_api.py,static/index.html}`, `web/src/PairingGrid.tsx` and related web tests, plus new `report_projection.py`/tests. Coordinate interface only, never edit Task 3 files. Root owns general README/packaging/CI.

Build a single resolved projection for professional and quick publication. Resolve by claim IDs when present; support old nested `primary_observation` reviews. Reject/insufficient evidence updates scene findings, primary observations, device profile summaries, attribution and report sections. Preserve separate devices with identical text. No JSON/HTML/DOCX publication of rejected claims. If source report JSON is missing, fail safely rather than bypassing resolution by HTML text replacement.

Quick mode is allowed but must explicitly state unresolved/skipped review and actual adapter/evidence provenance. Display heuristic/synthetic status and no invented device quality score. Export anonymized relative/asset references, pairing confidence/review notes and snapshot/trace lineage; no absolute diagnostic paths. Pairing UI shows thumbnails/confidence/review flags/notes and real manual-confirmation visibility. Test actual API finalize and real renderer content, privacy export, React interaction and type/build. Report `evidence/build/task-4.md`.

### Task 5: Root integration and independent verification

Root owns pyproject/Docker/bootstrap/CI/general documentation, cross-module end-to-end tests, source provenance and task ledger. Install IQA before Front, configure CI to test/build both independently, add executable synthetic and real-image local canary without paid endpoints. Keep existing Python 3.10/3.12 IQA contract where the source requires it; Front remains Python 3.12. Produce a standalone branch containing the repaired IQA project and an integration branch containing reviewed Front changes.

Run baseline and final full tests, compile, lint/format, applicable type checks, package builds/isolated installs, old and new CLI surfaces, Node test/build and deterministic rerun. Dispatch independent task/whole-system Critics after Mechanical Gate. Return structured findings to owners and use fresh non-editing reviewers after fixes. Delegate fresh Evaluator only after PASS_TO_EVALUATOR. Upload only verified new branches; no main merge or deployment. Record exact remote commit and check GitHub Actions for each branch.
