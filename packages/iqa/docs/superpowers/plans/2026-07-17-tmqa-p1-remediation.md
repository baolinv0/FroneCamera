# TMQA P1 Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all eight P1 findings against frozen TMQA specification 0.2.0 and produce exact-branch-head CI evidence suitable for independent Critic and Evaluator adjudication.

**Architecture:** Preserve the existing package boundaries. Add explicit dataset, prompt, image-payload, review-time, calibration-policy and reviewer-safe data contracts; separate the engineering dashboard from the blind-review client; use GitHub Actions as the authoritative execution environment because the current controller cannot reach GitHub from its local container.

**Tech Stack:** Python 3.10/3.12, Pydantic v2, Pillow, NumPy, httpx, stdlib HTTP server, pytest, Ruff, setuptools/build, HTML/JavaScript, GitHub Actions.

## Global Constraints

- Frozen specification version: `0.2.0`; do not modify requirements or thresholds.
- Python support: exactly 3.10 and 3.12 in required CI jobs.
- Default levels: `a_m100,a_m075,a_m050,a_m025,a_000,a_p025,a_p050,a_p075,a_p100`.
- Decision vocabulary: `KEEP`, `REGENERATE`, `REVIEW`, `REJECT`.
- Historical prompt versions are immutable; new six-score contract uses `tmqa.sequence@3.3` and `tmqa.sequence.v4`.
- Review server default bind address remains `127.0.0.1`.
- No test deletion, skip, xfail, threshold relaxation, synthetic-to-real relabeling, or silent fallback.
- No merge to `main`; human authorization remains required.

---

### Task 1: Enforce the nine-level dataset contract

**Files:**
- Modify: `src/qwen_tmqa/config.py`
- Modify: `src/qwen_tmqa/dataset.py`
- Modify: `configs/default.yaml`
- Modify: `tests/test_dataset_metrics.py`
- Modify: `tests/test_domain_config.py`

**Interfaces:**
- Produces: `DEFAULT_EXPECTED_LEVELS: tuple[str, ...]` and `DatasetConfig.expected_levels: list[str]`.
- `discover_scenes(root, config)` validates root-level and per-scene level completeness.

- [ ] Add tests that strict mode rejects one missing directory, one unknown parseable level, a duplicated configured level, a baseline not in expected levels, and a missing per-scene image.
- [ ] Add a non-strict test showing available parseable levels remain usable while baseline is mandatory.
- [ ] Run `python -m pytest -q -k 'strict_complete or expected_levels or non_image or recursive'`; expected pre-implementation result: new tests fail.
- [ ] Implement `DEFAULT_EXPECTED_LEVELS`, Pydantic validation and deterministic strict discovery.
- [ ] Add `expected_levels` to `configs/default.yaml`.
- [ ] Re-run the focused command; expected result: PASS.
- [ ] Commit `fix: enforce nine-level dataset completeness`.

### Task 2: Restore prompt immutability and trace exact image payloads

**Files:**
- Modify: `src/qwen_tmqa/config.py`
- Modify: `src/qwen_tmqa/domain.py`
- Modify: `src/qwen_tmqa/image_io.py`
- Modify: `src/qwen_tmqa/prompts.py`
- Modify: `src/qwen_tmqa/judges/openai_compatible.py`
- Modify: `configs/default.yaml`
- Modify: `tests/test_prompts.py`
- Modify: `tests/test_judges_evaluation.py`

**Interfaces:**
- Produces: `EncodedImagePayload(data: bytes, mime_type: str, width: int, height: int, encoding: dict[str, object], sha256: str)`.
- Produces: `encode_image_payload(path, sent_width, sent_height, jpeg_quality=92)`.
- `ImageManifestItem` records `source_sha256`, `payload_sha256`, `payload_mime`, and `payload_encoding`; retains a read-compatible `sha256` alias only if needed for old artifacts.

- [ ] Add immutable fixture assertions for historical `3.2` text/hash and schema version.
- [ ] Add tests for new `3.3` six-score contract and `tmqa.sequence.v4`.
- [ ] Add tests that base64 request bytes hash exactly equals `payload_sha256`, including EXIF transpose and resize.
- [ ] Add a mismatch test proving a real judge fails closed if the trace payload hash is stale.
- [ ] Run `python -m pytest -q -k 'prompt_version or immutable_prompt or output_schema_version or payload_hash or input_manifest or exif_orientation or image_payload'`; expected pre-implementation result: FAIL.
- [ ] Restore historical `3.2`; add `3.3`; switch defaults to `3.3`.
- [ ] Implement the shared encoder and make manifest generation and request construction use it.
- [ ] Re-run the focused tests; expected result: PASS.
- [ ] Commit `fix: version prompt contract and trace sent payloads`.

### Task 3: Make unanimous model decisions authoritative

**Files:**
- Modify: `src/qwen_tmqa/evaluation.py`
- Modify: `tests/test_judges_evaluation.py`

**Interfaces:**
- Produces: `_aggregate_available_decisions(models: list[ModelEvaluation]) -> Decision | None` where `None` means no unanimous decision.

- [ ] Add high-score unanimous `REJECT`, high-score unanimous `REGENERATE`, unanimous `REVIEW`, and unanimous `KEEP` with low objective score tests.
- [ ] Run `python -m pytest -q -k 'fatal or all_unavailable or decision_disagreement or unanimous_reject or unanimous_regenerate'`; expected pre-implementation result: unanimous non-KEEP tests fail.
- [ ] Implement precedence: fatal, all unavailable, disagreement, unanimous REJECT, unanimous REGENERATE, unanimous REVIEW, score logic.
- [ ] Record unanimous decision evidence in the model-judge stage and review reasons when applicable.
- [ ] Re-run focused tests; expected result: PASS.
- [ ] Commit `fix: honor unanimous model decisions`.

### Task 4: Establish authoritative server timestamps and calibration policy

**Files:**
- Modify: `src/qwen_tmqa/domain.py`
- Modify: `src/qwen_tmqa/review.py`
- Modify: `src/qwen_tmqa/server.py`
- Modify: `src/qwen_tmqa/cli.py`
- Modify: `scripts/run_verified_experiment.sh`
- Modify: `tests/test_review_calibration.py`
- Modify: `tests/test_server_cli.py`

**Interfaces:**
- `HumanReview.received_at` is required for stored records and timezone-aware.
- `HumanReview.client_created_at` is optional metadata and never used for latest selection.
- Produces: `select_calibration_reviews(reviews, review_type, allow_synthetic) -> tuple[list[HumanReview], int, str]`.

- [ ] Add tests that client-supplied `received_at` is rejected or overwritten by the server.
- [ ] Add tests that future client metadata cannot dominate and malformed timezone values fail validation.
- [ ] Update latest-review tests to use authoritative `received_at` and JSONL-order tie breaking.
- [ ] Add all-real, synthetic-only default reject, synthetic explicit allow, and mixed-input reject/filter tests.
- [ ] Run `python -m pytest -q -k 'received_at or calibration and (synthetic or reviewer or reliability or fusion)'`; expected pre-implementation result: FAIL.
- [ ] Implement server timestamp injection immediately before append.
- [ ] Implement explicit CLI flags `--allow-synthetic` and `--review-type {real,synthetic}`; update verified experiment to opt in to synthetic evidence.
- [ ] Record selected/ignored counts, selected type, generation time and experimental warning in calibration JSON.
- [ ] Re-run focused tests; expected result: PASS.
- [ ] Commit `fix: secure review time and calibration evidence`.

### Task 5: Implement true reviewer-safe blind review

**Files:**
- Modify: `src/qwen_tmqa/visualization.py`
- Create: `src/qwen_tmqa/assets/review.html`
- Modify: `src/qwen_tmqa/assets/dashboard.html`
- Modify: `src/qwen_tmqa/server.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_visualization.py`
- Modify: `tests/test_server_cli.py`

**Interfaces:**
- `generate_dashboard()` writes `engineering_evaluations.json`, `reviewer_payload.json`, `review_queue.json`, and `reveal_payload.json`.
- `review.html` embeds/fetches only reviewer-safe data.
- `POST /api/reviews` persists and returns `{status, review_id, reveal_url}`.
- `GET /api/reveal?scene_id=...&review_id=...` returns model evidence only after matching persistence.

- [ ] Replace the old assertion that model names exist in blind-review HTML with forbidden-field scans over `review.html` and `reviewer_payload.json`.
- [ ] Add tests that `reviewer_payload.json` lacks decision, overall score, stages, review reasons, model evaluations, prompts, raw/parsed responses and reliability.
- [ ] Add HTTP tests proving reveal is denied before persistence, allowed after matching persistence, denied for mismatched review ID/scene, and unavailable when append fails.
- [ ] Add a test that direct static access to `data/reveal_payload.json` and `data/engineering_evaluations.json` through the review server is denied.
- [ ] Run `python -m pytest -q -k 'blind_review or reviewer_payload or reveal_api or review_server or focused_queue or received_at or dashboard or visualization'`; expected pre-implementation result: FAIL.
- [ ] Split engineering and reviewer templates/data; remove blind-review form and claim from engineering page.
- [ ] Implement reviewer-safe endpoints and post-persistence reveal authorization.
- [ ] Add both HTML templates to package data.
- [ ] Re-run focused tests and extract both inline scripts for `node --check`; expected result: PASS.
- [ ] Commit `fix: enforce server-side blind review isolation`.

### Task 6: Expand audit and verification tooling to R-001 through R-019

**Files:**
- Modify: `src/qwen_tmqa/audit.py`
- Modify: `scripts/audit_requirements.py`
- Create: `scripts/verify_wheel_install.py`
- Create: `scripts/verify_cli_contract.py`
- Create: `scripts/check_dashboard_javascript.py`
- Create: `scripts/verify_no_secret_leakage.py`
- Modify: `scripts/run_verified_experiment.sh`
- Modify: `tests/test_server_cli.py`
- Create or modify: `tests/test_audit.py`

**Interfaces:**
- `audit_verified_experiment()` returns exactly `R-001` through `R-019` statuses plus an overall result in the CLI artifact.

- [ ] Add audit tests with a valid artifact set and injected failures for reviewer leakage, missing payload hash, synthetic ambiguity and missing required files.
- [ ] Add deterministic CLI-contract, wheel-install, JavaScript and secret-scan scripts matching `04_VERIFICATION_PLAN.json` commands.
- [ ] Run focused audit/CLI tests; expected pre-implementation result: FAIL.
- [ ] Implement structural and content checks for all nineteen requirements, using artifact separation rather than HTML marker-only security checks.
- [ ] Update the verified experiment to run every generated helper and produce `requirements_audit.json` with 19/19 PASS.
- [ ] Re-run focused tests; expected result: PASS.
- [ ] Commit `test: verify all nineteen frozen requirements`.

### Task 7: Run the complete mechanical gate in exact-head CI

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/EXPERIMENT_REPORT.md`
- Modify: `docs/REQUIREMENTS_MATRIX.md`
- Create: `docs/verification/TMQA_P1_REMEDIATION_EVIDENCE.md`

**Interfaces:**
- Required GitHub checks are named distinctly for Python 3.10 and 3.12 and report on the exact PR head SHA.

- [ ] Make CI execute compileall, Ruff, full pytest, sdist/wheel build, isolated wheel import, CLI contract, synthetic E2E, both dashboard JS checks, requirements audit and secret scan.
- [ ] Ensure the workflow triggers on `pull_request` and records `${{ github.event.pull_request.head.sha || github.sha }}` in the evidence artifact.
- [ ] Update documentation to remove stale `43 passed` and `10/10` claims; only generated CI evidence may state actual counts.
- [ ] Open a draft PR from `fix/tmqa-p1-remediation-v0.2.0` to `main`.
- [ ] Wait for required checks; inspect logs/artifacts and fix failures without weakening tests or thresholds.
- [ ] Record exact head SHA, workflow run IDs, check conclusions and artifact names.
- [ ] Commit `ci: run frozen TMQA verification plan` if CI fixes are needed.

### Task 8: Independent Critic and Fresh Evaluator gates

**Files:**
- Create: `docs/reviews/TMQA_CRITIC_ROUND_1.md`
- Create: `docs/reviews/TMQA_CRITIC_REVERIFICATION.md` if remediation is required
- Create: `docs/reviews/TMQA_FINAL_EVALUATION.md`

- [ ] Build a branch diff/evidence package from baseline `46a9586...` to exact PR head.
- [ ] Critic checks all eight P1 closure criteria, anti-cheating, regression behavior, exact-head CI and R-001..R-019 evidence without editing code.
- [ ] Return every `REJECT`/`REVISE` finding to Builder; preserve IDs; repeat with a fresh review pass, maximum three rounds.
- [ ] Proceed only after Critic verdict is exactly `PASS_TO_EVALUATOR`.
- [ ] Fresh Evaluator maps every MUST to implementation, tests, execution result, threshold and evidence; no file edits.
- [ ] Final verdict must be exactly `PASS`, `REVISE`, `FAIL`, or `BLOCKED`.
- [ ] Do not merge. Report completion only if Evaluator returns `PASS`, exact-head CI is green, and unresolved P0/P1 is zero.
