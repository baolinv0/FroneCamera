# Qwen-TMQA Audit Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the audit improvements so Qwen-TMQA fails safely, preserves trustworthy human-review evidence, renders a readable blind-review workflow, and passes the release suite.

**Architecture:** Keep the existing package boundaries. Validate inputs at dataset, judge, review, server, and dashboard boundaries; preserve append-only review records while deduplicating only the calibration view; verify each behavior with focused tests and a six-scene synthetic experiment.

**Tech Stack:** Python 3.10+, Pydantic v2, Pillow, NumPy, httpx, pytest, Ruff, HTML/JavaScript.

---

### Task 1: Dataset and configuration boundaries

**Files:** `src/qwen_tmqa/dataset.py`, `src/qwen_tmqa/image_io.py`, `src/qwen_tmqa/config.py`, `tests/test_dataset_metrics.py`, `tests/test_domain_config.py`.

- [x] Add failing tests for non-image files, EXIF-oriented dimensions, and remote judges without `base_url`.
- [x] Filter discovery through the image-extension allowlist, use `ImageOps.exif_transpose`, and validate the endpoint with a Pydantic `model_validator`.
- [x] Run `python -m pytest tests/test_dataset_metrics.py tests/test_domain_config.py -q`; expected result: PASS.

### Task 2: Judge contract and decision safety

**Files:** `src/qwen_tmqa/prompts.py`, `src/qwen_tmqa/judges/openai_compatible.py`, `src/qwen_tmqa/evaluation.py`, `tests/test_prompts.py`, `tests/test_judges_evaluation.py`.

- [x] Add failing tests for the six-score JSON contract, 100-point normalization, fatal issues, conflicting decisions, partial failure, and all judges unavailable.
- [x] Require every documented score, preserve trace data on failure, reject fatal evidence, and route disagreement or unavailable judges to review.
- [x] Run `python -m pytest tests/test_prompts.py tests/test_judges_evaluation.py -q`; expected result: PASS.

### Task 3: Human-review integrity

**Files:** `src/qwen_tmqa/domain.py`, `src/qwen_tmqa/review.py`, `src/qwen_tmqa/server.py`, `tests/test_review_calibration.py`, `tests/test_server_cli.py`.

- [x] Add failing tests for independent reviewers, out-of-order resubmissions, empty reviewer IDs, and submissions outside the focused queue.
- [x] Select the latest record per `(scene_id, reviewer_id)` by ISO timestamp without rewriting JSONL, require a non-empty reviewer ID, and authorize POSTs against `review_queue.json`.
- [x] Run `python -m pytest tests/test_review_calibration.py tests/test_server_cli.py tests/test_domain_config.py -q`; expected result: PASS.

### Task 4: Blind-review dashboard

**Files:** `src/qwen_tmqa/assets/dashboard.html`, `src/qwen_tmqa/visualization.py`, `src/qwen_tmqa/audit.py`, `tests/test_visualization.py`.

- [x] Add failing tests for UTF-8 labels, complete image sequences, independent filmstrip state, all score dimensions, explicit reviewer ID, server-confirmed reveal, and absence of `localStorage` fallback.
- [x] Repair the template, pass the selected manifest explicitly, submit the reviewer ID, and reveal model data only after persistence succeeds.
- [x] Run `python -m pytest tests/test_visualization.py -q` and parse the inline script with Node; expected result: PASS.

### Task 5: Release verification and publication

**Files:** `README.md`, `docs/EXPERIMENT_REPORT.md`, `docs/REQUIREMENTS_MATRIX.md`.

- [x] Run `python -m compileall -q src tests`, `python -m ruff check .`, and `python -m pytest -q`; expected result: all commands exit zero.
- [x] Build the sdist and Wheel, install the Wheel into an isolated `--target`, and import `qwen_tmqa`; expected version: `0.2.0`.
- [x] Run the six-scene evaluate, simulate-human, calibrate, visualize, and requirements-audit chain; expected audit: 10/10 PASS.
- [ ] Review the final staged diff, commit `fix: harden TMQA audit and review workflow`, push `codex/tmqa-audit-improvements`, and verify the remote hash.
