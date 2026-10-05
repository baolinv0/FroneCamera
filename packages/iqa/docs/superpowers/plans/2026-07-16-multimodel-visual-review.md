# Qwen-TMQA Multi-Model Visual Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a runnable Qwen-TMQA v0.2 with multi-model scoring, Prompt Inspector, focused blind human review, reliability calibration, and verified dashboard output.

**Architecture:** A typed Python package evaluates alpha sequences deterministically, calls any number of judge adapters, persists complete prompt/model traces, generates a self-contained HTML dashboard, and optionally serves a review API. Human labels are appended to JSONL and converted into reliability statistics and model-fusion weights.

**Tech Stack:** Python 3.10+, NumPy, Pillow, OpenCV, SciPy, Pydantic v2, PyYAML, httpx, matplotlib, standard-library HTTP server, pytest.

## Global Constraints

- Core execution must not require a real VLM endpoint.
- Mock judges and simulated human labels must be marked synthetic.
- Every image sent to a model must have explicit role/alpha/level metadata.
- Human blind review must hide prompts, model identities, model scores, rationales, and final decision until submission.
- Review queues must prioritize disagreement, tails, and high-risk cases.
- Source, tests, and docs must be browsable in GitHub.

---

### Task 1: Package scaffold and typed contracts

**Files:** `pyproject.toml`, `src/qwen_tmqa/domain.py`, `src/qwen_tmqa/config.py`, `configs/default.yaml`, `tests/test_domain_config.py`.

- [ ] Write validation tests for scene ordering, model traces, blind reviews, and config loading.
- [ ] Run the tests and confirm import failures.
- [ ] Implement Pydantic models and YAML configuration.
- [ ] Re-run tests and commit.

### Task 2: Dataset discovery and deterministic metrics

**Files:** `src/qwen_tmqa/dataset.py`, `src/qwen_tmqa/image_io.py`, `src/qwen_tmqa/metrics.py`, `tests/test_dataset_metrics.py`.

- [ ] Write tests for nine alpha directories, file hashes, image dimensions, brightness trajectories, clipping, and sequence reversals.
- [ ] Verify failures.
- [ ] Implement discovery and metrics.
- [ ] Verify pass and commit.

### Task 3: Versioned prompts and input manifests

**Files:** `src/qwen_tmqa/prompts.py`, `tests/test_prompts.py`.

- [ ] Test explicit image labels, source/baseline roles, stable hashes, templates, rendered prompts, and version diffs.
- [ ] Verify failures.
- [ ] Implement the prompt registry and rendering.
- [ ] Verify pass and commit.

### Task 4: Multi-judge execution

**Files:** `src/qwen_tmqa/judges/base.py`, `mock.py`, `openai_compatible.py`, `src/qwen_tmqa/evaluation.py`, `tests/test_judges_evaluation.py`.

- [ ] Test four independent model outputs, preserved traces, malformed JSON retry, model failure isolation, and model-by-dimension gaps.
- [ ] Verify failures.
- [ ] Implement judge adapters and evaluation orchestration.
- [ ] Verify pass and commit.

### Task 5: Review queue and calibration

**Files:** `src/qwen_tmqa/review.py`, `tests/test_review_calibration.py`.

- [ ] Test disagreement/tail/high-risk routing, blind-review schema, append-only storage, human-model errors, decision agreement, and inverse-error fusion weights.
- [ ] Verify failures.
- [ ] Implement selection, storage, and calibration.
- [ ] Verify pass and commit.

### Task 6: Dashboard and Prompt Inspector

**Files:** `src/qwen_tmqa/visualization.py`, `src/qwen_tmqa/assets/dashboard.html`, `tests/test_visualization.py`.

- [ ] Test presence of overview, scene inspector, stage trace, model matrix, Prompt Inspector, rendered/template/input/diff/raw/parsed views, blind mode, queue, and reliability page.
- [ ] Verify failures.
- [ ] Implement self-contained dashboard generation.
- [ ] Verify pass and commit.

### Task 7: Review server and CLI

**Files:** `src/qwen_tmqa/server.py`, `src/qwen_tmqa/cli.py`, `scripts/run_verified_experiment.sh`, `tests/test_server_cli.py`.

- [ ] Test valid/invalid review POSTs, static serving, all CLI commands, and end-to-end experiment output.
- [ ] Verify failures.
- [ ] Implement server and CLI.
- [ ] Verify pass and commit.

### Task 8: Documentation, requirements audit, and experiment

**Files:** `README.md`, `docs/REQUIREMENTS_MATRIX.md`, `docs/EXPERIMENT_REPORT.md`, verified outputs under `examples/verified_experiment/`.

- [ ] Run compile, lint, tests, package build, clean-environment install, CLI experiment, HTML checks, and requirements audit.
- [ ] Fix every failed check and re-run the full suite.
- [ ] Commit verified source and outputs.
- [ ] Submit the verified tree to `baolinv0/IQA` only after every requirement is marked PASS.
