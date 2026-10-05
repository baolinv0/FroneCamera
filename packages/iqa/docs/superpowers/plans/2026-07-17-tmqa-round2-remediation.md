# TMQA Round 2 Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:verification-before-completion. Execute inline; delegation is not authorized.

**Goal:** Close CRIT-012 through CRIT-017 without changing frozen requirements or default thresholds.

**Architecture:** Keep policy in validated, versioned configuration; propagate provenance into scene/calibration/audit artifacts. Make adapters and the audit fail closed at their input boundaries, then bind audit PASS to behavioral and exact-head CI evidence.

**Tech Stack:** Python 3.10/3.12, Pydantic v2, pytest, Ruff, httpx, static HTML/JavaScript.

## Global Constraints

- Preserve TMQA spec version 0.2.0 and every frozen threshold.
- Do not skip, xfail, weaken, or bypass tests.
- Do not modify PR draft state or merge.

### Task 1: Complete judge response contract and synthetic lineage

**Files:** `tests/test_judges_evaluation.py`, `src/qwen_tmqa/judges/openai_compatible.py`

- [ ] Add parameterized missing/invalid top-level field tests and real/synthetic success/failure tests.
- [ ] Run focused tests and preserve expected RED output.
- [ ] Require scores, decision, confidence, issues, and rationale; propagate `JudgeConfig.synthetic`.
- [ ] Run V-012 focused tests GREEN.

### Task 2: Zero-judge safety and versioned decision thresholds

**Files:** `tests/test_zero_judge_visualization.py`, `tests/test_judges_evaluation.py`, `tests/test_domain_config.py`, `src/qwen_tmqa/config.py`, `src/qwen_tmqa/domain.py`, `src/qwen_tmqa/evaluation.py`, `configs/default.yaml`

- [ ] Add zero-judge REVIEW/reason/uncertainty and config-controlled threshold/provenance tests.
- [ ] Run focused tests and preserve expected RED output.
- [ ] Add validated named fields with policy version, use them for decision/stage/risk cutoffs, and persist provenance.
- [ ] Run V-015/V-017 focused tests GREEN.

### Task 3: Calibration sample sufficiency disclosure

**Files:** `tests/test_review_calibration.py`, `tests/test_server_cli.py`, `tests/test_visualization.py`, `src/qwen_tmqa/config.py`, `src/qwen_tmqa/cli.py`, `src/qwen_tmqa/visualization.py`, `src/qwen_tmqa/assets/dashboard.html`

- [ ] Add one/few-real-sample non-production tests and dashboard disclosure test.
- [ ] Run focused tests and preserve expected RED output.
- [ ] Add versioned configured minimum sample count and carry sufficiency/warning to calibration/dashboard.
- [ ] Run V-017/V-018 focused tests GREEN.

### Task 4: Evidence-bound nineteen-requirement audit

**Files:** `tests/test_audit_negative.py`, `src/qwen_tmqa/audit.py`, `src/qwen_tmqa/cli.py`, `scripts/run_verified_experiment.sh`, `.github/workflows/ci.yml`

- [ ] Add injected fatal/disagreement/zero-unavailable/route/dashboard/exact-head CI failure tests.
- [ ] Run focused tests and preserve expected RED output.
- [ ] Add per-requirement implementation/test/command/threshold/actual-evidence/status records and require behavioral plus exact-head CI evidence.
- [ ] Generate evidence in the verified experiment and run V-009 GREEN with 19/19 only for valid inputs.

### Task 5: Full mechanical verification and publish

- [ ] Run V-001..V-009 and V-011..V-019.
- [ ] Recheck exact remote branch head before publishing.
- [ ] Create logically grouped commits through the GitHub connector and update only `fix/tmqa-p1-remediation-v0.2.0`.
- [ ] Verify exact-head Actions jobs and artifact SHA/head binding.
