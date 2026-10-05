# TMQA V1 Usability Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Close the three direct V1 usability gaps without changing the established architecture or expanding product scope.

**Architecture:** Extend the existing self-contained engineering Dashboard, make existing Pydantic configuration models fail closed on unknown keys, and separate local V-007 execution from the existing exact-head aggregate audit. Preserve all current interfaces and reviewer isolation.

**Tech Stack:** Python 3.10/3.12, Pydantic v2, pytest, vanilla HTML/JavaScript, Bash, GitHub Actions.

## Global Constraints

- Preserve the frozen TMQA 0.2.0 requirements and thresholds.
- Do not modify reviewer-safe data separation or reveal authorization.
- Do not implement deferred CRIT-012, CRIT-015, or CRIT-016 in this task.
- No architectural refactor or new dependency.
- Every behavior change follows RED -> GREEN TDD.

---

### Task 1: Complete the existing engineering Dashboard

**Files:**
- Modify: `src/qwen_tmqa/assets/dashboard.html`
- Modify only if payload plumbing is missing: `src/qwen_tmqa/visualization.py`
- Test: `tests/test_visualization.py`

- [ ] Add failing tests that assert the rendered engineering UI exposes the required overview, alpha metrics, model metadata/gaps, prompt metadata, and reliability statistics.
- [ ] Run the focused tests and preserve the expected RED output.
- [ ] Extend existing render functions with the smallest markup/JavaScript changes; do not redesign navigation or reviewer UI.
- [ ] Run focused visualization and blind-review regression tests to GREEN.
- [ ] Run both Dashboard JavaScript syntax checks.

### Task 2: Reject unknown configuration keys

**Files:**
- Modify: `src/qwen_tmqa/config.py`
- Test: `tests/test_domain_config.py`

- [ ] Add parameterized failing tests for unknown root, dataset, judge, decision policy, review, calibration, and visualization keys.
- [ ] Run the focused tests and preserve the expected RED output.
- [ ] Apply strict extra-field rejection consistently to every config model without changing defaults.
- [ ] Run config and CLI validation tests to GREEN.

### Task 3: Restore the documented V-007 command

**Files:**
- Modify: `scripts/run_verified_experiment.sh`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify or create a focused deterministic comparison helper only if needed under `scripts/`
- Test: the existing CLI/E2E tests plus a focused script contract test

- [ ] Add a failing test showing the plain documented command currently requires CI-only variables or omits deterministic comparison evidence.
- [ ] Run it and preserve the expected RED output.
- [ ] Make the plain command execute V-007 only; leave formal V-009 exact-head audit in the aggregate CI job.
- [ ] Add a second deterministic run and write `deterministic_rerun_comparison.json` comparing decisions, review queue, and stable summary fields.
- [ ] Update CI to call the plain documented V-007 command and update README artifact documentation.
- [ ] Run the exact standard command to GREEN.

### Task 4: Mechanical verification and branch update

**Files:**
- Update the existing PR branch only.

- [ ] Run compileall, Ruff, full pytest, build, isolated wheel install, CLI contract, standard V-007, both JavaScript checks, and secret scan.
- [ ] Confirm no test was skipped, xfailed, deleted, or weakened.
- [ ] Recheck the remote head before push and fast-forward the existing branch with logically grouped commits.
- [ ] Wait for exact-head Python 3.10/3.12 CI and verify artifact/head binding.
- [ ] Report the new head, run ID, test counts, artifacts, and known deferred limitations. Do not merge or mark the PR ready.
