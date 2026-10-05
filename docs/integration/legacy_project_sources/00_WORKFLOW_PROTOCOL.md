---
document_id: WORK-PROTOCOL-001
document_version: 1.0.0
status: ACTIVE
---

# Builder–Critic–Evaluator Workflow Protocol

## 1. Scope and environment boundary

This protocol governs non-trivial development and technical-review tasks performed in ChatGPT Work using Project Sources.

Project Sources are snapshots. They do not automatically synchronize with later local-file or repository changes. When a source changes, replace the uploaded Project Source and update `workflow/WORK_PROJECT_SOURCES.json`.

A ChatGPT Project must not be represented as direct access to a local repository unless the active environment actually exposes that repository through an approved tool or Local Project. When live execution is unavailable, results must be labeled `NOT_RUN` or `BLOCKED`; code inspection is not equivalent to execution.

## 2. Phase 0 — Human Specification Gate

Before Builder starts, confirm:

- `workflow/01_SPEC_BASELINE.json` has `status: FROZEN`;
- `approved_by_type` is `HUMAN`;
- a human approver and approval time are recorded;
- the requirements, acceptance matrix, and verification plan match the baseline version;
- every `MUST` requirement has a stable Requirement ID, verification method, measurable pass threshold, and required evidence;
- goals, non-goals, invariants, constraints, interfaces, and deliverables are explicit.

If any condition fails:

1. do not modify product code;
2. prepare a proposed specification correction;
3. set the working state to `AWAITING_HUMAN_SPEC_APPROVAL`;
4. stop.

Agents may propose a specification but may not freeze it.

## 3. Phase 1 — Builder

Explicitly delegate a Builder subagent.

Builder must first produce:

- Current State;
- Gap Analysis;
- bounded Implementation Scope;
- requirement-to-change plan;
- planned verification checks.

Builder then:

- implements the smallest complete solution;
- adds or updates meaningful tests;
- preserves frozen invariants and compatibility constraints;
- executes applicable verification checks;
- records actual results and missing checks;
- reports residual risks.

A Builder statement such as “all tests pass” is not a completion signal.

## 4. Phase 2 — Mechanical Gate

Before Critic review, inspect all applicable evidence:

- implementation diff or patch;
- syntax/compile result;
- lint/format result;
- type-check result;
- unit and integration tests;
- clean build or installation;
- CLI/API smoke test;
- synthetic end-to-end execution;
- representative real or production-like canary;
- deterministic rerun;
- latency and memory checks;
- required artifacts and lineage.

Mechanical Gate fails when:

- an applicable required check fails;
- a required check is omitted without a documented blocker;
- a frozen requirement or threshold was changed;
- a test was deleted, skipped, xfailed, weakened, or replaced by a non-equivalent mock;
- a failure was converted into silent fallback;
- evidence is missing, stale, or inconsistent with the implementation.

Mechanical failure returns to Builder before Critic review.

## 5. Phase 3 — Independent Critic

Explicitly delegate a fresh Critic subagent.

Critic must not modify product code, tests, requirements, thresholds, or evidence.

Each finding must contain:

- Finding ID;
- Severity: `P0`, `P1`, `P2`, or `P3`;
- Requirement ID;
- Category;
- affected files, symbols, or artifacts;
- failure mechanism;
- reproduction command or precise probe;
- expected behavior;
- actual behavior;
- evidence;
- exact closure criteria;
- status.

Critic must check for missing mandatory behavior, incorrect formulas or data flow, unsupported assumptions, adversarial inputs, test gaps, test-specific hard coding, deleted/weakened tests, silent exception handling, data leakage, stale logs, false reproducibility claims, and scope expansion.

Critic verdict must be one of:

- `REJECT`
- `REVISE`
- `PASS_TO_EVALUATOR`
- `BLOCKED_BY_SPEC`

## 6. Phase 4 — Builder Remediation

For `REJECT` or `REVISE`, return the complete finding list to Builder.

Builder must classify each finding as:

- `FIXED`
- `DISPUTED`
- `BLOCKED`
- `REQUEST_HUMAN_RISK_ACCEPTANCE`

Rules:

- `FIXED` includes changed locations, tests, commands, and evidence;
- `DISPUTED` includes a technical argument and executable counter-evidence;
- Builder cannot close its own finding;
- only a human may accept residual P2 risk.

## 7. Phase 5 — Fresh Critic Re-verification

Delegate a fresh Critic, not Builder self-review.

The fresh Critic must:

1. reproduce every prior open finding;
2. verify the exact closure criteria;
3. run regression checks;
4. scan for new defects caused by remediation;
5. preserve Finding IDs across rounds;
6. issue a new verdict.

A prior finding remains open until a later Critic explicitly marks it `FIXED`.

Use at most three normal review rounds. Stop patching and report `ARCHITECTURAL_BLOCKER` when the same P0/P1 defect class recurs in two rounds, or any P0/P1 remains after round three.

## 8. Phase 6 — Fresh Evaluator

Only after the latest Critic returns `PASS_TO_EVALUATOR`, explicitly delegate a fresh Evaluator.

Evaluator receives:

- the human-frozen baseline;
- original requirements;
- acceptance matrix;
- verification plan;
- final implementation or patch;
- actual execution evidence;
- all Critic rounds;
- remediation evidence;
- human risk acceptances;
- unresolved risks and scope deviations.

Evaluator independently maps every `MUST` requirement to implementation location, test location, verification result, threshold result, evidence, and final status.

Evaluator verdict must be exactly one of:

- `PASS`
- `REVISE`
- `FAIL`
- `BLOCKED`

## 9. Phase 7 — Completion Gate

Completion requires all of the following:

- specification remains human-frozen;
- all `MUST` requirements are independently verified as `PASS`;
- all required verification checks passed, or the task is explicitly `BLOCKED`;
- unresolved P0/P1 count is zero;
- every unresolved P2 has explicit human acceptance;
- no unapproved requirement or threshold change occurred;
- no required test was weakened or bypassed;
- latest Critic verdict is `PASS_TO_EVALUATOR`;
- fresh Evaluator verdict is `PASS`;
- unresolved limitations are disclosed;
- Project Sources are the expected versions listed in `WORK_PROJECT_SOURCES.json`.

## 10. Change Control

An approved change request does not directly override the frozen specification.

For a requirement, scope, threshold, or verification change:

1. record the proposal in `workflow/05_CHANGE_REQUESTS.md`;
2. stop the current run;
3. update requirements, acceptance matrix, and verification plan;
4. increment `spec_version`;
5. update Project Sources and their manifest;
6. obtain new human approval;
7. restart from Phase 0.

This prevents old and new specifications from being simultaneously authoritative.
