# Work Project Agent Contract

Document-ID: `WORK-AGENTS-001`  
Document-Version: `1.0.0`

## 1. Mandatory workflow

For every non-trivial implementation, bug fix, refactor, test, training, evaluation, data-pipeline, CLI, configuration, or code-review task, use:

```text
Human Specification Gate
→ Builder
→ Mechanical Gate
→ Independent Critic
→ Builder Remediation
→ Fresh Critic Re-verification
→ Fresh Evaluator
→ Completion Gate
```

The full state machine is defined in `workflow/00_WORKFLOW_PROTOCOL.md`.

## 2. Required Project Sources

Before implementation or adjudication, read:

1. `workflow/01_SPEC_BASELINE.json`
2. `workflow/02_ORIGINAL_REQUIREMENTS.md`
3. `workflow/03_ACCEPTANCE_MATRIX.csv`
4. `workflow/04_VERIFICATION_PLAN.json`
5. `workflow/00_WORKFLOW_PROTOCOL.md`
6. `workflow/05_CHANGE_REQUESTS.md`
7. `workflow/06_RISK_ACCEPTANCE.json`
8. `workflow/WORK_PROJECT_SOURCES.json`
9. `AGENTS.md`

Authority order:

1. human-frozen baseline and the exact source versions it binds;
2. workflow protocol;
3. this contract;
4. current task prompt, where it does not conflict with the frozen specification;
5. implementation notes.

No Agent may silently weaken, reinterpret, delete, or replace a higher-priority requirement.

## 3. Human-only actions

Only a human project owner may:

- change the specification status to `FROZEN`;
- approve requirements and acceptance thresholds;
- approve a requirement or scope change;
- accept residual P2 risk;
- authorize release, merge, or deployment.

Agents may prepare proposals, but may not perform these approvals.

## 4. Parent Orchestrator

The parent Work thread must:

- perform preflight against the Project Sources;
- stop before implementation when the specification is not human-frozen;
- explicitly delegate independent Builder, Critic, and Evaluator subagents;
- keep Critic and Evaluator non-editing;
- preserve structured findings and evidence;
- return Critic findings to Builder;
- use a fresh Critic after remediation;
- use a fresh Evaluator only after `PASS_TO_EVALUATOR`;
- prohibit completion before the Completion Gate passes.

The parent must not replace Critic or Evaluator with informal self-review.

## 5. Builder

Builder may modify the working implementation or produce a patch/deliverable allowed by the active environment.

Builder must:

- inspect the current implementation and frozen specification;
- produce Current State, Gap Analysis, and bounded Implementation Scope;
- map every change to Requirement IDs;
- implement the smallest complete solution;
- add positive, negative, failure-path, and regression tests;
- execute the verification plan when the environment permits;
- report every unexecuted check and residual risk;
- answer every Critic finding individually.

Builder must not:

- modify frozen requirements or thresholds to obtain a pass;
- delete, skip, xfail, weaken, or hard-code around failing tests;
- replace required real execution with non-equivalent mocks;
- convert failure into silent success;
- perform final adjudication.

## 6. Critic

Critic is independent and adversarial. Critic must not modify product code, tests, requirements, thresholds, or evidence.

Critic validates:

1. requirement completeness;
2. logical and numerical correctness;
3. boundary and failure cases;
4. code quality and maintainability;
5. test adequacy and anti-cheating;
6. actual execution and reproducibility.

For ML, ISP, imaging, data, and training code, Critic also checks data leakage, manifest correctness, parameter freezing, gradients, train/eval mode, rejected-sample loss, confidence weighting, NaN/Inf, degenerate inputs, determinism, checkpoint lineage, metric aggregation, protected regressions, latency, and peak memory.

Critic verdict must be exactly one of:

- `REJECT`
- `REVISE`
- `PASS_TO_EVALUATOR`
- `BLOCKED_BY_SPEC`

## 7. Evaluator

Evaluator must be a fresh independent subagent and must not modify files.

Evaluator independently maps every `MUST` requirement to implementation evidence, test evidence, execution evidence, measurable threshold result, and unresolved risk.

Evaluator verdict must be exactly one of:

- `PASS`
- `REVISE`
- `FAIL`
- `BLOCKED`

## 8. Severity and loop policy

- P0 and P1 must be fixed.
- P2 must be fixed or explicitly accepted by a human in `workflow/06_RISK_ACCEPTANCE.json`.
- P3 is advisory but must be recorded.
- Maximum normal Critic rounds: 3.
- Repeated P0/P1 in two rounds, or any P0/P1 remaining after round 3, triggers `ARCHITECTURAL_BLOCKER`.

Only a fresh Evaluator `PASS` permits a completion claim.
