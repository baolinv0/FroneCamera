# SDD ledger — plan: docs/integration/IMPLEMENTATION_PLAN.md

Ruling: the live instruction authorizes execution of the preceding integration design and upload; preserve the separate 0.2 frozen source snapshots, do not silently replace them or claim 0.3 completion.
Ruling: parallel disjoint Builders are explicitly requested and override skill sequential-dispatch defaults.

| Tasks | Shared interface | Conflict resolution |
|---|---|---|
| 1/2 | Comparison evidence and training export | Task 1 new comparison files; Task 2 legacy/PGT files; root joins CLI if needed |
| 1/3 | ComparisonRequest/evaluate_comparison | Exact DESIGN signature; Front device mode never training |
| 3/4 | Stable claim IDs/review projection | Task 3 produces IDs; Task 4 owns projection |
| 1/5 | Packaging CLI entry | Root owns pyproject |
| 2/5 | PGT CLI/old tests | Task 2 owns CLI content; root owns integration smoke |
| 3/5 | Root runtime dependency | Root installs nested IQA before Front |
| 4/5 | UI/CI | Task 4 owns UI; root owns CI |

All task ownership is disjoint; tests may be added by owner under unique new filenames. Old tests are not deleted or weakened.

Baseline: Front89/89 passed after retrieving exact omitted binary template; IQA103/103 passed with venv PATH inherited by subprocesses. Neither environment repair changed product or tests.
Builders: /root/build_iqa_compare (Task1), /root/build_iqa_admission (Task2), /root/build_front_pipeline (Task3), /root/build_front_publication (Task4).

## Latest acceptance state

| Task | Remediation and independent review | State |
|---|---|---|
| 1, comparison core | TIFF orientation/precision, ROI scope collisions, truncated NPY, native HEIC and decompression guard; fresh Critic round 3 | PASS_TO_EVALUATOR; all findings closed |
| 2, training admission | Strict finite runtime bbox; split/group closure, distinct Judges, byte-bound human attestation and persistent reveal lifecycle; fresh Critic round 2 | PASS_TO_EVALUATOR; all findings closed |
| 3, Front pipeline | Semantic opposition, cross-scene counterexamples, invalid-core gating, uint16 normalization and scientific-schema privacy; score-schema residual fixed in round 4; fresh Critic round 5 | PASS_TO_EVALUATOR; all findings closed |
| 4, publication | Rejected raw text, ID/scoped matching, UNC paths, durable concurrent report allocation, postcommit cleanup and fallback HTML safety; fresh Critic round 4 | PASS_TO_EVALUATOR; all findings closed |
| 5, integration | Shared-core canary, packaging/Docker/CI and explicit branch installation docs; fresh Critic round 2 | PASS_TO_EVALUATOR; all findings closed |

Root's final local Front verification after Task 3 round 4: **190 tests pass**, global Ruff/format pass and 36 source files type checked. Web has 7 passing tests and a passing TypeScript/Vite build. Rebuilt installed Front wheel executes the real two-scene workflow with the shared IQA core, source binding, deterministic comparison and path-free export. The legacy CLI, deterministic synthetic experiment and secret/JavaScript checks pass. Historical prompts and assertions are preserved.

Standalone branch `feature/iqa-standalone-20261005` is published at `9dcc99b815092f0b110db42b668a1d0696569e78`. Actual CI run `37281059505` passes Python 3.10, Python 3.12 (254 tests each) and the 19/19 exact-head audit. Its receipt is `evidence/standalone-delivery.json`.

All five final Critics now return PASS_TO_EVALUATOR. Integration upload and actual CI are in progress; a new independent Evaluator follows completed CI. No main merge, paid-provider accuracy, human Gold or training improvement is claimed.
