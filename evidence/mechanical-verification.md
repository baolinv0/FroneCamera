# Final local mechanical verification

Root verified the stable product code after Task 3 round 4 remediation:

| Check | Actual result |
|---|---|
| Front full pytest | 190 passed in 29.46s; five dependency deprecation warnings |
| Front Ruff / formatting / mypy | Pass; 36 source files type checked |
| Web tests / TypeScript / Vite | 7 tests across four files; build passed |
| IQA Python 3.10 and 3.12 | 254 tests passed on each GitHub job |
| IQA exact-head audit | 19/19 MUST requirements passed |

The IQA CI results are bound to standalone commit `9dcc99b815092f0b110db42b668a1d0696569e78`, run `37281059505`. Historical assertions and prompts remain intact; one historical test's import formatting changed.

Both projects build sdists and wheels. A dedicated IQA wheel environment has no Front, FastAPI or SQLAlchemy and executes the comparison CLI. A separate Front environment installs both wheels and their actual dependencies. Python isolated mode imports them from site-packages and preserves the original 35,053-byte DOCX template. After the final score-schema change, root rebuilt the Front wheel, reinstalled it in that environment, and reran the actual workflow successfully; its portable receipt is `evidence/installed-wheel-canary.json`.

The actual workflow from installed wheels deterministically compares algorithm outputs, persists two shared-core Front scene results, binds source bytes and generates a final report. Exported ZIP data contains no host paths. The canary explicitly records `paid_endpoint_calls=0`, `training_runs=0` and `evidence_kind=synthetic_images_and_heuristic_adapters`. This establishes software integration, not model accuracy or training gains.

The legacy eight-command CLI contract, deterministic six-scene synthetic experiment, dashboard JavaScript syntax and generated-output secret scans pass. Final independent closure, Evaluator and integration exact-head CI are recorded separately. Runtime/build artifacts are ignored; Builder red/green logs and independent review records are retained.
