# Testing and acceptance

The software target is a usable, repeatable comparison workflow: register device folders, inspect and confirm scene correspondence, evaluate the confirmed image bytes, apply human review, and produce immutable reports with traceable evidence. Passing software tests does not establish real-model accuracy or training benefit.

Use Python 3.12 and Node 22 or 24. Install the two Python projects together and activate the environment so subprocesses also use its Python:

```bash
bash scripts/install_development.sh
source .venv/bin/activate
python -m pytest -q
(cd packages/iqa && python -m pytest -q)
(cd web && npm ci && npm test -- --run && npm run build)
ruff check src tests scripts
ruff format --check src scripts
mypy src/portrait_eval
```

Run the existing integration canary in a fresh directory. It creates synthetic image fixtures, uses declared heuristic adapters, exercises standalone IQA and the actual Front workflow, verifies persisted shared-core evidence and source binding, and checks the exported package for host paths:

```bash
acceptance_output=$(mktemp -d /tmp/fronecamera-acceptance.XXXXXXXX)
python scripts/run_integration_canary.py --output "$acceptance_output"
python scripts/run_http_acceptance.py
```

Required software acceptance checks:

| Goal | Failure case the test must catch | Expected result |
|---|---|---|
| Browser workflow | Project/device POST resolves after the React event ends | Form resets and the new project/device appears without an error |
| Pairing review | One of four devices has contradictory content, despite a high median confidence | Every affected group requires review |
| Explicit missing positions | Zero-padded `01..09` reference, other device only `01/02` | `02` maps to G002 and later cells remain missing |
| Stable run evidence | Pairing or source bytes change during search or rendering | Stale run cannot register a valid final or overwrite the new pairing state |
| Human decisions | All source observations for a high-grade strategy are rejected | Derived strategy is removed or recomputed; old grade/confidence cannot survive |
| Professional gate | Another session adds unresolved or rejecting evidence while publishing | Unsafe publication returns conflict and registers no final |
| Color correctness | Equivalent visual colors have different embedded ICC encodings | Normalize to a common space or explicitly refuse unsupported evidence |
| Bounded input handling | NPY header requests huge memory but the file is tiny | Reject before allocation, retain byte lineage, continue normal siblings |
| Public artifact privacy | Nested evidence contains source/diagnostic host paths | Newly generated draft and final HTML/JSON/CSV/DOCX contain no host locations; relative assets still work |

Existing regressions also cover precision, EXIF/HEIF orientation, source lineage, split admission, blind review exposure, schema validation, persistence, retries, concurrent report allocation and immutable historical bundles. All required tests must execute, not be skipped or replaced by a zero-test run. Record the tested Git commit/diff, commands, test counts and failures. A failing acceptance probe is a software target failure even when the old suite passes.

`run_http_acceptance.py` starts an actual local API and worker in an isolated temporary workspace. It creates four synthetic devices with six scenes and one missing middle capture, then queues quick and professional evaluations. It checks task completion, persisted shared IQA evidence, draft/final HTML/JSON/CSV/DOCX, signed HTML and DOCX access without administrator authorization, professional review gating, and unchanged source bytes. It uses heuristic adapters and saves logs and a JSON result under the printed evidence directory. An optional `--output` must name an empty directory.

For actual operation, use the documented API and worker. Quick mode may preserve unresolved items with an explicit warning; professional mode must enforce its gate. Regenerate bundles produced before the privacy fix before sharing them. Docker and optional PDF require their own checks when used. High-precision ICC and unsupported HEIF NCLX color profiles are explicitly refused; unprofiled inputs retain a visible sRGB assumption warning.

Real evaluation quality is a separate acceptance stage. Use actual matched phone captures with recorded settings, keep derivative captures within the same data split, and hold out the evaluation scenes. Collect independent human judgments with device names hidden and order randomized. Define acceptable measurement error and claim agreement thresholds before comparing model outputs; count unsupported claims and missed comparability warnings separately from stylistic preferences. Check contradictory scenes and insufficient-coverage cases, and require unsupported mechanism attribution to remain indeterminate. Save the dataset version, real model identity, prompt/input traces and human decisions. Without these captures and blind reference judgments, report real-world accuracy as unverified.

Local acceptance on 2026-10-05 tested the uncommitted repair of base `489744672c2768ca157ba72e69fcbb1e001b30e9` on `fix/iqa-acceptance-20261005`, using Python 3.12.14 and Node 24. Backend: 247 passed; shared IQA: 272 passed; frontend: 9 passed. No tests were skipped. Frontend production build, Ruff, formatting, mypy, Python sdist/wheel builds, isolated wheel loading and the IQA CLI contract passed. The standalone integration canary and the actual queued HTTP acceptance passed. Independent reviews approved the repaired UI/pairing, core, pipeline, publication and acceptance harness. These are local software results; Python 3.10, hosted CI, Docker/PDF, real-model accuracy and training benefit were not verified in this run.
