# Task 4 — round 3 remediation

Read the fresh Critic verdict in `evidence/review/task-4-r3.md` and coordinated both exact reproductions with that Critic. The final verdict identifies precisely T4-R3-01 and T4-R3-02; prior T4-R1-01..04 and T4-R2-01 are independently CLOSED. Builder status for the two new findings: **FIXED**, pending fresh independent Critic closure. No commit or push performed.

| Finding | Changed paths and behavior | Regression |
|---|---|---|
| T4-R3-01 — P2, INT-011/014 | `src/portrait_eval/report_publication.py` flushes the final row and captures all response values before commit. No expired ORM attribute is read after final durable commit. A committed guard prevents destructive cleanup of successfully committed publication artifacts. The publication ID used by precommit failure handling is also captured as a primitive before rollback/expiration. | `tests/test_report_publication.py::test_expiring_session_returns_committed_report_without_refresh_or_deleting_artifacts` uses actual SQLite, real HTML/JSON/CSV/source-template DOCX, conventional `Session(database.engine)` with default `expire_on_commit=True`, and the Critic's exact engine hook: SELECT reports fails only after the second successful commit. It verifies no postcommit report SELECT occurs, both commits complete, all four artifacts exist, the durable final row matches the response, and the reservation stays published with no error. |
| T4-R3-02 — P2, INT-014 | `src/portrait_eval/static/index.html` renders project names/status and review category/status/provider payload with created DOM nodes and `textContent`. Project open actions use event listeners capturing exact labels, with no data-derived inline handlers. Existing pairing strings remain escaped; version interpolation is now escaped as well. | `web/tests/StaticConsole.test.mjs` reads and executes the actual fallback HTML through JSDOM with `runScripts:dangerously`. It supplies executable img/onerror markup in model statements, review categories and project names/status; dispatching error events must not execute handlers, no injected nodes may exist, and the exact markup stays visible as text. A second regression clicks a project with apostrophes, quotes and angle brackets and verifies the exact label and normal API flow through an event listener. |

## Reproduction and verification

- Before changing the publication helper, the new conventional-session test failed at the expired `row.id` SELECT with `RuntimeError('connection lost after durable final commit')`, exactly reproducing T4-R3-01.
- The static tests initially hit Vite's external-file raw-import restriction; they now read the actual repository static file with Node fs and execute it in a dedicated JSDOM. To prove the behavioral baseline, extracted the original unsafe static source with `git show HEAD:src/portrait_eval/static/index.html` into scratch and ran `FRONE_STATIC_CONSOLE_TEST_SOURCE=/workspace/scratch/617f8bd555c4/task-4-r3-unsafe-static.html npm test -- --run tests/StaticConsole.test.mjs`: **2 failures**. Actual provider error handler changed its window marker to true; quoted project label generated a broken dynamic onclick. Default tests read the current production file, with no historical-source override.
- Fresh full Front suite, using `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python` and its bin on PATH: `python -m pytest tests -q` — **170 passed**, 25.22 seconds; five upstream Starlette/httpx/Alembic deprecation warnings.
- `npm test -- --run` from web — **7 passed across 4 files**, including both actual static-page tests.
- `npm run build` — **passed**, TypeScript and Vite (33 production modules).
- `python -m ruff check src/portrait_eval/report_publication.py tests/test_report_publication.py` — **passed**.
- `python -m ruff format --check` on the same paths — **2 files already formatted**.
- `python -m mypy src/portrait_eval/report_publication.py` — **passed**, no issues.
- Node `new Function` parsing of the production static script — **passed**.
- `git diff --check` — **passed**.

Evidence is local software behavior with actual SQLite/SQLAlchemy, source-derived DOCX and the production fallback JavaScript executed in a browser-like DOM. No external model or capture-quality validation is claimed.
