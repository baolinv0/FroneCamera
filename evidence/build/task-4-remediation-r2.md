# Task 4 — round 2 publication concurrency remediation

Finding T4-R2-01 (P1) was returned by the fresh Critic via the parent task. Builder status: **FIXED**, pending fresh independent Critic closure. Round 1 findings T4-R1-01 through T4-R1-04 remain addressed. No commits/pushes or additional agents were created.

## Change and requirement mapping

| Requirement | Implementation |
|---|---|
| Shared durable cross-process allocation | New `src/portrait_eval/report_publication.py::reserve_publication` creates a committed `ReportPublicationRow` under database unique `(project_id, ordinal)` constraint. Concurrent collisions/SQLite snapshot contention roll back and retry; a bounded exhausted contention returns `PublicationConflict` (professional API 409). Both professional API and quick worker workflow call the same publisher. There is no production process mutex. |
| Compatibility with existing report versions | Allocation considers durable reservations and existing final `1.N` report versions. Failed/abandoned allocations remain consumed rather than reusing paths/versions. Existing final 1.7 migrates and next allocation is 1.8. |
| Immutable complete artifacts | Each reservation owns unique staging and final directories keyed by its database ID. The renderer completes nonempty HTML/JSON/CSV/DOCX inside staging; all returned paths must resolve inside that directory. The complete directory is atomically renamed before the final ReportRow is registered. Relative images from the immutable draft are copied into the new bundle. |
| Failure and retry correctness | Renderer exception/incomplete bundle produces no successful ReportRow, removes partial artifacts, and records failed reservation/error. Retry gets a fresh version and directory. Final ReportRow registration, reservation status and project status commit together. A process crash can leave an unregistered staging/orphan directory, but its consumed reservation cannot overwrite a successful publication. |
| Latest publication after out-of-order completion | Final report created_at uses its committed allocation timestamp. If an earlier request finishes after a newer publication, report listing/latest does not replace the newer publication with the earlier review projection. Both concurrent projections remain addressable by immutable report IDs. |
| Controlled existing deployment schema | New `migrations/versions/20261005_0003_report_publications.py` provides additive idempotent Alembic upgrade and downgrade after 20260720_0002. `database.py` adds only the publication table and UniqueConstraint import; no existing table columns or Task 3 repository functions were changed. |

Changed implementation paths: `src/portrait_eval/{api.py,workflow.py,database.py,report_publication.py}` and the additive migration. New regressions: `tests/test_report_publication.py`.

## Regression evidence

The first four tests were added **before** implementation changes. `python -m pytest tests/test_report_publication.py -q` then produced **4 failures**, reproducing pro/pro and quick/pro reuse of 1.0, and renderer failure retries reusing 1.0.

Final regression file contains nine tests:

- Real concurrent FastAPI professional/professional and quick/professional publications using independent database sessions. First rejected-a projection pauses; another connection approves a; second publication returns a+b; first resumes and returns b. Versions/directories differ, every byte of the already returned second bundle (including CSV) remains unchanged, database rows differ, and latest endpoint retains the later allocation.
- Professional and quick renderer exceptions plus incomplete renderer result tests. Neither registers a final row; partial staging disappears; retry obtains a new ordinal and complete artifacts.
- Three spawned independent processes allocate unique database ordinals, establishing shared coordination beyond any thread/process mutex.
- Real image fixture verifies draft-relative assets survive staging/materialization into the final immutable directory.
- Existing-schema Alembic upgrade/downgrade/re-upgrade retains existing reports and creates the reservation table; legacy final1.7 results in next ordinal8.

## Commands and results

Interpreter `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python`; PATH contains that environment's bin.

- `python -m pytest tests/test_report_publication.py -q` — **9 passed** after implementation.
- Fresh complete Front suite `python -m pytest tests -q` — **164 passed**, 24.76 seconds. Five upstream Starlette/httpx/Alembic configuration deprecation warnings.
- `python -m ruff check src/portrait_eval/{database,report_publication,api,workflow}.py tests/test_report_publication.py migrations/versions/20261005_0003_report_publications.py` — **passed**.
- `python -m ruff format --check` on those six files — **6 files already formatted**.
- `python -m mypy src/portrait_eval/report_publication.py src/portrait_eval/database.py` — **passed**, no issues in two source files.
- `git diff --check` — **passed**.

No UI changes were made in this round. Software evidence uses actual local FastAPI/SQLite, separate spawned processes, source-derived DOCX rendering and Alembic migration; no paid-model/capture-quality claim is introduced. The database uniqueness invariant is dialect-independent SQLAlchemy; the exercised database is SQLite, not a separately provisioned PostgreSQL service.
