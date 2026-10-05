# Builder Task 4 — publication, privacy and pairing UI

Implementation follows `docs/integration/DESIGN.md` and `task-4-brief.md`. Changes are limited to Task 4 report/API/export/workflow/UI ownership and related tests. No commit or push was performed.

## Requirement mapping

| Requirement | Implementation / evidence |
|---|---|
| INT-011 one resolved publication | `report_projection.resolve_report_payload` is called by actual professional API finalization and quick workflow. Stable IDs remove rejected/insufficient evidence records recursively from findings, scenes, primary/reviewer observations, attribution, external/capture evidence and supported profiles. Profiles with legacy derived prose rebuild from surviving strategy evidence. |
| INT-011 legacy and device specificity | Nested `primary_observation` resolves with device, dimension and scene scope. Identical statements belonging to a different device remain. Unique text-only historical decisions resolve; ambiguous text-only decisions fail closed and require a stable ID. Actual FastAPI/SQLite finalization inspects final HTML, JSON and real-template DOCX for absence of rejected content and presence of accepted content. |
| INT-011 missing source JSON | Professional API returns 409 if source JSON is missing/invalid or review identity is ambiguous; it never bypasses resolution using an HTML banner replacement. Existing immutable-version API regression now supplies an actual structured draft. |
| INT-012 quick and provenance | Both renderers use shared truthful banners; quick records open counts and skipped mandatory review gates. Every HTML/JSON/DOCX bundle discloses heuristic/synthetic evidence, or unrecorded adapter provenance. `synthetic: false` is distinguished from actual synthetic evidence. ReportPayload retains extra lineage fields. Existing DOCX score fallback remains approved-evidence counts explicitly distinguished from quality scores. |
| INT-012 private export | Recursive sanitizer covers keys, values, nested lists, diagnostics, errors, POSIX and Windows paths; path-only values become opaque asset references. Relative assets and HTTP sources are retained. All export JSON members are sanitized, including added `pairing-snapshots.json`; existing snapshot, confidence/review notes/history/input hashes are preserved. Actual ZIP/SQLite/image test checks no source workspace paths remain. |
| INT-010 UI visibility | React pairing grid and static console expose matching confidence, review status and notes, image thumbnails, and independent manual confirmation state. React supports edit interactions and confirmed snapshots. Authenticated thumbnail requests send API token headers and revoke blob URLs. Task 3 owns independent backend confirmation semantics. |
| INT-014 regressions | Real source-derived DOCX template remains unchanged; tests use it, not a fabricated template. Focused backend/API/report/export regressions and React/type/build checks below. |

## Verification

Interpreter: `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python`; PATH includes its `bin` for subprocesses.

- `python -m pytest tests/test_report_projection.py tests/test_reporting.py tests/test_exporting.py tests/test_api.py -q`: **24 passed**, one upstream Starlette/httpx deprecation warning.
- `python -m ruff check src/portrait_eval/{api,reporting,report_projection,docx_reporting,exporting,workflow}.py tests/test_report_projection.py tests/test_exporting.py tests/test_api.py`: **passed**.
- `python -m ruff format --check` on the same files: **9 files already formatted**.
- `npm test -- --run` from `web`: **5 tests passed across 3 files**, including authenticated thumbnail transport.
- `npm run build`: **passed**, TypeScript and Vite production build (33 modules).
- Parse static console JavaScript with Node `new Function`: **passed**.

The initial backend collection attempt encountered Task 3's in-progress missing `iqa_bridge`; rerunning after the bridge appeared passed. An initial React test exposed missing automatic DOM cleanup between cases; adding explicit cleanup fixed it.

## Limits

Tests establish real local API/database publication and generated artifact behavior, React interaction and transport, and export privacy. No paid model call, real mobile capture accuracy or independent human evidence validity is claimed. Stable-ID decisions are authoritative. Historical ambiguous text-only decisions intentionally stop publication rather than suppressing a different device. PDF consumes the resolved HTML through the existing renderer; optional WeasyPrint PDF generation was not separately exercised here. Review history is preserved when present; creating a new immutable backend review-history system is outside this task's repository ownership.
