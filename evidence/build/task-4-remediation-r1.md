# Task 4 — round 1 remediation

Original Builder addressed all four findings in `evidence/review/task-4-r1.md`. Builder implementation status is **FIXED** for each item below; independent closure remains assigned to a fresh non-editing Critic. No commit or push performed.

## Findings

| ID | Builder status | Changed paths / behavior | Regression evidence |
|---|---|---|---|
| T4-R1-01 | FIXED | `src/portrait_eval/report_projection.py` projects raw provider transport into deterministic response SHA-256 plus allowlisted adapter/model/usage/trace/provenance metadata before applying claim decisions. Provider choices, messages/content, response text and prompts are omitted from publication. Original structured draft and SQLite audit analysis remain untouched. | `tests/test_report_projection.py::test_actual_api_excludes_provider_bodies_but_retains_immutable_audit_and_hashes`, parametrized professional and quick. Realistic serialized observations in `raw.choices[].message.content` and raw prompt are absent from final HTML/JSON/real-template DOCX. Accepted claim and heuristic disclosure remain. Exact original raw hash, ordered image hashes and prompt hash remain; original draft bytes and stored raw analysis remain equal. |
| T4-R1-02 | FIXED | `report_projection.py` separates authoritative ID reviews from historical text reviews. ID-derived descriptors can resolve only ID-less representations; distinct ID-bearing claims survive even when device/text overlap. Exact dimensions/scenes from identified original claims scope historical equivalents. | `test_authoritative_claim_id_preserves_distinct_id_and_filters_idless_equivalent` rejects a/tone, preserves b/color with identical device/text, and removes an ID-less tone equivalent. Existing nested historical/no-ID tests remain passing. |
| T4-R1-03 | FIXED | `report_projection.py` uses the same scene/dimension/URL scope predicate for legacy device discovery and final rejection. Device ambiguity is evaluated within the declared scope. | `test_legacy_ambiguity_respects_scene_and_dimension_scope` resolves nested G001/tone without device ID, preserves G002/tone and G001/color, and inspects top findings plus each scene's observations. Existing truly ambiguous same-scene test still fails closed. |
| T4-R1-04 | FIXED | `src/portrait_eval/exporting.py` recursively replaces embedded Windows UNC network paths in diagnostic/error strings, alongside existing whole-path, drive-letter and POSIX handling. | `tests/test_exporting.py::test_actual_zip_redacts_embedded_unc_and_keeps_error_provenance` creates actual SQLite analysis then actual ZIP export; embedded server/share, drive and POSIX paths disappear while attempt ID, input hash and diagnostic prefix remain. |

## Commands and results

Interpreter `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python`, with that environment's `bin` on PATH.

1. Added the four regressions before changing implementation. `python -m pytest tests/test_report_projection.py tests/test_exporting.py -q` reproduced **4 failed, 11 passed**, matching all four Critic findings.
2. After remediation, focused final publication/API/export regressions: `python -m pytest tests/test_report_projection.py tests/test_exporting.py tests/test_reporting.py tests/test_api.py -q` — **29 passed**, one upstream Starlette/httpx deprecation warning.
3. `python -m ruff check src/portrait_eval/report_projection.py src/portrait_eval/exporting.py tests/test_report_projection.py tests/test_exporting.py` — **passed**.
4. `python -m ruff format --check` on those four files — **4 files already formatted**.
5. `git diff --check` — **passed**.

No UI implementation changed in this remediation; the prior Task 4 React/type/Vite evidence remains applicable. Tests establish local software behavior with the actual source-derived DOCX template and local FastAPI/SQLite/image/ZIP flows; no real model/capture-quality claim is introduced.
