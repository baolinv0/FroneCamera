Read DESIGN.md for the shared contract.

### Task 4: Unified report projection, privacy and UI

Owner: `src/portrait_eval/{api.py,reporting.py,docx_reporting.py,exporting.py,workflow.py,product_api.py,static/index.html}`, `web/src/PairingGrid.tsx` and related web tests, plus new `report_projection.py`/tests. Coordinate interface only, never edit Task 3 files. Root owns general README/packaging/CI.

Build a single resolved projection for professional and quick publication. Resolve by claim IDs when present; support old nested `primary_observation` reviews. Reject/insufficient evidence updates scene findings, primary observations, device profile summaries, attribution and report sections. Preserve separate devices with identical text. No JSON/HTML/DOCX publication of rejected claims. If source report JSON is missing, fail safely rather than bypassing resolution by HTML text replacement.

Quick mode is allowed but must explicitly state unresolved/skipped review and actual adapter/evidence provenance. Display heuristic/synthetic status and no invented device quality score. Export anonymized relative/asset references, pairing confidence/review notes and snapshot/trace lineage; no absolute diagnostic paths. Pairing UI shows thumbnails/confidence/review flags/notes and real manual-confirmation visibility. Test actual API finalize and real renderer content, privacy export, React interaction and type/build. Report `evidence/build/task-4.md`.


Verification interpreter: /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python
