Read DESIGN.md for the shared contract.

### Task 3: Front pipeline, decoder, pairing and IQA bridge

Owner: `src/portrait_eval/{pipeline.py,models.py,vlm.py,strategy.py,adjudication.py,dataset.py,repository.py,database.py,worker.py,imaging.py,iqa_bridge.py}` and new/extended tests for these modules. Do not edit reporting/API/workflow or Task 1/2 files. May create focused helper modules for run/trace/claim validation.

Call Task 1 core through `iqa_bridge` in the actual pipeline and persist `iqa_evaluation`. Bind to confirmed snapshot version and scanned bytes, including retry attempts. Remove equal-count high-confidence shortcut when ordinal/content evidence conflicts; present unresolved pairing as review-required. Make failed model run retryable with confirmed snapshot intact. Add strict local/provider response validation for actual known devices, dimensions, scores and evidence; capture exact prompt and ordered encoded JPEG trace. Remove diagnostic paths from prompt context.

Conservative semantic adjudication: support/opposition/unknown, never field-only agreement, certainty/confidence honored, objective claims require actual facts. Cross-scene counterexample ratio and audit comparability affect grading; heuristic/synthetic results are provisional. Stable claim IDs travel in findings and review payloads for Task 4. Add tests reproducing every previously reported failure using actual pipeline/database/provider transport where appropriate. Report `evidence/build/task-3.md`.


Verification interpreter: /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python
