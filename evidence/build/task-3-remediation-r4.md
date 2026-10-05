# Task 3 remediation, round 4

Addresses the sole residual **T3-R2-02 — P2 — INT-004/008** in final independent record `evidence/review/task-3-r4.md`. Prior scientific-fact, pixel precision/orientation, opposition, uncertainty, HEIC and invalid-core changes remain unchanged. Fresh independent closure is pending.

## Fail-before

New `tests/test_task3_remediation_r4.py` reproduced **3 failures** (3.48 s) before the production change. Both actual SQLite pipelines use one device display alias for every canonical dimension and both compatibility dimensions (12 devices), actual PNG files, pairing confirmation and real HTTP adapter encoding with mocked responses. The models return all dimension scores as integer 68. Persisted scores are correct, but the subsequent metric-validation and reviewer-challenge prompt score keys become anonymous camera codes A–L. A direct/serialized non-provider model-result control fails in the same way.

## Contract audit and change

`ModelEvaluationResult.scores` is `dict[str, StrictInt]`, with a validator requiring known dimension keys and 0–100 integer values. It is always scientific schema, never a device-indexed map. The sanitizer now marks the complete scores subtree scientific and removes its identity-map classification.

The other previously listed identity-map entries were checked against their actual boundaries and narrowed:

- `metrics`: the **outer evaluation request** dictionary of device code → image metric record. Nested model/backend dictionaries with this name are schema; numeric dictionaries do not satisfy the declared record-map shape.
- `mapping`: the **outer repository anonymous-map payload** dictionary of device ID → anonymous code string. Nested model/backend fields and numeric maps remain schema.
- `devices`: the supported dictionary of device ID → device record, guarded by the record-map shape. Actual pairing device lists use explicit identity fields; scientific subtrees cannot be reclassified by this field name.

No scientific dimension alias was special-cased. Schema keys, composite fact references, explicit identity-value anonymization and location removal continue to use the existing recursive traversal.

## Additive coverage

Three new cases cover:

- Both actual HTTP providers through SQLite scan/confirm/evaluation, with every ten-dimension ID plus `global_exposure` and `highlight_retention` used as a device display alias.
- Exact scores=68 preservation in visual observation, primary result and independent reviewer result carried into validation/challenge prompts; exact score preservation in all four persisted result kinds.
- Stored prompt hashes match actual captured prompts; capture folder paths and the configured API credential are absent from those prompts.
- Direct and serialized typed model results, standalone scores and untyped backend `mapping`/`metrics`/`devices` numeric fields retain the scientific keys. Actual device-map keys still anonymize while their metric suffixes remain intact; diagnostic paths remain omitted.

No historical tests were edited or weakened. Only the sanitizer and this uniquely named regression module changed; no publication modules, commits or pushes were touched.

## Verification

- Fail-before: **3 failed** (3.48 s).
- Current R3/R4 cases: **5 passed** (4.43 s).
- Global mypy: **36 source files clean**.
- Ruff and formatting on changed source/new tests: clean.
- Final full Front suite: **190 passed**, 5 existing deprecation warnings (47.80 s). No code changes followed this pass.

The new persistence check initially referenced a nonexistent generic result category; it was corrected to verify the four actual categories (`primary_vlm_visual`, `primary_vlm`, `reviewer_independent`, `reviewer_challenge`). Score and prompt assertions were retained. No production change was needed for that test-contract correction.

Interpreter: `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python`.

## Limits

This establishes local schema integrity/privacy through actual SQLite/image evaluation and captured HTTP requests with mocked model responses. It does not establish paid-model accuracy, human Gold, physical hardware causation or training benefit. Independent fresh Critic closure is required for acceptance.
