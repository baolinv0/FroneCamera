# Task 3 remediation, round 3

Addresses the sole residual **T3-R2-02 — P2 — INT-004/008** in the final independent record `evidence/review/task-3-r3.md`, including the R1-04 context residual. R2-01 precision/orientation changes and previous opposition, uncertainty, HEIC and invalid-core gates remain unchanged. Fresh independent closure is pending.

## Fail-before evidence

Two additive tests failed before the implementation change (`2 failed`, 1.46 s). The actual SQLite workflow uses two 96×96 PNG device folders named `person_count` and `face_luminance_spread`, with two detected face boxes per capture. Persisted shared-core `multi_face_consistency` facts correctly contain `{person_count: 2, face_luminance_spread: 0}`; actual primary metric-validation context instead contains `{A: 2, B: 0}`. A separate future-schema control also establishes that the previous finite vocabulary changes unknown nested scientific fields and fails to anonymize a device-map key colliding with `whole`.

## Change

Removed the finite scientific dictionary-key allowlist. Dictionary keys are now preserved as schema by default. Only explicitly declared request identity maps (`metrics`, `devices`, `scores`, `mapping`) replace their device-key component; their scientific payloads retain all keys recursively. Scientific subtrees (`facts`, `objective`, `dimensions`, whole/region/per-face measurements) cannot accidentally become identity maps because a nested field happens to match one of those names. Composite ROI fact keys replace only a declared device prefix; reference namespaces, scene IDs and metric/dimension suffixes are preserved. Explicit identity field values remain anonymized, including aliases colliding with protocol vocabulary. Serialized JSON uses this same recursive traversal. Location/diagnostic filtering and embedded path redaction remain active.

No additional scientific keys were added to a global allowlist. Only `model_validation.py` and the new uniquely named regression module changed in this round. No historical tests were changed or weakened; no publication modules, commits or pushes were made.

## Regression coverage

`tests/test_task3_remediation_r3.py` contains two additive tests covering:

- Actual SQLite scan, pairing confirmation, shared-core evaluation and all primary/reviewer validation contexts with the two problematic aliases and two faces.
- Every current fact leaf and objective key discovered dynamically from that actual core result, each used as a device alias. Both direct and serialized contexts retain exact scientific schema/fact equality.
- Future nested fact, objective, metric, region and dimension names absent from the prior vocabulary; a nested `metrics` fact cannot relabel scientific keys. A device actually named `whole` is anonymized in device-map keys and explicit identity values while structural `whole` remains intact.
- Both actual HTTP provider encoders, final prompt context equality, declared asset-reference catalog, embedded UNC redaction and hash equality for the exact prompt sent.

## Verification

- Fail-before actual/future schema regressions: **2 failed** (1.46 s).
- All Task 3 runtime/R1/R2/R3 tests after the implementation change: **72 passed**, 1 existing deprecation warning (10.62 s), before the final provider-payload assertions.
- Final new R3 tests: **2 passed** (1.07 s).
- Final full Front suite: **187 passed**, 5 existing deprecation warnings (35.14 s).
- Global mypy: **36 source files clean**. Ruff and format checks for changed source/new tests: clean.

The newly added transport test initially omitted the adapter-added `allowed_evidence_refs` field from its expected context. Its expectation was corrected to explicitly verify the declared asset catalog, retaining full scientific payload equality. No production changes were needed for that test-contract correction. No changes followed the final full-suite pass.

Interpreter: `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python`.

## Limits

Verification establishes local schema integrity/privacy through actual SQLite, core image evaluation and captured HTTP payloads with mocked model responses. It does not establish paid-model accuracy, human Gold, physical hardware causation or training benefit. Fresh independent Critic closure is required for acceptance.
