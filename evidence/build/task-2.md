# Task 2 — IQA candidate selection, training admission, blind gold lifecycle

## Scope and contracts

Implemented selectively from Canary 61cbaed3 against the nested P1 IQA baseline. Historical tmqa.sequence@3.2 and @3.3 prompt text remains unchanged; the additional candidate preference contract is explicitly tmqa.sequence@3.4 / tmqa.sequence.v5. This work preserves frozen TMQA 0.2 behavior and does not claim the proposed full TMQA 0.3 laboratory specification is implemented.

Owned implementation paths: `packages/iqa/src/qwen_tmqa/{config,domain,review,server,cli,prompts,candidate_schema,lineage,pseudo_gt,training_data}.py`, `judges/openai_compatible.py`, and corresponding new tests. Small formatter-only cleanup occurred in the owned Judge files. No Front, packaging, Task 1 comparison implementation, or historical requirement documents were modified.

- Configured Judge IDs must be nonempty and unique, including disabled Judges. Candidate voting rejects repeated identities, whitespace aliases and blank IDs; counts distinct evidence, excludes unavailable/unconfirmed synthetic evidence, and preserves fatal/REJECT dominance.
- Candidate preferences require complete finite scores, evaluated IDs, a maximizing preferred candidate and runner-up, and arithmetically consistent baseline improvement. Runtime decoder rejects extra response/score/issue fields, bool/string/nonfinite/out-of-range numbers, and incomplete candidate evidence. P1 encoded-payload hashes and ordered traces remain in use.
- Canary lineage binds evaluated image source bytes and prompt versions. Explicit split CSV requires `scene_id,canonical_scene_id,group_id,split`; canonical or group derivatives cannot cross splits. CLI selection requires complete evaluation coverage and exact dataset/config manifest hashes.
- Model-only outputs are `candidate_suggestion` records with `training_weight=0`; no candidate selection path writes formal labels.

## Reusable training interface

`qwen_tmqa.training_data.TrainingCandidate` requires `scene_id,candidate_id,source_path,candidate_path,source_sha256,candidate_sha256,split,canonical_scene_id,group_id,eligible,judge_ids,evidence_mode,lineage`. Candidate IDs are generic version identifiers, independent of alpha levels. Optional fields include `synthetic=False,rejection_reasons=[]`; fixed record kind is `candidate_suggestion`, fixed training weight is zero.

`HumanConfirmation` requires `confirmation_id,reviewer_id,scene_id,candidate_id,canonical_scene_id,group_id,source_sha256,candidate_sha256,confirmed,confirmation_kind,synthetic,label_scope,confirmed_at`. Confirmation kind must be `human`; supported scopes are `tone_mapping_preference` and `global_rendering_preference`. `review_stage` is `blind` or `adjudication` (default adjudication): training adjudication is distinct from blind calibration gold.

`export_training_data(records, output_dir, *, confirmations, split_assignments)` revalidates models and validates the whole batch before creating output. It requires train split, complete explicit canonical/group metadata over `lineage.dataset_scene_ids`, at least two distinct Judge identities, eligible/nonfatal real same-source algorithm evidence, explicit real human confirmation, matching IDs/groups/source and candidate hashes, and unchanged actual file bytes. Device comparisons, unconfirmed suggestions, synthetic records, missing metadata, stale confirmations and unsupported scopes fail closed. Formal output is `training_manifest.jsonl` with `record_kind=human_confirmed_training_label`, the complete confirmation record and training weight one.

These JSON records are explicit operator-supplied human attestations. The exporter does not fabricate human confirmation or authenticate human identity; deployment authentication and human collection remain external responsibilities.

## CLI

- Existing `evaluate` optionally accepts `--source-root` and now writes `evaluation_run_manifest.json` alongside legacy outputs.
- `qwen-tmqa select-candidates --results evaluations.json --root LEVEL_ROOT --source-root SOURCE_ROOT --config config.yaml --run-manifest evaluation_run_manifest.json --splits splits.csv --output OUT` produces generic `candidate_suggestions.jsonl`, selection audit and summary, with zero training labels.
- `qwen-tmqa export-training --candidates candidate_suggestions.jsonl --confirmations human_confirmations.jsonl --splits splits.csv --output OUT` invokes the same library gate and cannot export unconfirmed or holdout examples.

Task 1 owns the authoritative representative audit sampler and CLI. Task 2 did not merge representative audit samples into the risk queue or dashboard; the risk queue default remains unchanged. Calibration weights remain scoped to the supplied selected blind-review set and retain the existing real sample sufficiency gate; there is no automatic fusion activation.

## Persistent blind-review lifecycle

The HTTP server appends and fsyncs submissions. Reveal appends/fsyncs `REVIEWS_PATH.exposure.jsonl` before returning model evidence. Exposure snapshots bind every pre-reveal submission for that reviewer/scene to a review-content hash. Before reveal, the latest received blind correction remains calibration gold; after reveal, blind resubmission returns 409, including after server restart. Loading history marks later/imported edits ineligible instead of rewriting or discarding history, so even a later forged blind flag cannot overwrite pre-reveal calibration gold. Revealing an older submission still preserves the latest blind revision. Duplicate review IDs are rejected and empty review/scene identities cannot poison the ledger.

Storage failures and malformed exposure logs fail closed with HTTP 500 and no disclosure or new gold. Engineering mode cannot create real blind gold. Existing reviewer route/queue restrictions and real/synthetic calibration isolation remain intact.

The server serializes mutations with its process-wide bound lock; this is a local single-server deployment, not a distributed multi-process identity/authentication service. Keep review JSONL and its exposure sidecar together when moving persisted review evidence.

## Verification

Test-first reproductions observed the original failures for post-reveal resubmission, missing exposure storage, duplicate submission IDs, imported exposed edits replacing calibration, malformed exposure HTTP status, empty identities, absent candidate/training modules and CLI, candidate decoder omissions, and whitespace Judge identity aliases. Positive and negative tests use real local HTTP servers, JSONL persistence and restart, actual filesystem obstructions, real image fixtures and actual CLI subprocesses. Provider transport is mocked without paid model calls.

New Task 2 tests total 56 across candidate contract/selection/CLI, training export and review exposure lifecycle. A scoped regression run passed 93 tests before final identity hardening. Owned-path Ruff check passed; owned-path Ruff formatter completed.

Final full nested IQA verification uses:

```sh
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH \
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest -q
```

Final result: **187 passed in 45.04s**, exit 0. An earlier in-progress full snapshot had four Task 1 comparison failures while that builder was editing and one legacy subprocess PATH failure; Task 1 repaired its cases and the verification command now supplies the specified development PATH. They were not silently omitted.

No real human collection, paid inference, mobile capture accuracy or training improvement is claimed. Test confirmation fixtures establish admission behavior, not empirical pseudo-GT quality. No commit, push or upload was performed by this builder.

## Independent review remediation — preliminary bbox probe

Critic 2's preliminary probe reproduced an INT-008 decoder gap: issue bounding boxes containing NaN, infinity, string or boolean coordinates were coerced/accepted while the Judge remained available. Added failing provider-transport probes, then strict runtime validation requiring null or a JSON list of exactly four finite numeric nonboolean coordinates. No coordinate normalization convention is inferred: both finite normalized and pixel-number boxes remain accepted. Historical prompt text and legacy tests were not changed. New regression probes exercise both new 3.4 and historical 3.3 decoder paths.

Targeted candidate-contract plus legacy Judge verification: **54 passed**, exit 0; scoped Ruff check passed. The full-suite 187-pass result above is the initial builder snapshot, before this remediation. Final structured finding: **T2-R1-01 (P2, INT-008 / legacy R-004)**. Builder remediation is recorded in `evidence/build/task-2-remediation-r1.md`; fresh full suite: **240 passed in 40.43s**, owned Ruff/format checks pass. Fresh independent re-review remains pending.
