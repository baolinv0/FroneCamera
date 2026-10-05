# TMQA P1 Remediation Design

Date: 2026-07-17
Branch: `fix/tmqa-p1-remediation-v0.2.0`
Frozen specification: TMQA `0.2.0`
Baseline commit: `46a9586e8aec39c3b9abc2660d8e59cb2000539d`

## 1. Decision

Implement the eight open P1 findings as bounded contract fixes inside the existing package boundaries. Do not change the frozen requirements, acceptance thresholds, public decision vocabulary, package version, or synthetic evidence claims.

The repair must leave two independently usable surfaces:

1. **Engineering dashboard** — full evaluations, model identities, prompts, raw responses, reliability and queue diagnostics.
2. **Reviewer dashboard** — reviewer-safe scene and image metadata only. It receives model evidence only from a server-side reveal endpoint after the review has been validated and durably appended.

## 2. Current state

The starting commit already provides image filtering, recursive discovery control, EXIF-normalized image loading, six-score judge validation, isolated judge failures, fatal hard gates, decision-disagreement routing, focused queue authorization, append-only JSONL storage, multi-reviewer calibration, and an engineering dashboard.

It does not satisfy eight frozen MUST conditions:

- P1-01 / R-010: complete model evidence is preloaded into the browser used for blind review.
- P1-02 / R-008: unanimous `REJECT` or `REGENERATE` can be overridden by a high mean score.
- P1-03 / R-007: prompt `3.2` was changed in place.
- P1-04 / R-001: `strict_complete` validates only discovered levels, not the required nine-level set.
- P1-05 / R-006: manifest records source-file hash but not the exact encoded payload hash.
- P1-06 / R-012, R-015: client `created_at` controls latest-review selection.
- P1-07 / R-013, R-014: synthetic and real reviews can be mixed implicitly.
- P1-08 / R-018: no exact-head CI evidence exists.

## 3. Considered approaches

### A. CSS/JavaScript strengthening

Keep one dashboard payload and make model objects harder to inspect before submit.

Rejected: this remains client-side hiding and cannot satisfy R-010.

### B. Separate reviewer page plus server reveal API — selected

Generate full engineering data and a separate reviewer-safe payload/page. The review server serves both, validates queue membership, writes a server timestamp, and returns the reveal object only after persistence succeeds.

Selected because it preserves the engineering dashboard and creates a testable security boundary without introducing authentication or production deployment scope.

### C. Fully server-rendered application

Move all dashboard data behind dynamic API endpoints.

Rejected for this release: it is a broader architectural rewrite and is unnecessary to close the frozen requirements.

## 4. Architecture

### 4.1 Dataset contract

`DatasetConfig` gains a versioned `expected_levels` list whose default is the canonical nine levels. Validation requires:

- unique level names;
- every level parseable;
- `baseline_level` included in `expected_levels`;
- in strict mode, the root contains every expected directory;
- in strict mode, unknown parseable level directories are rejected unless an explicit future policy is introduced;
- every baseline scene has one image in every expected directory.

Non-strict mode may evaluate the available parseable levels but must still require the baseline.

### 4.2 Prompt immutability

Restore `tmqa.sequence@3.2` to its historical text and schema semantics. Add `tmqa.sequence@3.3` with the six-score contract and a new output schema version `tmqa.sequence.v4`. Change new/default judge configuration to `3.3`. Preserve `3.1`, `3.2`, and their diffs.

### 4.3 Exact image payload lineage

Introduce one deterministic encoder that:

1. opens the source image;
2. applies EXIF transpose;
3. converts to RGB;
4. resizes to the requested sent dimensions;
5. encodes JPEG at the configured quality;
6. returns the exact bytes, MIME type, dimensions, encoding parameters and SHA-256.

`build_input_manifest()` uses this encoder to record `source_sha256`, `payload_sha256`, `payload_mime`, and encoding parameters. `OpenAICompatibleJudge` uses the same encoder and verifies that the generated payload hash equals the trace before sending it.

### 4.4 Decision aggregation

Decision precedence after fatal and availability checks:

1. fatal evidence -> `REJECT`;
2. all configured judges unavailable -> `REVIEW`;
3. conflicting available decisions -> `REVIEW`;
4. unanimous available `REJECT` -> `REJECT`;
5. unanimous available `REGENERATE` -> `REGENERATE`;
6. unanimous available `REVIEW` -> `REVIEW`;
7. otherwise apply score-gap and score thresholds.

Unanimous `KEEP` does not bypass objective score/risk thresholds.

### 4.5 Review timestamp integrity

`HumanReview` stores optional client metadata separately from authoritative `received_at`. Server input forbids or ignores an attempted authoritative timestamp and always creates a UTC `received_at` immediately before append. Calibration selects the latest record per `(scene_id, reviewer_id)` using parsed, timezone-aware `received_at`, with deterministic JSONL-order tie breaking.

Synthetic reviews produced by the CLI also receive a trusted process-generated `received_at`.

### 4.6 Synthetic/real calibration isolation

Calibration policy defaults to `real_only`:

- all-real input: accepted;
- all-synthetic input: rejected unless `--allow-synthetic` is explicit;
- mixed input: rejected unless an explicit `--review-type real|synthetic` selector filters to one type;
- `--allow-synthetic` permits synthetic-only experimental output, never an implicit mixed set.

The output records selected type, selected count, ignored count, generation time, and an experimental/non-production warning for synthetic evidence.

The verified synthetic experiment passes `--allow-synthetic` explicitly.

### 4.7 True blind-review data flow

`generate_dashboard()` writes:

- `data/engineering_evaluations.json`: complete engineering payload;
- `data/reviewer_payload.json`: queue-only reviewer-safe payload containing scene ID, copied image URLs, manifest role/alpha/level/index, and no model/system decision, scores, prompts, rationale or complete evaluation object;
- `data/review_queue.json`: authorized scene IDs;
- `data/reveal_payload.json`: server-only reveal source containing model comparison and prompt data for queue scenes.

It generates:

- `index.html`: engineering dashboard; no blind-review claim or review form;
- `review.html`: reviewer client embedding only reviewer-safe data.

The server blocks direct static access to `reveal_payload.json` and engineering data from the reviewer route, exposes `GET /api/reviewer-payload`, and exposes `GET /api/reveal?scene_id=...&review_id=...` only when the append-only review file contains that exact persisted review ID and scene ID. `POST /api/reviews` returns a one-time reveal URL only after append succeeds.

The reviewer page fetches reveal data after a successful POST. A failed append, invalid review, queue violation or unknown review ID reveals nothing.

### 4.8 Requirements audit

Replace the ten legacy string-presence checks with a nineteen-requirement result map. Security-relevant checks inspect separated artifacts and forbidden fields rather than only HTML markers. The audit remains a deterministic artifact check; behavioral server and browser tests remain the authoritative evidence for R-010/R-015.

### 4.9 CI and evidence

The PR workflow must execute Python 3.10 and 3.12 jobs on the exact branch head. Required checks include compile, Ruff, full pytest, sdist and wheel build, isolated wheel import, CLI contract, synthetic end-to-end run, JavaScript syntax check, secret scan, and nineteen-requirement audit. The PR is not merge-authorized until all required checks report success for the exact head SHA.

## 5. Error handling and security

- Unknown or duplicate expected levels fail configuration validation.
- Payload-trace mismatches fail the affected real judge closed as unavailable evidence.
- Review JSON, content length, queue membership, blind flag and reviewer ID are validated before append.
- `received_at` is server-generated and cannot be selected by the client.
- Static requests for reveal data return 404/403.
- Synthetic calibration requires explicit operator intent.
- API keys and authorization headers are never serialized.

## 6. Test strategy

Tests are added before implementation for each finding:

- missing/unknown/duplicate nine-level directories;
- immutable historical prompt hashes and new 3.3/v4 contract;
- exact payload hash equals sent base64 bytes, including EXIF and resize;
- unanimous REJECT and REGENERATE precedence;
- server timestamp overrides/forbids client timestamp and calibration uses `received_at`;
- real-only, synthetic-only and mixed calibration policies;
- reviewer HTML/data forbidden-field scan;
- reveal endpoint unavailable before persistence and available only after matching review persistence;
- storage failure never returns reveal data;
- nineteen-requirement audit negative probes;
- exact-head CI workflow jobs for Python 3.10 and 3.12.

## 7. Bounded implementation scope

Allowed files are limited to `src/qwen_tmqa/**`, `tests/**`, `configs/**`, `scripts/**`, `.github/workflows/**`, documentation and packaging metadata where required by the frozen specification.

No model training, real-model accuracy claim, production authentication, RAW/HDR source model, automatic regeneration, threshold relaxation, test deletion/skip/xfail, or merge to `main` is included.

## 8. Completion gate

Builder completion only means code and evidence are ready for independent review. Final completion requires:

- 19/19 MUST requirements pass;
- exact branch-head CI succeeds;
- unresolved P0/P1 equals zero;
- Fresh Critic returns `PASS_TO_EVALUATOR`;
- Fresh Evaluator returns `PASS`;
- human merge authorization remains separate.
