# Evaluation evidence upgrade implementation plan

**Goal:** Implement the reviewed input, adjudication and reporting corrections in the existing EvaluationPipeline.
**Architecture:** Extend the runtime contracts and append versioned records. Keep the existing API/worker/workflow entrypoints; reports share ReportPayload. No database replacement, silent historical overwrite or paid inference.
**Spec:** REPOSITORY_REVIEW.md, EVALUATION_AND_REPORT_PLAN.md, IMPLEMENTATION_CHECKLIST.md in this directory.
**Execution:** Native implementation, regression tests first, then independent whole-branch review.

## Review of the proposal
The five P0 risks are actionable in this pipeline. Two qualifications apply: ICC conversion standardizes a rendering, it does not calibrate camera color accuracy; multiple rounds of the same adapter are dependent evidence. A/B/C are legacy rule labels, not calibrated probabilities. Unverified scene independence and unknown attributes must remain unknown.

## Global constraints
- Preserve user-confirmed same-scene pairing, even when pixels/framing differ.
- Preserve originals and old records; new contract version 0.2.
- SDR frame zero only; record unknown color/HDR information and restrict related claims.
- Human calibration, Camera0717 replay and editing experiments require their actual data; never mark them complete using fixtures.

## Review focus
- Corrupt or unsupported ICC/containers must not authorize color claims.
- Unknown/missing metrics and duplicated scene families must not inflate support.
- Same dimension but opposite region/direction/reference must not count as agreement.
- Same text on two devices/versions must not share external evidence.
- Quick finalization must retain the model-assisted/unvalidated label.

### Task 1: Input evidence (T01-T03, T08-T09)
Files: imaging.py, vlm.py, dataset.py; tests/test_evaluation_inputs.py.
Interfaces: load_analysis_rgb(path) -> (RGB array, rendering record); analyze_image(path, artifact_dir, face_annotations=None) retains legacy fields and adds faces/rendering/evidence.
- [x] Write/run failing tests for ICC, all EXIF orientations, linear Y vs encoded Y prime, multi-frame selection, multi-face crop fidelity and scene challenge independence.
- [x] Implement explicit color conversion, all-face evidence and normalized previews.
- [x] Run input regression tests and commit.

### Task 2: Claim semantics and aggregation (T04-T07, T10)
Files: models.py, adjudication.py, strategy.py, pipeline.py, evidence.py; tests/test_evaluation_claims.py.
Interfaces: ModelObservation adds region/attribute/direction/comparison/conditions; ClaimCandidate accepts unknown objective support, evidence and scene families; AdjudicatedClaim preserves complete denominator.
- [x] Write/run failing opposite-direction, 3-support/21-counterexample, repeated-family, unknown-support and device-bound external tests.
- [x] Implement structured review relations, measurement applicability, family aggregation and immutable claim identity; wire actual pipeline.
- [x] Run regressions and commit.

### Task 3: Annotation and provenance (T03, T08-T09, P1-04)
Files: annotations.py, api.py, pipeline.py, attribution.py; tests/test_evaluation_annotations.py.
Interfaces: append-only annotations validated against project/group/device; latest annotation consumed by pipeline. Family/challenge/manual face data retain author and revision.
- [x] Write/run failing invalid-owner, missing-EXIF and annotation-consumption tests.
- [x] Implement explicit scene/face annotations and claim-scoped competing hypotheses.
- [x] Run regressions and commit.

### Task 4: Shared report contract (T11-T13)
Files: reporting.py, docx_reporting.py, pipeline.py, report_model.py; tests/test_evaluation_reports.py.
Interfaces: ReportPayload 0.2 includes all_claims, coverage, challenge matrix, review provenance; HTML/JSON/DOCX consume the same claims.
- [x] Write/run failing nonempty field, anchor/image, counterexample and quick-finalization label tests.
- [x] Implement conclusion-led HTML, traceable device profiles, complete appendix and field-correct DOCX.
- [x] Run offline pipeline/export tests; render/inspect report artifacts and commit.

### Task 5: Validation and delivery
Files: IMPLEMENTATION_RESULTS.md, docs/implementation-status.md, README.md.
- [x] Run full pytest, ruff lint/format, mypy; record actual results.
- [x] Obtain independent review and fix important findings with regression tests.
- [x] Record checklist item states; keep T14-T25 data/human/experiment gates explicitly pending.
- [ ] Push implementation branch and create PR against agent/openai-auto-pairing-final.
