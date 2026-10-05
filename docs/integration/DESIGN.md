# IQA standalone and FroneCamera integration

## Authorization and scope

The current human instruction authorizes implementation of the preceding comparison report's recommendations, parallel independent Builders, independent review/remediation, and upload to baolinv0/FroneCamera. This task records that execution authorization; it does not pretend that an agent approved or replaced the separately frozen TMQA 0.2 specification. Its existing contracts, historical prompts, tests, and compatibility checks remain regression requirements. The unimplemented 0.3 laboratory/DXOMARK proposal is not silently made a release prerequisite or claimed complete.

The task has two independently usable deliveries in the same repository: an installable `packages/iqa` Python project and standalone branch; and a FroneCamera integration branch that calls this project from the real evaluation pipeline. FroneCamera source is 2423af87; IQA core is P1 1d73bc0f. Canary and source-aware features are ported selectively, preserving P1 corrections.

## Product behavior

- **IQA algorithm mode:** evaluate a common source, baseline, and arbitrary candidates; compare version folders; report measurements, regressions, ROI/person results, trace, and uncertainty. No alpha sequence is needed for the new path. The existing nine-alpha CLI remains available.
- **IQA device mode:** report image/ROI observations and comparability for different captures. It never treats cross-phone winners as same-input training labels or as proof of algorithm/hardware causation.
- **Training selection:** model preferences generate candidates. Train export requires train split, closed canonical group metadata, distinct Judge identities, no fatal defect, explicit real human confirmation bound to the selected image, and a usable label scope. Synthetic evidence remains demonstration-only.
- **FroneCamera:** retain pairing, projects, reports, real model adapters, and professional/quick modes. Use the shared IQA core in actual scene evaluation; persist its result. Semantic opposition, missing evidence, unavailable dimensions and heuristic adapters cannot create high-grade consensus. Human reject applies consistently to every published representation.

## Interfaces

Package `qwen_tmqa.comparison_models` provides strict Pydantic models:

- `ComparisonAsset`: `id`, `path`, `encoding` (`srgb` or `linear`). IDs are nonempty/unique. Loaded pixel precision and source bytes are traced.
- `ComparisonROI`: `id`, `kind`, `bbox` (`x,y,w,h`), optional `person_id` and `asset_id`. Algorithm masks belong to baseline coordinates; device masks require explicit asset correspondence.
- `ComparisonRequest`: `scene_id`, `mode` (`algorithm` or `device`), `baseline`, `candidates`, optional `source`, `rois`, `split` (`train,audit,benchmark,holdout`), nonempty `group_id`.
- `ComparisonResult`: `scene_id`, `mode`, `baseline_id`, `assets`, `comparisons`, `warnings`, `input_trace`, schema version. Asset assessments expose `objective` measurements; candidate comparisons expose `candidate_id`, `baseline_id`, `status`, `relative_metrics`, `fatal_reasons`, `review_reasons`.

`qwen_tmqa.comparison.evaluate_comparison(request: ComparisonRequest) -> ComparisonResult` is a pure library entry except for reading declared image assets. It has no HTTP or database dependency. It returns finite values and explicit missing/invalid evidence states. `iqa-compare` is its independent CLI.

Front bridge `portrait_eval.iqa_bridge.evaluate_scene(scene_id, image_paths, metrics)` calls this function in device mode and returns serializable IQA evidence. The actual pipeline stores an `iqa_evaluation` analysis for each scene. No undeclared absolute quality score is fabricated from missing chart/reference/human evidence.

## Evaluation rules

Keep absolute quality, relative measurements, controllability, style preference, and observability separate. Support all ten Front dimension IDs with applicable evidence or explicit unobservability; single-scene adaptability is not applicable. Use fixed baseline ROI for algorithm comparisons; handle multiple persons independently. Preserve 16-bit/float source and one orientation convention. Alignment is a same-shape translation diagnostic with overlap/validity checks, not a physical HDR truth claim. Sparse/constant/NaN/Inf inputs cannot silently become perfect fidelity.

Replace field-only agreement with explicit support/opposition/unknown semantics and evidence references. Exact matching observation text can serve as conservative support; different unstructured statements require review, not presumed agreement. Cross-scene confidence must depend on comparable support AND counterexamples, not hardcoded agreement/objective/validity. Heuristic/synthetic output is marked provisional in data and every report.

Bind evaluation to a confirmed pairing snapshot and scan bytes; source change fails/requires re-scan. Keep retries possible after a transient adapter failure and preserve error provenance. Anonymize model inputs and exported paths recursively, record actual ordered encoded image/prompt hashes. Publish from a unified resolved projection with stable claim IDs; legacy nested review payloads remain supported.

## Requirements and acceptance

| ID | Requirement | Verification |
|---|---|---|
| INT-001 | IQA installs/runs without Front API/database/frontend | isolated IQA wheel install and CLI smoke |
| INT-002 | Same-input B/C and arbitrary candidate version comparisons | real image fixtures, multiple sizes/encodings, regression CSV/JSON |
| INT-003 | Precision/orientation/alignment and per-person ROI boundaries | uint16/float/EXIF, degenerate/constant/mismatched inputs, ROI validity tests |
| INT-004 | Ten dimensions expose measured/missing/applicability evidence | complete ID coverage, missing skin/face/reference, multi-face tests |
| INT-005 | PGT judges/split/lineage/real confirmation/scope gates | duplicate identities, holdout, missing canonical, synthetic, stale confirmation, positive train export |
| INT-006 | Blind exposure lifecycle cannot contaminate gold | submit/revise/reveal/resubmit HTTP tests and calibration regression |
| INT-007 | Representative audit sampling is separate from risk queue | deterministic stratified clean+risk sample tests and evidence labels |
| INT-008 | Runtime response decoder is strict and actual input trace exists | unknown devices/dimensions, 999/NaN/Inf/extra fields, captured request hash tests |
| INT-009 | Semantic conflict and cross-scene counterexamples reduce claims | opposite same-field, zero confidence, 3 supports/20 counters, confounded scenes |
| INT-010 | Pairing correctness, confirmed bytes/snapshot and retry correctness | 1/2 vs 1/9, UI warnings, changed bytes, repeated API failure/success |
| INT-011 | Rejected claims vanish from all final report projections | actual API+renderer JSON/HTML/DOCX, nested review payload and device-specific ID tests |
| INT-012 | Quick/professional and heuristic provenance truthfully disclosed | open reviews, synthetic/heuristic flags, privacy export path scans |
| INT-013 | Actual Front pipeline invokes and stores shared IQA evidence | FastAPI/SQLite/image integration tests and end-to-end canary |
| INT-014 | Existing TMQA/Front regressions and build/install/CLI/UI remain usable | full legacy tests, lint/format/type, package builds, Node tests/build |
| INT-015 | Independent reviewers, remediation, fresh evaluator and GitHub delivery | structured findings/closure, final evidence, two new branch refs; no main merge |

## Evidence limits

Functional probes, real image fixtures, actual local API/database execution and mocked provider transport establish software behavior. No paid model call, semantic checkpoint download, real mobile capture accuracy or TM training improvement is claimed. Those remain real-model/Gold/training experiments, not replaced by relabelled mocks.
