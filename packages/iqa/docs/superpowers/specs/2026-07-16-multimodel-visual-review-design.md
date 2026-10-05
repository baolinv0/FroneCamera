# Qwen-TMQA Multi-Model Visual Review Design

## Goal

Build a fully browsable, runnable v0.2 quality-evaluation system that preserves deterministic tone/control analysis and adds multi-model scoring visibility, prompt traceability, focused human blind review, disagreement prioritization, and reliability calibration.

## Product focus

The first purpose is evaluation observability: show dataset statistics, per-scene images, stage scores, per-alpha results, per-model scores, and the exact evidence that produced each decision. The second purpose is evaluator improvement: route only disagreement, tail, and high-risk cases to human blind review, then use those labels to measure and calibrate model reliability.

## Required behaviors

1. Every model result records model ID, role, checkpoint/version, prompt ID/version/hash, rendered prompt, input manifest, inference parameters, raw response, parsed result, latency, confidence, scores, decision, issues, and regions.
2. The Prompt Inspector displays the rendered prompt, template, variables, image order, role, alpha, level, file hash, resize dimensions, raw response, parsed JSON, and prompt-version diff.
3. In engineering/debug mode, prompts and model scores are visible immediately.
4. In human Gold Label mode, model identity, prompt, scores, rationales, and system decision remain hidden until the reviewer submits an independent decision.
5. Human review is limited to disagreement, low-tail, hard-risk, branch-conflict, or OOD-like cases. Random audit sampling may be configured separately.
6. The dashboard contains dataset overview, scene filmstrip, stage trace, model-by-dimension comparison, Prompt Inspector, blind-review queue, failure/action queue, and reliability report.
7. The dashboard can run as static HTML for inspection and through a local HTTP service that stores human labels as JSONL.
8. Reliability calibration reports overall and per-dimension MAE, decision agreement, pairwise model gaps, and learned inverse-error fusion weights. It must never imply that synthetic demo labels are real human evidence.
9. Multi-image prompts explicitly label every image with index, role, alpha, level, path hash, and sent dimensions.
10. The core must run without real VLM endpoints. OpenAI-compatible endpoints are supported; deterministic mock judges are used only for reproducible integration experiments and are visibly tagged as synthetic.

## Architecture

- `domain.py`: typed records for scenes, prompts, model outputs, human reviews, and dataset summaries.
- `dataset.py` and `metrics.py`: alpha-folder discovery and deterministic image/sequence metrics.
- `prompts.py`: versioned prompt registry, explicit image manifests, rendering, hashing, and diffs.
- `judges/`: common judge protocol, OpenAI-compatible adapter, deterministic mock adapters.
- `evaluation.py`: per-scene deterministic stages, multi-judge execution, disagreement and final decision.
- `review.py`: queue selection, blind-review validation, JSONL storage, and calibration metrics.
- `visualization.py`: self-contained static dashboard assets and JSON bundles.
- `server.py`: local HTTP server with safe static serving and review submission.
- `cli.py`: validate, make-example, evaluate, visualize, serve, simulate-human, and calibrate commands.

## Data flow

Dataset -> deterministic metrics -> versioned prompt rendering -> independent model judges -> stage/model result JSON -> review-priority calculation -> dashboard -> blind human review -> review JSONL -> reliability calibration.

## Error handling

- A scene error is recorded and does not abort the batch.
- A judge error produces an unavailable model record with the prompt trace preserved.
- Invalid model JSON is retried once by the OpenAI-compatible adapter.
- Review submissions are schema-validated and append-only.
- Dashboard generation fails loudly when required result files are missing.

## Verification contract

- Python compile succeeds.
- Unit and integration tests pass.
- Package builds and installs in a clean virtual environment.
- End-to-end demo creates a nine-level dataset, evaluates it with four synthetic judges, generates dashboard files, stores synthetic blind reviews, and produces calibration metrics.
- An automated requirements matrix proves all ten required behaviors above are represented by code and tests.
- Repository contains browsable source, tests, configs, docs, and verified experiment output; binary-only source delivery is not accepted.
