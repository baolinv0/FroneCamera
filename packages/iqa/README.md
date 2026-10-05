# Qwen-TMQA v0.2

Qwen-TMQA is a multi-model, traceable quality-evaluation and human-calibration system for controllable tone-mapping / RGB-enhancement datasets.

Its first purpose is **evaluation observability**: show the current image, per-alpha objective metrics, stage scores, every model's scores, the exact prompt sent to each model, the input-image order and exact encoded-payload lineage, raw model response, parsed JSON, final decision, and dataset statistics.

Its second purpose is **evaluator improvement**: send only model-disagreement, low-tail, unavailable-judge, and high-risk scenes to a server-enforced blind review flow, then calculate model-to-human error and reliability weights from an explicitly selected evidence type.

## Implemed capabilities

- Strict canonical nine-level alpha-folder discovery (`a_m100 ... a_p100`).
- Deterministic brightness, clipping, shadow, contrast, color-drift, edge-similarity, and sequence-control metrics.
- Any number of independent judge adapters through `judges: [...]` configuration.
- OpenAI-compatible VLM adapter with interleaved, explicitly labelled images.
- Exact source SHA-256 and sent JPEG payload SHA-256, MIME type, dimensions, and encoding parameters in every prompt trace.
- Immutable historical prompt `tmqa.sequence@3.2` plus six-score prompt `3.3` and output schema `tmqa.sequence.v4`.
- Deterministic synthetic judges for tests and reproducible demonstrations; all synthetic results are visibly tagged.
- Full model result trace: model/version/role, rendered prompt, template, variables, prompt hash, input manifest, inference parameters, raw response, parsed JSON, latency, confidence, issues, scores, and decision.
- Engineering dashboard with dataset overview, scene filmstrip, difference map, stage trace, model-by-dimension matrix, Prompt Inspector, focused queue, and reliability views.
- Separate reviewer-safe `review.html` payload that contains no model identity, prompt, score, rationale, model decision, system decision, stage trace, or complete evaluation object.
- Reviewer-only server mode by default: engineering HTML and sensitive JSON routes are denied, including URL-encoded and normalized aliases.
- Server-side post-persistence reveal API; reveal evidence is returned only for a matching persisted review.
- Append-only JSONL review storage with server-generated authoritative `received_at` timestamps.
- Explicit real/synthetic calibration policy; synthetic evidence requires opt-in and implicit mixed evidence is rejected.
- Human-model score deltas, model decision agreement, per-dimension MAE, pairwise model gaps, and normalized inverse-error fusion weights.
- Automated R-001 through R-019 requirements audit.

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
```

## Reproducible synthetic experiment

The default configuration uses four deterministic synthetic judge profiles. They test the full data flow and do not establish real VLM accuracy.

```bash
bash scripts/run_verified_experiment.sh runs/verified_experiment
```

This creates:

```text
runs/verified_experiment/
├── dataset/
├── results/
│   ├── evaluations.json
│   ├── summary.json
│   ├── errors.json
│   └── scenes.csv
├── reviews.jsonl
├── calibration.json
├── deterministic_rerun_comparison.json
└── dashboard/
    ├── index.html
    ├── review.html
    ├── assets/images/
    └── data/
        ├── engineering_evaluations.json
        ├── reviewer_payload.json
        ├── reveal_payload.json
        └── review_queue.json
```

The command above performs V-007 only and does not require CI metadata. It evaluates the
fixed synthetic experiment twice and records whether scene decisions, the focused review
queue, and stable summary fields match. The formal R-001 through R-019 exact-head audit is
performed separately by the aggregate GitHub Actions job.

The engineering dashboard is a self-contained artifact and can be opened directly from:

```text
runs/verified_experiment/dashboard/index.html
```

Start the reviewer-only server with persistence:

```bash
qwen-tmqa serve \
  --dashboard runs/verified_experiment/dashboard \
  --reviews runs/verified_experiment/human_reviews.jsonl \
  --port 8765
```

Then open `http://127.0.0.1:8765/`. The default server root is the reviewer-safe page. It rejects `index.html`, engineering-evaluation JSON, reveal JSON, and encoded or normalized aliases of those routes.

## Real model deployment

Edit `configs/real_models.yaml` so each endpoint, model name, checkpoint version, API key, prompt version, and output-schema contract match the deployed OpenAI-compatible endpoint.

```bash
qwen-tmqa evaluate \
  --root /path/to/alpha_dataset \
  --output runs/real_models \
  --config configs/real_models.yaml

qwen-tmqa visualize \
  --results runs/real_models/evaluations.json \
  --output runs/real_dashboard \
  --config configs/real_models.yaml
```

The project does not silently treat synthetic judge output as real evidence. `synthetic: true` is persisted per result. Production calibration defaults to real reviews; synthetic-only calibration requires `--allow-synthetic --review-type synthetic`.

## CLI

```text
qwen-tmqa validate-config
qwen-tmqa make-example
qwen-tmqa evaluate
qwen-tmqa visualize
qwen-tmqa serve
qwen-tmqa simulate-human
qwen-tmqa calibrate
qwen-tmqa audit
```

`simulate-human` is a deterministic experiment utility. It marks every generated review as synthetic and must not be used as a human Gold Set.

## Human review policy

The blind-review page can only select scenes in `review_queue.json`. The server independently enforces the same queue. Review reasons include:

- model score disagreement;
- model decision disagrement;
- unanimous non-KEEP model decision;
- low-tail score;
- hard/fatal risk;
- unavailable judge;
- configured high-risk evidence.

Clean scenes remain available to engineering users but are not offered to the blind reviewer.

## Verification

The exact pull-request head is verified on Python 3.10 and 3.12. Required CI steps are:

- compileall;
- Ruff;
- full pytest; the authoritative test count is recorded in the exact-head CI artifact;
- sdist and Wheel build plus SHA-256 output;
- isolated Wheel install and import;
- all eight CLI surfaces;
- deterministic synthetic end-to-end experiment;
- JavaScript syntax checks for engineering and reviewer pages;
- secret-leakage scan;
- R-001 through R-019 audit with 19/19 PASS;
- recorded branch-head SHA equal to the checked-out commit.

The GitHub Actions run and its exact-head artifacts are the authoritative execution evidence. Static documentation is not a substitute for a green run on the current PR head.

## Current evidence boundary

The included experiment uses synthetic judge profiles because no Qwen3-VL, InternVL, Ovis, or MiniCPM endpoints were used in the verification environment. The real adapter and request schema are implemented and mock-tested, but real-model quality conclusions require deployed endpoints and blind expert reviews. The synthetic calibration weights are experimental and are not production recommendations. The system is not a trained Reward Model and does not claim physical HDR/RAW source-aware quality evaluation.
# Standalone IQA integration

This project installs independently of the Front API, database and frontend:

```bash
python -m pip install ./packages/iqa
iqa-compare --help
qwen-tmqa --help
```

The new `iqa-compare` entry point compares arbitrary algorithm outputs, preserving source precision, orientation, per-person ROIs and byte traces. `qwen-tmqa` retains its nine-alpha workflow and adds candidate selection and separate human-confirmed training export. See [the complete integration guide](../../docs/integration/QUICKSTART.md) for manifests and commands. Missing reference, human or calibration evidence remains explicit; proxy measurements do not become an absolute quality score.

Candidate selection requires the result bindings written by the current `evaluate` command. Re-evaluate historical runs whose manifests lack these bindings. See [the standalone P1 fixes and test instructions](../../docs/integration/P1_FIXES_20261005.md) for the static review boundary and result/configuration pairing checks.
