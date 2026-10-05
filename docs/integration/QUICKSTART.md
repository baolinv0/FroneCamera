# Two independently usable workflows

## IQA only

Python 3.10+; no Front server, database or web application is needed:

```bash
git clone --branch feature/iqa-standalone-20261005 https://github.com/baolinv0/FroneCamera.git FroneCamera-iqa
cd FroneCamera-iqa
python -m venv .venv-iqa
source .venv-iqa/bin/activate
pip install './packages/iqa'
iqa-compare --help
qwen-tmqa --help
```

`qwen-tmqa` retains nine-alpha evaluation/trace/visualization. `iqa-compare` compares ordinary baseline/candidate outputs independently of the alpha sequence. See `iqa-compare evaluate --help` and `compare-versions --help` for the current interface. Formal training export is a separate human-confirmed workflow, not an automatic consequence of a higher proxy score.

For corresponding source/baseline/candidate directories (same relative filenames):

```bash
iqa-compare compare-versions --source-root /data/source \
  --baseline-root /data/version-A --candidate-root /data/version-B \
  --candidate-root /data/version-C --source-encoding linear \
  --encoding srgb --split audit --output results/version-diff.json
```

This writes JSON evidence and a companion CSV, recording missing counterparts. Filename correspondence alone does not prove common-source lineage. For explicit per-person regions, use `iqa-compare evaluate --manifest comparison.json --output results/comparison.json`. Paths in this manifest are relative to the manifest file:

```json
{
  "scene_id": "portrait-001", "group_id": "capture-001",
  "mode": "algorithm", "split": "audit",
  "source": {"id": "source", "path": "source.npy", "encoding": "linear"},
  "baseline": {"id": "A", "path": "A.png", "encoding": "srgb"},
  "candidates": [{"id": "B", "path": "B.png", "encoding": "srgb"}],
  "rois": [{"id": "person-1-face", "kind": "face", "person_id": "person-1", "bbox": [20, 30, 64, 64]}]
}
```

Algorithm ROIs use baseline coordinates; device-mode regions require explicit asset scope and independently checked correspondence. Optional `source_sha256` on rendered assets is an operator declaration binding them to the current source bytes, not verification of how those renders were generated.

For the existing sequence dataset, copy the default configuration to `selection.yaml`, set every selector Judge's `prompt_version` to `"3.4"`, and configure real endpoints and distinct model identities for production evidence. Historical 3.2/3.3 prompts remain usable for quality evaluation but do not supply candidate-preference evidence. Evaluate with source lineage, then propose and review candidates:

```bash
qwen-tmqa evaluate --root /data/levels --source-root /data/source \
  --config selection.yaml --output results/eval
qwen-tmqa select-candidates --results results/eval/evaluations.json \
  --root /data/levels --source-root /data/source \
  --config selection.yaml \
  --run-manifest results/eval/evaluation_run_manifest.json \
  --splits splits.csv --output results/candidates
qwen-tmqa export-training --candidates results/candidates/candidate_suggestions.jsonl \
  --confirmations human_confirmations.jsonl --splits splits.csv --output results/train
```

Keep `evaluations.json` and `evaluation_run_manifest.json` from the same evaluation together. Candidate selection verifies each scene's result digest as well as the configuration and input images. Historical manifests without result bindings require a fresh evaluation; do not retrofit them using a separately supplied results file. See [the two P1 fixes and regression instructions](P1_FIXES_20261005.md).

`splits.csv` requires `scene_id,canonical_scene_id,group_id,split` covering every dataset scene. Canonical/group derivatives must stay in one split. The default mock configuration creates demonstration suggestions that cannot pass formal training admission. Use real distinct Judge evidence and a separately collected human confirmation for production labels. Generic non-alpha pipelines can call `TrainingCandidate` and `export_training_data`; the same source/candidate-byte, split, group, Judge, human and label-scope gates apply. The exporter authenticates neither reviewer identity nor the candidate-generation process.

Representative auditing is separate from targeted risky cases:

```bash
iqa-compare audit-sample --manifest audit-records.json --sample-size 40 \
  --seed experiment-001 --output results/audit-sample.json
```

Input is a JSON list of unique `id` records with boolean `risk`, plus optional `scene_type` and `split`. Output contains `sample`, a separate complete `focused_risk_queue`, and stratum counts. Collect blind human reviews on the selected audit cases; do not treat this balanced sample as a population-weighted accuracy estimate. Keep review JSONL and its `.exposure.jsonl` sidecar together. Reveal closes blind submission for the reviewer/scene across restarts; later adjudication can support training but cannot replace blind calibration gold. Run the local review server as one process.

## Front with the shared core

Python 3.12:

```bash
git clone --branch feature/iqa-frontcamera-integration-20261005 https://github.com/baolinv0/FroneCamera.git FroneCamera
cd FroneCamera
bash scripts/install_development.sh
source .venv/bin/activate
portrait-eval-api --help
portrait-eval-worker --help
```

The actual Front pipeline stores an `iqa_evaluation` for each scene. Cross-device outputs are evaluated in device mode; they cannot be exported as same-input algorithm training pairs. The new standalone branch supplies IQA only; the integration branch also supplies corrected Front workflow and reporting.

## Verification

Run the complete workflow checks from the integration branch cloned above:

```bash
python -m pytest -q
(cd packages/iqa && python -m pytest -q)
(cd web && npm ci && npm test -- --run && npm run build)
```

Default model configurations use declared synthetic/heuristic evidence. To evaluate real model quality, configure real endpoints and collect real blind reviews. No real-model accuracy or TM training benefit is established by the local software canary.
