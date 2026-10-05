# Standalone IQA

This branch provides the independently installable IQA project in `packages/iqa`, for training-data screening and comparing algorithm versions. The existing Front source is retained from the parent snapshot. The corrected Front pipeline is delivered on [the integration branch](https://github.com/baolinv0/FroneCamera/tree/feature/iqa-frontcamera-integration-20261005).

```bash
python -m venv .venv-iqa
source .venv-iqa/bin/activate
pip install ./packages/iqa
iqa-compare --help
qwen-tmqa --help
```

See [the workflow guide](docs/integration/QUICKSTART.md) for version folders, per-person manifests, representative auditing, candidate selection and human-confirmed training export. No Front API/database/frontend dependency is required. Proxy measurements preserve explicit missing-evidence states; software fixtures do not establish DXOMARK lab equivalence, real-model accuracy or training gains.
