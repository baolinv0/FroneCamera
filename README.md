# FroneCamera

FroneCamera is a local, evidence-oriented evaluation system for comparing front-camera portrait photographs from multiple phones in matched real-world scenes.

It is designed for imaging engineers who need more than a subjective ranking. The system preserves the chain from source image to pairing, objective measurement, blind model observation, evidence adjudication, external professional-review corroboration, human review, and final report.

## Default four-step workflow

The web application now presents the product workflow directly:

```text
1. Create test data
2. Run model evaluation
3. Generate the test report
4. Open a signed read-only report from another computer
```

The internal pipeline remains detailed, but the user starts it with one **Run full evaluation** action. Advanced pairing correction, evidence inspection, model conflicts, and human review remain available under **Advanced review and engineering controls**.

### Quick mode

- Runs the complete internal pipeline.
- Automatically creates an immutable final report.
- Does not block on open review items.
- Adds an explicit limitation stating that unresolved low-confidence findings require review before external publication.

### Professional mode

- Runs the same internal pipeline.
- Creates a draft report.
- Requires key review items to be resolved before finalization.
- Is intended for formal research, patent, competitor-analysis, or externally distributed reports.

## Internal evidence workflow

```text
N device folders
→ natural-sort pairing proposal
→ manual pairing correction and confirmation
→ image/EXIF audit and adaptive ROI metrics
→ anonymous primary VLM analysis
→ independent reviewer VLM and challenge pass
→ deterministic claim adjudication
→ conservative cross-scene strategy inference
→ freeze internal result snapshot
→ hardware and professional-review search
→ capture-bias assessment and attribution review
→ HTML/JSON/CSV report
→ optional human review gate and immutable final report
```

The input does **not** need to follow a fixed 20-scene taxonomy. Arbitrary matched scene groups are the default. A group can contain missing devices, and repeats are only used when explicitly grouped.

## Implemented modules

- SQLite-backed projects, devices, images, matched groups, analyses, review items, tasks, and reports
- Natural-sort scan and explicit missing pairing cells
- Optimistic version checks for pairing edits and immutable pairing snapshots
- Safe read-only image delivery by registered asset ID
- Display-referred luminance, clipping, shadow, saturation, detail, and noise proxies
- Automatic primary-face detection, face/background/face-ring/highlight/shadow regions, and diagnostic overlays
- Scene comparability warnings and adaptive content tags
- OpenAI-compatible local VLM adapter for vLLM/SGLang endpoints
- Deterministic heuristic fallback when model servers are not configured
- Blind primary visual pass, metric-validation pass, independent reviewer, and adversarial challenge pass with per-scene anonymization
- A/B/C evidence grading and insufficient-scene-coverage gates
- Cross-scene output-strategy claims with explicit counterexamples
- Internal-result freeze before external research
- SearxNG-based hardware and professional-review retrieval plus reviewer-model corroboration classification
- Capture-bias and controlled-reshoot recommendations
- Conservative hardware/capture/reconstruction/rendering attribution cases
- Persistent atomic task claiming, bounded retries, and Linux worker
- Four-step React/Vite product workflow plus advanced engineering review console
- Signed read-only HTML/PDF report links for other computers on the same network
- HTML, JSON, CSV, optional PDF, privacy-preserving project export, and review-aware immutable final reports
- Native Python and Docker Compose deployment

## Quick start: native Linux

Requirements:

- Python 3.12
- `uv`
- Linux access to the image folders

```bash
git clone https://github.com/baolinv0/FroneCamera.git
cd FroneCamera
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,pdf]"
cp .env.example .env
```

Before LAN use, set secure values in `.env`:

```env
PORTRAIT_EVAL_API_TOKEN=replace-with-an-admin-token
PORTRAIT_EVAL_REPORT_SHARE_SECRET=replace-with-a-long-random-secret
```

Start the API and worker:

```bash
portrait-eval-api --host 127.0.0.1 --port 7860
```

In a second terminal:

```bash
source .venv/bin/activate
portrait-eval-worker
```

Open `http://127.0.0.1:7860`. From Windows, use an SSH tunnel:

```bash
ssh -L 7860:127.0.0.1:7860 user@linux-server
```

For direct LAN access, bind the API to the Linux server's LAN address and restrict access with firewall rules. The generated `/reports/<report-id>?token=<signature>` URL is read-only and does not expose the project-control API token.

## Quick start: Docker Compose

```bash
export FRONECAMERA_DATA_ROOT=/absolute/path/to/device-folders
export FRONECAMERA_MODEL_ROOT=/absolute/path/to/local-models
docker compose up --build
```

The API and built-in console are exposed on `127.0.0.1:7860`; the React console is exposed on `127.0.0.1:8080`. Nginx proxies both `/api/` and signed `/reports/` routes to the API service.

## Input layout

Example:

```text
/data/front-test/
├── iPhone17/
│   ├── IMG_0001.HEIC
│   ├── IMG_0002.HEIC
│   └── IMG_0003.HEIC
├── X300Pro/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
└── OtherPhone/
    ├── selfie-a.jpg
    ├── selfie-b.jpg
    └── selfie-c.jpg
```

Filenames may differ. Natural order only proposes correspondence. The user must inspect and confirm pairing before analysis. A missing capture remains an explicit empty cell; the system never silently shifts all later groups.

## Local VLM configuration

FroneCamera supports OpenAI-compatible multimodal endpoints. Run the two model families sequentially on the single GPU and configure:

```env
PORTRAIT_EVAL_PRIMARY_VLM_URL=http://127.0.0.1:8001
PORTRAIT_EVAL_PRIMARY_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
PORTRAIT_EVAL_REVIEWER_VLM_URL=http://127.0.0.1:8002
PORTRAIT_EVAL_REVIEWER_VLM_MODEL=OpenGVLab/InternVL3_5-38B
PORTRAIT_EVAL_VLM_MAX_IMAGE_EDGE=1536
```

When the endpoints are empty, the deterministic heuristic adapter keeps the entire pipeline runnable for infrastructure validation. It is not a substitute for the configured VLMs in a formal image-quality study.

## External corroboration

Internal image conclusions are frozen before external search. To retrieve hardware sources and professional reviews, configure a SearxNG endpoint:

```env
PORTRAIT_EVAL_SEARCH_PROVIDER=searxng
PORTRAIT_EVAL_SEARXNG_URL=http://127.0.0.1:8888
```

External sources do not change the score of the submitted images. The reviewer endpoint classifies each front-camera-specific excerpt as supporting, contradicting, incomparable, irrelevant, or unresolved. Unresolved cases enter the human-review queue. External evidence affects only generalization confidence, capture-bias warnings, and the need for a controlled reshoot.

## Quality checks

```bash
pytest -q
ruff check src tests scripts
ruff format --check src scripts
mypy src/portrait_eval
cd web && npm install && npm test -- --run && npm run build
```

## Documents

- [Current design specification](docs/design/front-camera-portrait-evaluation-system.md)
- [Foundation and pairing implementation plan](docs/plans/foundation-and-pairing.md)
- [Simplified four-step workflow plan](docs/plans/simple-four-step-workflow.md)
- [Deployment and operations](docs/operations.md)
- [Implementation status](docs/implementation-status.md)

## Important interpretation limits

- JPEG/HEIC outputs do not reveal true sensor radiance or a unique internal ISP implementation.
- A brighter face is not automatically better; a darker background is not automatically worse.
- Objective metrics describe output properties, not final aesthetic preference.
- Hardware specifications are evidence for attribution, not direct image-quality scores.
- Cross-scene strategy claims require repeated support and retain counterexamples.
- Quick-mode reports are operationally final but may still contain explicitly flagged unresolved review items.
- Results apply to the submitted devices, firmware, shooting modes, people, and environments.
