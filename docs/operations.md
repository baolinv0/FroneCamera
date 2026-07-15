# Deployment and Operations

## Native deployment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,pdf]"
cp .env.example .env
alembic upgrade head
portrait-eval-api --host 127.0.0.1 --port 7860
portrait-eval-worker
```

The API also creates missing tables at startup, so a new local database can start without a separate migration command. Alembic remains the controlled production upgrade path.

## Windows access

Default SSH tunnel:

```bash
ssh -L 7860:127.0.0.1:7860 user@linux-server
```

Open `http://127.0.0.1:7860` on Windows.

For LAN access, bind the API to a specific server interface rather than `0.0.0.0`, configure an API token, and restrict access with the host firewall.

## Worker lifecycle

The API stores evaluation tasks in SQLite. The worker atomically claims pending tasks, increments the attempt counter, retries retryable failures up to `PORTRAIT_EVAL_TASK_MAX_ATTEMPTS` (default 3), and persists terminal failures. It can be restarted without losing queued work.

```bash
portrait-eval-worker --once      # process one pending task
portrait-eval-worker --interval 2
```

The deployment assumes one worker for the single-GPU/single-user v0.1 configuration.

## VLM service order

With one approximately 98 GB GPU, run primary and reviewer services sequentially or use mutually exclusive service supervision. Validate BF16/quantized memory and image-token limits on the actual server before formal evaluation.

## Data mounts

- source image folders: read-only;
- model folder: read-only;
- workspace: writable;
- SQLite database: inside the workspace volume.

## Backup

Use the project export endpoint for structured state. Back up the workspace volume to retain diagnostics and report files. The export ZIP intentionally excludes original images by default.

## External research

A SearxNG instance is the supported v0.1 search interface. Search is disabled by default. When the reviewer model endpoint is configured, it also classifies retrieved excerpts for corroboration; otherwise the safe heuristic only filters front-camera scope and sends relevant excerpts to human review. If search is disabled or unavailable, the visual evaluation still completes, but external corroboration and hardware context remain incomplete.
