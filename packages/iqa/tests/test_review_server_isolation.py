from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image

from qwen_tmqa.config import load_config
from qwen_tmqa.dataset import discover_scenes
from qwen_tmqa.evaluation import EvaluationPipeline
from qwen_tmqa.server import create_server
from qwen_tmqa.visualization import generate_dashboard

LEVELS = [
    ("a_m100", -1.0),
    ("a_m075", -0.75),
    ("a_m050", -0.5),
    ("a_m025", -0.25),
    ("a_000", 0.0),
    ("a_p025", 0.25),
    ("a_p050", 0.5),
    ("a_p075", 0.75),
    ("a_p100", 1.0),
]


def _save(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((20, 28, 3), np.clip(value, 0, 1), dtype=np.float32)
    Image.fromarray(np.round(image * 255).astype(np.uint8)).save(path)


def _dashboard(tmp_path: Path) -> tuple[Path, str]:
    dataset = tmp_path / "dataset"
    for level, alpha in LEVELS:
        _save(dataset / level / "scene.png", 0.45 + 0.2 * alpha)
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(dataset, config.dataset)[0]
    evaluation = EvaluationPipeline(config).evaluate_scene(scene)
    dashboard = tmp_path / "dashboard"
    generate_dashboard(
        [evaluation],
        dashboard,
        title="TMQA",
        review_queue=[evaluation],
    )
    return dashboard, evaluation.scene_id


def test_default_review_server_exposes_only_reviewer_surface(tmp_path: Path) -> None:
    dashboard, _ = _dashboard(tmp_path)
    server = create_server(dashboard, tmp_path / "reviews.jsonl", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            body = response.read().decode("utf-8")
        assert "人工盲审" in body
        assert "Qwen3-VL-8B-Instruct" not in body

        for protected_path in [
            "/index.html",
            "/data/engineering_evaluations.json",
            "/data/reveal_payload.json",
            "/data/%65ngineering_evaluations.json",
            "/data/%72eveal_payload.json",
            "/data/../data/reveal_payload.json",
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(f"http://127.0.0.1:{port}{protected_path}")
            assert error.value.code == 403
    finally:
        server.shutdown()
        thread.join(timeout=3)


def test_engineering_mode_is_explicit_and_does_not_change_reviewer_default(
    tmp_path: Path,
) -> None:
    dashboard, _ = _dashboard(tmp_path)
    server = create_server(
        dashboard,
        tmp_path / "reviews.jsonl",
        port=0,
        mode="engineering",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            body = response.read().decode("utf-8")
        assert "Engineering evaluation observability" in body
        assert "Qwen3-VL-8B-Instruct" in body
    finally:
        server.shutdown()
        thread.join(timeout=3)


def test_reviewer_safe_api_remains_available_in_default_mode(tmp_path: Path) -> None:
    dashboard, scene_id = _dashboard(tmp_path)
    server = create_server(dashboard, tmp_path / "reviews.jsonl", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/reviewer-payload") as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert [scene["scene_id"] for scene in payload["scenes"]] == [scene_id]
        serialized = json.dumps(payload)
        assert "model_evaluations" not in serialized
        assert "prompt_trace" not in serialized
    finally:
        server.shutdown()
        thread.join(timeout=3)
