from __future__ import annotations

import base64
import hashlib
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx
import numpy as np
import pytest
from PIL import Image

from qwen_tmqa.cli import (
    build_parser,
    calibrate_command,
    evaluate_command,
    make_example,
    simulate_human,
)
from qwen_tmqa.config import DatasetConfig, JudgeConfig, TMQAConfig, load_config
from qwen_tmqa.dataset import discover_scenes
from qwen_tmqa.domain import ModelEvaluation, PromptTrace
from qwen_tmqa.evaluation import EvaluationPipeline
from qwen_tmqa.judges.openai_compatible import OpenAICompatibleJudge, _image_data_url
from qwen_tmqa.prompts import PromptRegistry, build_input_manifest, render_prompt
from qwen_tmqa.review import append_review, load_reviews
from qwen_tmqa.server import create_server
from qwen_tmqa.visualization import generate_dashboard

EXPECTED_LEVELS = [
    "a_m100",
    "a_m075",
    "a_m050",
    "a_m025",
    "a_000",
    "a_p025",
    "a_p050",
    "a_p075",
    "a_p100",
]


def _save(path: Path, value: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.05, value, 32, dtype=np.float32)
    image = np.tile(x[None, :, None], (24, 1, 3))
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def _write_dataset(root: Path, levels: list[str] | None = None) -> None:
    selected = levels or EXPECTED_LEVELS
    for index, level in enumerate(selected):
        _save(root / level / "scene.png", 0.2 + 0.06 * index)


def _evaluated_scene(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    config = load_config(Path("configs/default.yaml"))
    scene = discover_scenes(dataset, config.dataset)[0]
    return config, scene, EvaluationPipeline(config).evaluate_scene(scene)


def test_strict_complete_rejects_missing_expected_level_directory(tmp_path: Path) -> None:
    _write_dataset(tmp_path, EXPECTED_LEVELS[:-1])

    with pytest.raises(ValueError, match="missing expected level"):
        discover_scenes(tmp_path, DatasetConfig(strict_complete=True))


def test_strict_complete_rejects_unknown_parseable_level_directory(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _save(tmp_path / "a_p125" / "scene.png", 0.9)

    with pytest.raises(ValueError, match="unknown alpha level"):
        discover_scenes(tmp_path, DatasetConfig(strict_complete=True))


def test_dataset_config_rejects_duplicate_levels_and_missing_baseline() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        DatasetConfig(expected_levels=["a_000", "a_000"])

    with pytest.raises(ValueError, match="baseline"):
        DatasetConfig(expected_levels=["a_m100", "a_p100"], baseline_level="a_000")


def test_historical_prompt_32_is_immutable_and_new_contract_is_33() -> None:
    registry = PromptRegistry.default()
    historical_32 = (
        "Role: You are a professional tone-mapping quality evaluator.\n"
        "Scene ID: {{scene_id}}\n"
        "Model role: {{model_role}}\n"
        "Evaluation mode: {{evaluation_mode}}\n"
        "Evaluate tone, color, fidelity, control, and preference independently.\n"
        "Do not assume brighter means better.\n"
        "A fatal content mutation cannot be compensated by aesthetic scores.\n"
        "Image manifest:\n{{image_manifest}}\n"
        "Objective evidence:\n{{objective_evidence}}\n"
        "Return only JSON matching {{output_schema}}."
    )

    prompt_32 = registry.get("tmqa.sequence", "3.2")
    prompt_33 = registry.get("tmqa.sequence", "3.3")

    assert prompt_32.text == historical_32
    assert prompt_32.output_schema_version == "tmqa.sequence.v3"
    assert prompt_33.output_schema_version == "tmqa.sequence.v4"
    assert "All six score keys are required" in prompt_33.text
    assert '"overall"' in prompt_33.text


def test_manifest_payload_hash_matches_exact_bytes_sent_to_model(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    scene = discover_scenes(tmp_path, DatasetConfig())[0]
    manifest = build_input_manifest(scene, max_side=17)
    first = manifest[0]

    data_url = _image_data_url(first.path, first.sent_width, first.sent_height)
    sent_bytes = base64.b64decode(data_url.split(",", 1)[1])

    assert first.source_sha256 == hashlib.sha256(Path(first.path).read_bytes()).hexdigest()
    assert first.payload_sha256 == hashlib.sha256(sent_bytes).hexdigest()
    assert first.payload_mime == "image/jpeg"
    assert first.payload_encoding["format"] == "JPEG"
    assert first.payload_encoding["quality"] == 92


def test_real_judge_fails_closed_when_manifest_payload_hash_is_stale(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    scene = discover_scenes(tmp_path, DatasetConfig())[0]
    trace = render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version="3.2",
        scene=scene,
        model_role="primary",
        objective_evidence={},
        inference_parameters={},
    )
    trace_data = trace.model_dump(mode="json")
    trace_data["input_manifest"][0]["payload_sha256"] = "0" * 64
    tampered = PromptTrace.model_validate(trace_data)

    def handler(_: httpx.Request) -> httpx.Response:
        payload = {
            "scores": {
                "tone": 80,
                "color": 80,
                "fidelity": 80,
                "control": 80,
                "preference": 80,
                "overall": 80,
            },
            "decision": "KEEP",
            "confidence": 90,
            "issues": [],
            "rationale": "valid response",
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}]},
        )

    judge = OpenAICompatibleJudge(
        JudgeConfig(
            id="remote",
            role="primary",
            adapter="openai_compatible",
            model="remote",
            base_url="http://test/v1",
        ),
        transport=httpx.MockTransport(handler),
    )

    result = judge.evaluate(tampered)

    assert result.available is False
    assert "payload" in (result.error or "").lower()
    assert "hash" in (result.error or "").lower()


@pytest.mark.parametrize(
    ("model_decision", "expected"),
    [("REJECT", "REJECT"), ("REGENERATE", "REGENERATE"), ("REVIEW", "REVIEW")],
)
def test_unanimous_non_keep_decision_cannot_be_overridden_by_high_scores(
    tmp_path: Path,
    monkeypatch,
    model_decision: str,
    expected: str,
) -> None:
    _write_dataset(tmp_path)
    config = TMQAConfig(
        judges=[
            JudgeConfig(id="a", role="primary", model="a", synthetic=True),
            JudgeConfig(id="b", role="arbiter", model="b", synthetic=True),
        ]
    )
    scene = discover_scenes(tmp_path, config.dataset)[0]

    class FixedJudge:
        def __init__(self, model_id: str):
            self.model_id = model_id

        def evaluate(self, trace: PromptTrace) -> ModelEvaluation:
            return ModelEvaluation(
                model_id=self.model_id,
                model_role="primary" if self.model_id == "a" else "arbiter",
                model_version="v1",
                synthetic=True,
                prompt_trace=trace,
                scores={
                    "tone": 0.95,
                    "color": 0.95,
                    "fidelity": 0.95,
                    "control": 0.95,
                    "preference": 0.95,
                    "overall": 0.95,
                },
                decision=model_decision,
                confidence=0.95,
            )

    monkeypatch.setattr(
        "qwen_tmqa.evaluation._judge_from_config",
        lambda judge_config, _transport: FixedJudge(judge_config.id),
    )

    result = EvaluationPipeline(config).evaluate_scene(scene)

    assert result.decision == expected


def test_calibration_rejects_synthetic_reviews_without_explicit_opt_in(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    reviews = tmp_path / "reviews.jsonl"
    output = tmp_path / "calibration.json"
    make_example(dataset, 1)
    evaluate_command(dataset, results, Path("configs/default.yaml"))
    simulate_human(results / "evaluations.json", reviews, Path("configs/default.yaml"))

    with pytest.raises(ValueError, match="synthetic"):
        calibrate_command(results / "evaluations.json", reviews, output)

    calibrate_command(
        results / "evaluations.json",
        reviews,
        output,
        allow_synthetic=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_review_type"] == "synthetic"
    assert payload["experimental"] is True


def test_calibration_rejects_implicit_mixed_review_evidence(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    reviews_path = tmp_path / "reviews.jsonl"
    output = tmp_path / "calibration.json"
    make_example(dataset, 1)
    evaluate_command(dataset, results, Path("configs/default.yaml"))
    simulate_human(results / "evaluations.json", reviews_path, Path("configs/default.yaml"))
    synthetic = load_reviews(reviews_path)[0]
    append_review(
        reviews_path,
        synthetic.model_copy(
            update={
                "review_id": "real-review",
                "reviewer_id": "human-reviewer",
                "synthetic": False,
            }
        ),
    )

    with pytest.raises(ValueError, match="mixed"):
        calibrate_command(results / "evaluations.json", reviews_path, output)


def test_calibrate_cli_exposes_explicit_synthetic_policy_flags() -> None:
    args = build_parser().parse_args(
        [
            "calibrate",
            "--results",
            "results.json",
            "--reviews",
            "reviews.jsonl",
            "--output",
            "calibration.json",
            "--allow-synthetic",
            "--review-type",
            "synthetic",
        ]
    )

    assert args.allow_synthetic is True
    assert args.review_type == "synthetic"


def _post_json(url: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_reviewer_client_is_data_blind_until_persisted_reveal(tmp_path: Path) -> None:
    _, _, evaluation = _evaluated_scene(tmp_path)
    dashboard = tmp_path / "dashboard"
    generate_dashboard(
        [evaluation],
        dashboard,
        title="TMQA",
        review_queue=[evaluation],
    )

    reviewer_html = (dashboard / "review.html").read_text(encoding="utf-8")
    reviewer_payload = json.loads(
        (dashboard / "data" / "reviewer_payload.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps(reviewer_payload, sort_keys=True)
    forbidden = [
        "model_evaluations",
        "model_id",
        "prompt_trace",
        "rendered_prompt",
        "raw_response",
        "parsed_response",
        "overall_score",
        "review_reasons",
        "stages",
        "system_decision",
    ]
    assert all(token not in reviewer_html for token in forbidden)
    assert all(token not in serialized for token in forbidden)
    assert (dashboard / "data" / "engineering_evaluations.json").exists()
    assert (dashboard / "data" / "reveal_payload.json").exists()

    reviews = tmp_path / "reviews.jsonl"
    server = create_server(dashboard, reviews, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        before_url = (
            f"http://127.0.0.1:{port}/api/reveal"
            f"?scene_id={evaluation.scene_id}&review_id=r1"
        )
        with pytest.raises(HTTPError) as before:
            urlopen(before_url)
        assert before.value.code in {403, 404}

        future_client_time = "2099-01-01T00:00:00+00:00"
        status, response = _post_json(
            f"http://127.0.0.1:{port}/api/reviews",
            {
                "review_id": "r1",
                "scene_id": evaluation.scene_id,
                "reviewer_id": "human",
                "blind_review": True,
                "decision": "KEEP",
                "scores": {"overall": 0.8},
                "confidence": 0.9,
                "created_at": future_client_time,
            },
        )
        assert status == 201
        assert response["reveal_url"].startswith("/api/reveal?")

        stored = json.loads(reviews.read_text(encoding="utf-8").splitlines()[0])
        assert stored["created_at"] == future_client_time
        assert stored["received_at"] != future_client_time
        assert stored["received_at"].endswith("+00:00") or stored["received_at"].endswith("Z")

        with urlopen(f"http://127.0.0.1:{port}{response['reveal_url']}") as revealed_response:
            reveal = json.loads(revealed_response.read().decode("utf-8"))
        assert reveal["scene_id"] == evaluation.scene_id
        assert reveal["model_evaluations"]
        assert reveal["model_evaluations"][0]["prompt_trace"]["rendered_prompt"]

        for protected in [
            "/data/reveal_payload.json",
            "/data/engineering_evaluations.json",
        ]:
            with pytest.raises(HTTPError) as static_error:
                urlopen(f"http://127.0.0.1:{port}{protected}")
            assert static_error.value.code == 403
    finally:
        server.shutdown()
        thread.join(timeout=3)


def test_storage_failure_does_not_return_reveal_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, _, evaluation = _evaluated_scene(tmp_path)
    dashboard = tmp_path / "dashboard"
    generate_dashboard([evaluation], dashboard, title="TMQA", review_queue=[evaluation])

    def fail_append(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("qwen_tmqa.server.append_review", fail_append)
    server = create_server(dashboard, tmp_path / "reviews.jsonl", host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        request = Request(
            f"http://127.0.0.1:{port}/api/reviews",
            data=json.dumps(
                {
                    "review_id": "r-fail",
                    "scene_id": evaluation.scene_id,
                    "reviewer_id": "human",
                    "blind_review": True,
                    "decision": "KEEP",
                    "scores": {"overall": 0.8},
                    "confidence": 0.9,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 500
        body = json.loads(error.value.read().decode("utf-8"))
        assert "reveal_url" not in body
    finally:
        server.shutdown()
        thread.join(timeout=3)
