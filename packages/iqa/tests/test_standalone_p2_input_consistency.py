from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from qwen_tmqa import evaluation
from qwen_tmqa.config import JudgeConfig, TMQAConfig
from qwen_tmqa.domain import AlphaImage, SceneSpec
from qwen_tmqa.evaluation import EvaluationPipeline
from qwen_tmqa.image_io import file_sha256


@pytest.fixture
def scene(tmp_path: Path) -> SceneSpec:
    images = []
    for index, alpha in enumerate([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]):
        path = tmp_path / f"level_{index}.png"
        pixels = np.tile(
            np.linspace(45, 135 + index * 7, 60, dtype=np.uint8)[None, :, None], (42, 1, 3)
        )
        Image.fromarray(pixels).save(path)
        images.append(AlphaImage(alpha=alpha, level=f"level_{index}", path=path))
    source = tmp_path / "source.png"
    Image.fromarray(np.full((42, 60, 3), 100, dtype=np.uint8)).save(source)
    return SceneSpec(
        scene_id="mutable-scene",
        source_path=source,
        baseline_path=images[4].path,
        alpha_images=images,
    )


def _pipeline(scene: SceneSpec, mutate=None):
    response = {
        "scores": dict.fromkeys(["tone", "color", "fidelity", "control", "preference", "overall"], 95),
        "decision": "KEEP",
        "confidence": 95,
        "issues": [],
        "rationale": "high quality candidate",
        "preferred_level": "level_8",
        "runner_up_level": "level_7",
        "acceptable_levels": ["level_8"],
        "level_scores": {f"level_{index}": round(.1 + .09 * index, 2) for index in range(9)},
        "selection_confidence": .95,
        "baseline_improvement": .36,
    }
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1 and mutate is not None:
            mutate()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(response)}}]}
        )

    config = TMQAConfig(
        judges=[
            JudgeConfig(
                id=name,
                role="primary",
                adapter="openai_compatible",
                model="mock-http",
                base_url="http://local.test/v1",
                prompt_version="3.4",
            )
            for name in ["first", "second"]
        ]
    )
    pipeline = EvaluationPipeline(
        config, transports={name: httpx.MockTransport(handler) for name in ["first", "second"]}
    )
    return pipeline, requests


def _path(scene: SceneSpec, role: str) -> Path:
    if role == "source":
        assert scene.source_path is not None
        return scene.source_path
    if role == "baseline":
        return scene.baseline_path
    return scene.alpha_images[-1 if role == "candidate" else 0].path


def _mutate(path: Path, change: str) -> None:
    if change == "replace":
        replacement = path.with_suffix(".replacement.png")
        Image.fromarray(np.full((42, 60, 3), 255, dtype=np.uint8)).save(replacement)
        replacement.replace(path)
    elif change == "metadata":
        with Image.open(path) as image:
            pixels = image.copy()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("producer_version", "rewritten after first judge")
        pixels.save(path, pnginfo=metadata)
    elif change == "delete":
        path.unlink()
    else:
        path.write_bytes(b"not a decodable image")


def _assert_request_matches_trace(request, model) -> None:
    parts = request["messages"][0]["content"]
    assert parts[0]["text"] == model.prompt_trace.rendered_prompt
    urls = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
    assert len(urls) == len(model.prompt_trace.input_manifest)
    for url, item in zip(urls, model.prompt_trace.input_manifest, strict=True):
        payload = base64.b64decode(url.split(",", 1)[1])
        assert hashlib.sha256(payload).hexdigest() == item.payload_sha256
        with Image.open(io.BytesIO(payload)) as image:
            assert image.size == (item.sent_width, item.sent_height)
            if item.role != "source":
                assert np.asarray(image).max() < 250


def test_stable_inputs_keep_original_source_and_actual_payload_evidence(scene):
    original = {str(path): file_sha256(path) for path in [_path(scene, "source"), *[
        item.path for item in scene.alpha_images
    ]]}
    pipeline, requests = _pipeline(scene)
    result = pipeline.evaluate_scene(scene)
    assert result.decision == "KEEP"
    assert len(requests) == 2
    assert all(item.available for item in result.model_evaluations)
    assert all(stage.status == "PASS" for stage in result.stages[:2])
    for item in result.input_manifest:
        assert item.source_sha256 == item.sha256 == original[item.path]
    for request, model in zip(requests, result.model_evaluations, strict=True):
        assert model.prompt_trace.input_manifest == result.input_manifest
        evidence = json.loads(model.prompt_trace.variables["objective_evidence"])
        assert evidence["max_clipping_ratio"] == max(
            item.clipping_ratio for item in result.objective_by_level.values()
        ) == 0.0
        _assert_request_matches_trace(request, model)


@pytest.mark.parametrize("role", ["source", "baseline", "candidate", "lower_alpha"])
@pytest.mark.parametrize("change", ["replace", "metadata", "delete", "unreadable"])
def test_change_after_first_request_fails_scene_and_prevents_second_request(scene, role, change):
    path = _path(scene, role)
    original_hash = file_sha256(path)
    pipeline, requests = _pipeline(scene, lambda: _mutate(path, change))
    result = pipeline.evaluate_scene(scene)
    assert result.decision == "REJECT"
    assert {stage.stage_id: stage.status for stage in result.stages}["integrity"] == "FAIL"
    assert {stage.stage_id: stage.status for stage in result.stages}["hard_gate"] == "FAIL"
    assert "input_integrity_failure" in result.review_reasons
    assert len(requests) == 1
    _assert_request_matches_trace(requests[0], result.model_evaluations[0])
    assert len(result.model_evaluations) == 2
    assert not result.model_evaluations[1].available
    assert "input integrity" in result.model_evaluations[1].error.lower()
    assert next(item for item in result.input_manifest if item.path == str(path)).source_sha256 == (
        original_hash
    )
    assert all(model.prompt_trace.input_manifest == result.input_manifest
               for model in result.model_evaluations)


def test_explicit_baseline_outside_alpha_list_is_guarded(scene):
    baseline = scene.baseline_path.with_name("objective_baseline.png")
    baseline.write_bytes(scene.baseline_path.read_bytes())
    scene.baseline_path = baseline
    pipeline, requests = _pipeline(scene, lambda: _mutate(baseline, "replace"))
    result = pipeline.evaluate_scene(scene)
    assert result.decision == "REJECT"
    assert result.stages[0].status == result.stages[1].status == "FAIL"
    assert len(requests) == 1


@pytest.mark.parametrize("role", ["source", "baseline", "candidate", "lower_alpha"])
@pytest.mark.parametrize("change", ["delete", "unreadable"])
def test_invalid_initial_input_fails_explicitly_before_dispatch(scene, role, change):
    _mutate(_path(scene, role), change)
    pipeline, requests = _pipeline(scene)
    with pytest.raises(ValueError, match="input integrity"):
        pipeline.evaluate_scene(scene)
    assert requests == []


def test_source_hash_read_failure_after_first_request_fails_scene(scene, monkeypatch):
    original_open = Path.open
    fail_reads = False

    def guarded_open(path, *args, **kwargs):
        if fail_reads and path == scene.source_path:
            raise PermissionError("input fixture read denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    def deny_reads():
        nonlocal fail_reads
        fail_reads = True

    pipeline, requests = _pipeline(scene, deny_reads)
    result = pipeline.evaluate_scene(scene)
    assert result.decision == "REJECT"
    assert result.stages[0].status == result.stages[1].status == "FAIL"
    assert len(requests) == 1
    assert "read denied" in " ".join(result.stages[0].evidence)


def test_change_during_objective_loading_fails_before_any_request(scene, monkeypatch):
    original_load = evaluation.load_rgb
    candidate_path = scene.alpha_images[-1].path

    def mutate_after_load(path):
        image = original_load(path)
        if path == candidate_path:
            _mutate(path, "replace")
        return image

    monkeypatch.setattr(evaluation, "load_rgb", mutate_after_load)
    pipeline, requests = _pipeline(scene)
    with pytest.raises(ValueError, match="input integrity"):
        pipeline.evaluate_scene(scene)
    assert requests == []


def test_metadata_change_during_prompt_rendering_fails_before_dispatch(scene, monkeypatch):
    original_render = evaluation.render_prompt

    def mutate_after_render(**kwargs):
        trace = original_render(**kwargs)
        _mutate(scene.alpha_images[0].path, "metadata")
        return trace

    monkeypatch.setattr(evaluation, "render_prompt", mutate_after_render)
    pipeline, requests = _pipeline(scene)
    with pytest.raises(ValueError, match="input integrity"):
        pipeline.evaluate_scene(scene)
    assert requests == []
