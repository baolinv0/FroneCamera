from pathlib import Path

import numpy as np
from PIL import Image

from qwen_tmqa.domain import AlphaImage, SceneSpec
from qwen_tmqa.prompts import (
    PromptRegistry,
    build_input_manifest,
    prompt_version_diff,
    render_prompt,
)


def _save(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8)).save(path)


def _scene(tmp_path: Path, with_source: bool = False) -> SceneSpec:
    images = []
    for level, alpha, value in [
        ("a_m100", -1.0, 50),
        ("a_000", 0.0, 120),
        ("a_p100", 1.0, 220),
    ]:
        path = tmp_path / level / "scene.png"
        _save(path, value)
        images.append(AlphaImage(level=level, alpha=alpha, path=path))
    source = None
    if with_source:
        source = tmp_path / "source" / "scene.png"
        _save(source, 100)
    return SceneSpec(
        scene_id="scene.png",
        baseline_path=tmp_path / "a_000" / "scene.png",
        alpha_images=images,
        source_path=source,
    )


def test_manifest_labels_every_image_with_source_and_payload_lineage(tmp_path: Path) -> None:
    manifest = build_input_manifest(_scene(tmp_path, with_source=True), max_side=1024)
    assert [item.index for item in manifest] == [1, 2, 3, 4]
    assert [item.role for item in manifest] == ["source", "candidate", "baseline", "candidate"]
    assert manifest[1].alpha == -1.0
    assert manifest[2].level == "a_000"
    assert len(manifest[3].source_sha256) == 64
    assert len(manifest[3].payload_sha256) == 64
    assert manifest[3].payload_mime == "image/jpeg"
    assert manifest[3].payload_encoding == {"format": "JPEG", "quality": 92}
    assert manifest[0].width == 16 and manifest[0].height == 12
    assert manifest[0].sent_width == 16 and manifest[0].sent_height == 12


def test_rendered_prompt_32_contains_explicit_image_order_and_guardrails(tmp_path: Path) -> None:
    registry = PromptRegistry.default()
    trace = render_prompt(
        registry=registry,
        prompt_id="tmqa.sequence",
        prompt_version="3.2",
        scene=_scene(tmp_path),
        model_role="primary",
        objective_evidence={"clipping_growth": 0.1},
        inference_parameters={"temperature": 0.0},
    )
    text = trace.rendered_prompt
    assert "[Image 1] role=candidate, alpha=-1.00, level=a_m100" in text
    assert "[Image 2] role=baseline, alpha=+0.00, level=a_000" in text
    assert "[Image 3] role=candidate, alpha=+1.00, level=a_p100" in text
    assert "Do not assume brighter means better" in text
    assert "clipping_growth" in text
    assert '"scores"' not in text
    assert trace.output_schema_version == "tmqa.sequence.v3"
    assert trace.prompt_hash == render_prompt(
        registry=registry,
        prompt_id="tmqa.sequence",
        prompt_version="3.2",
        scene=_scene(tmp_path),
        model_role="primary",
        objective_evidence={"clipping_growth": 0.1},
        inference_parameters={"temperature": 0.0},
    ).prompt_hash


def test_prompt_version_diff_exposes_guardrails_and_new_contract() -> None:
    registry = PromptRegistry.default()
    old_diff = prompt_version_diff(registry, "tmqa.sequence", "3.1", "3.2")
    contract_diff = prompt_version_diff(registry, "tmqa.sequence", "3.2", "3.3")
    assert "+Do not assume brighter means better." in old_diff
    assert "-Evaluate whether the enhanced images look good." in old_diff
    assert '+  "scores":' in contract_diff
    assert "+All six score keys are required." in contract_diff


def test_prompt_33_contains_explicit_machine_readable_output_contract(tmp_path: Path) -> None:
    trace = render_prompt(
        registry=PromptRegistry.default(),
        prompt_id="tmqa.sequence",
        prompt_version="3.3",
        scene=_scene(tmp_path),
        model_role="primary",
        objective_evidence={"clipping_growth": 0.1},
        inference_parameters={"temperature": 0.0},
    )

    assert trace.output_schema_version == "tmqa.sequence.v4"
    assert '"scores"' in trace.rendered_prompt
    assert '"overall"' in trace.rendered_prompt
    assert '"decision"' in trace.rendered_prompt
    assert '"confidence"' in trace.rendered_prompt
    assert '"issues"' in trace.rendered_prompt
    assert '"fatal"' in trace.rendered_prompt
    assert "KEEP|REGENERATE|REVIEW|REJECT" in trace.rendered_prompt
