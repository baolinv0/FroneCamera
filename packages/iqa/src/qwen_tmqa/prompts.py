from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass

from .domain import ImageManifestItem, PromptTrace, SceneSpec
from .image_io import encode_image_payload, file_sha256, image_dimensions


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    version: str
    text: str
    output_schema_version: str


class PromptRegistry:
    def __init__(self, templates: list[PromptTemplate]):
        self._templates = {(item.prompt_id, item.version): item for item in templates}

    def get(self, prompt_id: str, version: str) -> PromptTemplate:
        try:
            return self._templates[(prompt_id, version)]
        except KeyError as exc:
            raise KeyError(f"prompt {prompt_id}@{version} is not registered") from exc

    @classmethod
    def default(cls) -> PromptRegistry:
        prompt_32 = (
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
        prompt_33 = (
            prompt_32[:-1] + " with this contract:\n"
            "{\n"
            '  "scores": {"tone": 0-100, "color": 0-100, '
            '"fidelity": 0-100, "control": 0-100, '
            '"preference": 0-100, "overall": 0-100},\n'
            '  "decision": "KEEP|REGENERATE|REVIEW|REJECT",\n'
            '  "confidence": 0-100,\n'
            '  "issues": [{"dimension": "string", '
            '"severity": 0-1, "description": "string", '
            '"fatal": false, "region": "string or null"}],\n'
            '  "rationale": "brief evidence-based explanation"\n'
            "}\n"
            "All six score keys are required. Do not add Markdown fences or extra text."
        )
        return cls(
            [
                PromptTemplate(
                    prompt_id="tmqa.sequence",
                    version="3.1",
                    output_schema_version="tmqa.sequence.v3",
                    text=(
                        "Role: You are a professional image-quality evaluator.\n"
                        "Scene ID: {{scene_id}}\n"
                        "Model role: {{model_role}}\n"
                        "Evaluate whether the enhanced images look good.\n"
                        "Image manifest:\n{{image_manifest}}\n"
                        "Objective evidence:\n{{objective_evidence}}\n"
                        "Return only JSON matching {{output_schema}}."
                    ),
                ),
                PromptTemplate(
                    prompt_id="tmqa.sequence",
                    version="3.2",
                    output_schema_version="tmqa.sequence.v3",
                    text=prompt_32,
                ),
                PromptTemplate(
                    prompt_id="tmqa.sequence",
                    version="3.3",
                    output_schema_version="tmqa.sequence.v4",
                    text=prompt_33,
                ),
                PromptTemplate(
                    prompt_id="tmqa.sequence",
                    version="3.4",
                    output_schema_version="tmqa.sequence.v5",
                    text=prompt_33
                    + "\nCandidate preference contract (additional required JSON keys):\n"
                    "preferred_level, runner_up_level: evaluated candidate IDs; "
                    "acceptable_levels: nonempty evaluated ID list containing preferred_level; "
                    "level_scores: complete evaluated candidate ID map with finite [0,1] scores; "
                    "selection_confidence: finite [0,1]; baseline_improvement: preferred score "
                    "minus baseline score. Preferred must maximize score, runner-up must maximize "
                    "remaining scores. This is a model suggestion, not a human-confirmed training label. "
                    "Do not add other keys.",
                ),
            ]
        )


def _sent_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    largest = max(width, height)
    if largest <= max_side:
        return width, height
    scale = max_side / largest
    return max(1, round(width * scale)), max(1, round(height * scale))


def build_input_manifest(scene: SceneSpec, max_side: int = 1024) -> list[ImageManifestItem]:
    records: list[tuple[str, float | None, str, object]] = []
    if scene.source_path is not None:
        records.append(("source", None, "source", scene.source_path))
    for item in scene.alpha_images:
        role = "baseline" if item.path == scene.baseline_path or item.alpha == 0 else "candidate"
        records.append((role, item.alpha, item.level, item.path))

    manifest: list[ImageManifestItem] = []
    for index, (role, alpha, level, raw_path) in enumerate(records, start=1):
        path = raw_path
        width, height = image_dimensions(path)
        sent_width, sent_height = _sent_dimensions(width, height, max_side)
        encoded = encode_image_payload(path, sent_width, sent_height)
        source_hash = file_sha256(path)
        manifest.append(
            ImageManifestItem(
                index=index,
                role=role,
                alpha=alpha,
                level=level,
                path=str(path),
                sha256=source_hash,
                source_sha256=source_hash,
                payload_sha256=encoded.sha256,
                payload_mime=encoded.mime_type,
                payload_encoding=encoded.encoding,
                width=width,
                height=height,
                sent_width=encoded.width,
                sent_height=encoded.height,
            )
        )
    return manifest


def _manifest_text(manifest: list[ImageManifestItem]) -> str:
    lines: list[str] = []
    for item in manifest:
        alpha_text = "none" if item.alpha is None else f"{item.alpha:+.2f}"
        payload_hash = (item.payload_sha256 or "missing")[:12]
        lines.append(
            f"[Image {item.index}] role={item.role}, alpha={alpha_text}, level={item.level}, "
            f"source_sha256={item.source_sha256[:12]}, payload_sha256={payload_hash}, "
            f"mime={item.payload_mime}, sent={item.sent_width}x{item.sent_height}"
        )
    return "\n".join(lines)


def render_prompt(
    *,
    registry: PromptRegistry,
    prompt_id: str,
    prompt_version: str,
    scene: SceneSpec,
    model_role: str,
    objective_evidence: dict[str, object],
    inference_parameters: dict[str, object],
    max_side: int = 1024,
) -> PromptTrace:
    template = registry.get(prompt_id, prompt_version)
    manifest = build_input_manifest(scene, max_side=max_side)
    variables: dict[str, object] = {
        "scene_id": scene.scene_id,
        "model_role": model_role,
        "evaluation_mode": "SOURCE_AWARE_TM" if scene.source_path else "RGB_ENHANCEMENT",
        "image_manifest": _manifest_text(manifest),
        "objective_evidence": json.dumps(objective_evidence, ensure_ascii=False, sort_keys=True),
        "output_schema": template.output_schema_version,
    }
    rendered = template.text
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return PromptTrace(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        template=template.text,
        variables=variables,
        rendered_prompt=rendered,
        prompt_hash=prompt_hash,
        input_manifest=manifest,
        output_schema_version=template.output_schema_version,
        inference_parameters=inference_parameters,
    )


def prompt_version_diff(
    registry: PromptRegistry,
    prompt_id: str,
    old_version: str,
    new_version: str,
) -> str:
    old = registry.get(prompt_id, old_version).text.splitlines()
    new = registry.get(prompt_id, new_version).text.splitlines()
    return "\n".join(
        difflib.unified_diff(old, new, fromfile=old_version, tofile=new_version, lineterm="")
    )
