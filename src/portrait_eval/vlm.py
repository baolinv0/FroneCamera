from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from portrait_eval.models import ModelEvaluationResult, ModelObservation


def encode_image_data_url(path: Path, max_edge: int = 1536) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_json_object(content: str) -> dict[str, Any]:
    """Extract one JSON object from common model wrappers without guessing field values."""
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response does not contain a valid JSON object")


class VisionModelAdapter(ABC):
    @abstractmethod
    def analyze(
        self, scene_id: str, image_paths: dict[str, Path], context: dict[str, Any]
    ) -> ModelEvaluationResult:
        raise NotImplementedError


class OpenAICompatibleVisionAdapter(VisionModelAdapter):
    def __init__(
        self,
        base_url: str,
        model: str,
        role: str,
        timeout: float = 300.0,
        max_image_edge: int = 1536,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.role = role
        self.timeout = timeout
        self.max_image_edge = max_image_edge

    def analyze(
        self, scene_id: str, image_paths: dict[str, Path], context: dict[str, Any]
    ) -> ModelEvaluationResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": self._prompt(scene_id, context)}]
        for code, path in image_paths.items():
            content.extend(
                [
                    {"type": "text", "text": f"Anonymous device {code}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_data_url(path, self.max_image_edge)},
                    },
                ]
            )
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
        }
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        raw = response.json()
        parsed = parse_json_object(raw["choices"][0]["message"]["content"])
        parsed.update({"scene_id": scene_id, "role": self.role, "raw": raw})
        return ModelEvaluationResult.model_validate(parsed)

    def _prompt(self, scene_id: str, context: dict[str, Any]) -> str:
        return (
            "Evaluate anonymous front-camera portraits. Separate visible observations, subjective preferences, "
            "and mechanism hypotheses. Do not identify brands. Return JSON with observations, scores, hypotheses, "
            f"confidence. Scene={scene_id}. Objective context={json.dumps(context, ensure_ascii=False)}"
        )


class HeuristicVisionAdapter(VisionModelAdapter):
    """Deterministic fallback that keeps the complete pipeline runnable without model weights."""

    def __init__(self, role: str) -> None:
        self.role = role

    def analyze(
        self, scene_id: str, image_paths: dict[str, Path], context: dict[str, Any]
    ) -> ModelEvaluationResult:
        observations: list[ModelObservation] = []
        metrics = context.get("metrics", {})
        values = {
            device: data.get("whole", {}).get("luma_mean", 0.0) for device, data in metrics.items()
        }
        if values:
            brightest = max(values, key=lambda device: values[device])
            observations.append(
                ModelObservation(
                    device_id=brightest,
                    dimension="global_exposure",
                    statement=(
                        "This output has the highest display-referred mean luminance in the "
                        "matched group."
                    ),
                    evidence_refs=[f"metric:{scene_id}:{brightest}:luma_mean"],
                    certainty=0.95,
                )
            )
        return ModelEvaluationResult(
            scene_id=scene_id,
            role=self.role,
            observations=observations,
            scores={},
            hypotheses=[],
            confidence=0.65,
            raw={"adapter": "heuristic"},
        )
