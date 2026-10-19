from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from PIL import Image

from portrait_eval.color import SRGB_PROFILE, load_analysis_rgb
from portrait_eval.models import ModelEvaluationResult, ModelObservation

if TYPE_CHECKING:
    from portrait_eval.config import Settings


def encode_image_data_url(path: Path, max_edge: int = 1536, *, lossless: bool = False) -> str:
    pixels, _ = load_analysis_rgb(path)
    image = Image.fromarray(pixels)
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    if lossless:
        image.save(buffer, format="PNG", icc_profile=SRGB_PROFILE)
    else:
        image.save(buffer, format="JPEG", quality=92, optimize=True, icc_profile=SRGB_PROFILE)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "png" if lossless else "jpeg"
    return f"data:image/{mime};base64,{encoded}"


def _visible_context(context: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in context.items() if key != "evidence_inputs"}
    clean["available_evidence"] = [
        {key: value for key, value in item.items() if key != "path"}
        for items in context.get("evidence_inputs", {}).values()
        for item in items
    ]
    return clean


def _image_inputs(
    image_paths: dict[str, Path], context: dict[str, Any], max_edge: int
) -> list[dict[str, Any]]:
    inputs = []
    for code, path in image_paths.items():
        entries = context.get("evidence_inputs", {}).get(code) or [
            {"evidence_id": f"asset:unknown:{code}", "kind": "whole", "path": str(path)}
        ]
        for item in entries:
            pixels, _ = load_analysis_rgb(Path(item["path"]))
            height, width = pixels.shape[:2]
            inputs.append(
                {
                    **item,
                    "anonymous_device": code,
                    "source_size": [width, height],
                    "sent_max_edge": max_edge,
                    "native_resolution_preserved": max(width, height) <= max_edge,
                    "encoding": "PNG" if item["kind"] == "native_face" else "JPEG q92",
                    "native_pixels_preserved": item["kind"] == "native_face"
                    and max(width, height) <= max_edge,
                }
            )
    return inputs


def _record_result_inputs(
    parsed: dict[str, Any], inputs: list[dict[str, Any]], model: str, endpoint: str
) -> None:
    parsed["model_id"] = f"{endpoint}:{model}"
    parsed["model_family"] = "unknown"
    parsed["viewed_evidence"] = [
        {key: value for key, value in item.items() if key != "path"} for item in inputs
    ]


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


def extract_responses_output_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from a raw Responses API payload."""
    for output in payload.get("output", []):
        if not isinstance(output, dict) or output.get("type") != "message":
            continue
        for item in output.get("content", []):
            if isinstance(item, dict) and item.get("type") == "output_text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    raise ValueError("OpenAI Responses API payload does not contain output text")


def resolve_openai_api_key(
    explicit_key: str | None = None,
    auth_file: Path | None = None,
) -> str | None:
    """Resolve an API key without exposing Codex OAuth tokens or logging secrets."""
    if explicit_key:
        return explicit_key
    environment_key = os.getenv("OPENAI_API_KEY")
    if environment_key:
        return environment_key
    path = (auth_file or Path.home() / ".codex" / "auth.json").expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    api_key = payload.get("OPENAI_API_KEY") if isinstance(payload, dict) else None
    return api_key if isinstance(api_key, str) and api_key.startswith("sk-") else None


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
        inputs = _image_inputs(image_paths, context, self.max_image_edge)
        for item in inputs:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Anonymous device {item['anonymous_device']}; evidence={item['evidence_id']}; type={item['kind']}; native pixels preserved={item['native_pixels_preserved']}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image_data_url(
                                Path(item["path"]),
                                self.max_image_edge,
                                lossless=item["kind"] == "native_face",
                            )
                        },
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
        _record_result_inputs(parsed, inputs, self.model, self.base_url)
        return ModelEvaluationResult.model_validate(parsed)

    def _prompt(self, scene_id: str, context: dict[str, Any]) -> str:
        return (
            "Evaluate anonymous front-camera portraits. Separate visible observations, subjective preferences, "
            "and mechanism hypotheses. Do not identify brands. Return JSON with observations, scores, hypotheses, "
            "Every observation must include region, attribute, direction (higher/lower/equal/unknown), comparison, strength, conditions and evidence_refs. Unknown evidence stays unknown. "
            f"confidence. Scene={scene_id}. Objective context={json.dumps(_visible_context(context), ensure_ascii=False)}"
        )


class OpenAIResponsesVisionAdapter(VisionModelAdapter):
    """Official OpenAI Responses API adapter for multimodal image evaluation."""

    def __init__(
        self,
        api_key: str,
        model: str,
        role: str,
        base_url: str = "https://api.openai.com",
        timeout: float = 300.0,
        max_image_edge: int = 1536,
        max_output_tokens: int = 4000,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is required")
        self.api_key = api_key
        self.model = model
        self.role = role
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_image_edge = max_image_edge
        self.max_output_tokens = max_output_tokens

    def analyze(
        self, scene_id: str, image_paths: dict[str, Path], context: dict[str, Any]
    ) -> ModelEvaluationResult:
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": self._prompt(scene_id, context)}
        ]
        inputs = _image_inputs(image_paths, context, self.max_image_edge)
        for item in inputs:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"Anonymous device {item['anonymous_device']}; evidence={item['evidence_id']}; type={item['kind']}; native pixels preserved={item['native_pixels_preserved']}",
                    },
                    {
                        "type": "input_image",
                        "image_url": encode_image_data_url(
                            Path(item["path"]),
                            self.max_image_edge,
                            lossless=item["kind"] == "native_face",
                        ),
                    },
                ]
            )
        endpoint = (
            f"{self.base_url}/responses"
            if self.base_url.endswith("/v1")
            else f"{self.base_url}/v1/responses"
        )
        response = httpx.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": self.max_output_tokens,
                "text": {"format": {"type": "json_object"}},
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                error_payload = response.json()
                error_message = error_payload.get("error", {}).get("message")
            except (ValueError, AttributeError):
                error_message = None
            detail = f": {error_message}" if error_message else ""
            raise RuntimeError(
                f"OpenAI Responses API returned HTTP {response.status_code}{detail}"
            ) from exc
        raw = response.json()
        parsed = parse_json_object(extract_responses_output_text(raw))
        parsed.update({"scene_id": scene_id, "role": self.role, "raw": raw})
        _record_result_inputs(parsed, inputs, self.model, self.base_url)
        return ModelEvaluationResult.model_validate(parsed)

    def _prompt(self, scene_id: str, context: dict[str, Any]) -> str:
        contract = {
            "observations": [
                {
                    "device_id": "anonymous device code such as A",
                    "dimension": "stable snake_case rendering dimension",
                    "region": "whole or face:F001 or a locatable region; unknown if not observed",
                    "attribute": "encoded_luma_mean, encoded_near_white_ratio, or specific visual attribute",
                    "direction": "higher|lower|equal|unknown",
                    "comparison": "peer_median or anonymous peer code or unknown",
                    "strength": "visible magnitude or unknown",
                    "conditions": ["applicable capture/rendering conditions"],
                    "statement": "visible, evidence-based comparison",
                    "evidence_refs": ["asset or metric reference"],
                    "certainty": "number from 0 to 1",
                }
            ],
            "scores": {"optional_dimension": "integer from 0 to 100"},
            "hypotheses": [
                {
                    "statement": "possible mechanism, never a proprietary implementation fact",
                    "evidence_refs": ["supporting observation or metric"],
                    "alternatives": ["plausible alternative explanation"],
                    "confidence": "number from 0 to 1",
                }
            ],
            "confidence": "number from 0 to 1",
        }
        return (
            "You are evaluating anonymous front-camera portrait JPEG/HEIC outputs. "
            "Judge only visible rendering properties: face readability, highlight integrity, shadow/black "
            "rendering, skin/white-balance relation, lighting dimensionality, face/background relation, "
            "local face lift naturalness, multi-face consistency, scene adaptability, and artifact/texture "
            "control when applicable. Separate quality observations, style preferences, and mechanism "
            "hypotheses. Do not identify brands or infer identity, ethnicity, health, attractiveness, or other "
            "sensitive personal traits. Do not claim that final images prove a proprietary ISP mechanism. "
            "Return one JSON object only and include every top-level field in this contract: "
            f"{json.dumps(contract, ensure_ascii=False)}. Scene={scene_id}. "
            f"Evaluation context={json.dumps(_visible_context(context), ensure_ascii=False)}"
        )


def build_vision_adapters(settings: Settings) -> tuple[VisionModelAdapter, VisionModelAdapter]:
    """Select the legacy local branch or the official OpenAI Responses branch."""
    if settings.vlm_provider == "openai":
        api_key = resolve_openai_api_key(
            settings.openai_api_key,
            getattr(settings, "openai_auth_file", None),
        )
        if not api_key:
            raise ValueError(
                "OpenAI vision provider selected but no API key is configured. Set "
                "PORTRAIT_EVAL_OPENAI_API_KEY or OPENAI_API_KEY, or provide an "
                "OPENAI_API_KEY field in ~/.codex/auth.json."
            )
        openai_primary = OpenAIResponsesVisionAdapter(
            api_key,
            settings.openai_primary_model,
            "primary",
            base_url=settings.openai_base_url,
            max_image_edge=settings.vlm_max_image_edge,
            max_output_tokens=settings.openai_max_output_tokens,
        )
        openai_reviewer = OpenAIResponsesVisionAdapter(
            api_key,
            settings.openai_reviewer_model,
            "reviewer",
            base_url=settings.openai_base_url,
            max_image_edge=settings.vlm_max_image_edge,
            max_output_tokens=settings.openai_max_output_tokens,
        )
        return openai_primary, openai_reviewer

    primary: VisionModelAdapter = (
        OpenAICompatibleVisionAdapter(
            settings.primary_vlm_url,
            settings.primary_vlm_model,
            "primary",
            max_image_edge=settings.vlm_max_image_edge,
        )
        if settings.primary_vlm_url
        else HeuristicVisionAdapter("primary")
    )
    reviewer: VisionModelAdapter = (
        OpenAICompatibleVisionAdapter(
            settings.reviewer_vlm_url,
            settings.reviewer_vlm_model,
            "reviewer",
            max_image_edge=settings.vlm_max_image_edge,
        )
        if settings.reviewer_vlm_url
        else HeuristicVisionAdapter("reviewer")
    )
    return primary, reviewer


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
                    statement="This output has higher encoded global luma than the group median; this is a rendered-output measurement, not a quality preference.",
                    region="whole",
                    attribute="encoded_luma_mean",
                    direction="higher",
                    comparison="peer_median",
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
            model_id="heuristic_metrics_0.2",
            model_family="deterministic_metrics",
        )
