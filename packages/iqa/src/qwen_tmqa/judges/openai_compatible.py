from __future__ import annotations

import base64
import json
import math
import re
import time

import httpx

from ..candidate_schema import CandidatePreferenceEvidence, validate_candidate_preference_payload
from ..config import JudgeConfig
from ..domain import ModelEvaluation, ModelIssue, PromptTrace
from ..image_io import encode_image_payload

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_REQUIRED_SCORE_KEYS = {
    "tone",
    "color",
    "fidelity",
    "control",
    "preference",
    "overall",
}
_REQUIRED_RESPONSE_KEYS = {
    "scores",
    "decision",
    "confidence",
    "issues",
    "rationale",
}


def _image_data_url(path: str, width: int | None, height: int | None) -> str:
    payload = encode_image_payload(path, width, height)
    encoded = base64.b64encode(payload.data).decode("ascii")
    return f"data:{payload.mime_type};base64,{encoded}"


def _extract_json(text: str) -> dict[str, object]:
    match = _JSON_FENCE.search(text)
    candidate = match.group(1) if match else text
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise TypeError("model response must be a JSON object")
    return parsed


class OpenAICompatibleJudge:
    def __init__(self, config: JudgeConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        self._client = httpx.Client(
            transport=transport,
            timeout=config.timeout_seconds,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    def _request(self, prompt_trace: PromptTrace) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt_trace.rendered_prompt}]
        for item in prompt_trace.input_manifest:
            payload = encode_image_payload(item.path, item.sent_width, item.sent_height)
            if item.payload_sha256 and payload.sha256 != item.payload_sha256:
                raise ValueError(
                    f"image payload hash mismatch for image {item.index}: "
                    f"trace={item.payload_sha256}, actual={payload.sha256}"
                )
            if item.payload_mime and payload.mime_type != item.payload_mime:
                raise ValueError(
                    f"image payload MIME mismatch for image {item.index}: "
                    f"trace={item.payload_mime}, actual={payload.mime_type}"
                )
            label = (
                f"Image {item.index}: role={item.role}, alpha={item.alpha}, "
                f"level={item.level}, source_sha256={item.source_sha256[:12]}, "
                f"payload_sha256={payload.sha256[:12]}"
            )
            content.append({"type": "text", "text": label})
            encoded = base64.b64encode(payload.data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{payload.mime_type};base64,{encoded}"},
                }
            )
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        response = self._client.post(
            self.config.base_url.rstrip("/") + "/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"]["content"])

    def evaluate(self, prompt_trace: PromptTrace) -> ModelEvaluation:
        started = time.perf_counter()
        raw = ""
        error: str | None = None
        parsed: dict[str, object] | None = None
        for _ in range(2):
            try:
                raw = self._request(prompt_trace)
                parsed = _extract_json(raw)
                break
            except Exception as exc:  # noqa: BLE001 - provider failures remain reviewable
                error = str(exc)
        if parsed is None:
            return ModelEvaluation(
                model_id=self.config.id,
                model_role=self.config.role,
                model_version=self.config.version,
                synthetic=self.config.synthetic,
                available=False,
                prompt_trace=prompt_trace,
                scores={},
                decision="REVIEW",
                confidence=0,
                raw_response=raw,
                parsed_response={},
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error or "unknown judge error",
            )
        try:
            missing_fields = _REQUIRED_RESPONSE_KEYS - set(parsed)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"missing required response fields: {missing}")
            allowed = set(_REQUIRED_RESPONSE_KEYS)
            if prompt_trace.prompt_version == "3.4":
                allowed.update(CandidatePreferenceEvidence.model_fields)
                baseline_ids = [
                    item.level for item in prompt_trace.input_manifest if item.role == "baseline"
                ]
                if len(baseline_ids) != 1:
                    raise ValueError("candidate preference requires one explicit baseline")
                validate_candidate_preference_payload(
                    parsed,
                    expected_levels={
                        item.level
                        for item in prompt_trace.input_manifest
                        if item.role in {"baseline", "candidate"}
                    },
                    baseline_level=baseline_ids[0],
                )
            if set(parsed) - allowed:
                raise ValueError(f"extra response fields: {sorted(set(parsed) - allowed)}")
            score_payload = parsed["scores"]
            if not isinstance(score_payload, dict):
                raise TypeError("scores must be a JSON object")
            missing_scores = _REQUIRED_SCORE_KEYS - set(score_payload)
            if missing_scores:
                missing = ", ".join(sorted(missing_scores))
                raise ValueError(f"missing required score keys: {missing}")
            if set(score_payload) != _REQUIRED_SCORE_KEYS:
                raise ValueError("scores contain unknown dimensions")
            for label, value in [*score_payload.items(), ("confidence", parsed["confidence"])]:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 100
                ):
                    raise ValueError(f"{label} must be finite numeric score in [0,100]")
            issue_payload = parsed["issues"]
            if not isinstance(issue_payload, list):
                raise TypeError("issues must be a JSON array")
            for item in issue_payload:
                if not isinstance(item, dict) or set(item) - set(ModelIssue.model_fields):
                    raise ValueError("issues contain invalid or extra fields")
                severity = item.get("severity")
                if (
                    isinstance(severity, bool)
                    or not isinstance(severity, (int, float))
                    or not math.isfinite(severity)
                    or not 0 <= severity <= 1
                ):
                    raise ValueError("issue severity must be finite numeric [0,1]")
                if "fatal" in item and not isinstance(item["fatal"], bool):
                    raise ValueError("issue fatal must be boolean")
                bbox = item.get("bbox")
                if bbox is not None and (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(
                        isinstance(coordinate, bool)
                        or not isinstance(coordinate, (int, float))
                        or not math.isfinite(coordinate)
                        for coordinate in bbox
                    )
                ):
                    raise ValueError("issue bbox must be null or four finite numeric coordinates")
            issues = [ModelIssue.model_validate(item) for item in issue_payload]
            rationale = parsed["rationale"]
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("rationale must be a non-empty string")
            return ModelEvaluation(
                model_id=self.config.id,
                model_role=self.config.role,
                model_version=self.config.version,
                synthetic=self.config.synthetic,
                prompt_trace=prompt_trace,
                scores=dict(score_payload),
                decision=parsed["decision"],
                confidence=parsed["confidence"],
                issues=issues,
                rationale=rationale,
                raw_response=raw,
                parsed_response=parsed,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - provider failures remain reviewable
            return ModelEvaluation(
                model_id=self.config.id,
                model_role=self.config.role,
                model_version=self.config.version,
                synthetic=self.config.synthetic,
                available=False,
                prompt_trace=prompt_trace,
                scores={},
                decision="REVIEW",
                confidence=0,
                raw_response=raw,
                parsed_response=parsed,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"validation error: {exc}",
            )
