from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from portrait_eval.models import ExternalEvidence
from portrait_eval.vlm import parse_json_object


class CorroborationDecision(BaseModel):
    verdict: Literal["supports", "contradicts", "incomparable", "irrelevant", "unresolved"]
    confidence: float = Field(ge=0, le=1)
    reason: str
    front_camera_specific: bool

    @property
    def supports_claim(self) -> bool | None:
        if self.verdict == "supports":
            return True
        if self.verdict == "contradicts":
            return False
        return None


class CorroborationAdapter(ABC):
    @abstractmethod
    def classify(
        self,
        claim: str,
        device_name: str,
        evidence: ExternalEvidence,
    ) -> CorroborationDecision:
        raise NotImplementedError


class HeuristicCorroborationAdapter(CorroborationAdapter):
    """Safe fallback: filters scope but never fabricates agreement from snippets."""

    def classify(
        self,
        claim: str,
        device_name: str,
        evidence: ExternalEvidence,
    ) -> CorroborationDecision:
        text = f"{evidence.title} {evidence.snippet}".casefold()
        front_specific = any(token in text for token in ("front camera", "selfie", "前置", "自拍"))
        rear_only = (
            any(token in text for token in ("rear camera", "main camera", "后置", "主摄"))
            and not front_specific
        )
        if rear_only or not front_specific:
            return CorroborationDecision(
                verdict="incomparable",
                confidence=0.95 if rear_only else 0.8,
                reason="The retrieved text is not clearly scoped to the front camera.",
                front_camera_specific=False,
            )
        return CorroborationDecision(
            verdict="unresolved",
            confidence=0.6,
            reason=(
                "The source is front-camera specific, but a deterministic keyword rule cannot "
                "reliably decide whether it supports or contradicts the claim."
            ),
            front_camera_specific=True,
        )


class OpenAICompatibleCorroborationAdapter(CorroborationAdapter):
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def classify(
        self,
        claim: str,
        device_name: str,
        evidence: ExternalEvidence,
    ) -> CorroborationDecision:
        prompt = {
            "task": "Classify whether a professional-review search result corroborates an internal front-camera claim.",
            "rules": [
                "Use only the supplied title and snippet.",
                "Return incomparable when the source is rear-camera, another model, or a materially different mode.",
                "Return unresolved when the excerpt is insufficient.",
                "Do not infer proprietary mechanisms.",
            ],
            "device": device_name,
            "claim": claim,
            "source": evidence.model_dump(mode="json"),
            "output_schema": {
                "verdict": "supports|contradicts|incomparable|irrelevant|unresolved",
                "confidence": "0..1",
                "reason": "brief evidence-based explanation",
                "front_camera_specific": "boolean",
            },
        }
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return CorroborationDecision.model_validate(parse_json_object(content))
