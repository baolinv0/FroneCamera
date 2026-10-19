from __future__ import annotations

from abc import ABC, abstractmethod
from urllib.parse import urlparse

import httpx

from portrait_eval.models import ExternalEvidence


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[ExternalEvidence]:
        raise NotImplementedError


class DisabledSearchProvider(SearchProvider):
    def search(self, query: str, limit: int = 5) -> list[ExternalEvidence]:
        return []


class SearxNGSearchProvider(SearchProvider):
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")

    def search(self, query: str, limit: int = 5) -> list[ExternalEvidence]:
        response = httpx.get(
            f"{self.endpoint}/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=30,
        )
        response.raise_for_status()
        evidence = []
        for item in response.json().get("results", [])[:limit]:
            url = item.get("url", "")
            evidence.append(
                ExternalEvidence(
                    title=item.get("title", "Untitled"),
                    url=url,
                    source_domain=urlparse(url).netloc,
                    snippet=item.get("content", ""),
                )
            )
        return evidence


def build_corroboration_queries(device_name: str, statement: str) -> list[str]:
    return [
        f'"{device_name}" front camera selfie review {statement}',
        f'"{device_name}" 前置摄像头 专业评测 {statement}',
        f'"{device_name}" selfie HDR skin tone low light review',
    ]


def grade_source(domain: str, title: str = "", snippet: str = "") -> dict[str, object]:
    text = f"{title} {snippet}".casefold()
    domain = domain.casefold()
    official_domains = {
        "apple.com",
        "vivo.com",
        "samsung.com",
        "google.com",
        "mi.com",
        "oppo.com",
        "oneplus.com",
        "honor.com",
        "huawei.com",
        "qualcomm.com",
        "mediatek.com",
        "sony.com",
    }
    laboratory_domains = {
        "dxomark.com",
        "image-engineering.de",
        "ifixit.com",
        "techinsights.com",
    }
    professional_domains = {
        "dpreview.com",
        "gsmarena.com",
        "notebookcheck.net",
        "tomsguide.com",
        "techradar.com",
    }
    if any(domain == item or domain.endswith(f".{item}") for item in official_domains):
        tier = 1
    elif any(domain == item or domain.endswith(f".{item}") for item in laboratory_domains):
        tier = 2
    elif any(domain == item or domain.endswith(f".{item}") for item in professional_domains):
        tier = 3
    else:
        tier = 4
    front_specific = any(token in text for token in ("front camera", "selfie", "前置", "自拍"))
    return {"source_tier": tier, "front_camera_specific": front_specific}
