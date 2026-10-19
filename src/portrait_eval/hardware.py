from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceTier(IntEnum):
    OFFICIAL = 1
    REGULATORY_OR_TEARDOWN = 2
    PROFESSIONAL_REVIEW = 3
    COMMUNITY = 4


class HardwareFact(BaseModel):
    field_name: str
    value: Any
    source_url: str
    source_tier: SourceTier
    source_title: str = ""
    region: str | None = None
    confidence: float = Field(default=0.8, ge=0, le=1)


class HardwareProfile(BaseModel):
    values: dict[str, Any]
    selected_sources: dict[str, str]
    conflicts: dict[str, list[Any]]
    unresolved_fields: list[str] = Field(default_factory=list)


def reconcile_hardware_facts(facts: list[HardwareFact]) -> HardwareProfile:
    grouped: dict[str, list[HardwareFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.field_name, []).append(fact)
    values: dict[str, Any] = {}
    selected_sources: dict[str, str] = {}
    conflicts: dict[str, list[Any]] = {}
    for field_name, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda item: (int(item.source_tier), -item.confidence))
        selected = ordered[0]
        values[field_name] = selected.value
        selected_sources[field_name] = selected.source_url
        distinct = []
        for candidate in ordered:
            if candidate.value not in distinct:
                distinct.append(candidate.value)
        if len(distinct) > 1:
            conflicts[field_name] = distinct
    return HardwareProfile(values=values, selected_sources=selected_sources, conflicts=conflicts)


def build_hardware_queries(device_name: str) -> list[str]:
    return [
        f'"{device_name}" official front camera specifications',
        f'"{device_name}" front camera sensor autofocus aperture field of view',
        f'"{device_name}" 前置摄像头 传感器 对焦 光圈 规格',
    ]
