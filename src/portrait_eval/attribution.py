from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AttributionCase(BaseModel):
    statement: str
    primary_layer: str
    candidate_causes: list[str]
    alternative_explanations: list[str]
    confidence: float = Field(ge=0, le=1)
    evidence_grade: str
    hardware_evidence_count: int = 0
    exif_evidence_keys: list[str] = Field(default_factory=list)


def attribute_output_claim(
    statement: str,
    hardware_context: dict[str, Any],
    exif_context: dict[str, Any],
) -> AttributionCase:
    text = statement.casefold()
    candidate_causes: list[str]
    alternatives: list[str]
    primary_layer = "indeterminate"
    confidence = 0.45

    if "global luminance" in text or "brighter" in text or "higher display-referred" in text:
        candidate_causes = [
            "capture exposure",
            "global tone mapping",
            "shadow and midtone rendering",
        ]
        alternatives = ["scene timing variation", "framing difference", "screen fill light"]
    elif "clipping" in text or "highlight" in text:
        candidate_causes = [
            "capture exposure",
            "sensor headroom",
            "HDR reconstruction",
            "highlight tone mapping",
        ]
        alternatives = [
            "different highlight area",
            "gray compression without real detail",
            "scene motion",
        ]
    elif "noise" in text or "detail" in text or "texture" in text:
        candidate_causes = [
            "sensor signal quality",
            "exposure time",
            "multi-frame denoising",
            "sharpening",
            "beautification",
        ]
        alternatives = ["focus error", "motion blur", "compression"]
    elif "skin" in text or "color" in text or "warm" in text:
        candidate_causes = [
            "white balance",
            "color correction",
            "skin-tone mapping",
            "local semantic rendering",
        ]
        alternatives = ["ambient light change", "display conversion", "makeup or skin reflectance"]
    else:
        candidate_causes = ["capture strategy", "reconstruction", "rendering"]
        alternatives = ["capture variation", "scene mismatch"]

    has_exif = bool(exif_context)
    hardware_sources = hardware_context.get("sources", [])
    hardware_evidence_count = len(hardware_sources) if isinstance(hardware_sources, list) else 0
    has_hardware = bool(hardware_context)
    exif_evidence_keys = sorted(
        key
        for key in ("ExposureTime", "ISO", "ISOSpeedRatings", "FNumber", "Flash")
        if key in exif_context
    )
    if has_exif and any(key in exif_context for key in ("ExposureTime", "ISOSpeedRatings", "ISO")):
        primary_layer = "capture_or_rendering"
        confidence = 0.62
    if has_hardware and "front_focus" in hardware_context and ("focus" in text or "detail" in text):
        primary_layer = "hardware_software_joint"
        confidence = 0.68
    evidence_grade = "B" if confidence >= 0.6 else "C"
    return AttributionCase(
        statement=statement,
        primary_layer=primary_layer,
        candidate_causes=candidate_causes,
        alternative_explanations=alternatives,
        confidence=confidence,
        evidence_grade=evidence_grade,
        hardware_evidence_count=hardware_evidence_count,
        exif_evidence_keys=exif_evidence_keys,
    )
