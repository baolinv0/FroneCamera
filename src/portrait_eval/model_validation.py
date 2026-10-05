"""Runtime model contracts and anonymous request traces shared by all transports."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any

from portrait_eval.models import KNOWN_DIMENSIONS, ModelEvaluationResult

_PATH_FIELDS = {
    "path",
    "paths",
    "filename",
    "filenames",
    "folder",
    "folders",
    "folder_path",
    "folder_paths",
    "exif",
    "artifact",
    "artifacts",
    "diagnostic",
    "diagnostics",
}


# Identity maps are part of the request contract, not inferred from a key's
# spelling. All other dictionary keys are schema, including future core facts.
_SCIENTIFIC_FIELDS = {"facts", "objective", "dimensions", "scores", "whole", "regions", "per_face"}
_IDENTITY_FIELDS = {"id", "asset_id", "device_id", "device_name", "name", "canonical_model"}
_PROTOCOL_FIELDS = {
    "dimension",
    "state",
    "mode",
    "role",
    "encoding",
    "scope",
    "pass",
    "person_correspondence",
    "correspondence",
    "orientation_convention",
}


_STRUCTURAL_VALUES = KNOWN_DIMENSIONS | {
    "srgb",
    "linear",
    "algorithm",
    "device",
    "unobservable",
    "invalid",
    "measured_proxy",
    "not_applicable",
    "valid",
    "primary",
    "reviewer",
    "heuristic",
    "synthetic",
    "unverified",
    "visual",
    "metric_validation",
    "independent_visual",
    "challenge",
    "exif_transposed",
    "heif_native_display_oriented",
    "front-image",
    "display_referred_srgb_fixed_dtype_range",
    "fixed_unit_range_to_uint8_rounding",
}


def _location_field(key: str) -> bool:
    key = key.casefold()
    return key in _PATH_FIELDS or key.endswith(
        ("_path", "_paths", "_filename", "_filenames", "_folder", "_folders")
    )


def _replace_identities(value: str, identities: dict[str, str]) -> str:
    def prose(text: str) -> str:
        for identity, code in sorted(identities.items(), key=lambda item: -len(item[0])):

            def replacement(match: re.Match[str], code: str = code) -> str:
                return code

            text = re.sub(r"(?<!\w)" + re.escape(identity) + r"(?!\w)", replacement, text)
        return text

    # Ref namespaces, scene IDs and metric/dimension vocabulary are structural.
    # Only the declared device component of a reference is an identity field.
    reference_pattern = re.compile(r"\b(?:asset|metric|iqa|roi):[^\s\"'<>]+")
    pieces = []
    offset = 0
    for match in reference_pattern.finditer(value):
        pieces.append(prose(value[offset : match.start()]))
        parts = match.group().split(":")
        index = 1 if parts[0] == "roi" else 2
        if len(parts) > index:
            parts[index] = identities.get(parts[index], parts[index])
        pieces.append(":".join(parts))
        offset = match.end()
    pieces.append(prose(value[offset:]))
    return "".join(pieces)


def _fact_key(key: str, identities: dict[str, str]) -> str:
    """Core ROI facts may prefix an identifier; the metric suffix is schema."""
    if key.startswith(("asset:", "metric:", "iqa:", "roi:")):
        return _replace_identities(key, identities)
    if ":" in key:
        device, suffix = key.split(":", 1)
        return identities.get(device, device) + ":" + suffix
    return key


def _identity_map_field(key: str, value: Any, container_field: str) -> bool:
    """Recognize declared device-map shapes at their contract boundaries."""
    if not isinstance(value, dict):
        return False
    if key == "metrics" and not container_field:
        # EvaluationPipeline's outer request metrics: device code -> image metrics.
        return all(isinstance(item, dict) for item in value.values())
    if key == "mapping" and not container_field:
        # Repository anonymous mapping payload: device ID -> anonymous code.
        return all(isinstance(item, str) for item in value.values())
    if key == "devices":
        # Supported record map; pairing's device list uses explicit ID fields.
        return all(isinstance(item, dict) for item in value.values())
    # ModelEvaluationResult.scores is always dimension -> integer, not device -> score.
    return False


def _anonymous_context(
    value: Any,
    identities: dict[str, str],
    *,
    field: str = "",
    scientific: bool = False,
    identity_map: bool = False,
) -> Any:
    if isinstance(value, dict):
        result = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            # A declared device-map key is an identity even when a camera is
            # named "whole" or "path". Else keys are schema, never prose.
            if identity_map:
                anonymous_key = identities.get(key, key)
            elif _location_field(key):
                continue
            elif field == "facts":
                anonymous_key = _fact_key(key, identities)
            else:
                anonymous_key = key
            child_scientific = scientific or identity_map or key in _SCIENTIFIC_FIELDS
            result[anonymous_key] = _anonymous_context(
                item,
                identities,
                field=key,
                scientific=child_scientific,
                identity_map=not scientific
                and not identity_map
                and _identity_map_field(key, item, field),
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _anonymous_context(item, identities, field=field, scientific=scientific)
            for item in value
        ]
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                serialized = json.loads(value)
            except ValueError:
                serialized = None
            if isinstance(serialized, (dict, list)):
                return json.dumps(
                    _anonymous_context(serialized, identities, field=field, scientific=scientific),
                    ensure_ascii=False,
                    allow_nan=False,
                )
        if field in _IDENTITY_FIELDS and value in identities:
            return identities[value]
        if value.startswith(("/", "file:", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            return "[location omitted]"
        if value in _STRUCTURAL_VALUES or field in _PROTOCOL_FIELDS:
            return value
        value = _replace_identities(value, identities)
        value = re.sub(
            r"(?:file://)?(?:\\\\[^\\\s]+[\\/]|[A-Za-z]:[\\/]|/)[^\s\"'<>]+",
            "[location omitted]",
            value,
        )
        return value
    return value


def anonymous_context(value: Any, identities: dict[str, str] | None = None) -> Any:
    """Redact locations/identities while preserving scientific schema recursively."""
    return _anonymous_context(value, identities or {})


def semantic_relation(primary: str, reviewer: str) -> str:
    first, second = " ".join(primary.casefold().split()), " ".join(reviewer.casefold().split())
    if first == second:
        return "support"
    # Explicit inverse phrasing can establish opposition; general free text is unknown.
    pairs = (
        ("brighter", "darker"),
        ("higher", "lower"),
        ("highest", "lowest"),
        ("more", "less"),
        ("better", "worse"),
    )
    for positive, negative in pairs:
        if first.replace(positive, "<polarity>") == second.replace(
            negative, "<polarity>"
        ) or first.replace(negative, "<polarity>") == second.replace(positive, "<polarity>"):
            return "opposition"
    if first == second.replace(" not ", " ") or second == first.replace(" not ", " "):
        return "opposition"
    return "unknown"


def request_trace(
    prompt: str, codes_and_urls: list[tuple[str, str]], payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "images": [
            {
                "device_code": code,
                "encoding": "image/jpeg",
                "encoded_sha256": hashlib.sha256(
                    base64.b64decode(url.split(",", 1)[1])
                ).hexdigest(),
                "data_url_sha256": hashlib.sha256(url.encode()).hexdigest(),
            }
            for code, url in codes_and_urls
        ],
        "request_sha256": hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode()
        ).hexdigest(),
    }


def evidence_catalog(scene_id: str, codes: list[str], context: dict[str, Any]) -> set[str]:
    refs = {f"asset:{scene_id}:{code}" for code in codes}

    def visit(value: Any, code: str, prefix: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                visit(item, code, name)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                refs.add(f"metric:{scene_id}:{code}:{name}")
                # Historic Front metric references omit the whole/face section.
                refs.add(f"metric:{scene_id}:{code}:{key}")

    for code, metrics in context.get("metrics", {}).items():
        if code in codes:
            visit(metrics, code)
    for asset in context.get("iqa_evidence", {}).get("assets", []):
        code = asset.get("id")
        if code not in codes or asset.get("state") != "valid":
            continue
        for dimension, observation in asset.get("dimensions", {}).items():
            if observation.get("state") != "measured_proxy":
                continue
            for fact, value in observation.get("facts", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    refs.add(f"iqa:{scene_id}:{code}:{dimension}:{fact}")
        for fact, value in asset.get("objective", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                refs.add(f"iqa:{scene_id}:{code}:objective:{fact}")
    return refs


def validate_model_result(
    result: ModelEvaluationResult, scene_id: str, codes: list[str], context: dict[str, Any]
) -> ModelEvaluationResult:
    # Revalidation also rejects model_copy updates that bypass Pydantic validation.
    result = ModelEvaluationResult.model_validate(result.model_dump())
    if result.scene_id != scene_id:
        raise ValueError("Model response scene does not match the request")
    known_refs = evidence_catalog(scene_id, codes, context)
    for observation in result.observations:
        if observation.device_id not in codes:
            raise ValueError("Model response contains an unknown device")
        if any(ref not in known_refs for ref in observation.evidence_refs):
            raise ValueError("Model response contains undeclared evidence")
    for hypothesis in result.hypotheses:
        if any(ref not in known_refs for ref in hypothesis.evidence_refs):
            raise ValueError("Model hypothesis contains undeclared evidence")
    return result


def decode_model_result(
    parsed: dict[str, Any],
    scene_id: str,
    role: str,
    codes: list[str],
    context: dict[str, Any],
    raw: dict[str, Any],
    trace: dict[str, Any],
) -> ModelEvaluationResult:
    if "scene_id" in parsed and parsed["scene_id"] != scene_id:
        raise ValueError("Model response scene does not match the request")
    if "role" in parsed and parsed["role"] != role:
        raise ValueError("Model response role does not match the request")
    # Transport metadata is local evidence, never accepted from a remote model.
    if any(key in parsed for key in ("raw", "input_trace", "provisional", "claim_id")):
        raise ValueError("Model response contains reserved transport fields")
    if any(
        item.get("claim_id") for item in parsed.get("observations", []) if isinstance(item, dict)
    ):
        raise ValueError("Claim identity must be computed locally")
    result = ModelEvaluationResult.model_validate(
        {**parsed, "scene_id": scene_id, "role": role, "raw": raw, "input_trace": trace}
    )
    return validate_model_result(result, scene_id, codes, context)
