"""One publication projection shared by professional and quick reports."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from portrait_eval.reporting import ReportPayload

REJECTED = {"rejected", "insufficient_evidence"}


def _claim_ids(item: dict[str, Any]) -> set[str]:
    ids = item.get("claim_ids") or item.get("supporting_claim_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    return {str(value) for value in ids} | (
        {str(item["claim_id"])} if item.get("claim_id") else set()
    )


def publication_notes(payload: ReportPayload) -> list[str]:
    """Disclose actual adapter evidence even for draft/direct renderer callers."""
    notes = list(payload.provenance_notes)
    markers: set[str] = set()

    def scan(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                scan(item, name.lower())
        elif isinstance(value, list):
            for item in value:
                scan(item, key)
        elif isinstance(value, str):
            for marker in ("heuristic", "synthetic"):
                if marker in value.lower():
                    markers.add(marker)
        elif value is True and key in {"heuristic", "synthetic"}:
            markers.add(key)

    scan(payload.model_dump())
    if "heuristic" in markers:
        notes.append(
            "Heuristic adapter evidence is provisional; it is not independent model consensus."
        )
    if "synthetic" in markers:
        notes.append(
            "Synthetic evidence is demonstration-only; no real capture accuracy is established."
        )
    if not notes:
        notes.append(
            "Adapter provenance is not recorded; evidence validity has not been independently established."
        )
    return list(dict.fromkeys(notes))


def publication_banner(payload: ReportPayload, status: str) -> str:
    if status != "final":
        return "DRAFT — verify review gates before external distribution."
    summary = payload.review_summary
    if payload.evaluation_mode == "quick":
        return f"FINAL REPORT — quick publication; {summary.get('open', 0)} open reviews; mandatory review gates skipped."
    if not summary:
        return "FINAL REPORT — review resolution not recorded."
    return f"FINAL REPORT — {summary.get('open', 0)} open reviews; decisions applied; evidence limitations remain."


def _project_provider_metadata(value: Any) -> Any:
    """Keep audit linkage, never publish unstructured provider bodies or prompts."""
    unsafe_keys = {
        "choices",
        "messages",
        "message",
        "content",
        "prompt",
        "response",
        "response_text",
        "output_text",
        "raw_response",
        "body",
    }
    if isinstance(value, list):
        return [_project_provider_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    projected = {}
    for key, item in value.items():
        if key == "raw":
            encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            metadata = {"response_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
            if isinstance(item, dict):
                for field in (
                    "adapter",
                    "model",
                    "model_name",
                    "provider",
                    "id",
                    "response_id",
                    "usage",
                    "input_trace",
                    "trace",
                    "provenance",
                    "heuristic",
                    "synthetic",
                    "provisional",
                ):
                    if field in item:
                        metadata[field] = _project_provider_metadata(item[field])
            projected[key] = metadata
        elif key not in unsafe_keys:
            projected[key] = _project_provider_metadata(item)
    return projected


def _scope_matches(item: dict[str, Any], target: dict[str, Any], scene: str | None) -> bool:
    if target.get("url") and item.get("url") != target["url"]:
        return False
    dimension = target.get("dimension") or target.get("dimension_id")
    item_dimension = item.get("dimension") or item.get("dimension_id")
    if dimension and item_dimension and dimension != item_dimension:
        return False
    group = target.get("group_id") or target.get("scene_id")
    groups = item.get("supporting_scene_ids") or item.get("evidence") or []
    return not (
        group
        and group != (item.get("group_id") or item.get("scene_id") or scene)
        and group not in groups
    )


def resolve_report_payload(
    payload: ReportPayload, reviews: list[dict[str, Any]], mode: str = "professional"
) -> ReportPayload:
    data = _project_provider_metadata(payload.model_dump(mode="json"))
    denied_ids: set[str] = set()
    legacy: list[dict[str, Any]] = []
    stable_targets: list[dict[str, Any]] = []
    for review in reviews:
        if review.get("status") not in REJECTED:
            continue
        item = review.get("payload")
        if not isinstance(item, dict):
            continue
        denied_ids.update(_claim_ids(item))
        observation = item.get("primary_observation", item)
        if isinstance(observation, dict):
            denied_ids.update(_claim_ids(observation))
            target = dict(observation)
            if item.get("group_id"):
                target.setdefault("group_id", item["group_id"])
            if target.get("statement") or target.get("claim_statement"):
                ids = _claim_ids(item) | _claim_ids(observation)
                if ids:
                    target["claim_ids"] = sorted(ids)
                    target["_stable"] = True
                    stable_targets.append(target)
                else:
                    legacy.append(target)

    def matching_devices(value: Any, target: dict[str, Any], scene: str | None = None) -> set[str]:
        if isinstance(value, list):
            return set().union(*(matching_devices(item, target, scene) for item in value))
        if isinstance(value, dict):
            scene = value.get("group_id") or value.get("scene_id") or scene
            devices = (
                {str(value["device_id"])}
                if (
                    value.get("device_id")
                    and (target.get("statement") or target.get("claim_statement"))
                    == (value.get("statement") or value.get("claim_statement"))
                    and _scope_matches(value, target, scene)
                )
                else set()
            )
            return devices | set().union(
                *(matching_devices(item, target, scene) for item in value.values())
            )
        return set()

    for target in legacy:
        if target.get("device_id") or _claim_ids(target):
            continue
        devices = matching_devices(data, target)
        if len(devices) > 1:
            raise ValueError(
                "Rejected legacy claim has ambiguous device identity; a stable claim ID is required"
            )
        if devices:
            target["device_id"] = devices.pop()

    located_ids: set[str] = set()

    def collect_denied(value: Any, scene: str | None = None) -> None:
        if isinstance(value, list):
            for item in value:
                collect_denied(item, scene)
        elif isinstance(value, dict):
            scene = value.get("group_id") or value.get("scene_id") or scene
            if _claim_ids(value) & denied_ids and (
                value.get("statement") or value.get("claim_statement")
            ):
                located_ids.update(_claim_ids(value) & denied_ids)
                target = dict(value)
                if scene:
                    target.setdefault("group_id", scene)
                target["_stable"] = True
                legacy.append(target)
            for item in value.values():
                collect_denied(item, scene)

    collect_denied(data)
    legacy.extend(target for target in stable_targets if not (_claim_ids(target) & located_ids))

    def rejected(item: dict[str, Any], scene: str | None = None) -> bool:
        if _claim_ids(item) & denied_ids:
            return True
        statement = item.get("statement") or item.get("claim_statement")
        for target in legacy:
            if target.get("_stable") and _claim_ids(item):
                continue
            if statement != (target.get("statement") or target.get("claim_statement")):
                continue
            # An ambiguous text-only legacy review must never reject a different device.
            if target.get("device_id") and item.get("device_id") != target["device_id"]:
                continue
            if not target.get("device_id") and item.get("device_id"):
                continue
            if not _scope_matches(item, target, scene):
                continue
            return True
        return False

    def clean(value: Any, scene: str | None = None) -> Any:
        if isinstance(value, list):
            return [
                clean(item, scene)
                for item in value
                if not isinstance(item, dict) or not rejected(item, scene)
            ]
        if isinstance(value, dict):
            scene = value.get("group_id") or value.get("scene_id") or scene
            return {
                key: clean(item, scene)
                for key, item in value.items()
                if not isinstance(item, dict) or not rejected(item, scene)
            }
        return value

    projected = clean(data)
    # Profiles are derived prose: rebuild affected device summaries from surviving claims.
    for profile in projected.get("device_profiles", []):
        identity = profile.get("device_id")
        affected = any(target.get("device_id") == identity for target in legacy) or bool(denied_ids)
        if affected:
            statements = [
                item["statement"]
                for item in projected["findings"]
                if item.get("device_id") == identity and item.get("claim_type") == "strategy"
            ]
            profile["summary"] = (
                "；".join(statements[:2]) or "No approved cross-scene profile evidence."
            )
            profile["label"] = "Evidence Profile"
    counts = Counter(str(item.get("status", "open")) for item in reviews)
    projected["review_summary"] = {
        "total": len(reviews),
        "open": counts["open"],
        "skipped_mandatory_gates": mode == "quick",
        "decisions": dict(counts),
    }
    projected["evaluation_mode"] = mode
    resolved = ReportPayload.model_validate(projected)
    resolved.provenance_notes = publication_notes(payload)
    if mode == "quick":
        resolved.limitations.append(
            f"Quick mode: {counts['open']} unresolved reviews; mandatory review gates skipped."
        )
    return resolved
