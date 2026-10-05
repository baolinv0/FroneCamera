"""Deterministic representative auditing is distinct from targeted risk triage."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any


def representative_audit_sample(
    records: list[dict[str, Any]],
    sample_size: int,
    seed: str = "iqa-audit-v1",
    strata_fields: tuple[str, ...] = ("risk", "scene_type", "split"),
) -> dict:
    """Balanced strata sample, with design metadata; never an unbiased accuracy claim.

    Input-order independent SHA ranking with deterministic round-robin strata allocation.
    Risk queue keeps all risky input records, independent of sample size. Missing risk
    is its own unknown stratum and is included in the focused triage queue.
    """
    if type(sample_size) is not int or sample_size < 0:
        raise ValueError("sample_size must be a nonnegative integer")
    if not seed or not strata_fields or len(set(strata_fields)) != len(strata_fields):
        raise ValueError("seed and unique strata fields must be nonempty")
    ids = [r.get("id") for r in records]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("audit records require unique nonempty string ids")
    groups = defaultdict(list)
    for row in records:
        if "risk" in row and type(row["risk"]) is not bool:
            raise ValueError("risk must be boolean when supplied")
        # Validate serializability and finite values, retain original records.
        json.dumps(row, allow_nan=False, sort_keys=True)
        key = json.dumps([row.get(field, "__unknown__") for field in strata_fields], sort_keys=True)
        rank = hashlib.sha256(f"{seed}\0{row['id']}".encode()).hexdigest()
        groups[key].append((rank, row))
    keys = sorted(groups)
    for key in keys:
        groups[key].sort(key=lambda item: (item[0], item[1]["id"]))
    selected, counts = [], {key: 0 for key in keys}
    limit = min(sample_size, len(records))
    # Guarantee clean/risk coverage when at least two draws are requested; ordering of
    # many scene strata otherwise could exhaust the quota before reaching risk rows.
    if limit >= 2 and "risk" in strata_fields:
        for risk in (False, True):
            eligible = [
                (items[0][0], key)
                for key, items in groups.items()
                if items and items[0][1].get("risk") is risk
            ]
            if eligible:
                _, key = min(eligible)
                selected.append(groups[key].pop(0)[1])
                counts[key] += 1
    while len(selected) < limit:
        available = [key for key in keys if groups[key]]
        key = min(available, key=lambda k: (counts[k], k))
        selected.append(groups[key].pop(0)[1])
        counts[key] += 1
    population = {key: len(groups[key]) + counts[key] for key in keys}
    return {
        "purpose": "representative_audit",
        "sample": selected,
        "focused_risk_queue": sorted(
            [r for r in records if r.get("risk") is not False], key=lambda r: r["id"]
        ),
        "sample_design": {
            "seed": seed,
            "strata_fields": list(strata_fields),
            "population_count": len(records),
            "requested_count": sample_size,
            "selected_count": len(selected),
            "stratum_population": population,
            "stratum_selected": counts,
            "allocation": "balanced_strata_sha256_rank",
            "limits": ["not_population_weighted", "no_accuracy_estimate_without_human_labels"],
        },
    }
