"""Read-only verification of a run against its confirmed pairing and source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from portrait_eval.database import PairingSnapshotRow, SceneGroupRow
from portrait_eval.repository import Repository


def validate_run_binding(repo: Repository, project_id: str, binding: dict[str, Any]) -> None:
    """Raise ValueError for malformed or stale run inputs without committing changes.

    Callers provide payload.input_trace.run_binding and decide compatibility for
    legacy payloads without it. The binding must contain pairing_snapshot_id,
    pairing_version, pairing_sha256 (canonical sorted compact JSON), and the exact
    image-id/source-checksum mapping from that snapshot. File paths come only from
    the persisted snapshot, so public payloads need not expose original paths.
    Project status can change as evaluation progresses; confirmation is verified
    through the immutable snapshot and the current pairing groups/version.
    """
    if not isinstance(binding, dict):
        raise ValueError("Invalid run binding")
    version = binding.get("pairing_version")
    snapshot_id = binding.get("pairing_snapshot_id")
    if type(version) is not int or version < 1 or not isinstance(snapshot_id, str):
        raise ValueError("Invalid run binding pairing version or snapshot")
    checksums = binding.get("source_checksums")
    if not isinstance(checksums, dict):
        raise ValueError("Invalid run binding source checksums")

    with repo.session.no_autoflush:
        try:
            project = repo.get_project(project_id)
        except KeyError as exc:
            raise ValueError("Run binding project not found") from exc
        repo.session.refresh(project)
        if project.version != version:
            raise ValueError("Pairing changed during evaluation; confirm the new pairing snapshot")
        row = repo.session.execute(
            select(PairingSnapshotRow.payload_json).where(
                PairingSnapshotRow.id == snapshot_id,
                PairingSnapshotRow.project_id == project_id,
                PairingSnapshotRow.version == version,
            )
        ).scalar_one_or_none()
        if row is None:
            raise ValueError("Confirmed pairing snapshot is missing or stale")
        try:
            pairing = json.loads(row)
            if (
                not pairing["confirmed"]
                or pairing["project_id"] != project_id
                or pairing["version"] != version
                or pairing["pairing_snapshot_id"] != snapshot_id
            ):
                raise ValueError("Confirmed pairing snapshot is missing or stale")
            digest = hashlib.sha256(
                json.dumps(pairing, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if binding.get("pairing_sha256") != digest:
                raise ValueError("Run binding pairing snapshot hash does not match")
            groups = repo.session.execute(
                select(SceneGroupRow.confirmed, SceneGroupRow.version).where(
                    SceneGroupRow.project_id == project_id
                )
            ).all()
            if not groups or any(
                not confirmed or current != version for confirmed, current in groups
            ):
                raise ValueError(
                    "Pairing changed during evaluation; confirm the new pairing snapshot"
                )
            cells = [
                cell for group in pairing["groups"] for cell in group["cells"].values() if cell
            ]
            expected = {cell["image_id"]: cell["checksum"] for cell in cells}
            if not expected or expected != checksums:
                raise ValueError("Run binding source checksums do not match the snapshot")
            for cell in cells:
                try:
                    actual = hashlib.sha256(Path(cell["path"]).read_bytes()).hexdigest()
                except OSError as exc:
                    raise ValueError(
                        "Source bytes changed or are unavailable; re-scan and confirm pairing"
                    ) from exc
                if not cell["checksum"] or actual != cell["checksum"]:
                    raise ValueError(
                        "Source bytes changed during evaluation; re-scan and confirm pairing"
                    )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid confirmed pairing snapshot") from exc
