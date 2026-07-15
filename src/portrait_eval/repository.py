from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from portrait_eval.database import (
    AnalysisRow,
    DeviceRow,
    ImageRow,
    PairingCellRow,
    PairingSnapshotRow,
    ProjectRow,
    ReportRow,
    ReviewItemRow,
    SceneGroupRow,
    json_dump,
    json_load,
)
from portrait_eval.dataset import inspect_image, propose_pairing, scan_folder
from portrait_eval.models import DeviceScan, ProjectStatus


class Repository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_project(self, name: str) -> ProjectRow:
        row = ProjectRow(name=name)
        self.session.add(row)
        self.session.commit()
        return row

    def list_projects(self) -> list[ProjectRow]:
        return list(self.session.scalars(select(ProjectRow).order_by(ProjectRow.created_at.desc())))

    def get_project(self, project_id: str) -> ProjectRow:
        row = self.session.get(ProjectRow, project_id)
        if row is None:
            raise KeyError(project_id)
        return row

    def add_device(
        self, project_id: str, name: str, folder_path: str, canonical_model: str | None = None
    ) -> DeviceRow:
        self.get_project(project_id)
        folder = Path(folder_path).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")
        row = DeviceRow(
            project_id=project_id,
            name=name,
            folder_path=str(folder),
            canonical_model=canonical_model,
        )
        self.session.add(row)
        self.session.commit()
        return row

    def scan_and_pair(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        devices = list(
            self.session.scalars(select(DeviceRow).where(DeviceRow.project_id == project_id))
        )
        if len(devices) < 2:
            raise ValueError("At least two devices are required")
        self.session.execute(
            delete(PairingCellRow).where(
                PairingCellRow.group_id.in_(
                    select(SceneGroupRow.id).where(SceneGroupRow.project_id == project_id)
                )
            )
        )
        self.session.execute(delete(SceneGroupRow).where(SceneGroupRow.project_id == project_id))
        scans: list[DeviceScan] = []
        image_lookup: dict[tuple[str, str], ImageRow] = {}
        for device in devices:
            self.session.execute(delete(ImageRow).where(ImageRow.device_id == device.id))
            scan = scan_folder(device.id, Path(device.folder_path))
            scans.append(scan)
            for index, path in enumerate(scan.files):
                info = inspect_image(path)
                image = ImageRow(
                    device_id=device.id,
                    path=str(info["path"]),
                    filename=str(info["filename"]),
                    sequence_index=index,
                    checksum=str(info["checksum"]),
                    width=int(info["width"]),
                    height=int(info["height"]),
                    exif_json=json_dump(info["exif"]),
                )
                self.session.add(image)
                self.session.flush()
                image_lookup[(device.id, str(path))] = image
        draft = propose_pairing(scans)
        for index, group in enumerate(draft.groups):
            row = SceneGroupRow(
                project_id=project_id, group_key=group.group_id, sequence_index=index
            )
            self.session.add(row)
            self.session.flush()
            for device in devices:
                paired_path = group.cells[device.id]
                paired_image = (
                    image_lookup.get((device.id, str(paired_path))) if paired_path else None
                )
                self.session.add(
                    PairingCellRow(
                        group_id=row.id,
                        device_id=device.id,
                        image_id=paired_image.id if paired_image else None,
                    )
                )
        project.status = ProjectStatus.PAIRING_REQUIRED.value
        project.version += 1
        self.session.commit()
        return self.get_pairing(project_id)

    def get_pairing(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        devices = list(
            self.session.scalars(
                select(DeviceRow).where(DeviceRow.project_id == project_id).order_by(DeviceRow.name)
            )
        )
        groups = list(
            self.session.scalars(
                select(SceneGroupRow)
                .where(SceneGroupRow.project_id == project_id)
                .order_by(SceneGroupRow.sequence_index)
            )
        )
        result_groups = []
        for group in groups:
            cells = list(
                self.session.scalars(
                    select(PairingCellRow).where(PairingCellRow.group_id == group.id)
                )
            )
            cell_map = {cell.device_id: cell for cell in cells}
            result_cells = {}
            for device in devices:
                cell = cell_map.get(device.id)
                image = (
                    self.session.get(ImageRow, cell.image_id) if cell and cell.image_id else None
                )
                result_cells[device.id] = (
                    None
                    if image is None
                    else {
                        "image_id": image.id,
                        "filename": image.filename,
                        "path": image.path,
                        "width": image.width,
                        "height": image.height,
                        "exif": json_load(image.exif_json),
                    }
                )
            result_groups.append(
                {
                    "id": group.id,
                    "group_id": group.group_key,
                    "label": group.label,
                    "cells": result_cells,
                    "analyzable": sum(v is not None for v in result_cells.values()) >= 2,
                }
            )
        available_images: dict[str, list[dict[str, Any]]] = {}
        for device in devices:
            images = self.session.scalars(
                select(ImageRow)
                .where(ImageRow.device_id == device.id)
                .order_by(ImageRow.sequence_index)
            )
            available_images[device.id] = [
                {
                    "image_id": image.id,
                    "filename": image.filename,
                    "path": image.path,
                    "width": image.width,
                    "height": image.height,
                    "exif": json_load(image.exif_json),
                }
                for image in images
            ]
        return {
            "project_id": project_id,
            "version": project.version,
            "confirmed": project.status == ProjectStatus.PAIRING_CONFIRMED.value,
            "devices": [
                {
                    "id": device.id,
                    "name": device.name,
                    "folder_path": device.folder_path,
                    "canonical_model": device.canonical_model,
                }
                for device in devices
            ],
            "groups": result_groups,
            "available_images": available_images,
        }

    def update_pairing_cell(
        self,
        project_id: str,
        group_id: str,
        device_id: str,
        image_id: str | None,
        expected_version: int,
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        if project.version != expected_version:
            raise RuntimeError("version_conflict")
        group = self.session.scalar(
            select(SceneGroupRow).where(
                SceneGroupRow.id == group_id, SceneGroupRow.project_id == project_id
            )
        )
        if group is None:
            raise KeyError(group_id)
        cell = self.session.scalar(
            select(PairingCellRow).where(
                PairingCellRow.group_id == group_id, PairingCellRow.device_id == device_id
            )
        )
        if cell is None:
            raise KeyError(device_id)
        if image_id:
            image = self.session.get(ImageRow, image_id)
            if image is None or image.device_id != device_id:
                raise ValueError("image does not belong to device")
        cell.image_id = image_id
        project.version += 1
        project.status = ProjectStatus.PAIRING_REQUIRED.value
        self.session.commit()
        return self.get_pairing(project_id)

    def confirm_pairing(self, project_id: str, expected_version: int) -> dict[str, Any]:
        project = self.get_project(project_id)
        if project.version != expected_version:
            raise RuntimeError("version_conflict")
        groups = list(
            self.session.scalars(
                select(SceneGroupRow).where(SceneGroupRow.project_id == project_id)
            )
        )
        if not groups:
            raise ValueError("No pairing groups")
        project.status = ProjectStatus.PAIRING_CONFIRMED.value
        project.version += 1
        for group in groups:
            group.confirmed = True
            group.version = project.version
        self.session.flush()
        snapshot_payload = self.get_pairing(project_id)
        self.session.add(
            PairingSnapshotRow(
                project_id=project_id,
                version=project.version,
                payload_json=json_dump(snapshot_payload),
            )
        )
        self.session.commit()
        return snapshot_payload

    def list_pairing_snapshots(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        rows = self.session.scalars(
            select(PairingSnapshotRow)
            .where(PairingSnapshotRow.project_id == project_id)
            .order_by(PairingSnapshotRow.version.desc())
        )
        return [
            {
                "id": row.id,
                "version": row.version,
                "payload": json_load(row.payload_json),
                "created_at": row.created_at,
            }
            for row in rows
        ]

    def save_analysis(
        self,
        project_id: str,
        kind: str,
        payload: object,
        scene_group_id: str | None = None,
        image_id: str | None = None,
    ) -> AnalysisRow:
        row = AnalysisRow(
            project_id=project_id,
            kind=kind,
            payload_json=json_dump(payload),
            scene_group_id=scene_group_id,
            image_id=image_id,
        )
        self.session.add(row)
        self.session.commit()
        return row

    def list_analysis(self, project_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        query = select(AnalysisRow).where(AnalysisRow.project_id == project_id)
        if kind:
            query = query.where(AnalysisRow.kind == kind)
        return [
            {
                "id": row.id,
                "kind": row.kind,
                "scene_group_id": row.scene_group_id,
                "image_id": row.image_id,
                "payload": json_load(row.payload_json),
            }
            for row in self.session.scalars(query)
        ]

    def create_review_item(
        self, project_id: str, category: str, payload: object, priority: str = "medium"
    ) -> ReviewItemRow:
        row = ReviewItemRow(
            project_id=project_id,
            category=category,
            priority=priority,
            payload_json=json_dump(payload),
        )
        self.session.add(row)
        self.session.commit()
        return row

    def list_review_items(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(ReviewItemRow).where(ReviewItemRow.project_id == project_id)
        )
        return [
            {
                "id": row.id,
                "category": row.category,
                "priority": row.priority,
                "status": row.status,
                "payload": json_load(row.payload_json),
            }
            for row in rows
        ]

    def resolve_review_item(
        self, review_id: str, status: str, note: str | None = None
    ) -> dict[str, Any]:
        row = self.session.get(ReviewItemRow, review_id)
        if row is None:
            raise KeyError(review_id)
        payload = cast(dict[str, Any], json_load(row.payload_json))
        if note:
            payload["review_note"] = note
        row.payload_json = json_dump(payload)
        row.status = status
        self.session.commit()
        return {"id": row.id, "status": row.status, "payload": payload}

    def list_reports(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(ReportRow)
            .where(ReportRow.project_id == project_id)
            .order_by(ReportRow.created_at.desc())
        )
        return [
            {
                "id": row.id,
                "version": row.version,
                "status": row.status,
                "html_path": row.html_path,
                "created_at": row.created_at,
            }
            for row in rows
        ]

    def save_report(self, project_id: str, version: str, status: str, html_path: str) -> ReportRow:
        row = ReportRow(project_id=project_id, version=version, status=status, html_path=html_path)
        self.session.add(row)
        self.session.commit()
        return row
