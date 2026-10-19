from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from portrait_eval.database import DeviceRow, ImageRow, ProjectRow
from portrait_eval.repository import Repository


@dataclass(frozen=True, slots=True)
class PersistedV2Graph:
    project: ProjectRow
    device_a: DeviceRow
    device_b: DeviceRow
    image_a: ImageRow
    image_b: ImageRow


def create_persisted_v2_graph(
    session: Session,
    *,
    prefix: str = "",
) -> PersistedV2Graph:
    project = Repository(session).create_project(f"{prefix or 'primary'} v2 project")
    device_a = DeviceRow(
        id=f"{prefix}device-a",
        project_id=project.id,
        name=f"{prefix}Device A",
        folder_path=f"/virtual/{prefix}device-a",
        canonical_model="model-a",
    )
    device_b = DeviceRow(
        id=f"{prefix}device-b",
        project_id=project.id,
        name=f"{prefix}Device B",
        folder_path=f"/virtual/{prefix}device-b",
        canonical_model="model-b",
    )
    session.add_all([device_a, device_b])
    session.flush()

    image_a = ImageRow(
        id=f"{prefix}image-a",
        device_id=device_a.id,
        path=f"/virtual/{prefix}device-a/image-a.jpg",
        filename="image-a.jpg",
        sequence_index=0,
        checksum="a" * 64,
        width=1024,
        height=768,
        exif_json="{}",
    )
    image_b = ImageRow(
        id=f"{prefix}image-b",
        device_id=device_b.id,
        path=f"/virtual/{prefix}device-b/image-b.jpg",
        filename="image-b.jpg",
        sequence_index=0,
        checksum="b" * 64,
        width=1024,
        height=768,
        exif_json="{}",
    )
    session.add_all([image_a, image_b])
    session.commit()
    return PersistedV2Graph(
        project=project,
        device_a=device_a,
        device_b=device_b,
        image_a=image_a,
        image_b=image_b,
    )
