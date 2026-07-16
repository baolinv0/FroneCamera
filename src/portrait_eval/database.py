from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from portrait_eval.models import ProjectStatus


class Base(DeclarativeBase):
    pass


class ProjectRow(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(50), default=ProjectStatus.CREATED.value)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    devices: Mapped[list[DeviceRow]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    groups: Mapped[list[SceneGroupRow]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class DeviceRow(Base):
    __tablename__ = "devices"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    folder_path: Mapped[str] = mapped_column(Text)
    canonical_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    project: Mapped[ProjectRow] = relationship(back_populates="devices")
    images: Mapped[list[ImageRow]] = relationship(
        back_populates="device", cascade="all, delete-orphan"
    )


class ImageRow(Base):
    __tablename__ = "images"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    path: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(String(500))
    sequence_index: Mapped[int] = mapped_column(Integer)
    checksum: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    exif_json: Mapped[str] = mapped_column(Text, default="{}")
    device: Mapped[DeviceRow] = relationship(back_populates="images")


class SceneGroupRow(Base):
    __tablename__ = "scene_groups"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    group_key: Mapped[str] = mapped_column(String(50))
    label: Mapped[str | None] = mapped_column(String(300), nullable=True)
    sequence_index: Mapped[int] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer, default=1)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    project: Mapped[ProjectRow] = relationship(back_populates="groups")
    cells: Mapped[list[PairingCellRow]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )


class PairingCellRow(Base):
    __tablename__ = "pairing_cells"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    group_id: Mapped[str] = mapped_column(
        ForeignKey("scene_groups.id", ondelete="CASCADE"), index=True
    )
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    image_id: Mapped[str | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True
    )
    group: Mapped[SceneGroupRow] = relationship(back_populates="cells")


class PairingSnapshotRow(Base):
    __tablename__ = "pairing_snapshots"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AnalysisRow(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    scene_group_id: Mapped[str | None] = mapped_column(
        ForeignKey("scene_groups.id", ondelete="CASCADE"), nullable=True
    )
    image_id: Mapped[str | None] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(80), index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class ReviewItemRow(Base):
    __tablename__ = "review_items"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    category: Mapped[str] = mapped_column(String(80))
    priority: Mapped[str] = mapped_column(String(30), default="medium")
    status: Mapped[str] = mapped_column(String(30), default="open")
    payload_json: Mapped[str] = mapped_column(Text)


class TaskRow(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(30), default="PENDING", index=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class ReportRow(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(30))
    html_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Database:
    def __init__(self, url: str) -> None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()

        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        from portrait_eval import persistence_v2 as _persistence_v2  # noqa: F401

        Base.metadata.create_all(self.engine)

    def session(self) -> Iterator[Session]:
        with self.session_factory() as session:
            yield session


def json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def json_load(value: str) -> object:
    return json.loads(value)
