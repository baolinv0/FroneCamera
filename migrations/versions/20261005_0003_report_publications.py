"""Durable report publication reservations across API and worker processes."""

from alembic import op
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    inspect,
)

revision = "20261005_0003"
down_revision = "20260720_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "report_publications" not in inspect(op.get_bind()).get_table_names():
        op.create_table(
            "report_publications",
            Column("id", String(), primary_key=True),
            Column(
                "project_id",
                String(),
                ForeignKey("projects.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("ordinal", Integer(), nullable=False),
            Column("status", String(30), nullable=False),
            Column("error", Text(), nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("project_id", "ordinal", name="uq_report_publication_project_ordinal"),
        )
        op.create_index("ix_report_publications_project_id", "report_publications", ["project_id"])


def downgrade() -> None:
    if "report_publications" in inspect(op.get_bind()).get_table_names():
        op.drop_table("report_publications")
