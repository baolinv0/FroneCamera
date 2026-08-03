"""Add automatic pairing confidence metadata."""

from alembic import op
from sqlalchemy import Boolean, Column, Float, Text, inspect

revision = "20260720_0002"
down_revision = "20260715_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("scene_groups")}
    if "matching_confidence" not in columns:
        op.add_column("scene_groups", Column("matching_confidence", Float(), nullable=True))
    if "review_required" not in columns:
        op.add_column(
            "scene_groups",
            Column("review_required", Boolean(), nullable=False, server_default="0"),
        )
    if "match_notes_json" not in columns:
        op.add_column(
            "scene_groups",
            Column("match_notes_json", Text(), nullable=False, server_default="[]"),
        )


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("scene_groups")}
    if "match_notes_json" in columns:
        op.drop_column("scene_groups", "match_notes_json")
    if "review_required" in columns:
        op.drop_column("scene_groups", "review_required")
    if "matching_confidence" in columns:
        op.drop_column("scene_groups", "matching_confidence")
