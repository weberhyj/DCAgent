"""Persist bounded structured spreadsheet previews.

Revision ID: 20260721_02
Revises: 20260721_01
Create Date: 2026-07-21
"""

import sqlalchemy as sa

from alembic import op

revision = "20260721_02"
down_revision = "20260721_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "structured_previews",
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["knowledge_sources.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("source_id"),
    )


def downgrade() -> None:
    op.drop_table("structured_previews")
