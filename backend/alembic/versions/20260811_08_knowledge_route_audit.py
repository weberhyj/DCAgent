"""Persist bounded knowledge-answer route audit metadata.

Revision ID: 20260811_08
Revises: 20260811_07
Create Date: 2026-08-11
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260811_08"
down_revision = "20260811_07"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("route_type", sa.String(length=40), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("route_metadata", sa.JSON(), nullable=True),
    )
    op.execute(sa.text("UPDATE agent_runs SET route_type = 'document_qa' WHERE route_type IS NULL"))
    op.execute(sa.text("UPDATE agent_runs SET route_metadata = '{}' WHERE route_metadata IS NULL"))
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.alter_column("route_type", existing_type=sa.String(length=40), nullable=False)
        batch_op.alter_column("route_metadata", existing_type=sa.JSON(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_column("route_metadata")
        batch_op.drop_column("route_type")
