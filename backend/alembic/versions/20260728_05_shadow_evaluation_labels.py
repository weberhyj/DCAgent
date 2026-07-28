"""Add explicit evaluation labels to Shadow retrieval audits.

Revision ID: 20260728_05
Revises: 20260727_04
Create Date: 2026-07-28
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260728_05"
down_revision = "20260727_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "retrieval_shadow_comparisons",
        sa.Column("evaluation_case_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "retrieval_shadow_comparisons",
        sa.Column(
            "relevant_chunk_ids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("retrieval_shadow_comparisons", "relevant_chunk_ids")
    op.drop_column("retrieval_shadow_comparisons", "evaluation_case_id")
