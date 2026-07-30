"""Bind retrieval publications to complete embedding fingerprints.

Revision ID: 20260730_06
Revises: 20260728_05
Create Date: 2026-07-30
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260730_06"
down_revision = "20260728_05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "retrieval_publications",
        sa.Column("embedding_model_name", sa.String(length=240), nullable=True),
    )
    op.add_column(
        "retrieval_publications",
        sa.Column("embedding_model_sha256", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "retrieval_publications",
        sa.Column("embedding_normalized", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "retrieval_publications",
        sa.Column(
            "embedding_encoding_profile_sha256",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.add_column(
        "retrieval_publications",
        sa.Column("embedding_protocol_version", sa.String(length=120), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("retrieval_publications", "embedding_protocol_version")
    op.drop_column("retrieval_publications", "embedding_encoding_profile_sha256")
    op.drop_column("retrieval_publications", "embedding_normalized")
    op.drop_column("retrieval_publications", "embedding_model_sha256")
    op.drop_column("retrieval_publications", "embedding_model_name")
