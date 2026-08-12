"""Add a durable index for Word-document facts.

Revision ID: 20260811_07
Revises: 20260730_06
Create Date: 2026-08-11
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260811_07"
down_revision = "20260730_06"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_facts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("entity", sa.String(length=240), nullable=False),
        sa.Column("entity_normalized", sa.String(length=240), nullable=False),
        sa.Column("field", sa.String(length=120), nullable=False),
        sa.Column("field_normalized", sa.String(length=120), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("locator", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["knowledge_chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_facts_source_id", "knowledge_facts", ["source_id"])
    op.create_index("ix_knowledge_facts_chunk_id", "knowledge_facts", ["chunk_id"])
    op.create_index(
        "ix_knowledge_facts_entity_field",
        "knowledge_facts",
        ["entity_normalized", "field_normalized"],
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_facts_entity_field", table_name="knowledge_facts")
    op.drop_index("ix_knowledge_facts_chunk_id", table_name="knowledge_facts")
    op.drop_index("ix_knowledge_facts_source_id", table_name="knowledge_facts")
    op.drop_table("knowledge_facts")
