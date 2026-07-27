"""Add Qwen3 retrieval publication and Shadow audit persistence.

Revision ID: 20260727_04
Revises: 20260722_03
Create Date: 2026-07-27
"""

import sqlalchemy as sa

from alembic import op

revision = "20260727_04"
down_revision = "20260722_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_chunks",
        sa.Column(
            "metadata",
            sa.JSON(),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
    )
    op.create_table(
        "retrieval_publications",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("collection_name", sa.String(length=240), nullable=False),
        sa.Column("alias_name", sa.String(length=240), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("embedding_model_version", sa.String(length=120), nullable=False),
        sa.Column("sparse_profile_sha256", sa.String(length=64), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("point_count", sa.BigInteger(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.Column("completed_at", sa.String(length=40), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_name"),
    )
    op.create_index(
        "ix_retrieval_publications_status",
        "retrieval_publications",
        ["status"],
        unique=False,
    )
    op.create_index(
        "uq_retrieval_publications_active_alias",
        "retrieval_publications",
        ["alias_name"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        sqlite_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "retrieval_source_indexes",
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("publication_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("indexed_chunk_count", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(
            ["publication_id"],
            ["retrieval_publications.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["knowledge_sources.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("source_id"),
    )
    op.create_index(
        "ix_retrieval_source_indexes_status",
        "retrieval_source_indexes",
        ["status"],
        unique=False,
    )
    op.create_table(
        "retrieval_shadow_comparisons",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("routing_key_hash", sa.String(length=64), nullable=False),
        sa.Column("query_hash", sa.String(length=64), nullable=False),
        sa.Column("legacy_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("qwen_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("legacy_ms", sa.Float(), nullable=False),
        sa.Column("qwen_ms", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("fallback_reason", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    op.create_index(
        "ix_retrieval_shadow_comparisons_created_at",
        "retrieval_shadow_comparisons",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_retrieval_shadow_comparisons_status",
        "retrieval_shadow_comparisons",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("retrieval_shadow_comparisons")
    op.drop_table("retrieval_source_indexes")
    op.drop_index(
        "uq_retrieval_publications_active_alias",
        table_name="retrieval_publications",
    )
    op.drop_table("retrieval_publications")
    with op.batch_alter_table("knowledge_chunks") as batch:
        batch.drop_column("metadata")
