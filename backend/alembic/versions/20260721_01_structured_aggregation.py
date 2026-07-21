"""Add structured spreadsheet aggregation metadata.

Revision ID: 20260721_01
Revises: 20260715_00
Create Date: 2026-07-21
"""

import sqlalchemy as sa

from alembic import op

revision = "20260721_01"
down_revision = "20260715_00"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "structured_datasets",
        sa.Column("dataset_id", sa.String(length=128), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("worksheet_name", sa.String(length=240), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("schema_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.PrimaryKeyConstraint("dataset_id"),
        sa.UniqueConstraint(
            "source_id",
            "worksheet_name",
            "schema_version",
            name="uq_structured_datasets_source_worksheet_version",
        ),
    )
    op.create_table(
        "structured_columns",
        sa.Column("id", sa.String(length=160), nullable=False),
        sa.Column("dataset_id", sa.String(length=128), nullable=False),
        sa.Column("physical_name", sa.String(length=160), nullable=False),
        sa.Column("original_name", sa.String(length=240), nullable=False),
        sa.Column("display_name", sa.String(length=240), nullable=False),
        sa.Column("data_type", sa.String(length=20), nullable=False),
        sa.Column("aliases", sa.JSON(), nullable=False),
        sa.Column("allow_aggregate", sa.Boolean(), nullable=False),
        sa.Column("allow_filter", sa.Boolean(), nullable=False),
        sa.Column("null_policy", sa.String(length=40), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["structured_datasets.dataset_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_id",
            "physical_name",
            name="uq_structured_columns_dataset_physical_name",
        ),
    )
    op.create_table(
        "structured_ingestion_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=128), nullable=False),
        sa.Column("publication_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("lease_token", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.String(length=40), nullable=True),
        sa.Column("checkpoint_row", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("next_attempt_at", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["structured_datasets.dataset_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "structured_publications",
        sa.Column("publication_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=128), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("physical_table_name", sa.String(length=240), nullable=False),
        sa.Column("row_count", sa.BigInteger(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["structured_datasets.dataset_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("publication_id"),
    )

    op.create_index(
        "ix_structured_datasets_source_id",
        "structured_datasets",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        "ix_structured_datasets_status",
        "structured_datasets",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_structured_columns_dataset_id",
        "structured_columns",
        ["dataset_id"],
        unique=False,
    )
    op.create_index(
        "ix_structured_ingestion_jobs_source_id",
        "structured_ingestion_jobs",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        "ix_structured_ingestion_jobs_dataset_id",
        "structured_ingestion_jobs",
        ["dataset_id"],
        unique=False,
    )
    op.create_index(
        "ix_structured_ingestion_jobs_status",
        "structured_ingestion_jobs",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_structured_publications_dataset_id",
        "structured_publications",
        ["dataset_id"],
        unique=False,
    )
    op.create_index(
        "ix_structured_publications_status",
        "structured_publications",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_structured_publications_status", table_name="structured_publications")
    op.drop_index("ix_structured_publications_dataset_id", table_name="structured_publications")
    op.drop_index("ix_structured_ingestion_jobs_status", table_name="structured_ingestion_jobs")
    op.drop_index("ix_structured_ingestion_jobs_dataset_id", table_name="structured_ingestion_jobs")
    op.drop_index("ix_structured_ingestion_jobs_source_id", table_name="structured_ingestion_jobs")
    op.drop_index("ix_structured_columns_dataset_id", table_name="structured_columns")
    op.drop_index("ix_structured_datasets_status", table_name="structured_datasets")
    op.drop_index("ix_structured_datasets_source_id", table_name="structured_datasets")
    op.drop_table("structured_publications")
    op.drop_table("structured_ingestion_jobs")
    op.drop_table("structured_columns")
    op.drop_table("structured_datasets")
