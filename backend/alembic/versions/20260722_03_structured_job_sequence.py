"""Add per-source structured publication job sequence.

Revision ID: 20260722_03
Revises: 20260721_02
Create Date: 2026-07-22
"""

import sqlalchemy as sa

from alembic import op

revision = "20260722_03"
down_revision = "20260721_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "structured_ingestion_jobs",
        sa.Column("sequence", sa.BigInteger(), nullable=True),
    )
    connection = op.get_bind()
    jobs = sa.table(
        "structured_ingestion_jobs",
        sa.column("id", sa.String()),
        sa.column("source_id", sa.String()),
        sa.column("sequence", sa.BigInteger()),
    )
    rows = connection.execute(
        sa.select(jobs.c.id, jobs.c.source_id).order_by(jobs.c.source_id, jobs.c.id)
    ).all()
    source_sequences: dict[str, int] = {}
    for job_id, source_id in rows:
        sequence = source_sequences.get(source_id, 0) + 1
        source_sequences[source_id] = sequence
        connection.execute(sa.update(jobs).where(jobs.c.id == job_id).values(sequence=sequence))
    with op.batch_alter_table("structured_ingestion_jobs") as batch:
        batch.alter_column("sequence", existing_type=sa.BigInteger(), nullable=False)
    op.create_index(
        "uq_structured_ingestion_jobs_source_sequence",
        "structured_ingestion_jobs",
        ["source_id", "sequence"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_structured_ingestion_jobs_source_sequence",
        table_name="structured_ingestion_jobs",
    )
    with op.batch_alter_table("structured_ingestion_jobs") as batch:
        batch.drop_column("sequence")
