"""Freeze the existing DC-Agent relational schema.

Revision ID: 20260715_00
Revises:
Create Date: 2026-07-15
"""

import sqlalchemy as sa

from alembic import op

revision = "20260715_00"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("topic", sa.String(length=80), nullable=False),
        sa.Column("group_name", sa.String(length=40), nullable=False),
        sa.Column("updated_at", sa.String(length=40), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("context_summary", sa.Text(), nullable=False),
        sa.Column("turn_count", sa.Integer(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("time", sa.String(length=40), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("paragraphs", sa.JSON(), nullable=False),
        sa.Column("artifacts", sa.JSON(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("started_at", sa.String(length=40), nullable=False),
        sa.Column("completed_at", sa.String(length=40), nullable=False),
        sa.Column("answer_message_id", sa.String(length=64), nullable=False),
        sa.Column("evidence_count", sa.Integer(), nullable=False),
        sa.Column("source_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "agent_steps",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("input_summary", sa.Text(), nullable=False),
        sa.Column("output_summary", sa.Text(), nullable=False),
        sa.Column("source_ids", sa.JSON(), nullable=False),
        sa.Column("read_only", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.String(length=40), nullable=False),
        sa.Column("completed_at", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "evaluation_cases",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_source_ids", sa.JSON(), nullable=False),
        sa.Column("expected_terms", sa.JSON(), nullable=False),
        sa.Column("expect_answer", sa.Boolean(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.Column("updated_at", sa.String(length=40), nullable=False),
        sa.Column("category", sa.String(length=80), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("external_key", sa.String(length=120), nullable=True),
        sa.Column("import_batch_id", sa.String(length=64), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "evaluation_import_batches",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("file_name", sa.String(length=240), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("total_rows", sa.Integer(), nullable=False),
        sa.Column("valid_rows", sa.Integer(), nullable=False),
        sa.Column("invalid_rows", sa.Integer(), nullable=False),
        sa.Column("duplicate_rows", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.Column("completed_at", sa.String(length=40), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "evaluation_counters",
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("next_value", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "evaluation_batches",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("case_ids", sa.JSON(), nullable=False),
        sa.Column("retrieval_min_score", sa.Float(), nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("completed_count", sa.Integer(), nullable=False),
        sa.Column("passed_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("false_positive_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.String(length=40), nullable=False),
        sa.Column("completed_at", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("batch_id", sa.String(length=64), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("expect_answer", sa.Boolean(), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False),
        sa.Column("false_positive", sa.Boolean(), nullable=False),
        sa.Column("expected_source_ids", sa.JSON(), nullable=False),
        sa.Column("matched_source_ids", sa.JSON(), nullable=False),
        sa.Column("missing_source_ids", sa.JSON(), nullable=False),
        sa.Column("expected_terms", sa.JSON(), nullable=False),
        sa.Column("found_terms", sa.JSON(), nullable=False),
        sa.Column("missing_terms", sa.JSON(), nullable=False),
        sa.Column("source_recall", sa.Float(), nullable=False),
        sa.Column("term_recall", sa.Float(), nullable=False),
        sa.Column("top_score", sa.Float(), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.String(length=40), nullable=False),
        sa.Column("completed_at", sa.String(length=40), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("hits", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["batch_id"], ["evaluation_batches.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["case_id"], ["evaluation_cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "knowledge_sources",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=240), nullable=False),
        sa.Column("source_type", sa.String(length=80), nullable=False),
        sa.Column("records", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("updated_at", sa.String(length=40), nullable=False),
        sa.Column("classification", sa.String(length=80), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("mime_type", sa.String(length=120), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_conversations_sort_order", "conversations", ["sort_order"], unique=False)
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"], unique=False)
    op.create_index(
        "ix_agent_runs_conversation_id", "agent_runs", ["conversation_id"], unique=False
    )
    op.create_index("ix_agent_steps_run_id", "agent_steps", ["run_id"], unique=False)
    op.create_index("ix_evaluation_cases_category", "evaluation_cases", ["category"], unique=False)
    op.create_index(
        "ix_evaluation_cases_external_key",
        "evaluation_cases",
        ["external_key"],
        unique=False,
    )
    op.create_index(
        "ix_evaluation_cases_import_batch_id",
        "evaluation_cases",
        ["import_batch_id"],
        unique=False,
    )
    op.create_index(
        "ix_evaluation_cases_sort_order",
        "evaluation_cases",
        ["sort_order"],
        unique=False,
    )
    op.create_index("ix_evaluation_batches_status", "evaluation_batches", ["status"], unique=False)
    op.create_index(
        "ix_evaluation_batches_started_at",
        "evaluation_batches",
        ["started_at"],
        unique=False,
    )
    op.create_index("ix_evaluation_runs_case_id", "evaluation_runs", ["case_id"], unique=False)
    op.create_index("ix_evaluation_runs_batch_id", "evaluation_runs", ["batch_id"], unique=False)
    op.create_index("ix_evaluation_runs_sequence", "evaluation_runs", ["sequence"], unique=True)
    op.create_index(
        "ix_evaluation_runs_case_id_sequence",
        "evaluation_runs",
        ["case_id", "sequence"],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_sources_sort_order",
        "knowledge_sources",
        ["sort_order"],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_chunks_source_id", "knowledge_chunks", ["source_id"], unique=False
    )

    evaluation_counters = sa.table(
        "evaluation_counters",
        sa.column("name", sa.String(length=80)),
        sa.column("next_value", sa.BigInteger()),
    )
    op.bulk_insert(
        evaluation_counters,
        [{"name": "evaluation_runs", "next_value": 1}],
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_chunks_source_id", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_sources_sort_order", table_name="knowledge_sources")
    op.drop_index("ix_evaluation_runs_case_id_sequence", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_sequence", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_batch_id", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_case_id", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_batches_started_at", table_name="evaluation_batches")
    op.drop_index("ix_evaluation_batches_status", table_name="evaluation_batches")
    op.drop_index("ix_evaluation_cases_sort_order", table_name="evaluation_cases")
    op.drop_index("ix_evaluation_cases_import_batch_id", table_name="evaluation_cases")
    op.drop_index("ix_evaluation_cases_external_key", table_name="evaluation_cases")
    op.drop_index("ix_evaluation_cases_category", table_name="evaluation_cases")
    op.drop_index("ix_agent_steps_run_id", table_name="agent_steps")
    op.drop_index("ix_agent_runs_conversation_id", table_name="agent_runs")
    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_index("ix_conversations_sort_order", table_name="conversations")

    op.drop_table("knowledge_chunks")
    op.drop_table("knowledge_sources")
    op.drop_table("evaluation_runs")
    op.drop_table("evaluation_batches")
    op.drop_table("evaluation_counters")
    op.drop_table("evaluation_import_batches")
    op.drop_table("evaluation_cases")
    op.drop_table("agent_steps")
    op.drop_table("agent_runs")
    op.drop_table("messages")
    op.drop_table("conversations")
