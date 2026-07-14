from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from threading import RLock
from typing import Iterator

from sqlalchemy import BigInteger, JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool


DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:123456@127.0.0.1:5432/dc_agent"


def resolve_database_url(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return source.get("DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    pass


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    topic: Mapped[str] = mapped_column(String(80), nullable=False)
    group_name: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    context_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)

    messages: Mapped[list["MessageRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageRecord.sort_order",
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    time: Mapped[str] = mapped_column(String(40), nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    paragraphs: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    artifacts: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    conversation: Mapped[ConversationRecord] = relationship(back_populates="messages")


class AgentRunRecord(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[str] = mapped_column(String(40), nullable=False)
    completed_at: Mapped[str] = mapped_column(String(40), nullable=False)
    answer_message_id: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    steps: Mapped[list["AgentStepRecord"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AgentStepRecord.step_index",
    )


class AgentStepRecord(Base):
    __tablename__ = "agent_steps"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    input_summary: Mapped[str] = mapped_column(Text, nullable=False)
    output_summary: Mapped[str] = mapped_column(Text, nullable=False)
    source_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    read_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    started_at: Mapped[str] = mapped_column(String(40), nullable=False)
    completed_at: Mapped[str] = mapped_column(String(40), nullable=False)

    run: Mapped[AgentRunRecord] = relationship(back_populates="steps")


class EvaluationCaseRecord(Base):
    __tablename__ = "evaluation_cases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_source_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    expected_terms: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    expect_answer: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    category: Mapped[str | None] = mapped_column(String(80), index=True)
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    external_key: Mapped[str | None] = mapped_column(String(120), index=True)
    import_batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)

    runs: Mapped[list["EvaluationRunRecord"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )


class EvaluationImportBatchRecord(Base):
    __tablename__ = "evaluation_import_batches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(240), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    valid_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    invalid_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    duplicate_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String(40))


class EvaluationCounterRecord(Base):
    __tablename__ = "evaluation_counters"

    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    next_value: Mapped[int] = mapped_column(BigInteger, nullable=False)


class EvaluationBatchRecord(Base):
    __tablename__ = "evaluation_batches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    retrieval_min_score: Mapped[float] = mapped_column(Float, nullable=False)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    passed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    false_positive_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    completed_at: Mapped[str | None] = mapped_column(String(40))
    error_message: Mapped[str | None] = mapped_column(Text)

    runs: Mapped[list["EvaluationRunRecord"]] = relationship(back_populates="batch")


class EvaluationRunRecord(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index(
            "ix_evaluation_runs_sequence",
            "sequence",
            unique=True,
        ),
        Index(
            "ix_evaluation_runs_case_id_sequence",
            "case_id",
            "sequence",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    batch_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_batches.id", ondelete="SET NULL"),
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    expect_answer: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    answerable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    false_positive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    expected_source_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    matched_source_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    missing_source_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    expected_terms: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    found_terms: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    missing_terms: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    source_recall: Mapped[float] = mapped_column(Float, nullable=False)
    term_recall: Mapped[float] = mapped_column(Float, nullable=False)
    top_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[str] = mapped_column(String(40), nullable=False)
    completed_at: Mapped[str] = mapped_column(String(40), nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    hits: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)

    case: Mapped[EvaluationCaseRecord] = relationship(back_populates="runs")
    batch: Mapped[EvaluationBatchRecord | None] = relationship(back_populates="runs")


class KnowledgeSourceRecord(Base):
    __tablename__ = "knowledge_sources"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(240), nullable=False)
    source_type: Mapped[str] = mapped_column(String(80), nullable=False)
    records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    classification: Mapped[str] = mapped_column(String(80), nullable=False)
    file_path: Mapped[str | None] = mapped_column(Text)
    file_size: Mapped[int | None] = mapped_column(Integer)
    mime_type: Mapped[str | None] = mapped_column(String(120))
    error_message: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)

    chunks: Mapped[list["KnowledgeChunkRecord"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        order_by="KnowledgeChunkRecord.chunk_index",
    )


class KnowledgeChunkRecord(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(JSON)

    source: Mapped[KnowledgeSourceRecord] = relationship(back_populates="chunks")


class Database:
    def __init__(self, database_url: str) -> None:
        self.is_sqlite_memory = database_url == "sqlite+pysqlite:///:memory:"
        self.write_lock = RLock()
        engine_options = {}
        if self.is_sqlite_memory:
            engine_options = {
                "connect_args": {"check_same_thread": False},
                "poolclass": StaticPool,
            }
        self.engine = create_engine(database_url, future=True, pool_pre_ping=True, **engine_options)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, class_=Session)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        self._ensure_knowledge_source_upload_columns()
        self._ensure_knowledge_chunk_embedding_column()
        self._ensure_evaluation_columns()
        self._ensure_evaluation_run_counter()

    def _ensure_knowledge_source_upload_columns(self) -> None:
        inspector = inspect(self.engine)
        if not inspector.has_table("knowledge_sources"):
            return

        columns = {column["name"] for column in inspector.get_columns("knowledge_sources")}
        statements = []
        if "file_path" not in columns:
            statements.append("ALTER TABLE knowledge_sources ADD COLUMN file_path TEXT")
        if "file_size" not in columns:
            statements.append("ALTER TABLE knowledge_sources ADD COLUMN file_size INTEGER")
        if "mime_type" not in columns:
            statements.append("ALTER TABLE knowledge_sources ADD COLUMN mime_type VARCHAR(120)")
        if "error_message" not in columns:
            statements.append("ALTER TABLE knowledge_sources ADD COLUMN error_message TEXT")

        if not statements:
            return

        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))

    def _ensure_knowledge_chunk_embedding_column(self) -> None:
        inspector = inspect(self.engine)
        if not inspector.has_table("knowledge_chunks"):
            return

        columns = {column["name"] for column in inspector.get_columns("knowledge_chunks")}
        if "embedding" in columns:
            return

        with self.engine.begin() as connection:
            connection.execute(text("ALTER TABLE knowledge_chunks ADD COLUMN embedding JSON"))

    def _ensure_evaluation_columns(self) -> None:
        inspector = inspect(self.engine)
        statements = []
        has_evaluation_runs = inspector.has_table("evaluation_runs")
        evaluation_run_sequence_column_created = False
        backfill_evaluation_run_sequence = False
        create_evaluation_run_sequence_index = False
        evaluation_run_sequence_index_is_unique = False
        if inspector.has_table("evaluation_cases"):
            case_columns = {column["name"] for column in inspector.get_columns("evaluation_cases")}
            case_indexes = {index["name"] for index in inspector.get_indexes("evaluation_cases")}
            if "expect_answer" not in case_columns:
                statements.append(
                    "ALTER TABLE evaluation_cases ADD COLUMN expect_answer BOOLEAN NOT NULL DEFAULT TRUE"
                )
            if "category" not in case_columns:
                statements.append("ALTER TABLE evaluation_cases ADD COLUMN category VARCHAR(80)")
            if "tags" not in case_columns:
                statements.append(
                    "ALTER TABLE evaluation_cases ADD COLUMN tags JSON NOT NULL DEFAULT '[]'"
                )
            if "external_key" not in case_columns:
                statements.append("ALTER TABLE evaluation_cases ADD COLUMN external_key VARCHAR(120)")
            if "import_batch_id" not in case_columns:
                statements.append("ALTER TABLE evaluation_cases ADD COLUMN import_batch_id VARCHAR(64)")
            if "ix_evaluation_cases_category" not in case_indexes:
                statements.append(
                    "CREATE INDEX IF NOT EXISTS ix_evaluation_cases_category "
                    "ON evaluation_cases (category)"
                )
            if "ix_evaluation_cases_external_key" not in case_indexes:
                statements.append(
                    "CREATE INDEX IF NOT EXISTS ix_evaluation_cases_external_key "
                    "ON evaluation_cases (external_key)"
                )
            if "ix_evaluation_cases_import_batch_id" not in case_indexes:
                statements.append(
                    "CREATE INDEX IF NOT EXISTS ix_evaluation_cases_import_batch_id "
                    "ON evaluation_cases (import_batch_id)"
                )

        if has_evaluation_runs:
            run_columns = {column["name"] for column in inspector.get_columns("evaluation_runs")}
            run_indexes = {
                index["name"]: index
                for index in inspector.get_indexes("evaluation_runs")
            }
            if "expect_answer" not in run_columns:
                statements.append(
                    "ALTER TABLE evaluation_runs ADD COLUMN expect_answer BOOLEAN NOT NULL DEFAULT TRUE"
                )
            if "answerable" not in run_columns:
                statements.append(
                    "ALTER TABLE evaluation_runs ADD COLUMN answerable BOOLEAN NOT NULL DEFAULT FALSE"
                )
            if "false_positive" not in run_columns:
                statements.append(
                    "ALTER TABLE evaluation_runs ADD COLUMN false_positive BOOLEAN NOT NULL DEFAULT FALSE"
                )
            if "batch_id" not in run_columns:
                statements.append(
                    "ALTER TABLE evaluation_runs ADD COLUMN batch_id VARCHAR(64) "
                    "REFERENCES evaluation_batches(id) ON DELETE SET NULL"
                )
            if "sequence" not in run_columns:
                statements.append(
                    "ALTER TABLE evaluation_runs ADD COLUMN sequence BIGINT"
                )
                evaluation_run_sequence_column_created = True
                backfill_evaluation_run_sequence = True
            sequence_index = run_indexes.get("ix_evaluation_runs_sequence")
            if sequence_index is None:
                create_evaluation_run_sequence_index = True
            elif not sequence_index.get("unique"):
                statements.append(
                    "DROP INDEX IF EXISTS ix_evaluation_runs_sequence"
                )
                create_evaluation_run_sequence_index = True
            else:
                evaluation_run_sequence_index_is_unique = True
            if "ix_evaluation_runs_case_id_sequence" not in run_indexes:
                statements.append(
                    "CREATE INDEX IF NOT EXISTS ix_evaluation_runs_case_id_sequence "
                    "ON evaluation_runs (case_id, sequence)"
                )
            if "ix_evaluation_runs_batch_id" not in run_indexes:
                statements.append(
                    "CREATE INDEX IF NOT EXISTS ix_evaluation_runs_batch_id "
                    "ON evaluation_runs (batch_id)"
                )

        if (
            not statements
            and not has_evaluation_runs
            and not backfill_evaluation_run_sequence
            and not create_evaluation_run_sequence_index
        ):
            return
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
            if has_evaluation_runs and not evaluation_run_sequence_column_created:
                total_count, non_null_count, distinct_count = connection.execute(
                    text(
                        "SELECT COUNT(*), COUNT(sequence), "
                        "COUNT(DISTINCT sequence) FROM evaluation_runs"
                    )
                ).one()
                backfill_evaluation_run_sequence = (
                    total_count != non_null_count
                    or non_null_count != distinct_count
                )
                if (
                    backfill_evaluation_run_sequence
                    and evaluation_run_sequence_index_is_unique
                ):
                    connection.execute(
                        text("DROP INDEX IF EXISTS ix_evaluation_runs_sequence")
                    )
                    create_evaluation_run_sequence_index = True
            if backfill_evaluation_run_sequence:
                connection.execute(
                    text(
                        "WITH ranked AS ("
                        "SELECT id, "
                        "ROW_NUMBER() OVER ("
                        "ORDER BY completed_at ASC, started_at ASC, id ASC"
                        ") AS assigned_sequence "
                        "FROM evaluation_runs"
                        ") "
                        "UPDATE evaluation_runs "
                        "SET sequence = ("
                        "SELECT assigned_sequence FROM ranked "
                        "WHERE ranked.id = evaluation_runs.id"
                        ")"
                    )
                )
            if create_evaluation_run_sequence_index:
                connection.execute(
                    text(
                        "CREATE UNIQUE INDEX ix_evaluation_runs_sequence "
                        "ON evaluation_runs (sequence)"
                    )
                )

    def _ensure_evaluation_run_counter(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO evaluation_counters (name, next_value) "
                    "VALUES ('evaluation_runs', 1) "
                    "ON CONFLICT (name) DO NOTHING"
                )
            )
            connection.execute(
                text(
                    "UPDATE evaluation_counters "
                    "SET next_value = CASE "
                    "WHEN next_value < ("
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM evaluation_runs"
                    ") THEN ("
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM evaluation_runs"
                    ") ELSE next_value END "
                    "WHERE name = 'evaluation_runs'"
                )
            )

    @contextmanager
    def session(self) -> Iterator[Session]:
        session_lock = self.write_lock if self.is_sqlite_memory else nullcontext()
        with session_lock:
            session = self.session_factory()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()


def has_seed_data(session: Session) -> bool:
    return session.scalar(select(ConversationRecord.id).limit(1)) is not None
