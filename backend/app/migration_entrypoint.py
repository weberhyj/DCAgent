from __future__ import annotations

from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

from .database import resolve_database_url


BASELINE_REVISION = "20260715_00"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "alembic.ini"

TypeSignature = tuple[str, int | None]
ColumnSignature = tuple[str, TypeSignature, bool, bool]
ForeignKeySignature = tuple[tuple[str, ...], str, tuple[str, ...], str]


def _string(length: int | None) -> TypeSignature:
    return ("string", length)


TEXT: TypeSignature = ("text", None)
JSON: TypeSignature = ("json", None)
BOOLEAN: TypeSignature = ("boolean", None)
INTEGER: TypeSignature = ("integer", None)
BIGINT: TypeSignature = ("bigint", None)
FLOAT: TypeSignature = ("float", None)


BASELINE_COLUMNS: dict[str, tuple[ColumnSignature, ...]] = {
    "conversations": (
        ("id", _string(64), False, True),
        ("title", _string(200), False, False),
        ("topic", _string(80), False, False),
        ("group_name", _string(40), False, False),
        ("updated_at", _string(40), False, False),
        ("pinned", BOOLEAN, False, False),
        ("context_summary", TEXT, False, False),
        ("turn_count", INTEGER, False, False),
        ("sort_order", INTEGER, False, False),
    ),
    "messages": (
        ("id", _string(64), False, True),
        ("conversation_id", _string(64), False, False),
        ("role", _string(20), False, False),
        ("time", _string(40), False, False),
        ("content", TEXT, True, False),
        ("paragraphs", JSON, False, False),
        ("artifacts", JSON, False, False),
        ("sort_order", INTEGER, False, False),
    ),
    "agent_runs": (
        ("id", _string(64), False, True),
        ("conversation_id", _string(64), False, False),
        ("query", TEXT, False, False),
        ("mode", _string(20), False, False),
        ("status", _string(20), False, False),
        ("started_at", _string(40), False, False),
        ("completed_at", _string(40), False, False),
        ("answer_message_id", _string(64), False, False),
        ("evidence_count", INTEGER, False, False),
        ("source_count", INTEGER, False, False),
    ),
    "agent_steps": (
        ("id", _string(64), False, True),
        ("run_id", _string(64), False, False),
        ("step_index", INTEGER, False, False),
        ("tool_name", _string(80), False, False),
        ("status", _string(20), False, False),
        ("input_summary", TEXT, False, False),
        ("output_summary", TEXT, False, False),
        ("source_ids", JSON, False, False),
        ("read_only", BOOLEAN, False, False),
        ("started_at", _string(40), False, False),
        ("completed_at", _string(40), False, False),
    ),
    "evaluation_cases": (
        ("id", _string(64), False, True),
        ("question", TEXT, False, False),
        ("expected_source_ids", JSON, False, False),
        ("expected_terms", JSON, False, False),
        ("expect_answer", BOOLEAN, False, False),
        ("top_k", INTEGER, False, False),
        ("created_at", _string(40), False, False),
        ("updated_at", _string(40), False, False),
        ("category", _string(80), True, False),
        ("tags", JSON, False, False),
        ("external_key", _string(120), True, False),
        ("import_batch_id", _string(64), True, False),
        ("sort_order", INTEGER, False, False),
    ),
    "evaluation_import_batches": (
        ("id", _string(64), False, True),
        ("file_name", _string(240), False, False),
        ("status", _string(20), False, False),
        ("total_rows", INTEGER, False, False),
        ("valid_rows", INTEGER, False, False),
        ("invalid_rows", INTEGER, False, False),
        ("duplicate_rows", INTEGER, False, False),
        ("created_at", _string(40), False, False),
        ("completed_at", _string(40), True, False),
    ),
    "evaluation_counters": (
        ("name", _string(80), False, True),
        ("next_value", BIGINT, False, False),
    ),
    "evaluation_batches": (
        ("id", _string(64), False, True),
        ("name", _string(120), False, False),
        ("status", _string(20), False, False),
        ("case_ids", JSON, False, False),
        ("retrieval_min_score", FLOAT, False, False),
        ("case_count", INTEGER, False, False),
        ("completed_count", INTEGER, False, False),
        ("passed_count", INTEGER, False, False),
        ("failed_count", INTEGER, False, False),
        ("false_positive_count", INTEGER, False, False),
        ("started_at", _string(40), False, False),
        ("completed_at", _string(40), True, False),
        ("error_message", TEXT, True, False),
    ),
    "evaluation_runs": (
        ("id", _string(64), False, True),
        ("case_id", _string(64), False, False),
        ("batch_id", _string(64), True, False),
        ("question", TEXT, False, False),
        ("status", _string(20), False, False),
        ("expect_answer", BOOLEAN, False, False),
        ("answerable", BOOLEAN, False, False),
        ("false_positive", BOOLEAN, False, False),
        ("expected_source_ids", JSON, False, False),
        ("matched_source_ids", JSON, False, False),
        ("missing_source_ids", JSON, False, False),
        ("expected_terms", JSON, False, False),
        ("found_terms", JSON, False, False),
        ("missing_terms", JSON, False, False),
        ("source_recall", FLOAT, False, False),
        ("term_recall", FLOAT, False, False),
        ("top_score", FLOAT, False, False),
        ("hit_count", INTEGER, False, False),
        ("started_at", _string(40), False, False),
        ("completed_at", _string(40), False, False),
        ("sequence", BIGINT, False, False),
        ("hits", JSON, False, False),
    ),
    "knowledge_sources": (
        ("id", _string(64), False, True),
        ("name", _string(240), False, False),
        ("source_type", _string(80), False, False),
        ("records", INTEGER, False, False),
        ("status", _string(40), False, False),
        ("updated_at", _string(40), False, False),
        ("classification", _string(80), False, False),
        ("file_path", TEXT, True, False),
        ("file_size", INTEGER, True, False),
        ("mime_type", _string(120), True, False),
        ("error_message", TEXT, True, False),
        ("sort_order", INTEGER, False, False),
    ),
    "knowledge_chunks": (
        ("id", _string(64), False, True),
        ("source_id", _string(64), False, False),
        ("chunk_index", INTEGER, False, False),
        ("text", TEXT, False, False),
        ("token_count", INTEGER, False, False),
        ("embedding", JSON, True, False),
    ),
}

BASELINE_INDEXES: dict[str, dict[str, tuple[tuple[str, ...], bool]]] = {
    "conversations": {"ix_conversations_sort_order": (("sort_order",), False)},
    "messages": {"ix_messages_conversation_id": (("conversation_id",), False)},
    "agent_runs": {"ix_agent_runs_conversation_id": (("conversation_id",), False)},
    "agent_steps": {"ix_agent_steps_run_id": (("run_id",), False)},
    "evaluation_cases": {
        "ix_evaluation_cases_category": (("category",), False),
        "ix_evaluation_cases_external_key": (("external_key",), False),
        "ix_evaluation_cases_import_batch_id": (("import_batch_id",), False),
        "ix_evaluation_cases_sort_order": (("sort_order",), False),
    },
    "evaluation_import_batches": {},
    "evaluation_counters": {},
    "evaluation_batches": {
        "ix_evaluation_batches_status": (("status",), False),
        "ix_evaluation_batches_started_at": (("started_at",), False),
    },
    "evaluation_runs": {
        "ix_evaluation_runs_case_id": (("case_id",), False),
        "ix_evaluation_runs_batch_id": (("batch_id",), False),
        "ix_evaluation_runs_sequence": (("sequence",), True),
        "ix_evaluation_runs_case_id_sequence": (("case_id", "sequence"), False),
    },
    "knowledge_sources": {"ix_knowledge_sources_sort_order": (("sort_order",), False)},
    "knowledge_chunks": {"ix_knowledge_chunks_source_id": (("source_id",), False)},
}

BASELINE_FOREIGN_KEYS: dict[str, set[ForeignKeySignature]] = {
    "messages": {(("conversation_id",), "conversations", ("id",), "CASCADE")},
    "agent_steps": {(("run_id",), "agent_runs", ("id",), "CASCADE")},
    "evaluation_runs": {
        (("case_id",), "evaluation_cases", ("id",), "CASCADE"),
        (("batch_id",), "evaluation_batches", ("id",), "SET NULL"),
    },
    "knowledge_chunks": {
        (("source_id",), "knowledge_sources", ("id",), "CASCADE")
    },
}


class BaselineSchemaMismatch(RuntimeError):
    """Raised when an unmanaged database is not the exact frozen baseline."""


def _normalize_type(column_type: Any) -> TypeSignature:
    if isinstance(column_type, sa.Text):
        return TEXT
    if isinstance(column_type, sa.String):
        return _string(getattr(column_type, "length", None))
    if isinstance(column_type, sa.JSON):
        return JSON
    if isinstance(column_type, sa.Boolean):
        return BOOLEAN
    if isinstance(column_type, sa.BigInteger):
        return BIGINT
    if isinstance(column_type, sa.Integer):
        return INTEGER
    if isinstance(column_type, sa.Float):
        return FLOAT
    return (column_type.__class__.__name__.lower(), None)


def _schema_fingerprint(connection: Any) -> dict[str, Any]:
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    columns: dict[str, tuple[ColumnSignature, ...]] = {}
    indexes: dict[str, dict[str, tuple[tuple[str, ...], bool]]] = {}
    foreign_keys: dict[str, set[ForeignKeySignature]] = {}
    defaults: dict[str, tuple[tuple[str, Any], ...]] = {}
    unique_constraints: dict[str, tuple[Any, ...]] = {}
    check_constraints: dict[str, tuple[Any, ...]] = {}

    for table_name in sorted(tables):
        table_columns = inspector.get_columns(table_name)
        columns[table_name] = tuple(
            (
                column["name"],
                _normalize_type(column["type"]),
                bool(column["nullable"]),
                bool(column["primary_key"]),
            )
            for column in table_columns
        )
        defaults[table_name] = tuple(
            (column["name"], column.get("default")) for column in table_columns
        )
        indexes[table_name] = {
            index["name"]: (tuple(index["column_names"]), bool(index["unique"]))
            for index in inspector.get_indexes(table_name)
        }
        foreign_keys[table_name] = {
            (
                tuple(foreign_key["constrained_columns"]),
                foreign_key["referred_table"],
                tuple(foreign_key["referred_columns"]),
                str(foreign_key.get("options", {}).get("ondelete") or "").upper(),
            )
            for foreign_key in inspector.get_foreign_keys(table_name)
        }
        unique_constraints[table_name] = tuple(
            (
                constraint.get("name"),
                tuple(constraint.get("column_names") or ()),
            )
            for constraint in inspector.get_unique_constraints(table_name)
        )
        check_constraints[table_name] = tuple(
            (
                constraint.get("name"),
                constraint.get("sqltext"),
            )
            for constraint in inspector.get_check_constraints(table_name)
        )

    return {
        "tables": tables,
        "columns": columns,
        "indexes": indexes,
        "foreign_keys": foreign_keys,
        "defaults": defaults,
        "unique_constraints": unique_constraints,
        "check_constraints": check_constraints,
    }


def _validate_counter_row(connection: Any, errors: list[str]) -> None:
    try:
        counter = connection.execute(
            text(
                "SELECT next_value FROM evaluation_counters "
                "WHERE name = 'evaluation_runs'"
            )
        ).scalar_one_or_none()
        max_sequence = connection.execute(
            text("SELECT COALESCE(MAX(sequence), 0) FROM evaluation_runs")
        ).scalar_one()
    except Exception as error:
        errors.append(f"counter invariant could not be inspected: {error}")
        return
    if counter is None:
        errors.append("evaluation_counters is missing the evaluation_runs row")
    elif int(counter) < int(max_sequence) + 1:
        errors.append(
            "evaluation_counters.evaluation_runs.next_value must be greater than "
            "all existing evaluation_runs.sequence values"
        )


def _validate_baseline(connection: Any) -> None:
    actual = _schema_fingerprint(connection)
    errors: list[str] = []
    expected_tables = set(BASELINE_COLUMNS)
    if actual["tables"] != expected_tables:
        missing = sorted(expected_tables - actual["tables"])
        extra = sorted(actual["tables"] - expected_tables)
        if missing:
            errors.append(f"missing tables: {', '.join(missing)}")
        if extra:
            errors.append(f"extra tables: {', '.join(extra)}")

    for table_name in sorted(expected_tables & actual["tables"]):
        if actual["columns"][table_name] != BASELINE_COLUMNS[table_name]:
            errors.append(f"columns differ for table {table_name}")
        if actual["indexes"][table_name] != BASELINE_INDEXES[table_name]:
            errors.append(f"indexes differ for table {table_name}")
        if actual["foreign_keys"].get(table_name, set()) != BASELINE_FOREIGN_KEYS.get(
            table_name, set()
        ):
            errors.append(f"foreign keys differ for table {table_name}")
        if any(default is not None for _, default in actual["defaults"][table_name]):
            errors.append(f"server defaults are not allowed for table {table_name}")
        if actual["unique_constraints"][table_name]:
            errors.append(f"unexpected unique constraints for table {table_name}")
        if actual["check_constraints"][table_name]:
            errors.append(f"unexpected check constraints for table {table_name}")

    _validate_counter_row(connection, errors)
    if errors:
        raise BaselineSchemaMismatch(
            "Existing database does not match frozen revision "
            f"{BASELINE_REVISION}:\n- " + "\n- ".join(errors)
        )


def _make_config(database_url: str, config_path: Path) -> Config:
    config = Config(str(config_path))
    config.set_main_option("script_location", str(config_path.parent / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def run_migrations(
    database_url: str | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Upgrade an empty/managed DB or safely stamp the exact pre-Alembic baseline."""

    resolved_url = database_url or resolve_database_url()
    resolved_config_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    config = _make_config(resolved_url, resolved_config_path)

    engine: Engine = create_engine(resolved_url)
    try:
        with engine.connect() as connection:
            tables = set(inspect(connection).get_table_names())
            if not tables:
                action = "upgrade"
            elif "alembic_version" in tables:
                action = "upgrade"
            else:
                _validate_baseline(connection)
                action = "stamp"
    finally:
        engine.dispose()

    if action == "stamp":
        command.stamp(config, BASELINE_REVISION)
    command.upgrade(config, "head")


def main() -> None:
    run_migrations()


if __name__ == "__main__":
    main()
