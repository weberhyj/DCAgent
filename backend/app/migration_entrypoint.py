from __future__ import annotations

import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy import Engine, create_engine, inspect, text

from .database import resolve_database_url


BASELINE_REVISION = "20260715_00"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "alembic.ini"
POSTGRES_MIGRATION_LOCK_ID = 0x44434147454E54


@dataclass(frozen=True, slots=True)
class TypeSignature:
    family: str
    length: int | None = None
    collation: str | None = None
    precision: int | None = None
    scale: int | None = None
    asdecimal: bool | None = None


@dataclass(frozen=True, slots=True)
class IndexSignature:
    columns: tuple[str, ...]
    unique: bool
    semantic_options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PrimaryKeySignature:
    columns: tuple[str, ...]
    semantic_options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ForeignKeySignature:
    constrained_columns: tuple[str, ...]
    referred_schema: str | None
    referred_table: str
    referred_columns: tuple[str, ...]
    ondelete: str | None = None
    onupdate: str | None = None
    deferrable: bool | None = None
    initially: str | None = None
    match: str | None = None
    semantic_options: tuple[str, ...] = ()


ColumnSignature = tuple[Any, ...]


def _string(length: int | None, collation: str | None = None) -> TypeSignature:
    return TypeSignature("varchar", length, collation=collation)


TEXT = TypeSignature("text")
JSON = TypeSignature("json")
BOOLEAN = TypeSignature("boolean")
INTEGER = TypeSignature("integer")
BIGINT = TypeSignature("bigint")
FLOAT = TypeSignature("float", asdecimal=False)


def _index(columns: tuple[str, ...], unique: bool = False) -> IndexSignature:
    return IndexSignature(columns=columns, unique=unique)


def _foreign_key(
    constrained_columns: tuple[str, ...],
    referred_table: str,
    referred_columns: tuple[str, ...],
    ondelete: str,
) -> ForeignKeySignature:
    return ForeignKeySignature(
        constrained_columns=constrained_columns,
        referred_schema=None,
        referred_table=referred_table,
        referred_columns=referred_columns,
        ondelete=ondelete,
        onupdate="NO ACTION",
        deferrable=False,
        match="SIMPLE",
    )


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

BASELINE_PRIMARY_KEYS: dict[str, PrimaryKeySignature] = {
    table_name: PrimaryKeySignature(
        columns=tuple(
            column_name
            for column_name, _column_type, _nullable, primary_key in table_columns
            if primary_key
        )
    )
    for table_name, table_columns in BASELINE_COLUMNS.items()
}

BASELINE_INDEXES: dict[str, dict[str, IndexSignature]] = {
    "conversations": {"ix_conversations_sort_order": _index(("sort_order",))},
    "messages": {"ix_messages_conversation_id": _index(("conversation_id",))},
    "agent_runs": {"ix_agent_runs_conversation_id": _index(("conversation_id",))},
    "agent_steps": {"ix_agent_steps_run_id": _index(("run_id",))},
    "evaluation_cases": {
        "ix_evaluation_cases_category": _index(("category",)),
        "ix_evaluation_cases_external_key": _index(("external_key",)),
        "ix_evaluation_cases_import_batch_id": _index(("import_batch_id",)),
        "ix_evaluation_cases_sort_order": _index(("sort_order",)),
    },
    "evaluation_import_batches": {},
    "evaluation_counters": {},
    "evaluation_batches": {
        "ix_evaluation_batches_status": _index(("status",)),
        "ix_evaluation_batches_started_at": _index(("started_at",)),
    },
    "evaluation_runs": {
        "ix_evaluation_runs_case_id": _index(("case_id",)),
        "ix_evaluation_runs_batch_id": _index(("batch_id",)),
        "ix_evaluation_runs_sequence": _index(("sequence",), unique=True),
        "ix_evaluation_runs_case_id_sequence": _index(("case_id", "sequence")),
    },
    "knowledge_sources": {"ix_knowledge_sources_sort_order": _index(("sort_order",))},
    "knowledge_chunks": {"ix_knowledge_chunks_source_id": _index(("source_id",))},
}

BASELINE_FOREIGN_KEYS: dict[str, tuple[ForeignKeySignature, ...]] = {
    "messages": (
        _foreign_key(("conversation_id",), "conversations", ("id",), "CASCADE"),
    ),
    "agent_steps": (
        _foreign_key(("run_id",), "agent_runs", ("id",), "CASCADE"),
    ),
    "evaluation_runs": (
        _foreign_key(("case_id",), "evaluation_cases", ("id",), "CASCADE"),
        _foreign_key(("batch_id",), "evaluation_batches", ("id",), "SET NULL"),
    ),
    "knowledge_chunks": (
        _foreign_key(("source_id",), "knowledge_sources", ("id",), "CASCADE"),
    ),
}


class BaselineSchemaMismatch(RuntimeError):
    """Raised when an unmanaged database is not the exact frozen baseline."""


class ManagedRevisionStateError(RuntimeError):
    """Raised when an Alembic-managed database has an unsafe revision state."""


def _normalize_type(column_type: Any) -> TypeSignature:
    visit_name = str(getattr(column_type, "__visit_name__", "")).lower()
    family_by_visit_name = {
        "string": "varchar",
        "varchar": "varchar",
        "nvarchar": "nvarchar",
        "char": "char",
        "nchar": "nchar",
        "text": "text",
        "json": "json",
        "jsonb": "jsonb",
        "boolean": "boolean",
        "integer": "integer",
        "int": "integer",
        "small_integer": "smallint",
        "smallint": "smallint",
        "big_integer": "bigint",
        "bigint": "bigint",
        "float": "float",
        "double_precision": "float",
        "real": "real",
        "numeric": "numeric",
        "decimal": "numeric",
    }
    family = family_by_visit_name.get(visit_name)
    if family is None:
        family = (
            "unsupported:"
            f"{column_type.__class__.__module__}.{column_type.__class__.__qualname__}"
        )

    string_family = family in {"varchar", "nvarchar", "char", "nchar", "text"}
    numeric_family = family in {"numeric", "float", "real"}
    precision = getattr(column_type, "precision", None) if numeric_family else None
    if visit_name in {"double_precision", "real"}:
        precision = None
    elif visit_name == "float" and isinstance(precision, int):
        if 1 <= precision <= 24:
            family = "real"
            precision = None
        elif 25 <= precision <= 53:
            precision = None
    return TypeSignature(
        family=family,
        length=getattr(column_type, "length", None) if string_family else None,
        collation=getattr(column_type, "collation", None) if string_family else None,
        precision=precision,
        scale=getattr(column_type, "scale", None) if family == "numeric" else None,
        asdecimal=getattr(column_type, "asdecimal", None) if numeric_family else None,
    )


def _has_semantic_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (tuple, list, dict, set, frozenset)):
        return bool(value)
    return True


def _semantic_value(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{key}:{_semantic_value(item)}" for key, item in sorted(value.items())
        ) + "}"
    if isinstance(value, (tuple, list, set, frozenset)):
        return "[" + ",".join(_semantic_value(item) for item in value) + "]"
    return str(value).strip()


def _normalize_column(
    column: Any,
    primary_key_column_names: set[str],
) -> ColumnSignature:
    signature: ColumnSignature = (
        str(column["name"]),
        _normalize_type(column["type"]),
        bool(column["nullable"]),
        str(column["name"]) in primary_key_column_names,
    )
    semantic_options: list[str] = []
    for field in ("computed", "identity"):
        if field in column and column.get(field) is not None:
            semantic_options.append(
                f"{field}={_semantic_value(column.get(field))}"
            )
    autoincrement = column.get("autoincrement")
    if autoincrement is not None and autoincrement is not False:
        semantic_options.append(
            f"autoincrement={_semantic_value(autoincrement)}"
        )
    if semantic_options:
        return (*signature, tuple(sorted(set(semantic_options))))
    return signature


def _normalize_primary_key(primary_key: Any) -> PrimaryKeySignature:
    semantic_options: list[str] = []
    for name, value in (primary_key.get("dialect_options") or {}).items():
        if _has_semantic_value(value):
            semantic_options.append(f"{name}={_semantic_value(value)}")
    known_fields = {
        "name",
        "constrained_columns",
        "comment",
        "dialect_options",
    }
    semantic_options.extend(
        f"{name}={_semantic_value(value)}"
        for name, value in primary_key.items()
        if name not in known_fields and _has_semantic_value(value)
    )
    return PrimaryKeySignature(
        columns=tuple(primary_key.get("constrained_columns") or ()),
        semantic_options=tuple(sorted(set(semantic_options))),
    )


def _is_default_dialect_index_option(name: str, value: Any) -> bool:
    normalized_name = name.lower()
    if not _has_semantic_value(value):
        return True
    if normalized_name.endswith("_using") and str(value).lower() == "btree":
        return True
    return False


def _normalize_index(index: Any) -> IndexSignature:
    columns = tuple(index.get("column_names") or ())
    semantic_options: list[str] = []
    if any(column is None for column in columns):
        semantic_options.append("expression-column")

    for field in ("expressions", "column_sorting", "include_columns"):
        value = index.get(field)
        if _has_semantic_value(value):
            semantic_options.append(f"{field}={_semantic_value(value)}")

    duplicates_constraint = index.get("duplicates_constraint")
    if _has_semantic_value(duplicates_constraint):
        semantic_options.append(
            f"duplicates_constraint={_semantic_value(duplicates_constraint)}"
        )

    for name, value in (index.get("dialect_options") or {}).items():
        if not _is_default_dialect_index_option(str(name), value):
            semantic_options.append(f"{name}={_semantic_value(value)}")

    known_fields = {
        "name",
        "column_names",
        "unique",
        "dialect_options",
        "expressions",
        "column_sorting",
        "include_columns",
        "duplicates_constraint",
    }
    for name, value in index.items():
        if name not in known_fields and _has_semantic_value(value):
            semantic_options.append(f"{name}={_semantic_value(value)}")

    return IndexSignature(
        columns=tuple(str(column) for column in columns if column is not None),
        unique=bool(index.get("unique")),
        semantic_options=tuple(sorted(set(semantic_options))),
    )


_PLAIN_POSTGRES_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_$]*$")
_QUOTED_POSTGRES_IDENTIFIER = re.compile(r'^"((?:[^"]|"")+)"$')


def _plain_postgres_index_column(definition: Any) -> str | None:
    candidate = str(definition or "").strip()
    if _PLAIN_POSTGRES_IDENTIFIER.fullmatch(candidate):
        return candidate
    quoted = _QUOTED_POSTGRES_IDENTIFIER.fullmatch(candidate)
    if quoted:
        return quoted.group(1).replace('""', '"')
    return None


def _normalize_postgres_index_catalog_row(row: Any) -> IndexSignature:
    semantic_options: list[str] = []
    if row.get("is_primary") is True:
        if (
            "primary_constraint_oid" not in row
            or row.get("primary_constraint_oid") is None
        ):
            semantic_options.append("missing-primary-constraint")
        if "primary_constraint_deferrable" not in row:
            semantic_options.append("missing-primary-constraint-deferrable")
        elif row.get("primary_constraint_deferrable") is not False:
            semantic_options.append("primary-constraint-deferrable")
        if "primary_constraint_deferred" not in row:
            semantic_options.append("missing-primary-constraint-deferred")
        elif row.get("primary_constraint_deferred") is not False:
            semantic_options.append("primary-constraint-deferred")
    if row.get("is_valid") is not True:
        semantic_options.append("invalid")
    if row.get("is_ready") is not True:
        semantic_options.append("not-ready")
    if row.get("is_live") is not True:
        semantic_options.append("not-live")
    if "nulls_not_distinct" not in row:
        semantic_options.append("missing-nulls-not-distinct")
    elif row.get("nulls_not_distinct") is not False:
        semantic_options.append("nulls-not-distinct")
    if "reloptions" not in row:
        semantic_options.append("missing-reloptions")
    elif row.get("reloptions") is not None:
        semantic_options.append(
            f"reloptions={_semantic_value(row.get('reloptions'))}"
        )
    if "tablespace" not in row:
        semantic_options.append("missing-tablespace")
    elif row.get("tablespace") is not None:
        semantic_options.append(
            f"tablespace={_semantic_value(row.get('tablespace'))}"
        )
    if str(row.get("access_method") or "").lower() != "btree":
        semantic_options.append(f"access_method={row.get('access_method')}")
    for field in ("predicate", "expressions"):
        if _has_semantic_value(row.get(field)):
            semantic_options.append(f"{field}={_semantic_value(row.get(field))}")

    key_count = row.get("key_attribute_count")
    total_count = row.get("total_attribute_count")
    if key_count is None or total_count is None:
        semantic_options.append("missing-attribute-counts")
    elif int(key_count) != int(total_count):
        semantic_options.append("included-columns")

    definitions = tuple(row.get("key_definitions") or ())
    if key_count is None or len(definitions) != int(key_count):
        semantic_options.append("key-definition-count")
    columns: list[str] = []
    for definition in definitions:
        column = _plain_postgres_index_column(definition)
        if column is None:
            semantic_options.append(f"non-plain-key={_semantic_value(definition)}")
        else:
            columns.append(column)

    return IndexSignature(
        columns=tuple(columns),
        unique=bool(row.get("is_unique")),
        semantic_options=tuple(sorted(set(semantic_options))),
    )


def _normalize_action(value: Any, default: str) -> str:
    candidate = str(value or "").strip().upper()
    return candidate or default


def _normalize_foreign_key(
    foreign_key: Any,
    default_schema_name: str | None,
    current_schema_name: str | None = None,
) -> ForeignKeySignature:
    referred_schema = foreign_key.get("referred_schema")
    inspected_schema_name = current_schema_name or default_schema_name
    if referred_schema in {None, "", inspected_schema_name}:
        referred_schema = None

    options = {str(name).lower(): value for name, value in (
        foreign_key.get("options") or {}
    ).items()}
    ondelete = _normalize_action(options.pop("ondelete", None), "NO ACTION")
    onupdate = _normalize_action(options.pop("onupdate", None), "NO ACTION")
    deferrable_value = options.pop("deferrable", None)
    deferrable = bool(deferrable_value) if deferrable_value is not None else False
    initially_value = options.pop("initially", None)
    initially = (
        str(initially_value).strip().upper()
        if _has_semantic_value(initially_value)
        else None
    )
    match = _normalize_action(options.pop("match", None), "SIMPLE")

    semantic_options = [
        f"{name}={_semantic_value(value)}"
        for name, value in options.items()
        if _has_semantic_value(value)
    ]
    known_fields = {
        "name",
        "constrained_columns",
        "referred_schema",
        "referred_table",
        "referred_columns",
        "options",
    }
    semantic_options.extend(
        f"{name}={_semantic_value(value)}"
        for name, value in foreign_key.items()
        if name not in known_fields and _has_semantic_value(value)
    )

    return ForeignKeySignature(
        constrained_columns=tuple(foreign_key.get("constrained_columns") or ()),
        referred_schema=referred_schema,
        referred_table=str(foreign_key.get("referred_table") or ""),
        referred_columns=tuple(foreign_key.get("referred_columns") or ()),
        ondelete=ondelete,
        onupdate=onupdate,
        deferrable=deferrable,
        initially=initially,
        match=match,
        semantic_options=tuple(sorted(set(semantic_options))),
    )


def _get_foreign_keys(
    inspector: Any,
    table_name: str,
    dialect_name: str,
    schema_name: str | None = None,
) -> list[Any]:
    schema_arguments = {"schema": schema_name} if schema_name is not None else {}
    if dialect_name == "postgresql":
        return inspector.get_foreign_keys(
            table_name,
            **schema_arguments,
            postgresql_ignore_search_path=True,
        )
    return inspector.get_foreign_keys(table_name, **schema_arguments)


def _normalize_postgres_foreign_key_catalog_row(row: Any) -> tuple[str, ...]:
    semantic_options: list[str] = []
    if not str(row.get("constraint_name") or "").strip():
        semantic_options.append("missing-constraint-name")
    if "is_validated" not in row:
        semantic_options.append("missing-validated-state")
    elif row.get("is_validated") is not True:
        semantic_options.append("not-validated")
    if "is_enforced" not in row:
        semantic_options.append("missing-enforced-state")
    elif row.get("is_enforced") is not True:
        semantic_options.append("not-enforced")
    return tuple(sorted(set(semantic_options)))


POSTGRES_FOREIGN_KEY_CATALOG_SQL = text(
    """
    SELECT
        constraint_row.conname AS constraint_name,
        constraint_row.convalidated AS is_validated,
        COALESCE(
            (to_jsonb(constraint_row)->>'conenforced')::boolean,
            TRUE
        ) AS is_enforced
    FROM pg_constraint AS constraint_row
    JOIN pg_class AS table_class
      ON table_class.oid = constraint_row.conrelid
    JOIN pg_namespace AS table_namespace
      ON table_namespace.oid = table_class.relnamespace
    WHERE table_namespace.nspname = :schema_name
      AND table_class.relname = :table_name
      AND constraint_row.contype = 'f'
    """
)


POSTGRES_INDEX_CATALOG_SQL = text(
    """
    SELECT
        index_class.relname AS index_name,
        index_state.indisunique AS is_unique,
        index_state.indisvalid AS is_valid,
        index_state.indisready AS is_ready,
        index_state.indislive AS is_live,
        index_state.indisprimary AS is_primary,
        primary_constraint.oid AS primary_constraint_oid,
        primary_constraint.condeferrable AS primary_constraint_deferrable,
        primary_constraint.condeferred AS primary_constraint_deferred,
        index_state.indnullsnotdistinct AS nulls_not_distinct,
        index_class.reloptions AS reloptions,
        index_tablespace.spcname AS tablespace,
        access_method.amname AS access_method,
        pg_get_expr(index_state.indpred, index_state.indrelid) AS predicate,
        pg_get_expr(index_state.indexprs, index_state.indrelid) AS expressions,
        index_state.indnkeyatts AS key_attribute_count,
        index_state.indnatts AS total_attribute_count,
        ARRAY(
            SELECT pg_get_indexdef(index_state.indexrelid, position, true)
            FROM generate_series(1, index_state.indnkeyatts) AS position
            ORDER BY position
        ) AS key_definitions
    FROM pg_index AS index_state
    JOIN pg_class AS table_class
      ON table_class.oid = index_state.indrelid
    JOIN pg_class AS index_class
      ON index_class.oid = index_state.indexrelid
    JOIN pg_namespace AS table_namespace
      ON table_namespace.oid = table_class.relnamespace
    JOIN pg_am AS access_method
      ON access_method.oid = index_class.relam
    LEFT JOIN pg_constraint AS primary_constraint
      ON primary_constraint.contype = 'p'
     AND primary_constraint.conindid = index_state.indexrelid
    LEFT JOIN pg_tablespace AS index_tablespace
      ON index_tablespace.oid = index_class.reltablespace
    WHERE table_namespace.nspname = :schema_name
      AND table_class.relname = :table_name
    """
)


def _sqlite_index_list(connection: Any, table_name: str) -> list[Any]:
    quoted_table = table_name.replace('"', '""')
    return connection.execute(
        text(f'PRAGMA index_list("{quoted_table}")')
    ).mappings().all()


def _normalize_sqlite_index_catalog_row(
    connection: Any,
    index: Any,
) -> IndexSignature:
    index_name = str(index["name"])
    quoted_name = index_name.replace('"', '""')
    rows = connection.execute(
        text(f'PRAGMA index_xinfo("{quoted_name}")')
    ).mappings().all()
    semantic_options: list[str] = []
    if bool(index.get("partial")):
        semantic_options.append("sqlite-partial")
    key_rows = sorted(
        (row for row in rows if bool(row.get("key"))),
        key=lambda row: int(row.get("seqno", 0)),
    )
    columns: list[str] = []
    for row in key_rows:
        column_name = row.get("name")
        if column_name is None or int(row.get("cid", -2)) < 0:
            semantic_options.append("sqlite-expression-key")
        else:
            columns.append(str(column_name))
        if bool(row.get("desc")):
            semantic_options.append("sqlite-descending-key")
        if str(row.get("coll") or "BINARY").upper() != "BINARY":
            semantic_options.append(f"sqlite-collation={row.get('coll')}")
    return IndexSignature(
        columns=tuple(columns),
        unique=bool(index.get("unique")),
        semantic_options=tuple(sorted(set(semantic_options))),
    )


def _require_supported_postgres_version(connection: Any) -> None:
    if connection.dialect.name != "postgresql":
        return
    server_version = tuple(connection.dialect.server_version_info or ())
    if server_version < (15,):
        raise RuntimeError(
            "PostgreSQL 15+ required for exact index validation"
        )


def _validate_postgres_index_catalog(
    connection: Any,
    errors: list[str],
    schema_name: str,
) -> None:
    _require_supported_postgres_version(connection)
    for table_name, expected_indexes in BASELINE_INDEXES.items():
        rows = connection.execute(
            POSTGRES_INDEX_CATALOG_SQL,
            {"schema_name": schema_name, "table_name": table_name},
        ).mappings().all()
        primary_rows = [row for row in rows if row.get("is_primary") is True]
        expected_primary_key = BASELINE_PRIMARY_KEYS[table_name]
        expected_primary_index = IndexSignature(
            columns=expected_primary_key.columns,
            unique=True,
        )
        if (
            len(primary_rows) != 1
            or _normalize_postgres_index_catalog_row(primary_rows[0])
            != expected_primary_index
        ):
            errors.append(
                f"PostgreSQL primary index catalog differs for table {table_name}"
            )
        actual_indexes = {
            str(row["index_name"]): _normalize_postgres_index_catalog_row(row)
            for row in rows
            if row.get("is_primary") is not True
        }
        if actual_indexes != expected_indexes:
            errors.append(f"PostgreSQL index catalog differs for table {table_name}")


def _validate_postgres_foreign_key_catalog(
    connection: Any,
    errors: list[str],
    schema_name: str,
    actual_foreign_keys: dict[str, tuple[ForeignKeySignature, ...]],
    table_names: set[str],
) -> None:
    _require_supported_postgres_version(connection)
    for table_name in sorted(table_names):
        rows = connection.execute(
            POSTGRES_FOREIGN_KEY_CATALOG_SQL,
            {"schema_name": schema_name, "table_name": table_name},
        ).mappings().all()
        if (
            len(rows) != len(actual_foreign_keys.get(table_name, ()))
            or any(
                _normalize_postgres_foreign_key_catalog_row(row)
                for row in rows
            )
        ):
            errors.append(
                f"PostgreSQL foreign key catalog differs for table {table_name}"
            )


def _schema_fingerprint(
    connection: Any,
    ignored_tables: frozenset[str] = frozenset(),
    schema_name: str | None = None,
) -> dict[str, Any]:
    inspector = inspect(connection)
    current_schema_name = schema_name
    if connection.dialect.name == "postgresql" and current_schema_name is None:
        current_schema_name = connection.scalar(text("SELECT current_schema()"))
    schema_arguments = (
        {"schema": current_schema_name}
        if current_schema_name is not None
        else {}
    )
    tables = set(inspector.get_table_names(**schema_arguments)) - set(ignored_tables)
    columns: dict[str, tuple[ColumnSignature, ...]] = {}
    primary_keys: dict[str, PrimaryKeySignature] = {}
    indexes: dict[str, dict[str, IndexSignature]] = {}
    foreign_keys: dict[str, tuple[ForeignKeySignature, ...]] = {}
    defaults: dict[str, tuple[tuple[str, Any], ...]] = {}
    unique_constraints: dict[str, tuple[Any, ...]] = {}
    check_constraints: dict[str, tuple[Any, ...]] = {}
    default_schema_name = inspector.default_schema_name
    if current_schema_name is None:
        current_schema_name = default_schema_name

    for table_name in sorted(tables):
        table_columns = inspector.get_columns(table_name, **schema_arguments)
        primary_key = inspector.get_pk_constraint(table_name, **schema_arguments)
        primary_key_signature = _normalize_primary_key(primary_key)
        primary_keys[table_name] = primary_key_signature
        primary_key_column_names = set(primary_key_signature.columns)
        columns[table_name] = tuple(
            _normalize_column(column, primary_key_column_names)
            for column in table_columns
        )
        defaults[table_name] = tuple(
            (column["name"], column.get("default")) for column in table_columns
        )
        if connection.dialect.name == "sqlite":
            sqlite_indexes = _sqlite_index_list(connection, table_name)
            indexes[table_name] = {
                str(index["name"]): _normalize_sqlite_index_catalog_row(
                    connection, index
                )
                for index in sqlite_indexes
                if str(index.get("origin") or "").lower() == "c"
            }
        else:
            indexes[table_name] = {
                str(index["name"]): _normalize_index(index)
                for index in inspector.get_indexes(table_name, **schema_arguments)
            }
        foreign_keys[table_name] = tuple(
            _normalize_foreign_key(
                foreign_key,
                default_schema_name=default_schema_name,
                current_schema_name=current_schema_name,
            )
            for foreign_key in _get_foreign_keys(
                inspector,
                table_name,
                connection.dialect.name,
                schema_name=(
                    current_schema_name
                    if connection.dialect.name == "postgresql"
                    else None
                ),
            )
        )
        if connection.dialect.name == "sqlite":
            unique_constraints[table_name] = tuple(
                (
                    index.get("name"),
                    _normalize_sqlite_index_catalog_row(
                        connection, index
                    ).columns,
                )
                for index in sqlite_indexes
                if str(index.get("origin") or "").lower() == "u"
            )
        else:
            unique_constraints[table_name] = tuple(
                (
                    constraint.get("name"),
                    tuple(constraint.get("column_names") or ()),
                )
                for constraint in inspector.get_unique_constraints(
                    table_name, **schema_arguments
                )
            )
        check_constraints[table_name] = tuple(
            (
                constraint.get("name"),
                constraint.get("sqltext"),
            )
            for constraint in inspector.get_check_constraints(
                table_name, **schema_arguments
            )
        )

    return {
        "tables": tables,
        "columns": columns,
        "primary_keys": primary_keys,
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


def _validate_baseline(
    connection: Any,
    ignored_tables: frozenset[str] = frozenset(),
    schema_name: str | None = None,
) -> None:
    if connection.dialect.name == "postgresql" and schema_name is None:
        schema_name = connection.scalar(text("SELECT current_schema()"))
    actual = _schema_fingerprint(
        connection,
        ignored_tables=ignored_tables,
        schema_name=schema_name,
    )
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
        if actual["primary_keys"][table_name] != BASELINE_PRIMARY_KEYS[table_name]:
            errors.append(f"primary key differs for table {table_name}")
        if actual["indexes"][table_name] != BASELINE_INDEXES[table_name]:
            errors.append(f"indexes differ for table {table_name}")
        if Counter(actual["foreign_keys"].get(table_name, ())) != Counter(
            BASELINE_FOREIGN_KEYS.get(table_name, ())
        ):
            errors.append(f"foreign keys differ for table {table_name}")
        if any(default is not None for _, default in actual["defaults"][table_name]):
            errors.append(f"server defaults are not allowed for table {table_name}")
        if actual["unique_constraints"][table_name]:
            errors.append(f"unexpected unique constraints for table {table_name}")
        if actual["check_constraints"][table_name]:
            errors.append(f"unexpected check constraints for table {table_name}")

    if connection.dialect.name == "postgresql":
        try:
            if schema_name is None:
                raise RuntimeError("current PostgreSQL schema is unavailable")
            _validate_postgres_index_catalog(connection, errors, schema_name)
        except Exception as error:
            errors.append(
                "PostgreSQL index catalog could not be validated: "
                f"{error}"
            )
        try:
            if schema_name is None:
                raise RuntimeError("current PostgreSQL schema is unavailable")
            _validate_postgres_foreign_key_catalog(
                connection,
                errors,
                schema_name,
                actual["foreign_keys"],
                expected_tables & actual["tables"],
            )
        except Exception as error:
            errors.append(
                "PostgreSQL foreign key catalog could not be validated: "
                f"{error}"
            )

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


def _validate_alembic_version_table(
    connection: Any,
    schema_name: str | None = None,
) -> None:
    _require_supported_postgres_version(connection)
    inspector = inspect(connection)
    if connection.dialect.name == "postgresql" and schema_name is None:
        schema_name = connection.scalar(text("SELECT current_schema()"))
    schema_arguments = {"schema": schema_name} if schema_name is not None else {}
    postgres_primary_indexes: tuple[IndexSignature, ...] | None = None
    try:
        columns = inspector.get_columns("alembic_version", **schema_arguments)
        primary_key = inspector.get_pk_constraint(
            "alembic_version", **schema_arguments
        )
        unique_constraints = inspector.get_unique_constraints(
            "alembic_version", **schema_arguments
        )
        indexes = inspector.get_indexes("alembic_version", **schema_arguments)
        check_constraints = inspector.get_check_constraints(
            "alembic_version", **schema_arguments
        )
        foreign_keys = _get_foreign_keys(
            inspector,
            "alembic_version",
            connection.dialect.name,
            schema_name=schema_name,
        )
        if connection.dialect.name == "postgresql":
            catalog_rows = connection.execute(
                POSTGRES_INDEX_CATALOG_SQL,
                {
                    "schema_name": schema_name,
                    "table_name": "alembic_version",
                },
            ).mappings().all()
            postgres_primary_indexes = tuple(
                _normalize_postgres_index_catalog_row(row)
                for row in catalog_rows
                if row.get("is_primary") is True
            )
    except Exception as error:
        raise ManagedRevisionStateError(
            "Alembic revision table structure could not be inspected"
        ) from error

    primary_key_signature = _normalize_primary_key(primary_key)
    column_signatures = tuple(
        _normalize_column(column, set(primary_key_signature.columns))
        for column in columns
    )
    expected_columns: tuple[ColumnSignature, ...] = (
        ("version_num", _string(32), False, True),
    )
    expected_primary_index = IndexSignature(
        columns=("version_num",),
        unique=True,
    )
    has_server_default = any(column.get("default") is not None for column in columns)
    if (
        column_signatures != expected_columns
        or primary_key_signature != PrimaryKeySignature(("version_num",))
        or (
            connection.dialect.name == "postgresql"
            and postgres_primary_indexes != (expected_primary_index,)
        )
        or has_server_default
        or unique_constraints
        or indexes
        or check_constraints
        or foreign_keys
    ):
        raise ManagedRevisionStateError(
            "Alembic revision table does not match the required structure"
        )


def _read_managed_revision(
    connection: Any,
    script_directory: ScriptDirectory,
) -> str:
    try:
        revisions = tuple(MigrationContext.configure(connection).get_current_heads())
    except Exception as error:
        raise ManagedRevisionStateError(
            "Alembic revision state could not be read"
        ) from error
    if len(revisions) != 1:
        raise ManagedRevisionStateError(
            "Alembic revision table must contain exactly one revision"
        )

    revision = revisions[0]
    if not isinstance(revision, str) or not revision or revision != revision.strip():
        raise ManagedRevisionStateError("Alembic revision must be one exact revision ID")
    try:
        resolved_revision = script_directory.get_revision(revision)
    except CommandError as error:
        raise ManagedRevisionStateError(
            f"Alembic revision is unknown: {revision}"
        ) from error
    if resolved_revision is None or resolved_revision.revision != revision:
        raise ManagedRevisionStateError(
            "Alembic revision must not be symbolic or abbreviated"
        )
    return revision


def _classify_and_validate(
    connection: Any,
    config: Config,
) -> str:
    _require_supported_postgres_version(connection)
    schema_name: str | None = None
    if connection.dialect.name == "postgresql":
        schema_name = connection.scalar(text("SELECT current_schema()"))
    schema_arguments = {"schema": schema_name} if schema_name is not None else {}
    tables = set(inspect(connection).get_table_names(**schema_arguments))
    if not tables:
        return "upgrade"
    if "alembic_version" not in tables:
        _validate_baseline(connection, schema_name=schema_name)
        return "stamp"

    _validate_alembic_version_table(connection, schema_name=schema_name)
    script_directory = ScriptDirectory.from_config(config)
    revision = _read_managed_revision(connection, script_directory)
    if revision == BASELINE_REVISION:
        if tables == {"alembic_version"}:
            raise ManagedRevisionStateError(
                "Baseline revision exists without the frozen application schema"
            )
        _validate_baseline(
            connection,
            ignored_tables=frozenset({"alembic_version"}),
            schema_name=schema_name,
        )
    return "upgrade"


def _release_postgres_migration_lock(connection: Any) -> None:
    if connection.in_transaction():
        connection.rollback()
    connection.execute(
        text("SELECT pg_advisory_unlock(:lock_id)"),
        {"lock_id": POSTGRES_MIGRATION_LOCK_ID},
    )
    connection.commit()


@contextmanager
def _postgres_migration_lock(connection: Any) -> Any:
    if connection.dialect.name != "postgresql":
        yield
        return

    connection.execute(
        text("SELECT pg_advisory_lock(:lock_id)"),
        {"lock_id": POSTGRES_MIGRATION_LOCK_ID},
    )
    connection.commit()
    try:
        yield
    except BaseException:
        try:
            _release_postgres_migration_lock(connection)
        except Exception:
            connection.invalidate()
        raise
    else:
        _release_postgres_migration_lock(connection)


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
            config.attributes["connection"] = connection
            with _postgres_migration_lock(connection):
                with connection.begin():
                    action = _classify_and_validate(connection, config)
                    if action == "stamp":
                        command.stamp(config, BASELINE_REVISION)
                    command.upgrade(config, "head")
    finally:
        engine.dispose()


def main() -> None:
    run_migrations()


if __name__ == "__main__":
    main()
