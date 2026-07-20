from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from alembic import command

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REVISION = "20260715_00"

EXPECTED_COLUMNS: dict[str, tuple[tuple[str, str, bool, bool], ...]] = {
    "agent_runs": (
        ("id", "VARCHAR(64)", False, True),
        ("conversation_id", "VARCHAR(64)", False, False),
        ("query", "TEXT", False, False),
        ("mode", "VARCHAR(20)", False, False),
        ("status", "VARCHAR(20)", False, False),
        ("started_at", "VARCHAR(40)", False, False),
        ("completed_at", "VARCHAR(40)", False, False),
        ("answer_message_id", "VARCHAR(64)", False, False),
        ("evidence_count", "INTEGER", False, False),
        ("source_count", "INTEGER", False, False),
    ),
    "agent_steps": (
        ("id", "VARCHAR(64)", False, True),
        ("run_id", "VARCHAR(64)", False, False),
        ("step_index", "INTEGER", False, False),
        ("tool_name", "VARCHAR(80)", False, False),
        ("status", "VARCHAR(20)", False, False),
        ("input_summary", "TEXT", False, False),
        ("output_summary", "TEXT", False, False),
        ("source_ids", "JSON", False, False),
        ("read_only", "BOOLEAN", False, False),
        ("started_at", "VARCHAR(40)", False, False),
        ("completed_at", "VARCHAR(40)", False, False),
    ),
    "conversations": (
        ("id", "VARCHAR(64)", False, True),
        ("title", "VARCHAR(200)", False, False),
        ("topic", "VARCHAR(80)", False, False),
        ("group_name", "VARCHAR(40)", False, False),
        ("updated_at", "VARCHAR(40)", False, False),
        ("pinned", "BOOLEAN", False, False),
        ("context_summary", "TEXT", False, False),
        ("turn_count", "INTEGER", False, False),
        ("sort_order", "INTEGER", False, False),
    ),
    "evaluation_batches": (
        ("id", "VARCHAR(64)", False, True),
        ("name", "VARCHAR(120)", False, False),
        ("status", "VARCHAR(20)", False, False),
        ("case_ids", "JSON", False, False),
        ("retrieval_min_score", "FLOAT", False, False),
        ("case_count", "INTEGER", False, False),
        ("completed_count", "INTEGER", False, False),
        ("passed_count", "INTEGER", False, False),
        ("failed_count", "INTEGER", False, False),
        ("false_positive_count", "INTEGER", False, False),
        ("started_at", "VARCHAR(40)", False, False),
        ("completed_at", "VARCHAR(40)", True, False),
        ("error_message", "TEXT", True, False),
    ),
    "evaluation_cases": (
        ("id", "VARCHAR(64)", False, True),
        ("question", "TEXT", False, False),
        ("expected_source_ids", "JSON", False, False),
        ("expected_terms", "JSON", False, False),
        ("expect_answer", "BOOLEAN", False, False),
        ("top_k", "INTEGER", False, False),
        ("created_at", "VARCHAR(40)", False, False),
        ("updated_at", "VARCHAR(40)", False, False),
        ("category", "VARCHAR(80)", True, False),
        ("tags", "JSON", False, False),
        ("external_key", "VARCHAR(120)", True, False),
        ("import_batch_id", "VARCHAR(64)", True, False),
        ("sort_order", "INTEGER", False, False),
    ),
    "evaluation_counters": (
        ("name", "VARCHAR(80)", False, True),
        ("next_value", "BIGINT", False, False),
    ),
    "evaluation_import_batches": (
        ("id", "VARCHAR(64)", False, True),
        ("file_name", "VARCHAR(240)", False, False),
        ("status", "VARCHAR(20)", False, False),
        ("total_rows", "INTEGER", False, False),
        ("valid_rows", "INTEGER", False, False),
        ("invalid_rows", "INTEGER", False, False),
        ("duplicate_rows", "INTEGER", False, False),
        ("created_at", "VARCHAR(40)", False, False),
        ("completed_at", "VARCHAR(40)", True, False),
    ),
    "evaluation_runs": (
        ("id", "VARCHAR(64)", False, True),
        ("case_id", "VARCHAR(64)", False, False),
        ("batch_id", "VARCHAR(64)", True, False),
        ("question", "TEXT", False, False),
        ("status", "VARCHAR(20)", False, False),
        ("expect_answer", "BOOLEAN", False, False),
        ("answerable", "BOOLEAN", False, False),
        ("false_positive", "BOOLEAN", False, False),
        ("expected_source_ids", "JSON", False, False),
        ("matched_source_ids", "JSON", False, False),
        ("missing_source_ids", "JSON", False, False),
        ("expected_terms", "JSON", False, False),
        ("found_terms", "JSON", False, False),
        ("missing_terms", "JSON", False, False),
        ("source_recall", "FLOAT", False, False),
        ("term_recall", "FLOAT", False, False),
        ("top_score", "FLOAT", False, False),
        ("hit_count", "INTEGER", False, False),
        ("started_at", "VARCHAR(40)", False, False),
        ("completed_at", "VARCHAR(40)", False, False),
        ("sequence", "BIGINT", False, False),
        ("hits", "JSON", False, False),
    ),
    "knowledge_chunks": (
        ("id", "VARCHAR(64)", False, True),
        ("source_id", "VARCHAR(64)", False, False),
        ("chunk_index", "INTEGER", False, False),
        ("text", "TEXT", False, False),
        ("token_count", "INTEGER", False, False),
        ("embedding", "JSON", True, False),
    ),
    "knowledge_sources": (
        ("id", "VARCHAR(64)", False, True),
        ("name", "VARCHAR(240)", False, False),
        ("source_type", "VARCHAR(80)", False, False),
        ("records", "INTEGER", False, False),
        ("status", "VARCHAR(40)", False, False),
        ("updated_at", "VARCHAR(40)", False, False),
        ("classification", "VARCHAR(80)", False, False),
        ("file_path", "TEXT", True, False),
        ("file_size", "INTEGER", True, False),
        ("mime_type", "VARCHAR(120)", True, False),
        ("error_message", "TEXT", True, False),
        ("sort_order", "INTEGER", False, False),
    ),
    "messages": (
        ("id", "VARCHAR(64)", False, True),
        ("conversation_id", "VARCHAR(64)", False, False),
        ("role", "VARCHAR(20)", False, False),
        ("time", "VARCHAR(40)", False, False),
        ("content", "TEXT", True, False),
        ("paragraphs", "JSON", False, False),
        ("artifacts", "JSON", False, False),
        ("sort_order", "INTEGER", False, False),
    ),
}

EXPECTED_INDEXES = {
    "agent_runs": {"ix_agent_runs_conversation_id": (("conversation_id",), False)},
    "agent_steps": {"ix_agent_steps_run_id": (("run_id",), False)},
    "conversations": {"ix_conversations_sort_order": (("sort_order",), False)},
    "evaluation_batches": {
        "ix_evaluation_batches_started_at": (("started_at",), False),
        "ix_evaluation_batches_status": (("status",), False),
    },
    "evaluation_cases": {
        "ix_evaluation_cases_category": (("category",), False),
        "ix_evaluation_cases_external_key": (("external_key",), False),
        "ix_evaluation_cases_import_batch_id": (("import_batch_id",), False),
        "ix_evaluation_cases_sort_order": (("sort_order",), False),
    },
    "evaluation_counters": {},
    "evaluation_import_batches": {},
    "evaluation_runs": {
        "ix_evaluation_runs_batch_id": (("batch_id",), False),
        "ix_evaluation_runs_case_id": (("case_id",), False),
        "ix_evaluation_runs_case_id_sequence": (("case_id", "sequence"), False),
        "ix_evaluation_runs_sequence": (("sequence",), True),
    },
    "knowledge_chunks": {"ix_knowledge_chunks_source_id": (("source_id",), False)},
    "knowledge_sources": {"ix_knowledge_sources_sort_order": (("sort_order",), False)},
    "messages": {"ix_messages_conversation_id": (("conversation_id",), False)},
}

EXPECTED_FOREIGN_KEYS = {
    "agent_steps": {("run_id", "agent_runs", "id", "CASCADE")},
    "evaluation_runs": {
        ("batch_id", "evaluation_batches", "id", "SET NULL"),
        ("case_id", "evaluation_cases", "id", "CASCADE"),
    },
    "knowledge_chunks": {("source_id", "knowledge_sources", "id", "CASCADE")},
    "messages": {("conversation_id", "conversations", "id", "CASCADE")},
}


def make_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


class AlembicBaselineTest(unittest.TestCase):
    def test_env_uses_supplied_connection_without_building_an_engine(self) -> None:
        engine = create_engine("sqlite+pysqlite:///:memory:")
        try:
            with engine.connect() as connection:
                config = make_config("sqlite+pysqlite:///:memory:")
                config.attributes["connection"] = connection
                with patch(
                    "sqlalchemy.engine_from_config",
                    side_effect=AssertionError("external connection was ignored"),
                ):
                    command.upgrade(config, "head")
                self.assertIn("alembic_version", inspect(connection).get_table_names())
                self.assertFalse(connection.closed)
        finally:
            engine.dispose()

    def test_empty_database_upgrades_to_complete_frozen_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "baseline.db"
            database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
            command.upgrade(make_config(database_url), "head")

            engine = create_engine(database_url)
            inspector = inspect(engine)
            application_tables = set(inspector.get_table_names()) - {"alembic_version"}
            self.assertEqual(set(EXPECTED_COLUMNS), application_tables)
            version_columns = inspector.get_columns("alembic_version")
            self.assertEqual(1, len(version_columns))
            self.assertEqual("version_num", version_columns[0]["name"])
            self.assertEqual("VARCHAR(32)", str(version_columns[0]["type"]).upper())
            self.assertFalse(version_columns[0]["nullable"])
            self.assertIsNone(version_columns[0]["default"])
            self.assertEqual(
                ("version_num",),
                tuple(
                    inspector.get_pk_constraint("alembic_version").get("constrained_columns") or ()
                ),
            )

            for table_name, expected_columns in EXPECTED_COLUMNS.items():
                with self.subTest(table=table_name):
                    actual_columns = tuple(
                        (
                            column["name"],
                            str(column["type"]).upper(),
                            bool(column["nullable"]),
                            bool(column["primary_key"]),
                        )
                        for column in inspector.get_columns(table_name)
                    )
                    self.assertEqual(expected_columns, actual_columns)
                    self.assertTrue(
                        all(
                            column["default"] is None
                            for column in inspector.get_columns(table_name)
                        )
                    )
                    actual_indexes = {
                        index["name"]: (
                            tuple(index["column_names"]),
                            bool(index["unique"]),
                        )
                        for index in inspector.get_indexes(table_name)
                    }
                    self.assertEqual(EXPECTED_INDEXES[table_name], actual_indexes)

                    actual_foreign_keys = {
                        (
                            foreign_key["constrained_columns"][0],
                            foreign_key["referred_table"],
                            foreign_key["referred_columns"][0],
                            foreign_key["options"].get("ondelete"),
                        )
                        for foreign_key in inspector.get_foreign_keys(table_name)
                    }
                    self.assertEqual(
                        EXPECTED_FOREIGN_KEYS.get(table_name, set()),
                        actual_foreign_keys,
                    )

            with engine.connect() as connection:
                current_revision = connection.scalar(
                    text("SELECT version_num FROM alembic_version")
                )
                counter_row = connection.execute(
                    text(
                        "SELECT name, next_value FROM evaluation_counters "
                        "WHERE name = 'evaluation_runs'"
                    )
                ).one()
            engine.dispose()
            self.assertEqual(REVISION, current_revision)
            self.assertEqual(("evaluation_runs", 1), counter_row)

    def test_revision_is_explicit_and_does_not_depend_on_live_metadata(self) -> None:
        revision_path = BACKEND_ROOT / "alembic" / "versions" / "20260715_00_existing_schema.py"
        source = revision_path.read_text(encoding="utf-8")
        self.assertIn('revision = "20260715_00"', source)
        self.assertIn("down_revision = None", source)
        self.assertNotIn("Base.metadata", source)
        self.assertNotIn("create_all(", source)
        self.assertNotIn("app.database", source)
        for table_name in EXPECTED_COLUMNS:
            self.assertIn(f'op.create_table(\n        "{table_name}"', source)
        for indexes in EXPECTED_INDEXES.values():
            for index_name in indexes:
                self.assertIn(f'"{index_name}"', source)


if __name__ == "__main__":
    unittest.main()
