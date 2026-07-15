from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from app.database import Database
from app.migration_entrypoint import BaselineSchemaMismatch, run_migrations


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def sqlite_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def prepare_current_schema(database_url: str) -> None:
    database = Database(database_url)
    database.create_schema()
    database.engine.dispose()


def table_names(database_url: str) -> list[str]:
    engine = create_engine(database_url)
    try:
        return inspect(engine).get_table_names()
    finally:
        engine.dispose()


def column_names(database_url: str, table_name: str) -> set[str]:
    engine = create_engine(database_url)
    try:
        return {column["name"] for column in inspect(engine).get_columns(table_name)}
    finally:
        engine.dispose()


def index_names(database_url: str, table_name: str) -> set[str]:
    engine = create_engine(database_url)
    try:
        return {index["name"] for index in inspect(engine).get_indexes(table_name)}
    finally:
        engine.dispose()


def current_revision(database_url: str) -> str:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return connection.scalar(text("SELECT version_num FROM alembic_version"))
    finally:
        engine.dispose()


class MigrationEntrypointTest(unittest.TestCase):
    def test_existing_current_database_is_stamped_and_rerun_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "existing.db")
            prepare_current_schema(database_url)

            run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")
            first_tables = table_names(database_url)
            self.assertIn("alembic_version", first_tables)
            first_revision = current_revision(database_url)

            run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")
            second_tables = table_names(database_url)
            second_revision = current_revision(database_url)
            self.assertEqual(first_revision, "20260715_00")
            self.assertEqual(second_revision, first_revision)
            self.assertEqual(first_tables, second_tables)

    def test_missing_column_refuses_to_stamp_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "missing-column.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE conversations DROP COLUMN context_summary"))
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

            self.assertNotIn("alembic_version", table_names(database_url))
            self.assertNotIn(
                "context_summary",
                column_names(database_url, "conversations"),
            )

    def test_extra_column_refuses_to_stamp_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "extra-column.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE conversations ADD COLUMN unexpected TEXT"))
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

            self.assertNotIn("alembic_version", table_names(database_url))
            self.assertIn(
                "unexpected",
                column_names(database_url, "conversations"),
            )

    def test_extra_table_refuses_to_stamp_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "extra-table.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("CREATE TABLE unexpected_table (id INTEGER PRIMARY KEY)"))
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

            self.assertNotIn("alembic_version", table_names(database_url))

    def test_extra_index_refuses_to_stamp_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "extra-index.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE INDEX ix_conversations_unexpected "
                        "ON conversations (title)"
                    )
                )
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

            self.assertNotIn("alembic_version", table_names(database_url))
            self.assertIn(
                "ix_conversations_unexpected",
                index_names(database_url, "conversations"),
            )

    def test_non_unique_replacement_for_unique_index_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "wrong-unique-index.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("DROP INDEX ix_evaluation_runs_sequence"))
                connection.execute(
                    text(
                        "CREATE INDEX ix_evaluation_runs_sequence "
                        "ON evaluation_runs (sequence)"
                    )
                )
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")
            self.assertNotIn("alembic_version", table_names(database_url))

    def test_wrong_column_type_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "wrong-type.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE evaluation_counters"))
                connection.execute(
                    text(
                        "CREATE TABLE evaluation_counters "
                        "(name VARCHAR(80) NOT NULL PRIMARY KEY, next_value INTEGER NOT NULL)"
                    )
                )
                connection.execute(
                    text(
                        "INSERT INTO evaluation_counters (name, next_value) "
                        "VALUES ('evaluation_runs', 1)"
                    )
                )
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")
            self.assertNotIn("alembic_version", table_names(database_url))

    def test_missing_foreign_key_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "missing-foreign-key.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE messages"))
                connection.execute(
                    text(
                        "CREATE TABLE messages ("
                        "id VARCHAR(64) NOT NULL PRIMARY KEY, "
                        "conversation_id VARCHAR(64) NOT NULL, "
                        "role VARCHAR(20) NOT NULL, "
                        "time VARCHAR(40) NOT NULL, "
                        "content TEXT, paragraphs JSON NOT NULL, "
                        "artifacts JSON NOT NULL, sort_order INTEGER NOT NULL)"
                    )
                )
                connection.execute(
                    text(
                        "CREATE INDEX ix_messages_conversation_id "
                        "ON messages (conversation_id)"
                    )
                )
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")
            self.assertNotIn("alembic_version", table_names(database_url))

    def test_composite_foreign_key_drift_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "composite-foreign-key.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE messages"))
                connection.execute(
                    text(
                        "CREATE TABLE messages ("
                        "id VARCHAR(64) NOT NULL PRIMARY KEY, "
                        "conversation_id VARCHAR(64) NOT NULL, "
                        "role VARCHAR(20) NOT NULL, "
                        "time VARCHAR(40) NOT NULL, "
                        "content TEXT, paragraphs JSON NOT NULL, "
                        "artifacts JSON NOT NULL, sort_order INTEGER NOT NULL, "
                        "FOREIGN KEY (conversation_id, role) "
                        "REFERENCES conversations (id, topic) ON DELETE CASCADE)"
                    )
                )
                connection.execute(
                    text(
                        "CREATE INDEX ix_messages_conversation_id "
                        "ON messages (conversation_id)"
                    )
                )
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")
            self.assertNotIn("alembic_version", table_names(database_url))

    def test_partial_schema_is_not_treated_as_current_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "partial.db")
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("CREATE TABLE conversations (id VARCHAR(64) PRIMARY KEY)"))
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")
            self.assertNotIn("alembic_version", table_names(database_url))


if __name__ == "__main__":
    unittest.main()
