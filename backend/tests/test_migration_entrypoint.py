from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects import postgresql

from app import migration_entrypoint
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


class FingerprintNormalizationTest(unittest.TestCase):
    def test_type_signature_distinguishes_database_type_families(self) -> None:
        normalize = migration_entrypoint._normalize_type

        self.assertNotEqual(normalize(sa.CHAR(32)), normalize(sa.String(32)))
        self.assertNotEqual(normalize(sa.SmallInteger()), normalize(sa.Integer()))
        self.assertNotEqual(normalize(postgresql.JSONB()), normalize(sa.JSON()))
        self.assertNotEqual(normalize(sa.REAL()), normalize(sa.Float()))
        self.assertEqual(
            normalize(postgresql.DOUBLE_PRECISION()),
            normalize(sa.Float()),
        )
        equivalent_aliases = (
            (sa.String(32), postgresql.VARCHAR(32)),
            (sa.Text(), postgresql.TEXT()),
            (sa.JSON(), postgresql.JSON()),
            (sa.Boolean(), postgresql.BOOLEAN()),
            (sa.Integer(), postgresql.INTEGER()),
            (sa.BigInteger(), postgresql.BIGINT()),
        )
        for generic, postgres_type in equivalent_aliases:
            with self.subTest(generic=generic, postgres_type=postgres_type):
                self.assertEqual(normalize(generic), normalize(postgres_type))

    def test_type_signature_keeps_collation_precision_and_scale(self) -> None:
        normalize = migration_entrypoint._normalize_type

        self.assertNotEqual(
            normalize(sa.String(32, collation="C")),
            normalize(sa.String(32, collation="POSIX")),
        )
        self.assertNotEqual(
            normalize(sa.Numeric(precision=10, scale=2)),
            normalize(sa.Numeric(precision=12, scale=2)),
        )
        self.assertNotEqual(
            normalize(sa.Numeric(precision=10, scale=2)),
            normalize(sa.Numeric(precision=10, scale=4)),
        )
        self.assertNotEqual(
            normalize(sa.Numeric(precision=10, scale=2, asdecimal=True)),
            normalize(sa.Numeric(precision=10, scale=2, asdecimal=False)),
        )

    def test_generic_index_signature_fails_closed_on_semantic_options(self) -> None:
        plain = {
            "name": "ix_evaluation_runs_sequence",
            "column_names": ["sequence"],
            "unique": True,
            "dialect_options": {},
        }
        normalized = migration_entrypoint._normalize_index(plain)
        self.assertEqual(("sequence",), normalized.columns)
        self.assertTrue(normalized.unique)
        self.assertEqual((), normalized.semantic_options)

        variants = (
            {**plain, "expressions": ["lower(sequence)"]},
            {**plain, "column_sorting": {"sequence": ("desc",)}},
            {**plain, "include_columns": ["case_id"]},
            {**plain, "dialect_options": {"sqlite_where": "sequence > 0"}},
            {**plain, "dialect_options": {"postgresql_ops": {"sequence": "int8_ops"}}},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    (),
                    migration_entrypoint._normalize_index(variant).semantic_options,
                )

    def test_postgres_catalog_index_signature_requires_plain_ready_btree(self) -> None:
        plain = {
            "index_name": "ix_evaluation_runs_sequence",
            "is_unique": True,
            "is_valid": True,
            "is_ready": True,
            "is_live": True,
            "access_method": "btree",
            "predicate": None,
            "expressions": None,
            "key_attribute_count": 1,
            "total_attribute_count": 1,
            "key_definitions": ["sequence"],
        }
        normalized = migration_entrypoint._normalize_postgres_index_catalog_row(plain)
        self.assertEqual(("sequence",), normalized.columns)
        self.assertTrue(normalized.unique)
        self.assertEqual((), normalized.semantic_options)

        variants = (
            {**plain, "is_valid": False},
            {**plain, "is_ready": False},
            {**plain, "is_live": False},
            {name: value for name, value in plain.items() if name != "is_live"},
            {**plain, "access_method": "hash"},
            {**plain, "predicate": "sequence > 0"},
            {**plain, "expressions": "lower(sequence)"},
            {**plain, "total_attribute_count": 2},
            {**plain, "key_definitions": ["sequence DESC"]},
            {**plain, "key_definitions": ["sequence int8_ops"]},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    (),
                    migration_entrypoint._normalize_postgres_index_catalog_row(
                        variant
                    ).semantic_options,
                )

    def test_foreign_key_signature_includes_all_semantic_options(self) -> None:
        plain = {
            "name": "generated_name_is_ignored",
            "constrained_columns": ["conversation_id"],
            "referred_schema": None,
            "referred_table": "conversations",
            "referred_columns": ["id"],
            "options": {"ondelete": "CASCADE"},
        }
        normalized = migration_entrypoint._normalize_foreign_key(
            plain, default_schema_name="main"
        )
        same_default_schema = {
            **plain,
            "name": "different_generated_name",
            "referred_schema": "main",
        }
        self.assertEqual(
            normalized,
            migration_entrypoint._normalize_foreign_key(
                same_default_schema, default_schema_name="main"
            ),
        )

        variants = (
            {**plain, "referred_schema": "other"},
            {**plain, "options": {"ondelete": "CASCADE", "onupdate": "SET NULL"}},
            {**plain, "options": {"ondelete": "CASCADE", "deferrable": True}},
            {**plain, "options": {"ondelete": "CASCADE", "initially": "DEFERRED"}},
            {**plain, "options": {"ondelete": "CASCADE", "match": "FULL"}},
            {**plain, "options": {"ondelete": "CASCADE", "custom": "value"}},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    normalized,
                    migration_entrypoint._normalize_foreign_key(
                        variant, default_schema_name="main"
                    ),
                )

    def test_postgres_fk_reflection_ignores_search_path(self) -> None:
        inspector = MagicMock()
        inspector.get_foreign_keys.return_value = []
        migration_entrypoint._get_foreign_keys(
            inspector,
            "messages",
            dialect_name="postgresql",
        )
        inspector.get_foreign_keys.assert_called_once_with(
            "messages",
            postgresql_ignore_search_path=True,
        )


class MigrationEntrypointTest(unittest.TestCase):
    def _create_version_table(
        self,
        database_url: str,
        revisions: list[str],
        *,
        primary_key: bool = True,
    ) -> None:
        engine = create_engine(database_url)
        with engine.begin() as connection:
            primary_key_sql = " PRIMARY KEY" if primary_key else ""
            connection.execute(
                text(
                    "CREATE TABLE alembic_version ("
                    f"version_num VARCHAR(32) NOT NULL{primary_key_sql})"
                )
            )
            for revision in revisions:
                connection.execute(
                    text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
                    {"revision": revision},
                )
        engine.dispose()

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

    def test_baseline_version_without_application_schema_refuses_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "version-only.db")
            self._create_version_table(database_url, ["20260715_00"])

            with self.assertRaises(migration_entrypoint.ManagedRevisionStateError):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

    def test_managed_baseline_missing_schema_refuses_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "managed-missing.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE conversations DROP COLUMN context_summary"))
            engine.dispose()
            self._create_version_table(database_url, ["20260715_00"])

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

    def test_managed_baseline_extra_schema_refuses_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "managed-extra.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE conversations ADD COLUMN unexpected TEXT"))
            engine.dispose()
            self._create_version_table(database_url, ["20260715_00"])

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

    def test_empty_managed_version_row_refuses_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "empty-version.db")
            prepare_current_schema(database_url)
            self._create_version_table(database_url, [])

            with self.assertRaises(migration_entrypoint.ManagedRevisionStateError):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

    def test_multiple_managed_version_rows_refuse_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "multiple-version.db")
            prepare_current_schema(database_url)
            self._create_version_table(
                database_url,
                ["20260715_00", "20260715_00"],
                primary_key=False,
            )

            with self.assertRaises(migration_entrypoint.ManagedRevisionStateError):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

    def test_unknown_managed_version_refuses_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "unknown-version.db")
            prepare_current_schema(database_url)
            self._create_version_table(database_url, ["future_revision"])

            with self.assertRaises(migration_entrypoint.ManagedRevisionStateError):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")

    def test_symbolic_or_abbreviated_managed_versions_refuse_to_start(self) -> None:
        for revision in ("head", "base", "20260715"):
            with self.subTest(revision=revision), tempfile.TemporaryDirectory() as temp_dir:
                database_url = sqlite_url(Path(temp_dir) / "symbolic-version.db")
                prepare_current_schema(database_url)
                self._create_version_table(database_url, [revision])

                with self.assertRaises(migration_entrypoint.ManagedRevisionStateError):
                    run_migrations(
                        database_url=database_url,
                        config_path=BACKEND_ROOT / "alembic.ini",
                    )

    def test_migration_commands_receive_the_classification_connection(self) -> None:
        connection = MagicMock()
        connection.dialect.name = "sqlite"
        connection.in_transaction.return_value = False
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection
        inspector = MagicMock()
        inspector.get_table_names.return_value = []

        received_connections: list[object] = []

        def fake_upgrade(config: object, revision: str) -> None:
            received_connections.append(config.attributes["connection"])

        with (
            patch("app.migration_entrypoint.create_engine", return_value=engine),
            patch("app.migration_entrypoint.inspect", return_value=inspector),
            patch("app.migration_entrypoint.command.upgrade", side_effect=fake_upgrade),
        ):
            run_migrations(database_url="sqlite+pysqlite:///:memory:", config_path=BACKEND_ROOT / "alembic.ini")

        self.assertEqual(received_connections, [connection])
        engine.connect.assert_called_once_with()

    def test_stamp_and_upgrade_share_the_validation_connection(self) -> None:
        connection = MagicMock()
        connection.dialect.name = "sqlite"
        connection.in_transaction.return_value = False
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection
        inspector = MagicMock()
        inspector.get_table_names.return_value = list(
            migration_entrypoint.BASELINE_COLUMNS
        )
        events: list[tuple[str, object]] = []

        def validate(active_connection: object) -> None:
            events.append(("validate", active_connection))

        def stamp(config: object, revision: str) -> None:
            events.append(("stamp", config.attributes["connection"]))

        def upgrade(config: object, revision: str) -> None:
            events.append(("upgrade", config.attributes["connection"]))

        with (
            patch("app.migration_entrypoint.create_engine", return_value=engine),
            patch("app.migration_entrypoint.inspect", return_value=inspector),
            patch("app.migration_entrypoint._validate_baseline", side_effect=validate),
            patch("app.migration_entrypoint.command.stamp", side_effect=stamp),
            patch("app.migration_entrypoint.command.upgrade", side_effect=upgrade),
        ):
            run_migrations(database_url="sqlite+pysqlite:///:memory:", config_path=BACKEND_ROOT / "alembic.ini")

        self.assertEqual(
            events,
            [
                ("validate", connection),
                ("stamp", connection),
                ("upgrade", connection),
            ],
        )

    def test_postgres_advisory_lock_is_released_on_migration_failure(self) -> None:
        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.in_transaction.return_value = False
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection
        inspector = MagicMock()
        inspector.get_table_names.return_value = []

        def fail_upgrade(config: object, revision: str) -> None:
            raise RuntimeError("migration failed")

        with (
            patch("app.migration_entrypoint.create_engine", return_value=engine),
            patch("app.migration_entrypoint.inspect", return_value=inspector),
            patch("app.migration_entrypoint.command.upgrade", side_effect=fail_upgrade),
            self.assertRaises(RuntimeError),
        ):
            run_migrations(database_url="postgresql+psycopg://db", config_path=BACKEND_ROOT / "alembic.ini")

        executed_sql = " ".join(str(call.args[0]) for call in connection.execute.call_args_list)
        self.assertIn("pg_advisory_lock", executed_sql)
        self.assertIn("pg_advisory_unlock", executed_sql)

    def test_postgres_lock_transaction_order_is_explicit(self) -> None:
        events: list[tuple[str, str | None]] = []
        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.in_transaction.return_value = False

        def record_execute(statement: object, *args: object, **kwargs: object) -> MagicMock:
            events.append(("execute", str(statement)))
            return MagicMock()

        class Transaction:
            def __enter__(self) -> "Transaction":
                events.append(("begin", None))
                return self

            def __exit__(self, *_args: object) -> None:
                events.append(("end", None))

        connection.execute.side_effect = record_execute
        connection.commit.side_effect = lambda: events.append(("commit", None))
        connection.begin.side_effect = lambda: Transaction()
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection
        inspector = MagicMock()
        inspector.get_table_names.return_value = []

        def record_upgrade(config: object, revision: str) -> None:
            events.append(("upgrade", None))

        with (
            patch("app.migration_entrypoint.create_engine", return_value=engine),
            patch("app.migration_entrypoint.inspect", return_value=inspector),
            patch("app.migration_entrypoint.command.upgrade", side_effect=record_upgrade),
        ):
            run_migrations(database_url="postgresql+psycopg://db", config_path=BACKEND_ROOT / "alembic.ini")

        event_names = [name for name, _ in events]
        self.assertEqual(
            event_names,
            ["execute", "commit", "begin", "upgrade", "end", "execute", "commit"],
        )
        self.assertIn("pg_advisory_lock", events[0][1] or "")
        self.assertIn("pg_advisory_unlock", events[-2][1] or "")

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

    def test_extra_expression_index_refuses_to_stamp_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "extra-expression-index.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE INDEX ix_conversations_title_expression "
                        "ON conversations (lower(title))"
                    )
                )
                sqlite_indexes = {
                    row["name"]
                    for row in connection.execute(
                        text('PRAGMA index_list("conversations")')
                    ).mappings()
                }
            engine.dispose()

            self.assertIn("ix_conversations_title_expression", sqlite_indexes)
            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(
                    database_url=database_url,
                    config_path=BACKEND_ROOT / "alembic.ini",
                )
            self.assertNotIn("alembic_version", table_names(database_url))

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

    def test_partial_unique_index_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "partial-unique-index.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("DROP INDEX ix_evaluation_runs_sequence"))
                connection.execute(
                    text(
                        "CREATE UNIQUE INDEX ix_evaluation_runs_sequence "
                        "ON evaluation_runs (sequence) WHERE sequence > 0"
                    )
                )
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(database_url=database_url, config_path=BACKEND_ROOT / "alembic.ini")
            self.assertNotIn("alembic_version", table_names(database_url))

    def test_descending_index_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "descending-index.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("DROP INDEX ix_conversations_sort_order"))
                connection.execute(
                    text(
                        "CREATE INDEX ix_conversations_sort_order "
                        "ON conversations (sort_order DESC)"
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

    def test_char_replacement_for_varchar_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "wrong-char-type.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE evaluation_counters"))
                connection.execute(
                    text(
                        "CREATE TABLE evaluation_counters "
                        "(name CHAR(80) NOT NULL PRIMARY KEY, next_value BIGINT NOT NULL)"
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

    def test_foreign_key_update_and_deferrable_drift_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "foreign-key-options.db")
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
                        "FOREIGN KEY (conversation_id) REFERENCES conversations (id) "
                        "ON DELETE CASCADE ON UPDATE SET NULL "
                        "DEFERRABLE INITIALLY DEFERRED)"
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
