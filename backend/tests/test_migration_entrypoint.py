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
    def _postgres_version_table_mocks(
        self,
        catalog_row: dict[str, object],
    ) -> tuple[MagicMock, MagicMock]:
        inspector = MagicMock()
        inspector.get_columns.return_value = [
            {
                "name": "version_num",
                "type": postgresql.VARCHAR(32),
                "nullable": False,
                "default": None,
            }
        ]
        inspector.get_pk_constraint.return_value = {
            "name": "alembic_version_pkc",
            "constrained_columns": ["version_num"],
            "dialect_options": {},
        }
        inspector.get_unique_constraints.return_value = []
        inspector.get_indexes.return_value = []
        inspector.get_check_constraints.return_value = []
        inspector.get_foreign_keys.return_value = []

        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.dialect.server_version_info = (15, 8)
        connection.execute.return_value.mappings.return_value.all.return_value = [
            catalog_row
        ]
        return connection, inspector

    def test_type_signature_distinguishes_database_type_families(self) -> None:
        normalize = migration_entrypoint._normalize_type

        self.assertNotEqual(normalize(sa.CHAR(32)), normalize(sa.String(32)))
        self.assertNotEqual(normalize(sa.SmallInteger()), normalize(sa.Integer()))
        self.assertNotEqual(normalize(postgresql.JSONB()), normalize(sa.JSON()))
        self.assertNotEqual(normalize(sa.REAL()), normalize(sa.Float()))
        double_precision_types = (
            sa.Float(),
            sa.Float(precision=25),
            sa.Float(precision=53),
            postgresql.DOUBLE_PRECISION(),
            postgresql.DOUBLE_PRECISION(precision=53),
        )
        for double_precision_type in double_precision_types:
            with self.subTest(double_precision_type=double_precision_type):
                self.assertEqual(
                    normalize(double_precision_type),
                    normalize(sa.Float()),
                )
        self.assertEqual(normalize(sa.Float(24)), normalize(postgresql.REAL()))
        self.assertNotEqual(normalize(sa.Float(24)), normalize(sa.Float()))
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

    def test_postgres_reflected_double_precision_matches_float_baseline(
        self,
    ) -> None:
        reflected_column = {
            "name": "retrieval_min_score",
            "type": postgresql.DOUBLE_PRECISION(precision=53),
            "nullable": False,
            "default": None,
        }

        self.assertEqual(
            migration_entrypoint.BASELINE_COLUMNS["evaluation_batches"][4],
            migration_entrypoint._normalize_column(reflected_column, set()),
        )

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
            "nulls_not_distinct": False,
            "reloptions": None,
            "tablespace": None,
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
            {**plain, "nulls_not_distinct": True},
            {
                name: value
                for name, value in plain.items()
                if name != "nulls_not_distinct"
            },
            {**plain, "reloptions": ["fillfactor=90"]},
            {name: value for name, value in plain.items() if name != "reloptions"},
            {**plain, "tablespace": "fastspace"},
            {name: value for name, value in plain.items() if name != "tablespace"},
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

        catalog_sql = str(migration_entrypoint.POSTGRES_INDEX_CATALOG_SQL)
        self.assertIn("indnullsnotdistinct", catalog_sql)
        self.assertIn("reloptions", catalog_sql)
        self.assertIn("pg_tablespace", catalog_sql)
        self.assertIn("indisprimary", catalog_sql)
        self.assertIn(":schema_name", catalog_sql)
        self.assertNotIn("current_schema()", catalog_sql)
        self.assertNotIn("NOT index_state.indisprimary", catalog_sql)

    def test_postgres_primary_catalog_row_requires_plain_constraint(self) -> None:
        plain = {
            "index_name": "evaluation_runs_pkey",
            "is_unique": True,
            "is_valid": True,
            "is_ready": True,
            "is_live": True,
            "is_primary": True,
            "nulls_not_distinct": False,
            "reloptions": None,
            "tablespace": None,
            "access_method": "btree",
            "predicate": None,
            "expressions": None,
            "key_attribute_count": 1,
            "total_attribute_count": 1,
            "key_definitions": ["id"],
            "primary_constraint_oid": 123,
            "primary_constraint_deferrable": False,
            "primary_constraint_deferred": False,
        }
        self.assertEqual(
            (),
            migration_entrypoint._normalize_postgres_index_catalog_row(
                plain
            ).semantic_options,
        )

        variants = (
            {
                name: value
                for name, value in plain.items()
                if name != "primary_constraint_oid"
            },
            {**plain, "primary_constraint_oid": None},
            {
                name: value
                for name, value in plain.items()
                if name != "primary_constraint_deferrable"
            },
            {**plain, "primary_constraint_deferrable": True},
            {
                name: value
                for name, value in plain.items()
                if name != "primary_constraint_deferred"
            },
            {**plain, "primary_constraint_deferred": True},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    (),
                    migration_entrypoint._normalize_postgres_index_catalog_row(
                        variant
                    ).semantic_options,
                )

    def test_postgres_catalog_sql_joins_primary_constraint(self) -> None:
        catalog_sql = str(migration_entrypoint.POSTGRES_INDEX_CATALOG_SQL)

        self.assertIn("LEFT JOIN pg_constraint AS primary_constraint", catalog_sql)
        self.assertIn("primary_constraint.contype = 'p'", catalog_sql)
        self.assertIn(
            "primary_constraint.conindid = index_state.indexrelid",
            catalog_sql,
        )
        self.assertIn("primary_constraint.oid AS primary_constraint_oid", catalog_sql)
        self.assertIn(
            "primary_constraint.condeferrable AS primary_constraint_deferrable",
            catalog_sql,
        )
        self.assertIn(
            "primary_constraint.condeferred AS primary_constraint_deferred",
            catalog_sql,
        )

    def test_postgres_alembic_version_accepts_exact_primary_catalog_row(self) -> None:
        catalog_row = {
            "index_name": "alembic_version_pkc",
            "is_unique": True,
            "is_valid": True,
            "is_ready": True,
            "is_live": True,
            "is_primary": True,
            "primary_constraint_oid": 123,
            "primary_constraint_deferrable": False,
            "primary_constraint_deferred": False,
            "nulls_not_distinct": False,
            "reloptions": None,
            "tablespace": None,
            "access_method": "btree",
            "predicate": None,
            "expressions": None,
            "key_attribute_count": 1,
            "total_attribute_count": 1,
            "key_definitions": ["version_num"],
        }
        connection, inspector = self._postgres_version_table_mocks(catalog_row)

        with patch("app.migration_entrypoint.inspect", return_value=inspector):
            migration_entrypoint._validate_alembic_version_table(
                connection,
                schema_name="tenant",
            )

        connection.execute.assert_called_once_with(
            migration_entrypoint.POSTGRES_INDEX_CATALOG_SQL,
            {"schema_name": "tenant", "table_name": "alembic_version"},
        )

    def test_postgres_alembic_version_rejects_deferrable_primary_catalog_row(
        self,
    ) -> None:
        catalog_row = {
            "index_name": "alembic_version_pkc",
            "is_unique": True,
            "is_valid": True,
            "is_ready": True,
            "is_live": True,
            "is_primary": True,
            "primary_constraint_oid": 123,
            "primary_constraint_deferrable": True,
            "primary_constraint_deferred": True,
            "nulls_not_distinct": False,
            "reloptions": None,
            "tablespace": None,
            "access_method": "btree",
            "predicate": None,
            "expressions": None,
            "key_attribute_count": 1,
            "total_attribute_count": 1,
            "key_definitions": ["version_num"],
        }
        connection, inspector = self._postgres_version_table_mocks(catalog_row)

        with (
            patch("app.migration_entrypoint.inspect", return_value=inspector),
            self.assertRaises(migration_entrypoint.ManagedRevisionStateError),
        ):
            migration_entrypoint._validate_alembic_version_table(
                connection,
                schema_name="tenant",
            )

    def test_primary_key_signature_fails_closed_on_semantic_options(self) -> None:
        def fingerprint(primary_key: dict[str, object]) -> object:
            inspector = MagicMock()
            inspector.default_schema_name = "public"
            inspector.get_table_names.return_value = ["items"]
            inspector.get_columns.return_value = [
                {"name": "id", "type": sa.Integer(), "nullable": False},
            ]
            inspector.get_pk_constraint.return_value = primary_key
            inspector.get_indexes.return_value = []
            inspector.get_foreign_keys.return_value = []
            inspector.get_unique_constraints.return_value = []
            inspector.get_check_constraints.return_value = []
            connection = MagicMock()
            connection.dialect.name = "postgresql"
            connection.scalar.return_value = "tenant"
            with patch("app.migration_entrypoint.inspect", return_value=inspector):
                return migration_entrypoint._schema_fingerprint(connection)[
                    "primary_keys"
                ]

        plain = {
            "name": "items_pkey",
            "constrained_columns": ["id"],
            "dialect_options": {},
        }
        normalized = fingerprint(plain)
        variants = (
            {**plain, "dialect_options": {"postgresql_include": ["label"]}},
            {**plain, "deferrable": True},
            {**plain, "initially": "DEFERRED"},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(normalized, fingerprint(variant))

    def test_postgres_index_catalog_requires_version_15_or_newer(self) -> None:
        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.dialect.server_version_info = (14, 12)

        with self.assertRaisesRegex(RuntimeError, r"PostgreSQL 15\+ required"):
            migration_entrypoint._validate_postgres_index_catalog(
                connection,
                [],
                "public",
            )
        connection.execute.assert_not_called()

    def test_postgres_version_requirement_survives_baseline_wrapping(self) -> None:
        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.dialect.server_version_info = (14, 12)

        with (
            patch(
                "app.migration_entrypoint._schema_fingerprint",
                return_value={"tables": set()},
            ),
            patch("app.migration_entrypoint._validate_counter_row"),
            self.assertRaisesRegex(
                BaselineSchemaMismatch,
                r"PostgreSQL 15\+ required",
            ),
        ):
            migration_entrypoint._validate_baseline(
                connection,
                schema_name="public",
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
            schema_name="tenant",
        )
        inspector.get_foreign_keys.assert_called_once_with(
            "messages",
            schema="tenant",
            postgresql_ignore_search_path=True,
        )

    def test_foreign_key_only_normalizes_the_inspected_schema(self) -> None:
        foreign_key = {
            "name": "messages_conversation_id_fkey",
            "constrained_columns": ["conversation_id"],
            "referred_schema": "public",
            "referred_table": "conversations",
            "referred_columns": ["id"],
            "options": {"ondelete": "CASCADE"},
        }

        normalized = migration_entrypoint._normalize_foreign_key(
            foreign_key,
            default_schema_name="public",
            current_schema_name="tenant",
        )
        self.assertEqual("public", normalized.referred_schema)
        self.assertIsNone(
            migration_entrypoint._normalize_foreign_key(
                {**foreign_key, "referred_schema": "tenant"},
                default_schema_name="public",
                current_schema_name="tenant",
            ).referred_schema
        )

    def test_postgres_fingerprint_uses_pk_constraint_and_locked_schema(self) -> None:
        inspector = MagicMock()
        inspector.default_schema_name = "public"
        inspector.get_table_names.return_value = ["items"]
        inspector.get_columns.return_value = [
            {"name": "tenant_id", "type": sa.Integer(), "nullable": False},
            {"name": "item_id", "type": sa.Integer(), "nullable": False},
            {"name": "label", "type": sa.String(20), "nullable": False},
        ]
        inspector.get_pk_constraint.return_value = {
            "name": "items_pkey",
            "constrained_columns": ["item_id", "tenant_id"],
        }
        inspector.get_indexes.return_value = []
        inspector.get_foreign_keys.return_value = []
        inspector.get_unique_constraints.return_value = []
        inspector.get_check_constraints.return_value = []

        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.scalar.return_value = "tenant"

        with patch("app.migration_entrypoint.inspect", return_value=inspector):
            fingerprint = migration_entrypoint._schema_fingerprint(connection)

        self.assertEqual(
            fingerprint["primary_keys"],
            {
                "items": migration_entrypoint.PrimaryKeySignature(
                    ("item_id", "tenant_id")
                )
            },
        )
        self.assertEqual(
            tuple(column[3] for column in fingerprint["columns"]["items"]),
            (True, True, False),
        )
        inspector.get_table_names.assert_called_once_with(schema="tenant")
        inspector.get_columns.assert_called_once_with("items", schema="tenant")
        inspector.get_pk_constraint.assert_called_once_with(
            "items", schema="tenant"
        )
        inspector.get_indexes.assert_called_once_with("items", schema="tenant")
        inspector.get_foreign_keys.assert_called_once_with(
            "items",
            schema="tenant",
            postgresql_ignore_search_path=True,
        )
        inspector.get_unique_constraints.assert_called_once_with(
            "items", schema="tenant"
        )
        inspector.get_check_constraints.assert_called_once_with(
            "items", schema="tenant"
        )

    def test_postgres_classification_locks_table_lookup_to_current_schema(self) -> None:
        inspector = MagicMock()
        inspector.get_table_names.return_value = []
        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.dialect.server_version_info = (15, 8)
        connection.scalar.return_value = "tenant"

        with patch("app.migration_entrypoint.inspect", return_value=inspector):
            action = migration_entrypoint._classify_and_validate(
                connection,
                MagicMock(),
            )

        self.assertEqual("upgrade", action)
        inspector.get_table_names.assert_called_once_with(schema="tenant")

    def test_postgres_14_empty_classification_fails_before_inspection(self) -> None:
        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.dialect.server_version_info = (14, 12)

        with (
            patch("app.migration_entrypoint.inspect") as inspect_database,
            self.assertRaisesRegex(RuntimeError, r"PostgreSQL 15\+ required"),
        ):
            migration_entrypoint._classify_and_validate(
                connection,
                MagicMock(),
            )

        connection.scalar.assert_not_called()
        inspect_database.assert_not_called()

    def test_column_generation_metadata_changes_fingerprint(self) -> None:
        base_column = {
            "name": "value",
            "type": sa.Integer(),
            "nullable": False,
            "default": None,
        }

        def fingerprint(column: dict[str, object]) -> dict[str, object]:
            inspector = MagicMock()
            inspector.default_schema_name = "public"
            inspector.get_table_names.return_value = ["items"]
            inspector.get_columns.return_value = [column]
            inspector.get_pk_constraint.return_value = {
                "name": "items_pkey",
                "constrained_columns": [],
            }
            inspector.get_indexes.return_value = []
            inspector.get_foreign_keys.return_value = []
            inspector.get_unique_constraints.return_value = []
            inspector.get_check_constraints.return_value = []
            connection = MagicMock()
            connection.dialect.name = "postgresql"
            connection.scalar.return_value = "tenant"
            with patch("app.migration_entrypoint.inspect", return_value=inspector):
                return migration_entrypoint._schema_fingerprint(connection)

        plain = fingerprint(base_column)
        variants = (
            {**base_column, "computed": {"sqltext": "1", "persisted": True}},
            {**base_column, "identity": {"start": 1, "increment": 1}},
            {**base_column, "autoincrement": True},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    plain["columns"],
                    fingerprint(variant)["columns"],
                )


class MigrationEntrypointTest(unittest.TestCase):
    def _create_version_table(
        self,
        database_url: str,
        revisions: list[str],
        *,
        primary_key: bool = True,
        column_type: str = "VARCHAR(32)",
        extra_column: bool = False,
        foreign_key: bool = False,
    ) -> None:
        engine = create_engine(database_url)
        try:
            with engine.begin() as connection:
                primary_key_sql = " PRIMARY KEY" if primary_key else ""
                extra_column_sql = ", unexpected INTEGER" if extra_column else ""
                foreign_key_sql = (
                    ", FOREIGN KEY (version_num) "
                    "REFERENCES alembic_version(version_num)"
                    if foreign_key
                    else ""
                )
                connection.execute(
                    text(
                        "CREATE TABLE alembic_version ("
                        f"version_num {column_type} NOT NULL{primary_key_sql}"
                        f"{extra_column_sql}{foreign_key_sql})"
                    )
                )
                for revision in revisions:
                    connection.execute(
                        text(
                            "INSERT INTO alembic_version (version_num) "
                            "VALUES (:revision)"
                        ),
                        {"revision": revision},
                    )
        finally:
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

    def test_malformed_alembic_version_schema_refuses_to_start(self) -> None:
        variants = (
            {"primary_key": False},
            {"column_type": "TEXT"},
            {"extra_column": True},
            {"foreign_key": True},
        )
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp_dir:
                database_url = sqlite_url(Path(temp_dir) / "malformed-version.db")
                prepare_current_schema(database_url)
                self._create_version_table(
                    database_url,
                    ["20260715_00"],
                    **variant,
                )

                with self.assertRaises(
                    migration_entrypoint.ManagedRevisionStateError
                ):
                    run_migrations(
                        database_url=database_url,
                        config_path=BACKEND_ROOT / "alembic.ini",
                    )

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

        def validate(
            active_connection: object,
            schema_name: str | None = None,
        ) -> None:
            self.assertIsNone(schema_name)
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

    def test_postgres_14_empty_database_refuses_upgrade_without_mutation(
        self,
    ) -> None:
        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.dialect.server_version_info = (14, 12)
        connection.in_transaction.return_value = False
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection

        with (
            patch("app.migration_entrypoint.create_engine", return_value=engine),
            patch("app.migration_entrypoint.inspect") as inspect_database,
            patch("app.migration_entrypoint.command.stamp") as stamp,
            patch("app.migration_entrypoint.command.upgrade") as upgrade,
            self.assertRaisesRegex(RuntimeError, r"PostgreSQL 15\+ required"),
        ):
            run_migrations(
                database_url="postgresql+psycopg://db",
                config_path=BACKEND_ROOT / "alembic.ini",
            )

        connection.scalar.assert_not_called()
        inspect_database.assert_not_called()
        stamp.assert_not_called()
        upgrade.assert_not_called()
        executed_sql = tuple(
            str(call.args[0]) for call in connection.execute.call_args_list
        )
        self.assertEqual(2, len(executed_sql))
        self.assertIn("pg_advisory_lock", executed_sql[0])
        self.assertIn("pg_advisory_unlock", executed_sql[1])

    def test_postgres_advisory_lock_is_released_on_migration_failure(self) -> None:
        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.dialect.server_version_info = (15, 8)
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
        connection.dialect.server_version_info = (15, 8)
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

    def test_generated_column_drift_refuses_to_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = sqlite_url(Path(temp_dir) / "generated-column.db")
            prepare_current_schema(database_url)
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE evaluation_counters"))
                connection.execute(
                    text(
                        "CREATE TABLE evaluation_counters ("
                        "name VARCHAR(80) NOT NULL PRIMARY KEY, "
                        "next_value BIGINT GENERATED ALWAYS AS (1000000) "
                        "STORED NOT NULL)"
                    )
                )
                connection.execute(
                    text(
                        "INSERT INTO evaluation_counters (name) "
                        "VALUES ('evaluation_runs')"
                    )
                )
            engine.dispose()

            with self.assertRaises(BaselineSchemaMismatch):
                run_migrations(
                    database_url=database_url,
                    config_path=BACKEND_ROOT / "alembic.ini",
                )
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
