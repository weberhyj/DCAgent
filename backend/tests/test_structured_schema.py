from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from alembic import command
from app.database import (
    Database,
    StructuredColumnRecord,
    StructuredDatasetRecord,
    StructuredIngestionJobRecord,
    StructuredPublicationRecord,
)
from app.models import KnowledgeStatus
from app.schemas import KnowledgeSource
from app.structured_models import (
    StructuredColumnSchema,
    StructuredColumnType,
    StructuredDatasetSchema,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "20260715_00"
STRUCTURED_REVISION = "20260721_01"
PREVIEW_REVISION = "20260721_02"


def make_alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


class StructuredSchemaContractTest(unittest.TestCase):
    def test_structured_column_and_dataset_contracts_are_typed(self) -> None:
        column = StructuredColumnSchema(
            physical_name="order_amount",
            original_name="订单金额",
            display_name="订单金额",
            data_type=StructuredColumnType.DECIMAL,
            aliases=("金额",),
            allow_aggregate=True,
            allow_filter=True,
        )
        dataset = StructuredDatasetSchema(
            dataset_id="ds-sales",
            source_id="kb-sales",
            worksheet_name="明细",
            schema_version=1,
            columns=(column,),
            schema_hash="a" * 64,
        )

        self.assertIs(dataset.columns[0].data_type, StructuredColumnType.DECIMAL)
        self.assertEqual(dataset.schema_hash, "a" * 64)
        self.assertEqual(StructuredColumnType.DECIMAL.value, "decimal")
        self.assertIn("待确认表结构", KnowledgeStatus.__args__)
        self.assertIn("结构化导入中", KnowledgeStatus.__args__)
        self.assertIn("待确认表结构", KnowledgeSource.model_fields["status"].annotation.__args__)
        self.assertIn("结构化导入中", KnowledgeSource.model_fields["status"].annotation.__args__)

    def test_database_creates_structured_tables(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        tables = set(inspect(database.engine).get_table_names())

        self.assertTrue(
            {
                "structured_datasets",
                "structured_columns",
                "structured_ingestion_jobs",
                "structured_publications",
            }.issubset(tables)
        )

    def test_structured_table_constraints_and_indexes_are_present(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        inspector = inspect(database.engine)

        datasets = inspector.get_indexes("structured_datasets")
        columns = inspector.get_indexes("structured_columns")
        jobs = inspector.get_indexes("structured_ingestion_jobs")
        publications = inspector.get_indexes("structured_publications")
        self.assertTrue(
            any(index["name"] == "ix_structured_datasets_source_id" for index in datasets)
        )
        self.assertTrue(any(index["name"] == "ix_structured_datasets_status" for index in datasets))
        self.assertTrue(
            any(index["name"] == "ix_structured_columns_dataset_id" for index in columns)
        )
        self.assertTrue(
            any(index["name"] == "ix_structured_ingestion_jobs_status" for index in jobs)
        )
        self.assertTrue(
            any(index["name"] == "ix_structured_ingestion_jobs_dataset_id" for index in jobs)
        )
        self.assertTrue(
            any(index["name"] == "ix_structured_publications_dataset_id" for index in publications)
        )

        unique_constraints = inspector.get_unique_constraints("structured_datasets")
        self.assertTrue(
            any(
                tuple(constraint["column_names"])
                == ("source_id", "worksheet_name", "schema_version")
                for constraint in unique_constraints
            )
        )
        self.assertEqual(
            tuple(inspector.get_pk_constraint("structured_datasets")["constrained_columns"]),
            ("dataset_id", "schema_version"),
        )
        for table_name in (
            "structured_columns",
            "structured_ingestion_jobs",
            "structured_publications",
        ):
            with self.subTest(table=table_name):
                self.assertTrue(
                    any(
                        tuple(foreign_key["constrained_columns"])
                        == ("dataset_id", "schema_version")
                        and tuple(foreign_key["referred_columns"])
                        == ("dataset_id", "schema_version")
                        for foreign_key in inspector.get_foreign_keys(table_name)
                    )
                )

    def test_dataset_versions_keep_distinct_metadata_relationships(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()

        with database.session() as session:
            version_one = StructuredDatasetRecord(
                dataset_id="ds-sales",
                source_id="kb-sales",
                worksheet_name="明细",
                schema_version=1,
                schema_hash="a" * 64,
                status="published",
            )
            version_two = StructuredDatasetRecord(
                dataset_id="ds-sales",
                source_id="kb-sales",
                worksheet_name="明细",
                schema_version=2,
                schema_hash="b" * 64,
                status="confirmed",
            )
            version_one.columns.append(
                StructuredColumnRecord(
                    id="col-v1",
                    dataset_id="ds-sales",
                    schema_version=1,
                    physical_name="order_amount",
                    original_name="订单金额",
                    display_name="订单金额",
                    data_type="decimal",
                    aliases=["金额"],
                    allow_aggregate=True,
                    allow_filter=True,
                    null_policy="ignore",
                    sort_order=0,
                )
            )
            version_two.columns.append(
                StructuredColumnRecord(
                    id="col-v2",
                    dataset_id="ds-sales",
                    schema_version=2,
                    physical_name="net_amount",
                    original_name="净金额",
                    display_name="净金额",
                    data_type="decimal",
                    aliases=["净额"],
                    allow_aggregate=True,
                    allow_filter=True,
                    null_policy="ignore",
                    sort_order=0,
                )
            )
            version_two.ingestion_jobs.append(
                StructuredIngestionJobRecord(
                    id="job-v2",
                    source_id="kb-sales",
                    dataset_id="ds-sales",
                    schema_version=2,
                    status="queued",
                    checkpoint_row=0,
                    attempt=0,
                )
            )
            version_one.publications.append(
                StructuredPublicationRecord(
                    publication_id="pub-v1",
                    dataset_id="ds-sales",
                    schema_version=1,
                    physical_table_name="structured_ds_sales_v1",
                    row_count=3,
                    content_hash="c" * 64,
                    status="published",
                )
            )
            session.add_all((version_one, version_two))

        with database.session() as session:
            version_one = session.get(StructuredDatasetRecord, ("ds-sales", 1))
            version_two = session.get(StructuredDatasetRecord, ("ds-sales", 2))
            self.assertIsNotNone(version_one)
            self.assertIsNotNone(version_two)
            self.assertEqual(
                [column.physical_name for column in version_one.columns], ["order_amount"]
            )
            self.assertEqual(
                [column.physical_name for column in version_two.columns], ["net_amount"]
            )
            self.assertEqual(version_two.ingestion_jobs[0].schema_version, 2)
            self.assertEqual(version_two.ingestion_jobs[0].dataset.schema_version, 2)
            self.assertEqual(version_one.publications[0].schema_version, 1)
            self.assertEqual(version_one.publications[0].dataset.schema_version, 1)

    def test_alembic_structured_revision_round_trips_references_and_indexes(self) -> None:
        structured_tables = {
            "structured_datasets",
            "structured_columns",
            "structured_ingestion_jobs",
            "structured_publications",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "structured-migration.db"
            database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
            config = make_alembic_config(database_url)
            command.upgrade(config, BASE_REVISION)
            command.upgrade(config, STRUCTURED_REVISION)

            engine = create_engine(database_url)
            try:
                inspector = inspect(engine)
                self.assertTrue(structured_tables.issubset(inspector.get_table_names()))
                self.assertEqual(
                    tuple(
                        inspector.get_pk_constraint("structured_datasets")["constrained_columns"]
                    ),
                    ("dataset_id", "schema_version"),
                )
                expected_foreign_keys = {
                    "structured_datasets": {
                        (("source_id",), "knowledge_sources", ("id",), "CASCADE"),
                    },
                    "structured_columns": {
                        (
                            ("dataset_id", "schema_version"),
                            "structured_datasets",
                            ("dataset_id", "schema_version"),
                            "CASCADE",
                        ),
                    },
                    "structured_ingestion_jobs": {
                        (("source_id",), "knowledge_sources", ("id",), "CASCADE"),
                        (
                            ("dataset_id", "schema_version"),
                            "structured_datasets",
                            ("dataset_id", "schema_version"),
                            "CASCADE",
                        ),
                        (
                            ("publication_id",),
                            "structured_publications",
                            ("publication_id",),
                            "SET NULL",
                        ),
                    },
                    "structured_publications": {
                        (
                            ("dataset_id", "schema_version"),
                            "structured_datasets",
                            ("dataset_id", "schema_version"),
                            "CASCADE",
                        ),
                    },
                }
                expected_indexes = {
                    "structured_datasets": {
                        "ix_structured_datasets_source_id",
                        "ix_structured_datasets_status",
                    },
                    "structured_columns": {"ix_structured_columns_dataset_id"},
                    "structured_ingestion_jobs": {
                        "ix_structured_ingestion_jobs_source_id",
                        "ix_structured_ingestion_jobs_dataset_id",
                        "ix_structured_ingestion_jobs_status",
                    },
                    "structured_publications": {
                        "ix_structured_publications_dataset_id",
                        "ix_structured_publications_status",
                    },
                }
                for table_name in structured_tables:
                    with self.subTest(table=table_name):
                        actual_foreign_keys = {
                            (
                                tuple(foreign_key["constrained_columns"]),
                                foreign_key["referred_table"],
                                tuple(foreign_key["referred_columns"]),
                                foreign_key["options"].get("ondelete"),
                            )
                            for foreign_key in inspector.get_foreign_keys(table_name)
                        }
                        self.assertEqual(expected_foreign_keys[table_name], actual_foreign_keys)
                        self.assertEqual(
                            expected_indexes[table_name],
                            {index["name"] for index in inspector.get_indexes(table_name)},
                        )
            finally:
                engine.dispose()

            command.downgrade(config, BASE_REVISION)
            engine = create_engine(database_url)
            try:
                inspector = inspect(engine)
                tables_after_downgrade = set(inspector.get_table_names())
                self.assertTrue(structured_tables.isdisjoint(tables_after_downgrade))
                self.assertIn("knowledge_sources", tables_after_downgrade)
                with engine.connect() as connection:
                    self.assertEqual(
                        connection.scalar(text("SELECT version_num FROM alembic_version")),
                        BASE_REVISION,
                    )
            finally:
                engine.dispose()

    def test_alembic_preview_revision_round_trips_source_cascade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "preview-migration.db"
            database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
            config = make_alembic_config(database_url)
            command.upgrade(config, PREVIEW_REVISION)

            engine = create_engine(database_url)
            try:
                inspector = inspect(engine)
                self.assertIn("structured_previews", inspector.get_table_names())
                self.assertEqual(
                    inspector.get_pk_constraint("structured_previews")["constrained_columns"],
                    ["source_id"],
                )
                self.assertEqual(
                    [
                        (
                            foreign_key["constrained_columns"],
                            foreign_key["referred_table"],
                            foreign_key["referred_columns"],
                            foreign_key["options"].get("ondelete"),
                        )
                        for foreign_key in inspector.get_foreign_keys("structured_previews")
                    ],
                    [(["source_id"], "knowledge_sources", ["id"], "CASCADE")],
                )
                with engine.connect() as connection:
                    connection.exec_driver_sql("PRAGMA foreign_keys=ON")
                    connection.execute(
                        text(
                            "INSERT INTO knowledge_sources ("
                            "id, name, source_type, records, status, updated_at, "
                            "classification, sort_order"
                            ") VALUES ("
                            "'cascade-source', 'Cascade source', 'XLSX', 0, 'ready', "
                            "'2026-07-22T00:00:00Z', 'internal', 0"
                            ")"
                        )
                    )
                    connection.execute(
                        text(
                            "INSERT INTO structured_previews (source_id, payload) "
                            "VALUES ('cascade-source', '{}')"
                        )
                    )
                    connection.execute(
                        text("DELETE FROM knowledge_sources WHERE id = 'cascade-source'")
                    )
                    self.assertEqual(
                        connection.scalar(
                            text(
                                "SELECT COUNT(*) FROM structured_previews "
                                "WHERE source_id = 'cascade-source'"
                            )
                        ),
                        0,
                    )
                    connection.commit()
            finally:
                engine.dispose()

            command.downgrade(config, STRUCTURED_REVISION)
            engine = create_engine(database_url)
            try:
                inspector = inspect(engine)
                self.assertNotIn("structured_previews", inspector.get_table_names())
                self.assertIn("structured_datasets", inspector.get_table_names())
                with engine.connect() as connection:
                    self.assertEqual(
                        connection.scalar(text("SELECT version_num FROM alembic_version")),
                        STRUCTURED_REVISION,
                    )
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
