from __future__ import annotations

import unittest

from sqlalchemy import inspect

from app.database import Database
from app.models import KnowledgeStatus
from app.schemas import KnowledgeSource
from app.structured_models import (
    StructuredColumnSchema,
    StructuredColumnType,
    StructuredDatasetSchema,
)


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


if __name__ == "__main__":
    unittest.main()
