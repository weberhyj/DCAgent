from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from decimal import Decimal

from app.clickhouse_gateway import (
    ClickHouseGateway,
    ClickHousePublicationTarget,
    StructuredStorageError,
    StructuredValidationStatistics,
)
from tests.support.structured_fakes import sample_columns


class RecordingIngestClient:
    def __init__(self) -> None:
        self.ddl: list[tuple[str, dict[str, object]]] = []
        self.inserts: list[tuple[str, object, dict[str, object]]] = []

    def command(self, statement: str, *, settings: dict[str, object]) -> None:
        self.ddl.append((statement, settings))

    def insert_arrow(self, table: str, batch, *, settings: dict[str, object]) -> None:
        self.inserts.append((table, batch, settings))


class RenameCollisionIngestClient(RecordingIngestClient):
    def command(self, statement: str, *, settings: dict[str, object]) -> None:
        super().command(statement, settings=settings)
        if statement.startswith("RENAME TABLE"):
            raise RuntimeError("target table already exists")


class RecordingQueryClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.queries: list[tuple[str, dict[str, object]]] = []

    def query(self, statement: str, *, settings: dict[str, object]):
        self.queries.append((statement, settings))
        return self.responses.pop(0)


def sample_confirmed_schema_pathless():
    from app.structured_models import StructuredDatasetSchema

    return StructuredDatasetSchema(
        dataset_id="ds-sales",
        source_id="kb-sales",
        worksheet_name="明细",
        schema_version=1,
        columns=sample_columns(),
        schema_hash="a" * 64,
    )


def content_observation(row_count: int) -> tuple[str, dict[str, int]]:
    sums = (11, 12, 13, 14)
    xors = (21, 22, 23, 24)
    payload = json.dumps(
        {"rowCount": row_count, "sums": sums, "xors": xors},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    content_hash = hashlib.sha256(payload).hexdigest()
    values = {
        **{f"content_sum_{index}": value for index, value in enumerate(sums)},
        **{f"content_xor_{index}": value for index, value in enumerate(xors)},
    }
    return content_hash, values


def empty_content_observation() -> tuple[str, dict[str, int]]:
    payload = json.dumps(
        {"rowCount": 0, "sums": (0, 0, 0, 0), "xors": (0, 0, 0, 0)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    values = {
        **{f"content_sum_{index}": 0 for index in range(4)},
        **{f"content_xor_{index}": 0 for index in range(4)},
    }
    return hashlib.sha256(payload).hexdigest(), values


class ClickHouseGatewayTest(unittest.TestCase):
    def test_gateway_rejects_untrusted_identifiers(self) -> None:
        gateway = ClickHouseGateway(RecordingIngestClient())

        with self.assertRaises(StructuredStorageError):
            gateway.create_table("sales; DROP TABLE users", sample_columns())

    def test_prepare_uses_versioned_safe_names_and_bounded_settings(self) -> None:
        ingest = RecordingIngestClient()
        gateway = ClickHouseGateway(ingest)

        target = gateway.prepare_publication(sample_confirmed_schema_pathless(), "pub-1", "a" * 64)

        self.assertRegex(target.staging_table, r"^[a-z0-9_]+$")
        self.assertRegex(target.physical_table_name, r"^[a-z0-9_]+$")
        self.assertNotEqual(target.staging_table, target.physical_table_name)
        self.assertIn("DROP TABLE IF EXISTS", ingest.ddl[0][0])
        statement, settings = ingest.ddl[1]
        self.assertIn("CREATE TABLE", statement)
        self.assertIn("Nullable(Decimal(38, 9))", statement)
        self.assertEqual(settings["max_execution_time"], 30)
        self.assertEqual(settings["max_memory_usage"], 512 * 1024 * 1024)
        self.assertEqual(settings["max_result_rows"], 10_000)
        self.assertEqual(settings["overflow_mode"], "break")

    def test_each_publication_attempt_uses_an_isolated_staging_table(self) -> None:
        ingest = RecordingIngestClient()
        gateway = ClickHouseGateway(ingest)
        schema = sample_confirmed_schema_pathless()

        first = gateway.prepare_publication(schema, "pub-1", "a" * 64)
        second = gateway.prepare_publication(schema, "pub-1", "a" * 64)

        self.assertEqual(first.physical_table_name, second.physical_table_name)
        self.assertNotEqual(first.staging_table, second.staging_table)

    def test_rejects_the_same_client_for_ingest_and_read_only_queries(self) -> None:
        client = RecordingIngestClient()

        with self.assertRaisesRegex(StructuredStorageError, "separate read-only"):
            ClickHouseGateway(client, query_client=client)

    def test_prepare_rejects_schema_version_that_breaks_identifier_policy(self) -> None:
        ingest = RecordingIngestClient()
        gateway = ClickHouseGateway(ingest)

        with self.assertRaisesRegex(StructuredStorageError, "identifier"):
            gateway.prepare_publication(
                replace(sample_confirmed_schema_pathless(), schema_version=-1),
                "pub-1",
                "a" * 64,
            )

        self.assertEqual(ingest.ddl, [])

    def test_validation_mismatch_never_promotes(self) -> None:
        ingest = RecordingIngestClient()
        query = RecordingQueryClient(
            [
                [
                    ("order_amount", "Nullable(Decimal(38, 9))"),
                    ("region", "Nullable(String)"),
                ],
                {"row_count": 2, "content_hash_versions": 1, "content_hash": "a" * 64},
            ]
        )
        gateway = ClickHouseGateway(ingest, query_client=query)
        target = gateway.prepare_publication(sample_confirmed_schema_pathless(), "pub-1", "a" * 64)

        with self.assertRaisesRegex(StructuredStorageError, "row count"):
            gateway.validate_and_promote(
                target,
                statistics=StructuredValidationStatistics(
                    row_count=3,
                    column_count=3,
                    null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                    numeric_ranges={"order_amount": (10.0, 30.0)},
                    content_hash="a" * 64,
                ),
            )

        self.assertFalse(any("RENAME TABLE" in statement for statement, _ in ingest.ddl))

    def test_validation_requires_a_separate_read_only_query_client(self) -> None:
        gateway = ClickHouseGateway(RecordingIngestClient())
        target = gateway.prepare_publication(sample_confirmed_schema_pathless(), "pub-1", "a" * 64)

        with self.assertRaisesRegex(StructuredStorageError, "separate read-only"):
            gateway.validate_and_promote(
                target,
                statistics=StructuredValidationStatistics(
                    row_count=0,
                    column_count=3,
                    null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                    numeric_ranges={},
                    content_hash="a" * 64,
                ),
            )

    def test_validation_rechecks_public_target_identifiers(self) -> None:
        schema = sample_confirmed_schema_pathless()
        target = ClickHousePublicationTarget(
            schema=schema,
            staging_table="safe_staging; DROP TABLE users",
            physical_table_name="safe_final",
            content_hash="a" * 64,
        )
        gateway = ClickHouseGateway(
            RecordingIngestClient(),
            query_client=RecordingQueryClient([]),
        )

        with self.assertRaisesRegex(StructuredStorageError, "identifier"):
            gateway.validate_and_promote(
                target,
                statistics=StructuredValidationStatistics(
                    row_count=0,
                    column_count=3,
                    null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                    numeric_ranges={"order_amount": (None, None)},
                    content_hash="a" * 64,
                ),
            )

    def test_decimal_range_validation_is_exact_at_the_smallest_scale(self) -> None:
        expected = Decimal("12345678901234567890123456789.123456789")
        observed = Decimal("12345678901234567890123456789.123456788")
        content_hash, content_values = content_observation(1)
        ingest = RecordingIngestClient()
        query = RecordingQueryClient(
            [
                [
                    ("order_amount", "Nullable(Decimal(38, 9))"),
                    ("region", "Nullable(String)"),
                    ("order_date", "Nullable(Date)"),
                    ("_source_id", "String"),
                    ("_dataset_id", "String"),
                    ("_schema_version", "UInt64"),
                    ("_worksheet", "String"),
                    ("_row_number", "UInt64"),
                    ("_content_hash", "String"),
                ],
                {
                    "row_count": 1,
                    "content_hash_versions": 1,
                    "content_hash": content_hash,
                    **content_values,
                    "null_order_amount": 0,
                    "null_region": 0,
                    "null_order_date": 0,
                    "count_order_amount": 1,
                    "min_order_amount": observed,
                    "max_order_amount": expected,
                },
            ]
        )
        gateway = ClickHouseGateway(ingest, query_client=query)
        target = gateway.prepare_publication(
            sample_confirmed_schema_pathless(), "pub-1", content_hash
        )

        with self.assertRaisesRegex(StructuredStorageError, "minimum"):
            gateway.validate_and_promote(
                target,
                statistics=StructuredValidationStatistics(
                    row_count=1,
                    column_count=3,
                    null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                    numeric_ranges={"order_amount": (expected, expected)},
                    content_hash=content_hash,
                ),
            )

    def test_actual_row_content_digest_mismatch_blocks_promotion(self) -> None:
        content_hash, content_values = content_observation(1)
        corrupted_values = dict(content_values)
        corrupted_values["content_sum_0"] += 1
        ingest = RecordingIngestClient()
        query = RecordingQueryClient(
            [
                [
                    ("order_amount", "Nullable(Decimal(38, 9))"),
                    ("region", "Nullable(String)"),
                    ("order_date", "Nullable(Date)"),
                    ("_source_id", "String"),
                    ("_dataset_id", "String"),
                    ("_schema_version", "UInt64"),
                    ("_worksheet", "String"),
                    ("_row_number", "UInt64"),
                    ("_content_hash", "String"),
                ],
                {
                    "row_count": 1,
                    "content_hash_versions": 1,
                    "content_hash": content_hash,
                    **corrupted_values,
                    "null_order_amount": 0,
                    "null_region": 0,
                    "null_order_date": 0,
                    "count_order_amount": 1,
                    "min_order_amount": Decimal("10.000000000"),
                    "max_order_amount": Decimal("10.000000000"),
                },
            ]
        )
        gateway = ClickHouseGateway(ingest, query_client=query)
        target = gateway.prepare_publication(
            sample_confirmed_schema_pathless(), "pub-1", content_hash
        )

        with self.assertRaisesRegex(StructuredStorageError, "content hash"):
            gateway.validate_and_promote(
                target,
                statistics=StructuredValidationStatistics(
                    row_count=1,
                    column_count=3,
                    null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                    numeric_ranges={
                        "order_amount": (
                            Decimal("10.000000000"),
                            Decimal("10.000000000"),
                        )
                    },
                    content_hash=content_hash,
                ),
            )

    def test_numeric_count_mismatch_blocks_promotion(self) -> None:
        content_hash, content_values = content_observation(3)
        ingest = RecordingIngestClient()
        query = RecordingQueryClient(
            [
                [
                    ("order_amount", "Nullable(Decimal(38, 9))"),
                    ("region", "Nullable(String)"),
                    ("order_date", "Nullable(Date)"),
                    ("_source_id", "String"),
                    ("_dataset_id", "String"),
                    ("_schema_version", "UInt64"),
                    ("_worksheet", "String"),
                    ("_row_number", "UInt64"),
                    ("_content_hash", "String"),
                ],
                {
                    "row_count": 3,
                    "content_hash_versions": 1,
                    "content_hash": content_hash,
                    **content_values,
                    "null_order_amount": 0,
                    "null_region": 0,
                    "null_order_date": 0,
                    "count_order_amount": 2,
                    "min_order_amount": Decimal("10.000000000"),
                    "max_order_amount": Decimal("30.000000000"),
                },
            ]
        )
        gateway = ClickHouseGateway(ingest, query_client=query)
        target = gateway.prepare_publication(
            sample_confirmed_schema_pathless(), "pub-1", content_hash
        )

        with self.assertRaisesRegex(StructuredStorageError, "count"):
            gateway.validate_and_promote(
                target,
                statistics=StructuredValidationStatistics(
                    row_count=3,
                    column_count=3,
                    null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                    numeric_ranges={"order_amount": (10.0, 30.0)},
                    content_hash=content_hash,
                ),
            )

    def test_empty_publication_validates_without_a_stored_content_hash_row(self) -> None:
        content_hash, content_values = empty_content_observation()
        ingest = RecordingIngestClient()
        query = RecordingQueryClient(
            [
                [
                    ("order_amount", "Nullable(Decimal(38, 9))"),
                    ("region", "Nullable(String)"),
                    ("order_date", "Nullable(Date)"),
                    ("_source_id", "String"),
                    ("_dataset_id", "String"),
                    ("_schema_version", "UInt64"),
                    ("_worksheet", "String"),
                    ("_row_number", "UInt64"),
                    ("_content_hash", "String"),
                ],
                {
                    "row_count": 0,
                    "content_hash_versions": 0,
                    "content_hash": None,
                    **content_values,
                    "null_order_amount": 0,
                    "null_region": 0,
                    "null_order_date": 0,
                    "count_order_amount": 0,
                    "min_order_amount": None,
                    "max_order_amount": None,
                },
            ]
        )
        gateway = ClickHouseGateway(ingest, query_client=query)
        target = gateway.prepare_publication(
            sample_confirmed_schema_pathless(), "pub-empty", content_hash
        )

        table = gateway.validate_and_promote(
            target,
            statistics=StructuredValidationStatistics(
                row_count=0,
                column_count=3,
                null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                numeric_ranges={"order_amount": (None, None)},
                content_hash=content_hash,
            ),
        )

        self.assertEqual(table, target.physical_table_name)

    def test_rename_collision_recovers_when_existing_final_table_matches(self) -> None:
        content_hash, content_values = content_observation(1)
        description = [
            ("order_amount", "Nullable(Decimal(38, 9))"),
            ("region", "Nullable(String)"),
            ("order_date", "Nullable(Date)"),
            ("_source_id", "String"),
            ("_dataset_id", "String"),
            ("_schema_version", "UInt64"),
            ("_worksheet", "String"),
            ("_row_number", "UInt64"),
            ("_content_hash", "String"),
        ]
        observation = {
            "row_count": 1,
            "content_hash_versions": 1,
            "content_hash": content_hash,
            **content_values,
            "null_order_amount": 0,
            "null_region": 0,
            "null_order_date": 0,
            "count_order_amount": 1,
            "min_order_amount": Decimal("10.000000000"),
            "max_order_amount": Decimal("10.000000000"),
        }
        ingest = RenameCollisionIngestClient()
        query = RecordingQueryClient([description, observation, description, observation])
        gateway = ClickHouseGateway(ingest, query_client=query)
        target = gateway.prepare_publication(
            sample_confirmed_schema_pathless(), "pub-retry", content_hash
        )

        table = gateway.validate_and_promote(
            target,
            statistics=StructuredValidationStatistics(
                row_count=1,
                column_count=3,
                null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                numeric_ranges={"order_amount": (10.0, 10.0)},
                content_hash=content_hash,
            ),
        )

        self.assertEqual(table, target.physical_table_name)
        self.assertIn(target.physical_table_name, query.queries[2][0])
        self.assertTrue(
            any(
                statement == f"DROP TABLE IF EXISTS {target.staging_table}"
                for statement, _ in ingest.ddl
            )
        )

    def test_validated_table_is_promoted_with_separate_read_client(self) -> None:
        content_hash, content_values = content_observation(3)
        ingest = RecordingIngestClient()
        query = RecordingQueryClient(
            [
                [
                    ("order_amount", "Nullable(Decimal(38, 9))"),
                    ("region", "Nullable(String)"),
                    ("order_date", "Nullable(Date)"),
                    ("_source_id", "String"),
                    ("_dataset_id", "String"),
                    ("_schema_version", "UInt64"),
                    ("_worksheet", "String"),
                    ("_row_number", "UInt64"),
                    ("_content_hash", "String"),
                ],
                {
                    "row_count": 3,
                    "content_hash_versions": 1,
                    "content_hash": content_hash,
                    **content_values,
                    "null_order_amount": 0,
                    "null_region": 0,
                    "null_order_date": 0,
                    "count_order_amount": 3,
                    "min_order_amount": 10.0,
                    "max_order_amount": 30.0,
                },
            ]
        )
        gateway = ClickHouseGateway(ingest, query_client=query)
        target = gateway.prepare_publication(
            sample_confirmed_schema_pathless(), "pub-1", content_hash
        )

        table = gateway.validate_and_promote(
            target,
            statistics=StructuredValidationStatistics(
                row_count=3,
                column_count=3,
                null_counts={"order_amount": 0, "region": 0, "order_date": 0},
                numeric_ranges={"order_amount": (10.0, 30.0)},
                content_hash=content_hash,
            ),
        )

        self.assertEqual(table, target.physical_table_name)
        self.assertTrue(any("RENAME TABLE" in statement for statement, _ in ingest.ddl))
        self.assertEqual(len(query.queries), 2)
        validation_statement = query.queries[1][0]
        self.assertIn("SHA256", validation_statement)
        self.assertIn("toDecimalString(order_amount, 9)", validation_statement)
        self.assertIn("content_sum_0", validation_statement)
        self.assertIn("content_xor_3", validation_statement)
        for _, settings in query.queries:
            self.assertEqual(settings["overflow_mode"], "break")
            self.assertEqual(settings["readonly"], 1)


if __name__ == "__main__":
    unittest.main()
