from __future__ import annotations

import os
import unittest
import uuid
from dataclasses import replace
from datetime import date
from decimal import Decimal

from app.clickhouse_gateway import ClickHouseGateway
from app.structured_models import (
    StructuredFilter,
    StructuredIntent,
    StructuredMetricIntent,
    StructuredMultiAggregateIntent,
)
from app.structured_query import StructuredQueryExecutor, StructuredQueryPlanner
from tests.support.structured_fakes import (
    sample_catalog,
    sample_multi_metric_catalog,
    sample_publication,
)


def _integration_target_available() -> bool:
    return os.getenv("RUN_OFFLINE_INTEGRATION") == "1" and bool(os.getenv("CLICKHOUSE_HOST"))


@unittest.skipUnless(
    _integration_target_available(),
    "requires RUN_OFFLINE_INTEGRATION=1 and an explicit CLICKHOUSE_HOST target",
)
class StructuredQueryClickHouseIntegrationTest(unittest.TestCase):
    def test_parameterized_multi_summary_executes_as_one_legacy_safe_select(self) -> None:
        import clickhouse_connect

        host = os.environ["CLICKHOUSE_HOST"]
        port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
        query_username = os.getenv("CLICKHOUSE_QUERY_USER", "default")
        query_password = os.getenv("CLICKHOUSE_QUERY_PASSWORD", "")
        ingest_username = os.getenv("CLICKHOUSE_INGEST_USER", "default")
        ingest_password = os.getenv("CLICKHOUSE_INGEST_PASSWORD", "")
        table_name = f"structured_multi_query_it_{uuid.uuid4().hex[:16]}"
        admin = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=ingest_username,
            password=ingest_password,
        )
        query_client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=query_username,
            password=query_password,
        )
        try:
            admin.command(
                f"CREATE TABLE {table_name} ("
                "sales_amount Nullable(Decimal(38, 9)), "
                "cost_amount Nullable(Decimal(38, 9)), "
                "profit_amount Nullable(Decimal(38, 9)), "
                "region Nullable(String)"
                ") ENGINE = Memory"
            )
            admin.insert(
                table_name,
                [
                    [Decimal("100.25"), Decimal("50.10"), Decimal("50.15"), "华东"],
                    [Decimal("200"), Decimal("70"), Decimal("130"), "华东"],
                    [None, Decimal("10"), None, "华东"],
                    [Decimal("400"), Decimal("150"), Decimal("250"), "华南"],
                ],
                column_names=(
                    "sales_amount",
                    "cost_amount",
                    "profit_amount",
                    "region",
                ),
            )
            catalog = sample_multi_metric_catalog()
            publication = replace(sample_publication(), physical_table_name=table_name)
            catalog = replace(
                catalog,
                datasets=(replace(catalog.datasets[0], active_publication=publication),),
            )
            plan = StructuredQueryPlanner(catalog).plan_multi(
                StructuredMultiAggregateIntent(
                    "ds-sales",
                    (
                        StructuredMetricIntent("sum", "sales_amount"),
                        StructuredMetricIntent("sum", "cost_amount"),
                        StructuredMetricIntent("sum", "profit_amount"),
                    ),
                    (StructuredFilter("region", "eq", "华东"),),
                    False,
                ),
                publication,
            )

            result = StructuredQueryExecutor(
                catalog,
                ClickHouseGateway(admin, query_client=query_client, max_result_rows=1),
            ).execute_multi(plan)

            self.assertEqual(plan.sql.count("SELECT"), 1)
            self.assertNotIn("toDecimalString", plan.sql)
            self.assertNotIn(" JOIN ", plan.sql.upper())
            self.assertEqual(result.total_count, 3)
            self.assertEqual(
                [metric.value for metric in result.metrics],
                [Decimal("300.25"), Decimal("130.10"), Decimal("180.15")],
            )
            self.assertEqual(
                [
                    (metric.valid_count, metric.null_count)
                    for metric in result.metrics
                ],
                [(2, 1), (3, 0), (2, 1)],
            )
        finally:
            admin.command(f"DROP TABLE IF EXISTS {table_name}")
            admin.close()
            query_client.close()

    def test_parameterized_average_matches_published_rows(self) -> None:
        import clickhouse_connect

        host = os.environ["CLICKHOUSE_HOST"]
        port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
        query_username = os.getenv("CLICKHOUSE_QUERY_USER", "default")
        query_password = os.getenv("CLICKHOUSE_QUERY_PASSWORD", "")
        ingest_username = os.getenv("CLICKHOUSE_INGEST_USER", "default")
        ingest_password = os.getenv("CLICKHOUSE_INGEST_PASSWORD", "")
        table_name = f"structured_query_it_{uuid.uuid4().hex[:16]}"
        admin = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=ingest_username,
            password=ingest_password,
        )
        query_client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=query_username,
            password=query_password,
        )
        try:
            admin.command(
                f"CREATE TABLE {table_name} ("
                "order_amount Nullable(Decimal(38, 9)), "
                "region Nullable(String), "
                "order_date Nullable(Date)"
                ") ENGINE = Memory"
            )
            admin.insert(
                table_name,
                [
                    [Decimal("10"), "华东", date(2026, 1, 1)],
                    [Decimal("30"), "华东", date(2026, 1, 2)],
                    [Decimal("50"), "华南", date(2026, 1, 3)],
                ],
                column_names=("order_amount", "region", "order_date"),
            )
            catalog = sample_catalog()
            publication = replace(sample_publication(), physical_table_name=table_name)
            catalog = replace(
                catalog,
                datasets=(replace(catalog.datasets[0], active_publication=publication),),
            )
            plan = StructuredQueryPlanner(catalog).plan(
                StructuredIntent(
                    "ds-sales",
                    "avg",
                    "order_amount",
                    (StructuredFilter("region", "eq", "华东"),),
                ),
                publication,
            )

            result = StructuredQueryExecutor(
                catalog,
                ClickHouseGateway(admin, query_client=query_client, max_result_rows=1),
            ).execute(plan)

            self.assertEqual(result.value, Decimal("20"))
            self.assertEqual((result.total_count, result.valid_count, result.null_count), (2, 2, 0))
        finally:
            admin.command(f"DROP TABLE IF EXISTS {table_name}")
            admin.close()
            query_client.close()


if __name__ == "__main__":
    unittest.main()
