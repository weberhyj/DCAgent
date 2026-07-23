from __future__ import annotations

import os
import re
import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from openpyxl import Workbook

from app.database import Database
from app.main import create_app
from app.models import ChatMessageModel, ResponseParagraphModel
from app.sql_repository import SqlChatRepository
from app.structured_answer import StructuredAnswerService
from app.structured_ingestion import ArrowParquetSink, SpreadsheetPublisher
from app.structured_repository import StructuredRepository
from app.structured_worker import StructuredIngestionWorker
from tests.support.structured_fakes import sample_catalog

ROW_COUNT = 100_000
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _fixture_row(index: int) -> tuple[Decimal | None, str, date]:
    amount = None if index % 97 == 0 else Decimal(index % 100_000) / Decimal(100)
    region = ("华东", "华南", "华北", "西部")[index % 4]
    order_date = date(2025, 1, 1) + timedelta(days=index % 365)
    return amount, region, order_date


def _write_fixture(path: Path, rows: int = ROW_COUNT) -> Path:
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("Sales")
    sheet.append(("order_amount", "region", "order_date"))
    for index in range(rows):
        sheet.append(_fixture_row(index))
    workbook.save(path)
    workbook.close()
    return path


class PhysocFake:
    def __init__(self) -> None:
        self.calls = 0

    def generate_reply(self, _request: object) -> ChatMessageModel:
        self.calls += 1
        return ChatMessageModel(
            id=f"msg-physoc-{self.calls}",
            role="assistant",
            time="2026-07-23 12:00:00",
            paragraphs=[ResponseParagraphModel(text="legacy Physoc answer")],
        )


class SwitchableCatalog:
    def __init__(self, repository: StructuredRepository) -> None:
        self.repository = repository
        self.override = None

    def __call__(self):
        return self.override or self.repository.get_catalog()


class InMemoryAggregateGateway:
    def __init__(self) -> None:
        self.rows: list[tuple[Decimal | None, str, date]] = []
        self.query_calls: list[tuple[str, dict[str, object]]] = []
        self.last_result: dict[str, object] | None = None
        self.error: Exception | None = None
        self.batch_rows: list[int] = []

    def prepare_publication(
        self,
        _schema: object,
        _publication_id: str,
        content_hash: str,
        **_kwargs: object,
    ) -> object:
        self.rows.clear()
        self.batch_rows.clear()
        return SimpleNamespace(
            physical_table_name="structured_e2e_sales",
            staging_table="structured_e2e_sales_staging",
            content_hash=content_hash,
        )

    def insert_batch(self, _target: object, batch: object) -> None:
        amount_index = batch.schema.get_field_index("order_amount")
        region_index = batch.schema.get_field_index("region")
        date_index = batch.schema.get_field_index("order_date")
        amounts = batch.column(amount_index).to_pylist()
        regions = batch.column(region_index).to_pylist()
        dates = batch.column(date_index).to_pylist()
        self.rows.extend(zip(amounts, regions, dates, strict=True))
        self.batch_rows.append(batch.num_rows)

    def validate_and_promote(self, target: object, **statistics: object) -> str:
        if statistics["row_count"] != len(self.rows):
            raise AssertionError("published row count does not match inserted rows")
        return target.physical_table_name

    def discard_publication(self, _target: object) -> None:
        self.rows.clear()

    def query(self, statement: str, parameters: object) -> dict[str, object]:
        if self.error is not None:
            raise self.error
        bound = dict(parameters)
        self.query_calls.append((statement, bound))
        filtered = [row for row in self.rows if self._matches(statement, bound, row)]
        match = re.search(
            r"SELECT\s+(avg|sum|count|min|max)\(([^)]*)\)\s+AS aggregate_value",
            statement,
            re.IGNORECASE,
        )
        if match is None:
            raise AssertionError(statement)
        aggregate = match.group(1).lower()
        metric = match.group(2).strip()
        values = [row[0] for row in filtered if row[0] is not None]
        if aggregate == "count":
            aggregate_value: Decimal | int | None = len(filtered) if not metric else len(values)
        elif not values:
            aggregate_value = None
        elif aggregate == "avg":
            aggregate_value = sum(values, Decimal(0)) / len(values)
        elif aggregate == "sum":
            aggregate_value = sum(values, Decimal(0))
        elif aggregate == "min":
            aggregate_value = min(values)
        else:
            aggregate_value = max(values)
        self.last_result = {
            "aggregate_value": aggregate_value,
            "total_count": len(filtered),
            "valid_count": len(filtered) if not metric else len(values),
            "null_count": 0 if not metric else len(filtered) - len(values),
        }
        return self.last_result

    @staticmethod
    def _matches(
        statement: str,
        parameters: dict[str, object],
        row: tuple[Decimal | None, str, date],
    ) -> bool:
        values = {"order_amount": row[0], "region": row[1], "order_date": row[2]}
        for name, expected in parameters.items():
            pattern = re.compile(
                rf"([a-z0-9_]+)\s*(>=|<=|=|>|<)\s*\{{{re.escape(name)}:[^}}]+\}}",
                re.IGNORECASE,
            )
            match = pattern.search(statement)
            if match is None:
                raise AssertionError(f"parameter {name} is not bound in SQL: {statement}")
            actual = values[match.group(1)]
            if actual is None:
                return False
            operator = match.group(2)
            if operator == "=" and actual != expected:
                return False
            if operator == ">=" and actual < expected:
                return False
            if operator == "<=" and actual > expected:
                return False
            if operator == ">" and actual <= expected:
                return False
            if operator == "<" and actual >= expected:
                return False
        return True

    def close(self) -> None:
        return None


def _reference(
    aggregate: str,
    predicate,
) -> dict[str, Decimal | int | None]:
    filtered = [
        _fixture_row(index) for index in range(ROW_COUNT) if predicate(*_fixture_row(index))
    ]
    values = [amount for amount, _region, _order_date in filtered if amount is not None]
    if aggregate == "avg":
        value: Decimal | int | None = sum(values, Decimal(0)) / len(values)
    elif aggregate == "sum":
        value = sum(values, Decimal(0))
    elif aggregate == "count":
        value = len(values)
    elif aggregate == "min":
        value = min(values)
    else:
        value = max(values)
    return {
        "aggregate_value": value,
        "total_count": len(filtered),
        "valid_count": len(values),
        "null_count": len(filtered) - len(values),
    }


class StructuredAggregationEndToEndTest(unittest.TestCase):
    def test_upload_publish_query_and_fail_closed_routing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _write_fixture(root / "structured-100k.xlsx")
            database = Database("sqlite+pysqlite:///:memory:")
            database.create_schema()
            structured_repository = StructuredRepository(database)
            gateway = InMemoryAggregateGateway()
            catalogs = SwitchableCatalog(structured_repository)
            physoc = PhysocFake()
            chat_repository = SqlChatRepository(
                database,
                llm_provider=physoc,
                structured_service=StructuredAnswerService(catalogs, gateway),
            )
            app = create_app(
                repository=chat_repository,
                structured_repository=structured_repository,
                structured_query_enabled=True,
                upload_dir=root / "uploads",
            )
            try:
                with TestClient(app) as client, fixture.open("rb") as handle:
                    upload = client.post(
                        "/api/knowledge/uploads",
                        files={"files": (fixture.name, handle, XLSX_MIME)},
                    )
                    self.assertEqual(upload.status_code, 200, upload.text)
                    source_id = upload.json()[0]["id"]
                    client.get("/api/knowledge/sources")
                    preview_response = client.get(
                        f"/api/knowledge/sources/{source_id}/structured-preview"
                    )
                    self.assertEqual(preview_response.status_code, 200, preview_response.text)
                    preview = preview_response.json()
                    dataset = preview["datasets"][0]
                    columns = [
                        {
                            "physicalName": column["physicalName"],
                            "displayName": column["physicalName"],
                            "dataType": column["dataType"],
                            "aliases": column["aliases"],
                            "allowAggregate": column["physicalName"] == "order_amount",
                            "allowFilter": True,
                            "nullPolicy": "ignore",
                        }
                        for column in dataset["columns"]
                    ]
                    confirmation = client.put(
                        f"/api/knowledge/sources/{source_id}/structured-schema",
                        json={
                            "datasets": [{"datasetId": dataset["datasetId"], "columns": columns}]
                        },
                    )
                    self.assertEqual(confirmation.status_code, 200, confirmation.text)
                    enqueue = client.post(
                        f"/api/knowledge/sources/{source_id}/structured-publications"
                    )
                    self.assertEqual(enqueue.status_code, 202, enqueue.text)

                    worker = StructuredIngestionWorker(
                        structured_repository,
                        SpreadsheetPublisher(
                            sink=ArrowParquetSink(root / "parquet"),
                            clickhouse=gateway,
                            batch_rows=25_000,
                        ),
                        worker_id="e2e-worker",
                        lease_seconds=60,
                    )
                    self.assertTrue(worker.run_once())
                    status = client.get(f"/api/knowledge/sources/{source_id}/structured-status")
                    self.assertEqual(status.status_code, 200, status.text)
                    self.assertEqual(status.json()["job"]["status"], "published")
                    self.assertEqual(status.json()["activePublication"]["rowCount"], ROW_COUNT)
                    self.assertEqual(sum(gateway.batch_rows), ROW_COUNT)
                    self.assertLessEqual(max(gateway.batch_rows), 25_000)

                    cases = (
                        ("order_amount平均值", "avg", lambda _a, _r, _d: True),
                        ("order_amount总和，region为华东", "sum", lambda _a, r, _d: r == "华东"),
                        ("order_amount计数", "count", lambda _a, _r, _d: True),
                        (
                            "order_amount最小值，order_amount不少于250",
                            "min",
                            lambda amount, _r, _d: amount is not None and amount >= Decimal(250),
                        ),
                        (
                            "order_amount最大值，order_date2025-04-01至2025-06-30",
                            "max",
                            lambda _a, _r, order_date: (
                                date(2025, 4, 1) <= order_date <= date(2025, 6, 30)
                            ),
                        ),
                    )
                    for question, aggregate, predicate in cases:
                        with self.subTest(question=question):
                            conversation_id = client.post("/api/conversations").json()[
                                "activeConversationId"
                            ]
                            response = client.post(
                                f"/api/conversations/{conversation_id}/messages",
                                json={"content": question, "mode": "source"},
                            )
                            self.assertEqual(response.status_code, 200, response.text)
                            self.assertEqual(gateway.last_result, _reference(aggregate, predicate))
                            self.assertIn(
                                f"aggregate={aggregate}",
                                response.json()["messages"][-1]["paragraphs"][0]["text"],
                            )

                    document_conversation = client.post("/api/conversations").json()[
                        "activeConversationId"
                    ]
                    document = client.post(
                        f"/api/conversations/{document_conversation}/messages",
                        json={"content": "请介绍差旅报销政策", "mode": "source"},
                    )
                    self.assertEqual(document.status_code, 200, document.text)
                    self.assertEqual(physoc.calls, 1)
                    self.assertEqual(
                        document.json()["messages"][-1]["paragraphs"][0]["text"],
                        "legacy Physoc answer",
                    )

                    catalogs.override = sample_catalog(ambiguous=True)
                    query_count = len(gateway.query_calls)
                    ambiguous_conversation = client.post("/api/conversations").json()[
                        "activeConversationId"
                    ]
                    ambiguous = client.post(
                        f"/api/conversations/{ambiguous_conversation}/messages",
                        json={"content": "平均金额", "mode": "source"},
                    )
                    self.assertEqual(ambiguous.status_code, 200, ambiguous.text)
                    self.assertEqual(len(gateway.query_calls), query_count)
                    self.assertEqual(physoc.calls, 1)
                    ambiguous_run = next(
                        run
                        for run in chat_repository.list_agent_runs()
                        if run.conversation_id == ambiguous_conversation
                    )
                    self.assertIn(
                        "clarification",
                        ambiguous_run.steps[0].output_summary,
                    )

                    catalogs.override = None
                    gateway.error = TimeoutError("ClickHouse query timed out")
                    timeout_conversation = client.post("/api/conversations").json()[
                        "activeConversationId"
                    ]
                    timeout = client.post(
                        f"/api/conversations/{timeout_conversation}/messages",
                        json={"content": "order_amount总和", "mode": "source"},
                    )
                    self.assertEqual(timeout.status_code, 200, timeout.text)
                    self.assertEqual(physoc.calls, 1)
                    timeout_run = next(
                        run
                        for run in chat_repository.list_agent_runs()
                        if run.conversation_id == timeout_conversation
                    )
                    self.assertIn(
                        "unavailable",
                        timeout_run.steps[0].output_summary,
                    )
            finally:
                database.engine.dispose()


@unittest.skipUnless(
    os.getenv("RUN_OFFLINE_INTEGRATION") == "1" and bool(os.getenv("CLICKHOUSE_HOST")),
    "target-host gate requires RUN_OFFLINE_INTEGRATION=1 and CLICKHOUSE_HOST",
)
class StructuredAggregationTargetHostGateTest(unittest.TestCase):
    def test_clickhouse_target_is_explicitly_available(self) -> None:
        self.assertTrue(os.environ["CLICKHOUSE_HOST"].strip())


if __name__ == "__main__":
    unittest.main()
