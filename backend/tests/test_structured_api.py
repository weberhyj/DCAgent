from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
from openpyxl import Workbook
from sqlalchemy import event, insert, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import (
    Database,
    KnowledgeSourceRecord,
    StructuredDatasetRecord,
    StructuredPreviewRecord,
)
from app.infra.health import DependencyHealthRegistry
from app.llm import TemplateLLMProvider
from app.main import create_app, create_production_app
from app.offline_settings import OfflineSettings
from app.repository import InMemoryChatRepository
from app.seed import build_seed_state
from app.spreadsheet_schema import (
    MAX_COLUMNS_PER_DATASET,
    MAX_WORKSHEETS,
    infer_spreadsheet_schema,
)
from app.sql_repository import SqlChatRepository
from app.structured_models import (
    SpreadsheetPreview,
    StructuredColumnType,
    StructuredDiagnostic,
)
from app.structured_repository import (
    StructuredColumnConfirmation,
    StructuredConflictError,
    StructuredDatasetConfirmation,
    StructuredRepository,
    StructuredValidationError,
)

WAITING_FOR_SCHEMA = "\u5f85\u786e\u8ba4\u8868\u7ed3\u6784"
INDEXING = "\u89e3\u6790\u4e2d"
INDEXED = "\u5df2\u7d22\u5f15"
FAILED = "\u89e3\u6790\u5931\u8d25"


def workbook_bytes(rows: list[list[object]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


class StructuredApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Database("sqlite+pysqlite:///:memory:")
        self.database.create_schema()
        self.chat_repository = SqlChatRepository(self.database)
        self.structured_repository = StructuredRepository(self.database)

    def tearDown(self) -> None:
        self.database.engine.dispose()
        self.temp_dir.cleanup()

    def build_client(self, *, structured_query_enabled: bool = True) -> TestClient:
        return TestClient(
            create_app(
                repository=self.chat_repository,
                structured_repository=self.structured_repository,
                structured_query_enabled=structured_query_enabled,
                upload_dir=Path(self.temp_dir.name),
            )
        )

    def upload(
        self,
        client: TestClient,
        name: str,
        content: bytes,
        content_type: str,
    ) -> str:
        response = client.post(
            "/api/knowledge/uploads",
            files={"files": (name, content, content_type)},
        )
        self.assertEqual(response.status_code, 200, response.text)
        uploaded = response.json()[0]
        self.assertEqual(uploaded["status"], INDEXING)
        return uploaded["id"]

    def upload_xlsx(self, client: TestClient) -> str:
        return self.upload(
            client,
            "sales.xlsx",
            workbook_bytes([["amount", "region"], ["12.5", "East"], ["7", "West"]]),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def confirmation_payload(self, preview: dict[str, Any]) -> dict[str, Any]:
        datasets = []
        for dataset in preview["datasets"]:
            columns = []
            for column in dataset["columns"]:
                columns.append(
                    {
                        "physicalName": column["physicalName"],
                        "displayName": column["displayName"],
                        "dataType": column["dataType"],
                        "aliases": column["aliases"],
                        "allowAggregate": column["dataType"] in {"integer", "decimal"},
                        "allowFilter": True,
                        "nullPolicy": "ignore",
                    }
                )
            datasets.append({"datasetId": dataset["datasetId"], "columns": columns})
        return {"datasets": datasets}

    def test_enabled_xlsx_upload_exposes_camel_case_preview_without_chunks(self) -> None:
        client = self.build_client()
        source_id = self.upload_xlsx(client)

        sources = client.get("/api/knowledge/sources")
        source = next(item for item in sources.json() if item["id"] == source_id)
        self.assertEqual(source["status"], WAITING_FOR_SCHEMA)
        preview = client.get(f"/api/knowledge/sources/{source_id}/structured-preview")

        self.assertEqual(preview.status_code, 200, preview.text)
        body = preview.json()
        self.assertEqual(body["sourceId"], source_id)
        self.assertEqual(body["datasets"][0]["worksheetName"], "Sales")
        self.assertEqual(body["datasets"][0]["columns"][0]["dataType"], "decimal")
        self.assertIn("sampledRows", body["datasets"][0])
        self.assertIn("nullCount", body["datasets"][0]["columns"][0])
        chunks = client.get(f"/api/knowledge/sources/{source_id}/chunks")
        self.assertEqual(chunks.status_code, 200)
        self.assertEqual(chunks.json(), [])

    def test_enabled_csv_upload_enters_preview_state(self) -> None:
        client = self.build_client()
        source_id = self.upload(
            client,
            "sales.csv",
            b"amount,region\n12.5,East\n7,West\n",
            "text/csv",
        )

        sources = client.get("/api/knowledge/sources").json()
        source = next(item for item in sources if item["id"] == source_id)
        preview = client.get(f"/api/knowledge/sources/{source_id}/structured-preview")
        self.assertEqual(source["status"], WAITING_FOR_SCHEMA)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["datasets"][0]["columns"][0]["dataType"], "decimal")

    def test_confirmation_succeeds_and_reconfirmation_increments_version(self) -> None:
        client = self.build_client()
        source_id = self.upload_xlsx(client)
        client.get("/api/knowledge/sources")
        preview = client.get(f"/api/knowledge/sources/{source_id}/structured-preview").json()
        payload = self.confirmation_payload(preview)

        first = client.put(f"/api/knowledge/sources/{source_id}/structured-schema", json=payload)
        second = client.put(f"/api/knowledge/sources/{source_id}/structured-schema", json=payload)

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["status"], "confirmed")
        self.assertEqual(first.json()["datasets"][0]["schemaVersion"], 1)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["datasets"][0]["schemaVersion"], 2)
        self.assertEqual(first.json()["datasets"][0]["columns"][0]["allowAggregate"], True)

    def test_confirmation_rejects_missing_column_and_string_aggregate(self) -> None:
        client = self.build_client()
        source_id = self.upload_xlsx(client)
        client.get("/api/knowledge/sources")
        preview = client.get(f"/api/knowledge/sources/{source_id}/structured-preview").json()
        missing = self.confirmation_payload(preview)
        missing["datasets"][0]["columns"] = missing["datasets"][0]["columns"][:1]
        aggregate = self.confirmation_payload(preview)
        aggregate["datasets"][0]["columns"][1]["allowAggregate"] = True
        unknown = self.confirmation_payload(preview)
        unknown["datasets"][0]["columns"][0]["physicalName"] = "unknown_amount"

        missing_response = client.put(
            f"/api/knowledge/sources/{source_id}/structured-schema", json=missing
        )
        aggregate_response = client.put(
            f"/api/knowledge/sources/{source_id}/structured-schema", json=aggregate
        )
        unknown_response = client.put(
            f"/api/knowledge/sources/{source_id}/structured-schema", json=unknown
        )

        self.assertEqual(missing_response.status_code, 400, missing_response.text)
        self.assertEqual(aggregate_response.status_code, 400, aggregate_response.text)
        self.assertEqual(unknown_response.status_code, 400, unknown_response.text)
        self.assertIn("aggregate", aggregate_response.json()["detail"].lower())

    def test_confirmation_requires_explicit_capability_booleans(self) -> None:
        client = self.build_client()
        source_id = self.upload_xlsx(client)
        client.get("/api/knowledge/sources")
        preview = client.get(f"/api/knowledge/sources/{source_id}/structured-preview").json()
        payload = self.confirmation_payload(preview)
        del payload["datasets"][0]["columns"][0]["allowAggregate"]

        response = client.put(f"/api/knowledge/sources/{source_id}/structured-schema", json=payload)

        self.assertEqual(response.status_code, 422, response.text)

    def test_confirmation_request_rejects_empty_and_over_limit_collections(self) -> None:
        client = self.build_client()
        source_id = self.upload_xlsx(client)
        client.get("/api/knowledge/sources")
        column = {
            "physicalName": "amount",
            "displayName": "Amount",
            "dataType": "decimal",
            "aliases": ["amount"],
            "allowAggregate": True,
            "allowFilter": True,
            "nullPolicy": "ignore",
        }
        dataset = {"datasetId": "ds-sales", "columns": [column]}
        payloads = (
            {"datasets": []},
            {"datasets": [{"datasetId": "ds-sales", "columns": []}]},
            {"datasets": [dataset] * (MAX_WORKSHEETS + 1)},
            {
                "datasets": [
                    {
                        "datasetId": "ds-sales",
                        "columns": [column] * (MAX_COLUMNS_PER_DATASET + 1),
                    }
                ]
            },
            {
                "datasets": [
                    {
                        "datasetId": "ds-sales",
                        "columns": [{**column, "aliases": ["a" * 81]}],
                    }
                ]
            },
        )

        for payload in payloads:
            with self.subTest(payload_shape=str(payload)[:80]):
                response = client.put(
                    f"/api/knowledge/sources/{source_id}/structured-schema",
                    json=payload,
                )
                self.assertEqual(response.status_code, 422, response.text)

    def test_missing_and_conflicting_previews_map_to_404_and_409(self) -> None:
        client = self.build_client()
        missing = client.get("/api/knowledge/sources/missing/structured-preview")
        source_id = self.upload_xlsx(client)
        client.get("/api/knowledge/sources")
        with self.database.session() as session:
            preview_record = session.get(StructuredPreviewRecord, source_id)
            assert preview_record is not None
            preview_record.payload = {"invalid": True}
        conflict = client.get(f"/api/knowledge/sources/{source_id}/structured-preview")

        self.assertEqual(missing.status_code, 404, missing.text)
        self.assertEqual(conflict.status_code, 409, conflict.text)

    def test_disabled_spreadsheets_and_enabled_non_table_files_use_legacy_chunks(self) -> None:
        disabled_client = self.build_client(structured_query_enabled=False)
        xlsx_source_id = self.upload_xlsx(disabled_client)
        xlsx_sources = disabled_client.get("/api/knowledge/sources").json()
        xlsx_source = next(item for item in xlsx_sources if item["id"] == xlsx_source_id)
        self.assertEqual(xlsx_source["status"], INDEXED)
        self.assertGreater(
            len(disabled_client.get(f"/api/knowledge/sources/{xlsx_source_id}/chunks").json()),
            0,
        )
        csv_source_id = self.upload(
            disabled_client,
            "legacy.csv",
            b"amount,region\n12.5,East\n",
            "text/csv",
        )
        csv_sources = disabled_client.get("/api/knowledge/sources").json()
        csv_source = next(item for item in csv_sources if item["id"] == csv_source_id)
        self.assertEqual(csv_source["status"], INDEXED)
        self.assertGreater(
            len(disabled_client.get(f"/api/knowledge/sources/{csv_source_id}/chunks").json()),
            0,
        )

        enabled_client = self.build_client(structured_query_enabled=True)
        text_source_id = self.upload(
            enabled_client,
            "policy.txt",
            b"travel policy approval flow " * 40,
            "text/plain",
        )
        text_sources = enabled_client.get("/api/knowledge/sources").json()
        text_source = next(item for item in text_sources if item["id"] == text_source_id)
        self.assertEqual(text_source["status"], INDEXED)
        self.assertGreater(
            len(enabled_client.get(f"/api/knowledge/sources/{text_source_id}/chunks").json()),
            0,
        )

    def test_disabled_feature_returns_stable_not_found_for_preview_route(self) -> None:
        client = self.build_client(structured_query_enabled=False)
        source_id = self.upload_xlsx(client)
        client.get("/api/knowledge/sources")

        response = client.get(f"/api/knowledge/sources/{source_id}/structured-preview")

        self.assertEqual(response.status_code, 404, response.text)

    def test_enabled_feature_without_repository_returns_stable_unavailable_response(self) -> None:
        client = TestClient(
            create_app(
                repository=self.chat_repository,
                structured_query_enabled=True,
                upload_dir=Path(self.temp_dir.name),
            )
        )

        source_id = self.upload_xlsx(client)
        sources = client.get("/api/knowledge/sources").json()
        source = next(item for item in sources if item["id"] == source_id)
        response = client.get(f"/api/knowledge/sources/{source_id}/structured-preview")

        self.assertEqual(source["status"], FAILED)
        self.assertEqual(client.get(f"/api/knowledge/sources/{source_id}/chunks").json(), [])
        self.assertEqual(response.status_code, 503, response.text)


class StructuredWiringTest(unittest.TestCase):
    def test_offline_settings_parse_structured_query_flag_defaulting_false(self) -> None:
        self.assertFalse(OfflineSettings.from_environ({}).structured_query_enabled)
        self.assertTrue(
            OfflineSettings.from_environ(
                {"STRUCTURED_QUERY_ENABLED": "true"}
            ).structured_query_enabled
        )

    def test_production_app_injects_flag_without_breaking_legacy_queue_factory(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        custom_queue = SimpleNamespace(close=lambda: None)
        production = create_production_app(
            environ={"OFFLINE_MODE": "false", "STRUCTURED_QUERY_ENABLED": "false"},
            repository_factory=lambda: InMemoryChatRepository(build_seed_state()),
            database_factory=lambda _url: database,
            llm_provider_factory=lambda _environment: TemplateLLMProvider(),
            health_registry_factory=DependencyHealthRegistry,
            ingestion_queue_factory=lambda _repository: custom_queue,
        )
        with TestClient(production) as client:
            self.assertIs(client.app.state.knowledge_ingestion_queue, custom_queue)
            self.assertFalse(client.app.state.structured_query_enabled)
            self.assertIsInstance(client.app.state.structured_repository, StructuredRepository)

    def test_enabled_production_app_rejects_legacy_one_argument_queue_factory(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        production = create_production_app(
            environ={"OFFLINE_MODE": "false", "STRUCTURED_QUERY_ENABLED": "true"},
            repository_factory=lambda: InMemoryChatRepository(build_seed_state()),
            database_factory=lambda _url: database,
            llm_provider_factory=lambda _environment: TemplateLLMProvider(),
            health_registry_factory=DependencyHealthRegistry,
            ingestion_queue_factory=lambda _repository: SimpleNamespace(close=lambda: None),
        )

        with self.assertRaisesRegex(TypeError, "structured-aware|structured_repository"):
            with TestClient(production):
                pass

    def test_enabled_production_app_passes_context_to_structured_aware_factory(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        captured: dict[str, object] = {}
        custom_queue = SimpleNamespace(close=lambda: None)

        def build_queue(
            repository: object,
            structured_repository: object,
            structured_query_enabled: bool,
        ) -> object:
            captured.update(
                repository=repository,
                structured_repository=structured_repository,
                structured_query_enabled=structured_query_enabled,
            )
            return custom_queue

        production = create_production_app(
            environ={"OFFLINE_MODE": "false", "STRUCTURED_QUERY_ENABLED": "true"},
            repository_factory=lambda: InMemoryChatRepository(build_seed_state()),
            database_factory=lambda _url: database,
            llm_provider_factory=lambda _environment: TemplateLLMProvider(),
            health_registry_factory=DependencyHealthRegistry,
            ingestion_queue_factory=build_queue,
        )

        with TestClient(production) as client:
            self.assertIs(client.app.state.knowledge_ingestion_queue, custom_queue)
            self.assertIs(captured["repository"], client.app.state.repository)
            self.assertIs(captured["structured_repository"], client.app.state.structured_repository)
            self.assertIs(captured["structured_query_enabled"], True)

    def test_production_app_passes_flag_and_repository_to_default_queue(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        production = create_production_app(
            environ={"OFFLINE_MODE": "false", "STRUCTURED_QUERY_ENABLED": "true"},
            repository_factory=lambda: InMemoryChatRepository(build_seed_state()),
            database_factory=lambda _url: database,
            llm_provider_factory=lambda _environment: TemplateLLMProvider(),
            health_registry_factory=DependencyHealthRegistry,
        )
        with TestClient(production) as client:
            queue = client.app.state.knowledge_ingestion_queue
            self.assertTrue(queue._structured_query_enabled)
            self.assertIs(queue._structured_repository, client.app.state.structured_repository)


class StructuredRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Database("sqlite+pysqlite:///:memory:")
        self.database.create_schema()
        self.chat_repository = SqlChatRepository(self.database)
        self.repository = StructuredRepository(self.database)
        self.source_id = "kb-sales"
        self.path = Path(self.temp_dir.name) / "sales.xlsx"
        self.path.write_bytes(
            workbook_bytes([["amount", "region"], ["12.5", "East"], ["7", "West"]])
        )
        self.chat_repository.add_uploaded_knowledge_source(
            source_id=self.source_id,
            name=self.path.name,
            source_type="XLSX",
            classification="internal",
            records=2,
            file_path=str(self.path),
            file_size=self.path.stat().st_size,
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self.preview = infer_spreadsheet_schema(self.path, self.source_id)

    def tearDown(self) -> None:
        self.database.engine.dispose()
        self.temp_dir.cleanup()

    def confirmation(self, *, amount_display_name: str = "Order amount") -> tuple:
        dataset = self.preview.datasets[0]
        return (
            StructuredDatasetConfirmation(
                dataset_id=dataset.dataset_id,
                columns=(
                    StructuredColumnConfirmation(
                        physical_name="amount",
                        display_name=amount_display_name,
                        data_type=StructuredColumnType.DECIMAL,
                        aliases=("amount",),
                        allow_aggregate=True,
                        allow_filter=True,
                        null_policy="ignore",
                    ),
                    StructuredColumnConfirmation(
                        physical_name="region",
                        display_name="Region",
                        data_type=StructuredColumnType.STRING,
                        aliases=("region",),
                        allow_aggregate=False,
                        allow_filter=True,
                        null_policy="ignore",
                    ),
                ),
            ),
        )

    def test_saves_and_reloads_bounded_preview_and_updates_source_status(self) -> None:
        self.repository.save_preview(self.preview)

        reloaded = self.repository.get_preview(self.source_id)
        source = next(
            item
            for item in self.chat_repository.list_knowledge_sources()
            if item.id == self.source_id
        )
        self.assertEqual(source.status, WAITING_FOR_SCHEMA)
        self.assertEqual(reloaded.source_id, self.source_id)
        self.assertEqual(reloaded.datasets[0].columns[0].data_type, StructuredColumnType.DECIMAL)
        self.assertEqual(reloaded.datasets[0].columns[0].examples, ("12.5", "7"))
        self.assertEqual(reloaded.datasets[0].sampled_rows, 2)

    def test_preview_metadata_survives_repository_recreation(self) -> None:
        mixed_path = Path(self.temp_dir.name) / "mixed.xlsx"
        mixed_path.write_bytes(workbook_bytes([["value"], [1], ["text"]]))
        preview = infer_spreadsheet_schema(mixed_path, self.source_id)
        self.repository.save_preview(preview)

        reloaded = StructuredRepository(self.database).get_preview(self.source_id)

        self.assertEqual(reloaded, preview)

    def test_confirmation_rejects_a_stale_preview(self) -> None:
        class StalePreviewRepository(StructuredRepository):
            def get_preview(self, source_id: str) -> SpreadsheetPreview:
                preview = super().get_preview(source_id)
                dataset = preview.datasets[0]
                changed_column = replace(
                    dataset.columns[0], display_name="Changed after validation"
                )
                changed_dataset = replace(dataset, columns=(changed_column, *dataset.columns[1:]))
                StructuredRepository.save_preview(
                    self, replace(preview, datasets=(changed_dataset,))
                )
                return preview

        repository = StalePreviewRepository(self.database)
        repository.save_preview(self.preview)

        with self.assertRaisesRegex(StructuredConflictError, "preview"):
            repository.confirm_schema(self.source_id, self.confirmation())

    def test_confirmation_maps_schema_version_race_to_conflict(self) -> None:
        self.repository.save_preview(self.preview)
        self.repository.confirm_schema(self.source_id, self.confirmation())
        triggered = False

        def insert_competing_version(
            session: Session, _flush_context: object, _instances: object
        ) -> None:
            nonlocal triggered
            if triggered:
                return
            if any(
                isinstance(item, StructuredDatasetRecord)
                and item.status == "confirmed"
                and item.schema_version == 2
                for item in session.new
            ):
                triggered = True
                session.execute(
                    insert(StructuredDatasetRecord).values(
                        dataset_id=self.preview.datasets[0].dataset_id,
                        source_id=self.source_id,
                        worksheet_name=self.preview.datasets[0].worksheet_name,
                        schema_version=2,
                        schema_hash="b" * 64,
                        status="confirmed",
                    )
                )

        event.listen(Session, "before_flush", insert_competing_version)
        try:
            with self.assertRaisesRegex(StructuredConflictError, "version"):
                self.repository.confirm_schema(self.source_id, self.confirmation())
        finally:
            event.remove(Session, "before_flush", insert_competing_version)

    def test_postgres_writes_lock_source_before_preview_and_datasets(self) -> None:
        self.repository.save_preview(self.preview)
        with self.database.session() as session:
            source = session.get(KnowledgeSourceRecord, self.source_id)
            preview_record = session.get(StructuredPreviewRecord, self.source_id)
            records = session.scalars(
                select(StructuredDatasetRecord).where(
                    StructuredDatasetRecord.source_id == self.source_id,
                    StructuredDatasetRecord.schema_version == 0,
                )
            ).all()
            for record in records:
                tuple(record.columns)
        assert source is not None
        assert preview_record is not None

        class ScalarRows:
            def __init__(self, rows: list[StructuredDatasetRecord]) -> None:
                self._rows = rows

            def all(self) -> list[StructuredDatasetRecord]:
                return self._rows

        class RecordingSession:
            def __init__(self, *, include_preview: bool, rows: list[Any]) -> None:
                self.operations: list[str] = []
                self.include_preview = include_preview
                self.rows = rows

            def get_bind(self) -> object:
                return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

            def get(self, record_type: type[object], _key: object) -> object | None:
                self.operations.append(f"GET {record_type.__tablename__}")
                return source

            def scalar(self, statement: object) -> object | None:
                sql = str(statement.compile(dialect=postgresql.dialect()))
                self.operations.append(sql)
                if "FROM knowledge_sources" in sql:
                    return source
                if "FROM structured_previews" in sql:
                    return preview_record if self.include_preview else None
                if "max(structured_datasets.schema_version)" in sql:
                    return 0
                raise AssertionError(f"Unexpected scalar SQL: {sql}")

            def scalars(self, statement: object) -> ScalarRows:
                sql = str(statement.compile(dialect=postgresql.dialect()))
                self.operations.append(sql)
                return ScalarRows(self.rows)

            def add(self, _record: object) -> None:
                return None

            def delete(self, _record: object) -> None:
                return None

            def flush(self) -> None:
                return None

        class RecordingDatabase:
            def __init__(self, session: RecordingSession) -> None:
                self._session = session

            @contextmanager
            def session(self):
                yield self._session

        def assert_lock_order(test_case: unittest.TestCase, operations: list[str]) -> None:
            test_case.assertIn("FROM knowledge_sources", operations[0])
            test_case.assertIn("FOR UPDATE", operations[0])
            test_case.assertIn("FROM structured_previews", operations[1])
            test_case.assertIn("FOR UPDATE", operations[1])
            test_case.assertIn("FROM structured_datasets", operations[2])
            test_case.assertIn("FOR UPDATE", operations[2])

        save_session = RecordingSession(include_preview=False, rows=[])
        StructuredRepository(RecordingDatabase(save_session)).save_preview(self.preview)  # type: ignore[arg-type]
        assert_lock_order(self, save_session.operations)

        class CachedPreviewRepository(StructuredRepository):
            def get_preview(self, _source_id: str) -> SpreadsheetPreview:
                return self_preview

        self_preview = self.preview
        confirm_session = RecordingSession(include_preview=True, rows=records)
        CachedPreviewRepository(RecordingDatabase(confirm_session)).confirm_schema(  # type: ignore[arg-type]
            self.source_id, self.confirmation()
        )
        assert_lock_order(self, confirm_session.operations)

    def test_non_version_integrity_error_uses_generic_conflict_message(self) -> None:
        self.repository.save_preview(self.preview)

        def raise_foreign_key_error(
            _session: Session, _flush_context: object, _instances: object
        ) -> None:
            raise IntegrityError("INSERT", {}, Exception("FOREIGN KEY constraint failed"))

        event.listen(Session, "before_flush", raise_foreign_key_error)
        try:
            with self.assertRaises(StructuredConflictError) as caught:
                self.repository.confirm_schema(self.source_id, self.confirmation())
        finally:
            event.remove(Session, "before_flush", raise_foreign_key_error)

        self.assertNotIn("version", str(caught.exception).lower())
        self.assertIn("database state", str(caught.exception).lower())

    def test_corrupt_preview_payload_always_raises_conflict(self) -> None:
        corruptions = {
            "source id": lambda payload: payload.update({"source_id": "other-source"}),
            "dataset source id": lambda payload: payload["datasets"][0].update(
                {"source_id": "other-source"}
            ),
            "datasets type": lambda payload: payload.update({"datasets": "invalid"}),
            "dataset object": lambda payload: payload.update({"datasets": [None]}),
            "columns object": lambda payload: payload["datasets"][0].update({"columns": [None]}),
            "enum": lambda payload: payload["datasets"][0]["columns"][0].update(
                {"data_type": "invalid"}
            ),
            "examples type": lambda payload: payload["datasets"][0]["columns"][0].update(
                {"examples": "invalid"}
            ),
            "sampled rows": lambda payload: payload["datasets"][0].update({"sampled_rows": []}),
            "column sampled rows": lambda payload: payload["datasets"][0]["columns"][0].update(
                {"sampled_rows": []}
            ),
            "null count": lambda payload: payload["datasets"][0]["columns"][0].update(
                {"null_count": []}
            ),
            "diagnostic object": lambda payload: payload.update({"diagnostics": [None]}),
            "diagnostic row number": lambda payload: payload.update(
                {
                    "diagnostics": [
                        {
                            "code": "mixed_type",
                            "message": "Mixed values",
                            "worksheet_name": "Sales",
                            "column_name": "amount",
                            "row_number": [],
                        }
                    ]
                }
            ),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(corruption=name):
                self.repository.save_preview(self.preview)
                with self.database.session() as session:
                    record = session.get(StructuredPreviewRecord, self.source_id)
                    assert record is not None
                    payload = deepcopy(record.payload)
                    corrupt(payload)
                    record.payload = payload

                with self.assertRaises(StructuredConflictError):
                    self.repository.get_preview(self.source_id)

    def test_reconfirmation_creates_an_immutable_new_schema_version(self) -> None:
        self.repository.save_preview(self.preview)

        first = self.repository.confirm_schema(self.source_id, self.confirmation())
        second = self.repository.confirm_schema(
            self.source_id,
            self.confirmation(amount_display_name="Net order amount"),
        )

        dataset_id = self.preview.datasets[0].dataset_id
        self.assertEqual(first.datasets[0].schema_version, 1)
        self.assertEqual(second.datasets[0].schema_version, 2)
        self.assertEqual(
            self.repository.get_schema(dataset_id, 1).columns[0].display_name,
            "Order amount",
        )
        self.assertEqual(
            self.repository.get_schema(dataset_id, 2).columns[0].display_name,
            "Net order amount",
        )
        self.assertTrue(first.datasets[0].columns[0].allow_aggregate)
        self.assertFalse(first.datasets[0].columns[1].allow_aggregate)

    def test_confirmation_rejects_incomplete_columns_and_invalid_capabilities(self) -> None:
        self.repository.save_preview(self.preview)
        dataset = self.confirmation()[0]
        missing_column = StructuredDatasetConfirmation(
            dataset_id=dataset.dataset_id,
            columns=dataset.columns[:1],
        )
        invalid_aggregate = StructuredDatasetConfirmation(
            dataset_id=dataset.dataset_id,
            columns=(
                dataset.columns[0],
                StructuredColumnConfirmation(
                    physical_name="region",
                    display_name="Region",
                    data_type=StructuredColumnType.STRING,
                    aliases=("region",),
                    allow_aggregate=True,
                    allow_filter=True,
                    null_policy="ignore",
                ),
            ),
        )

        with self.assertRaisesRegex(StructuredValidationError, "physical columns"):
            self.repository.confirm_schema(self.source_id, (missing_column,))
        with self.assertRaisesRegex(StructuredValidationError, "aggregate"):
            self.repository.confirm_schema(self.source_id, (invalid_aggregate,))

    def test_confirmation_rejects_overlong_display_name(self) -> None:
        self.repository.save_preview(self.preview)

        with self.assertRaisesRegex(StructuredValidationError, "display name"):
            self.repository.confirm_schema(
                self.source_id,
                self.confirmation(amount_display_name="x" * 241),
            )

    def test_confirmation_rejects_blank_alias_and_generated_header_display_name(self) -> None:
        blank_path = Path(self.temp_dir.name) / "blank-header.xlsx"
        blank_path.write_bytes(workbook_bytes([[None, "region"], ["12.5", "East"]]))
        blank_preview = infer_spreadsheet_schema(blank_path, self.source_id)
        self.repository.save_preview(blank_preview)
        dataset = blank_preview.datasets[0]
        submission = StructuredDatasetConfirmation(
            dataset_id=dataset.dataset_id,
            columns=(
                StructuredColumnConfirmation(
                    physical_name="column_1",
                    display_name="",
                    data_type=StructuredColumnType.DECIMAL,
                    aliases=("",),
                    allow_aggregate=True,
                    allow_filter=True,
                    null_policy="ignore",
                ),
                StructuredColumnConfirmation(
                    physical_name="region",
                    display_name="Region",
                    data_type=StructuredColumnType.STRING,
                    aliases=(),
                    allow_aggregate=False,
                    allow_filter=True,
                    null_policy="ignore",
                ),
            ),
        )

        with self.assertRaisesRegex(StructuredValidationError, "display name|alias"):
            self.repository.confirm_schema(self.source_id, (submission,))

    def test_confirmation_rejects_blocking_preview_diagnostics(self) -> None:
        blocked = SpreadsheetPreview(
            source_id=self.source_id,
            datasets=(),
            diagnostics=(
                StructuredDiagnostic(
                    code="empty_sheet",
                    message="Worksheet is empty",
                    worksheet_name="Sales",
                ),
            ),
        )
        self.repository.save_preview(blocked)

        with self.assertRaisesRegex(StructuredValidationError, "blocking diagnostic"):
            self.repository.confirm_schema(self.source_id, ())


if __name__ == "__main__":
    unittest.main()
