from __future__ import annotations

import unittest
from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

from app.database import (
    Database,
    KnowledgeSourceRecord,
    StructuredColumnRecord,
    StructuredDatasetRecord,
    StructuredPublicationRecord,
)
from app.infra.health import DependencyHealthRegistry
from app.main import create_default_repository, create_production_app
from app.models import ChatMessageModel, ChatState, ResponseParagraphModel
from app.repository import InMemoryChatRepository
from app.sql_repository import SqlChatRepository
from app.structured_answer import StructuredAnswerService
from app.structured_repository import StructuredRepository
from tests.support.structured_fakes import sample_catalog


class RecordingLLMProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate_reply(self, request: object) -> ChatMessageModel:
        self.calls += 1
        return ChatMessageModel(
            id=f"msg-legacy-{self.calls}",
            role="assistant",
            time="2026-07-23 10:00:00",
            paragraphs=[ResponseParagraphModel(text="legacy Physoc answer")],
        )


class RecordingClickHouseGateway:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = result or {
            "aggregate_value": Decimal("20.1250"),
            "total_count": 4,
            "valid_count": 3,
            "null_count": 1,
        }
        self.error = error
        self.calls: list[tuple[str, object]] = []

    def query(self, statement: str, parameters: object) -> object:
        self.calls.append((statement, parameters))
        if self.error is not None:
            raise self.error
        return self.result


class CountingStructuredService:
    def __init__(self, service: StructuredAnswerService) -> None:
        self.service = service
        self.calls = 0

    def try_answer(self, *args: object, **kwargs: object):
        self.calls += 1
        return self.service.try_answer(*args, **kwargs)


def empty_state() -> ChatState:
    return ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])


class StructuredAnswerServiceTest(unittest.TestCase):
    def test_default_repository_does_not_construct_clickhouse_when_disabled(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        client_calls: list[object] = []

        repository = create_default_repository(
            environ={"OFFLINE_MODE": "true", "STRUCTURED_QUERY_ENABLED": "false"},
            database_factory=lambda _url: database,
            clickhouse_client_factory=lambda **kwargs: client_calls.append(kwargs),
        )

        self.assertIsInstance(repository, SqlChatRepository)
        self.assertIsNone(repository._structured_service)
        self.assertEqual(client_calls, [])

    def test_default_repository_injects_structured_service_when_enabled(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        clients: list[object] = []

        class Client:
            def query(self, *_args: object, **_kwargs: object) -> object:
                return {"value": 1}

        def build_client(**kwargs: object) -> object:
            clients.append(kwargs)
            return Client()

        repository = create_default_repository(
            environ={"OFFLINE_MODE": "true", "STRUCTURED_QUERY_ENABLED": "true"},
            database_factory=lambda _url: database,
            clickhouse_client_factory=build_client,
        )

        self.assertIsNotNone(repository._structured_service)
        self.assertEqual(clients, [])
        repository._structured_service._clickhouse_gateway.query("SELECT 1", {})
        self.assertEqual(len(clients), 2)

    def test_production_default_does_not_construct_clickhouse_when_disabled(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repository = SqlChatRepository(database)
        client_calls: list[object] = []

        app = create_production_app(
            environ={"OFFLINE_MODE": "true", "STRUCTURED_QUERY_ENABLED": "false"},
            database_factory=lambda _url: database,
            repository_factory=lambda: repository,
            health_registry_factory=lambda: DependencyHealthRegistry([]),
            ingestion_queue_factory=lambda _repository: object(),
            storage_factory=lambda _root: object(),
            evaluation_import_service_factory=lambda: object(),
            clickhouse_client_factory=lambda **kwargs: client_calls.append(kwargs),
        )

        with patch("app.main.create_structured_repository") as structured_factory:
            from fastapi.testclient import TestClient

            with TestClient(app):
                pass

        self.assertEqual(client_calls, [])
        structured_factory.assert_called_once()

    def test_production_enabled_injects_structured_service_and_builds_two_clients(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        repository = SqlChatRepository(database)
        clients: list[object] = []

        class Client:
            def __init__(self) -> None:
                self.close_calls = 0

            def query(self, *_args: object, **_kwargs: object) -> object:
                return {"value": 1}

            def close(self) -> None:
                self.close_calls += 1

        def build_client(**kwargs: object) -> Client:
            clients.append(kwargs)
            client = Client()
            clients.append(client)
            return client

        app = create_production_app(
            environ={"OFFLINE_MODE": "true", "STRUCTURED_QUERY_ENABLED": "true"},
            database_factory=lambda _url: database,
            repository_factory=lambda: repository,
            health_registry_factory=lambda: DependencyHealthRegistry([]),
            ingestion_queue_factory=lambda **_kwargs: object(),
            storage_factory=lambda _root: object(),
            evaluation_import_service_factory=lambda: object(),
            clickhouse_client_factory=build_client,
        )
        from fastapi.testclient import TestClient

        with TestClient(app):
            self.assertIsNotNone(repository._structured_service)
            self.assertEqual(len([item for item in clients if isinstance(item, Client)]), 0)
            repository._structured_service._clickhouse_gateway.query("SELECT 1", {})
        self.assertEqual(len([item for item in clients if isinstance(item, Client)]), 2)
        self.assertTrue(all(item.close_calls == 1 for item in clients if isinstance(item, Client)))

    def test_structured_repository_builds_catalog_from_active_publication(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        with database.session() as session:
            session.add(
                KnowledgeSourceRecord(
                    id="kb-catalog",
                    name="sales.xlsx",
                    source_type="XLSX",
                    records=4,
                    status="已索引",
                    updated_at="2026-07-23 10:00:00",
                    classification="内部",
                    sort_order=0,
                )
            )
            dataset = StructuredDatasetRecord(
                dataset_id="ds-catalog",
                source_id="kb-catalog",
                worksheet_name="明细",
                schema_version=2,
                schema_hash="a" * 64,
                status="published",
            )
            dataset.columns = [
                StructuredColumnRecord(
                    id="column-catalog-amount",
                    dataset_id="ds-catalog",
                    schema_version=2,
                    physical_name="amount",
                    original_name="金额",
                    display_name="订单金额",
                    data_type="decimal",
                    aliases=["金额"],
                    allow_aggregate=True,
                    allow_filter=True,
                    null_policy="ignore",
                    sort_order=0,
                )
            ]
            session.add(dataset)
            session.add(
                StructuredPublicationRecord(
                    publication_id="pub-catalog-2",
                    dataset_id="ds-catalog",
                    schema_version=2,
                    physical_table_name="structured_ds_catalog_v2",
                    row_count=4,
                    content_hash="b" * 64,
                    status="published",
                )
            )

        catalog = StructuredRepository(database).get_catalog()

        self.assertEqual(len(catalog.datasets), 1)
        item = catalog.datasets[0]
        self.assertEqual(item.source_name, "sales.xlsx")
        self.assertEqual(item.schema.schema_version, 2)
        self.assertEqual(item.schema.columns[0].display_name, "订单金额")
        self.assertEqual(item.active_publication.publication_id, "pub-catalog-2")

    def test_average_query_routes_to_clickhouse_without_llm(self) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        structured = CountingStructuredService(
            StructuredAnswerService(lambda: sample_catalog(), gateway)
        )
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=structured,
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "订单金额平均值", "source")

        self.assertEqual(structured.calls, 1)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)
        answer = messages[-1].paragraphs[0].text
        for expected in (
            "sales.xlsx",
            "明细",
            "avg",
            "订单金额",
            "20.1250",
            "total=4",
            "valid=3",
            "null=1",
            "filters=none",
            "schema_version=1",
            "publication_version=pub-sales-1",
            "publication_id=pub-sales-1",
            "elapsed_ms=",
            "audit_id=",
        ):
            self.assertIn(expected, answer)
        runs = repository.list_agent_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual([step.tool_name for step in runs[0].steps], ["query_structured_data"])

    def test_non_structured_query_keeps_legacy_agent_path(self) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        structured = CountingStructuredService(
            StructuredAnswerService(lambda: sample_catalog(), gateway)
        )
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=structured,
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "请总结报销制度", "source")

        self.assertEqual(structured.calls, 1)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(gateway.calls, [])
        self.assertEqual(messages[-1].paragraphs[0].text, "legacy Physoc answer")

    def test_ambiguous_structured_query_clarifies_without_clickhouse_or_llm(self) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        structured = CountingStructuredService(
            StructuredAnswerService(lambda: sample_catalog(ambiguous=True), gateway)
        )
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=structured,
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "平均金额", "quick")

        self.assertEqual(structured.calls, 1)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(gateway.calls, [])
        self.assertIn("请", messages[-1].paragraphs[0].text)
        self.assertEqual(
            [step.tool_name for step in repository.list_agent_runs()[0].steps],
            ["query_structured_data"],
        )

    def test_structured_unavailable_does_not_fall_back_to_rag(self) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway(error=RuntimeError("clickhouse offline"))
        structured = CountingStructuredService(
            StructuredAnswerService(lambda: sample_catalog(), gateway)
        )
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=structured,
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "订单金额总和", "deep")

        self.assertEqual(structured.calls, 1)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)
        self.assertIn("暂时不可用", messages[-1].paragraphs[0].text)

    def test_unpublished_structured_dataset_does_not_fall_back_to_rag(self) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        catalog = sample_catalog()
        unpublished = replace(
            catalog,
            datasets=(replace(catalog.datasets[0], active_publication=None),),
        )
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: unpublished, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "sales订单金额平均值", "source")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(gateway.calls, [])
        self.assertIn("不可用", messages[-1].paragraphs[0].text)

    def test_in_memory_repository_persists_messages_and_audit_exactly_once(self) -> None:
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            structured_service=StructuredAnswerService(lambda: sample_catalog(), gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "订单金额平均值", "source")

        self.assertEqual(len(repository.get_messages(conversation_id)), 2)
        runs = repository.list_agent_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0].steps), 1)
        self.assertEqual(runs[0].steps[0].tool_name, "query_structured_data")

    def test_sql_repository_persists_messages_and_audit_exactly_once(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        gateway = RecordingClickHouseGateway()
        repository = SqlChatRepository(
            database,
            structured_service=StructuredAnswerService(lambda: sample_catalog(), gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "订单金额平均值", "source")

        reloaded = SqlChatRepository(database)
        self.assertEqual(len(reloaded.get_messages(conversation_id)), 2)
        runs = reloaded.list_agent_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0].steps), 1)
        self.assertEqual(runs[0].steps[0].tool_name, "query_structured_data")

    def test_repository_constructors_default_structured_service_to_none(self) -> None:
        memory = InMemoryChatRepository(empty_state())
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        sql = SqlChatRepository(database)

        self.assertIsNone(memory._structured_service)
        self.assertIsNone(sql._structured_service)


if __name__ == "__main__":
    unittest.main()
