from __future__ import annotations

import unittest
from dataclasses import replace
from decimal import Decimal
from threading import Event, Lock, Thread
from unittest.mock import patch

from app.clickhouse_gateway import StructuredStorageError
from app.database import (
    Database,
    KnowledgeSourceRecord,
    StructuredColumnRecord,
    StructuredDatasetRecord,
    StructuredPublicationRecord,
)
from app.infra.health import DependencyHealthRegistry
from app.main import (
    _LazyStructuredQueryGateway,
    create_app,
    create_default_repository,
    create_production_app,
)
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


class LifecycleClickHouseClient:
    def __init__(self, query_handler=None) -> None:
        self.close_calls = 0
        self.query_calls = 0
        self.query_handler = query_handler

    def query(self, *args: object, **kwargs: object) -> object:
        self.query_calls += 1
        if self.query_handler is not None:
            return self.query_handler(*args, **kwargs)
        return {"value": 1}

    def close(self) -> None:
        self.close_calls += 1


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
    def test_default_repository_close_disposes_only_its_owned_database_once(self) -> None:
        owned_database = Database("sqlite+pysqlite:///:memory:")
        with patch.object(
            owned_database.engine,
            "dispose",
            wraps=owned_database.engine.dispose,
        ) as owned_dispose:
            repository = create_default_repository(
                environ={"OFFLINE_MODE": "true", "STRUCTURED_QUERY_ENABLED": "false"},
                database_factory=lambda _url: owned_database,
            )
            repository.close()
            repository.close()

        owned_dispose.assert_called_once_with()

        external_database = Database("sqlite+pysqlite:///:memory:")
        external_database.create_schema()
        with patch.object(external_database.engine, "dispose") as external_dispose:
            SqlChatRepository(external_database).close()
        external_dispose.assert_not_called()

    def test_lazy_gateway_closes_same_client_when_gateway_rejects_identity_reuse(self) -> None:
        client = LifecycleClickHouseClient()
        gateway = _LazyStructuredQueryGateway(lambda **_kwargs: client, "http://clickhouse")

        with self.assertRaises(StructuredStorageError):
            gateway.query("SELECT 1", {})

        self.assertEqual(client.close_calls, 1)
        self.assertIsNone(gateway._clients)

    def test_lazy_gateway_constructor_failure_closes_clients_and_allows_clean_retry(self) -> None:
        clients: list[LifecycleClickHouseClient] = []

        def build_client(**_kwargs: object) -> LifecycleClickHouseClient:
            client = LifecycleClickHouseClient()
            clients.append(client)
            return client

        gateway = _LazyStructuredQueryGateway(build_client, "http://clickhouse")
        with patch("app.main.ClickHouseGateway", side_effect=RuntimeError("constructor failed")):
            with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                gateway.query("SELECT 1", {})

        self.assertEqual([client.close_calls for client in clients], [1, 1])
        self.assertIsNone(gateway._clients)
        self.assertEqual(gateway.query("SELECT 1", {}), {"value": 1})
        self.assertEqual(len(clients), 4)

    def test_lazy_gateway_close_is_permanent_and_idempotent(self) -> None:
        clients: list[LifecycleClickHouseClient] = []

        def build_client(**_kwargs: object) -> LifecycleClickHouseClient:
            client = LifecycleClickHouseClient()
            clients.append(client)
            return client

        gateway = _LazyStructuredQueryGateway(build_client, "http://clickhouse")
        gateway.query("SELECT 1", {})

        gateway.close()
        gateway.close()

        self.assertEqual([client.close_calls for client in clients], [1, 1])
        with self.assertRaisesRegex(StructuredStorageError, "closed"):
            gateway.query("SELECT 1", {})
        self.assertEqual(len(clients), 2)

    def test_lazy_gateway_close_waits_for_inflight_query_and_blocks_new_queries(self) -> None:
        query_started = Event()
        release_query = Event()
        close_returned = Event()
        clients: list[LifecycleClickHouseClient] = []

        def query_handler(*_args: object, **_kwargs: object) -> object:
            query_started.set()
            self.assertTrue(release_query.wait(2))
            return {"value": 1}

        def build_client(**_kwargs: object) -> LifecycleClickHouseClient:
            handler = query_handler if len(clients) == 1 else None
            client = LifecycleClickHouseClient(handler)
            clients.append(client)
            return client

        gateway = _LazyStructuredQueryGateway(build_client, "http://clickhouse")
        query_thread = Thread(target=lambda: gateway.query("SELECT 1", {}))
        query_thread.start()
        self.assertTrue(query_started.wait(2))
        close_thread = Thread(target=lambda: (gateway.close(), close_returned.set()))
        close_thread.start()

        self.assertFalse(close_returned.wait(0.1))
        with self.assertRaisesRegex(StructuredStorageError, "closed"):
            gateway.query("SELECT 2", {})
        self.assertEqual([client.close_calls for client in clients], [0, 0])

        release_query.set()
        query_thread.join(2)
        close_thread.join(2)
        self.assertFalse(query_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertEqual([client.close_calls for client in clients], [1, 1])

    def test_lazy_gateway_allows_fifteen_queries_to_run_concurrently(self) -> None:
        entered = 0
        entered_lock = Lock()
        all_entered = Event()
        release = Event()
        errors: list[Exception] = []
        clients: list[LifecycleClickHouseClient] = []

        def query_handler(*_args: object, **_kwargs: object) -> object:
            nonlocal entered
            with entered_lock:
                entered += 1
                if entered == 15:
                    all_entered.set()
            release.wait(2)
            return {"value": 1}

        def build_client(**_kwargs: object) -> LifecycleClickHouseClient:
            handler = query_handler if len(clients) == 1 else None
            client = LifecycleClickHouseClient(handler)
            clients.append(client)
            return client

        gateway = _LazyStructuredQueryGateway(build_client, "http://clickhouse")

        def run_query() -> None:
            try:
                gateway.query("SELECT 1", {})
            except Exception as error:
                errors.append(error)

        threads = [Thread(target=run_query) for _ in range(15)]
        for thread in threads:
            thread.start()
        self.assertTrue(all_entered.wait(2))
        release.set()
        for thread in threads:
            thread.join(2)
        self.assertEqual(errors, [])

    def test_default_repository_close_releases_lazy_clickhouse_clients_once(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        clients: list[LifecycleClickHouseClient] = []

        def build_client(**_kwargs: object) -> LifecycleClickHouseClient:
            client = LifecycleClickHouseClient()
            clients.append(client)
            return client

        repository = create_default_repository(
            environ={"OFFLINE_MODE": "true", "STRUCTURED_QUERY_ENABLED": "true"},
            database_factory=lambda _url: database,
            clickhouse_client_factory=build_client,
        )
        repository._structured_service._clickhouse_gateway.query("SELECT 1", {})

        repository.close()
        repository.close()

        self.assertEqual([client.close_calls for client in clients], [1, 1])

    def test_create_app_closes_owned_default_repository(self) -> None:
        from fastapi.testclient import TestClient

        class Repository:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        repository = Repository()
        with patch("app.main.create_default_repository", return_value=repository):
            application = create_app()
            with TestClient(application):
                pass

        self.assertEqual(repository.close_calls, 1)

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

    def test_structured_decimal_value_uses_deterministic_thousands_separator(self) -> None:
        gateway = RecordingClickHouseGateway(
            result={
                "aggregate_value": Decimal("12345.67"),
                "total_count": 1,
                "valid_count": 1,
                "null_count": 0,
            }
        )
        repository = InMemoryChatRepository(
            empty_state(),
            structured_service=StructuredAnswerService(lambda: sample_catalog(), gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "订单金额总和", "source")

        self.assertIn("value=12,345.67", messages[-1].paragraphs[0].text)

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

    def test_row_count_language_without_catalog_reference_keeps_legacy_agent_path(self) -> None:
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

        _, _, messages = repository.send_message(
            conversation_id,
            "合同有多少条付款条款",
            "source",
        )

        self.assertEqual(structured.calls, 1)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(gateway.calls, [])
        self.assertEqual(messages[-1].paragraphs[0].text, "legacy Physoc answer")

    def test_english_aggregate_substring_keeps_legacy_agent_path(self) -> None:
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

        _, _, messages = repository.send_message(
            conversation_id,
            "summarize sales policy",
            "source",
        )

        self.assertEqual(structured.calls, 1)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(gateway.calls, [])
        self.assertEqual(messages[-1].paragraphs[0].text, "legacy Physoc answer")

    def test_english_average_query_keeps_legacy_agent_path(self) -> None:
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

        repository.send_message(conversation_id, "average sales amount", "source")

        self.assertEqual(provider.calls, 1)
        self.assertEqual(gateway.calls, [])

    def test_anchored_implicit_row_count_uses_single_published_dataset(self) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: sample_catalog(), gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "总共有多少条", "source")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)
        self.assertIn("count() AS aggregate_value", gateway.calls[0][0])
        self.assertIn("aggregate=count", messages[-1].paragraphs[0].text)

    def test_catalog_failure_only_marks_strong_structured_shape_unavailable(self) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()

        def failing_catalog():
            raise RuntimeError("catalog down")

        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(failing_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "订单金额的平均值", "source")

        self.assertEqual(provider.calls, 0)
        self.assertIn("不可用", messages[-1].paragraphs[0].text)

    def test_catalog_failure_keeps_weak_document_count_question_on_legacy_path(self) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()

        def failing_catalog():
            raise RuntimeError("catalog down")

        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(failing_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "合同有多少条付款条款", "source")

        self.assertEqual(provider.calls, 1)

    def test_catalog_failure_keeps_aggregate_concept_questions_on_legacy_path(self) -> None:
        def failing_catalog():
            raise RuntimeError("catalog down")

        for question in ("什么是平均值", "平均值是什么"):
            with self.subTest(question=question):
                provider = RecordingLLMProvider()
                repository = InMemoryChatRepository(
                    empty_state(),
                    llm_provider=provider,
                    structured_service=StructuredAnswerService(
                        failing_catalog,
                        RecordingClickHouseGateway(),
                    ),
                )
                _, conversation_id, _ = repository.create_conversation()

                repository.send_message(conversation_id, question, "source")

                self.assertEqual(provider.calls, 1)

    def _assert_catalog_failure_uses_legacy_path(self, question: str) -> None:
        def failing_catalog():
            raise RuntimeError("catalog down")

        provider = RecordingLLMProvider()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(
                failing_catalog,
                RecordingClickHouseGateway(),
            ),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, question, "source")

        self.assertEqual(provider.calls, 1)

    def _assert_catalog_failure_is_strong_candidate(self, question: str) -> None:
        def failing_catalog():
            raise RuntimeError("catalog down")

        provider = RecordingLLMProvider()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(
                failing_catalog,
                RecordingClickHouseGateway(),
            ),
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, question, "source")

        self.assertEqual(provider.calls, 0)
        self.assertIn("不可用", messages[-1].paragraphs[0].text)

    def test_catalog_failure_concept_grammar_handles_full_explanatory_sentences(self) -> None:
        questions = (
            "请说明因为平均值过高会有什么影响",
            "请介绍被称为算术平均值的概念",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_uses_legacy_path(question)

    def test_catalog_failure_field_prefixes_can_contain_concept_words(self) -> None:
        questions = (
            "产品说明平均值",
            "商品说明订单金额平均值",
            "介绍费总和",
            "定义值总和",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_is_strong_candidate(question)

    def test_catalog_failure_copula_fragments_remain_on_legacy_path(self) -> None:
        questions = (
            "平均值为常用统计指标",
            "平均值作为统计指标",
            "因为平均值过高会有什么影响",
            "被称为算术平均值的概念",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_uses_legacy_path(question)

    def test_catalog_failure_explicit_filter_grammars_are_strong_candidates(self) -> None:
        questions = (
            "订单金额大于10的平均值",
            "订单日期2026-01-01至2026-01-31订单金额平均值",
            "地区=华东订单金额平均值",
            "地区为华东订单金额平均值",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_is_strong_candidate(question)

    def test_catalog_failure_concept_phrases_can_appear_anywhere(self) -> None:
        questions = (
            "什么是平均值",
            "我想了解什么是平均值",
            "能帮我讲讲什么是平均值",
            "通俗地解释一下平均值",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_uses_legacy_path(question)

    def test_catalog_failure_metric_qualified_concept_wording_is_strong_candidate(
        self,
    ) -> None:
        for question in (
            "什么是订单金额平均值",
            "订单金额平均值是什么",
            "什么是订单金额加权平均值",
            "订单金额加权平均值是什么",
            "什么是订单金额加权平均值呢",
            "我的订单金额平均值是什么",
        ):
            with self.subTest(question=question):
                self._assert_catalog_failure_is_strong_candidate(question)

    def test_catalog_failure_named_average_concepts_keep_legacy_agent_path(self) -> None:
        questions = (
            "什么是算术平均值",
            "算术平均值是什么",
            "什么是加权平均值",
            "什么是移动平均值",
            "什么是几何平均值",
            "什么是调和平均值",
            "我想了解什么是加权平均值",
            "能帮我讲讲什么是移动平均值",
            "请问调和平均值是什么",
            "什么是加权平均值呢",
            "我想了解什么是移动平均值吗",
            "能帮我说说加权平均值是什么",
            "麻烦告诉我调和平均值是什么",
            "请介绍加权平均值是什么",
            "我想请教一下加权平均值是什么",
            "可否聊聊移动平均值是什么",
            "请简单说一下调和平均值是什么",
            "请问平均值是什么",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_uses_legacy_path(question)

    def test_named_average_metric_phrases_route_to_structured_query(self) -> None:
        for question in (
            "什么是订单金额加权平均值",
            "订单金额加权平均值是什么",
            "什么是订单金额加权平均值呢",
        ):
            with self.subTest(question=question):
                provider = RecordingLLMProvider()
                gateway = RecordingClickHouseGateway()
                repository = InMemoryChatRepository(
                    empty_state(),
                    llm_provider=provider,
                    structured_service=StructuredAnswerService(lambda: sample_catalog(), gateway),
                )
                _, conversation_id, _ = repository.create_conversation()

                repository.send_message(conversation_id, question, "source")

                self.assertEqual(provider.calls, 0)
                self.assertEqual(len(gateway.calls), 1)

    def test_aggregate_looking_equality_values_keep_legacy_agent_path(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        level = replace(
            dataset.schema.columns[1],
            physical_name="level",
            original_name="等级",
            display_name="等级",
            aliases=(),
        )
        level_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(
                            dataset.schema.columns[0],
                            level,
                            dataset.schema.columns[2],
                        ),
                    ),
                ),
            ),
        )

        questions = (
            "等级为最高",
            "等级为最高的记录",
            "等级为最高.",
            "等级为最高！",
            "等级为最高!",
            "等级为最高吧",
            "等级为最高呀",
            "等级为最高吧！",
            "等级为最高标准",
            *(
                f"等级为{value}{tail}"
                for value in ("最高", "最低", "平均值", "总和")
                for tail in ("？", "?", "吗", "呢")
            ),
        )
        for question in questions:
            with self.subTest(question=question):
                provider = RecordingLLMProvider()
                gateway = RecordingClickHouseGateway()
                repository = InMemoryChatRepository(
                    empty_state(),
                    llm_provider=provider,
                    structured_service=StructuredAnswerService(lambda: level_catalog, gateway),
                )
                _, conversation_id, _ = repository.create_conversation()

                repository.send_message(conversation_id, question, "source")

                self.assertEqual(provider.calls, 1)
                self.assertEqual(gateway.calls, [])

    def test_real_aggregate_natural_tails_remain_structured_candidates(self) -> None:
        for question in ("订单金额最高吗", "订单金额平均值呢", "订单金额最高吧！"):
            with self.subTest(question=question):
                provider = RecordingLLMProvider()
                gateway = RecordingClickHouseGateway()
                repository = InMemoryChatRepository(
                    empty_state(),
                    llm_provider=provider,
                    structured_service=StructuredAnswerService(lambda: sample_catalog(), gateway),
                )
                _, conversation_id, _ = repository.create_conversation()

                repository.send_message(conversation_id, question, "source")

                self.assertEqual(provider.calls, 0)
                self.assertEqual(len(gateway.calls), 1)

    def test_aggregate_looking_equality_value_allows_separate_aggregate_clause(
        self,
    ) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        level = replace(
            dataset.schema.columns[1],
            physical_name="level",
            original_name="等级",
            display_name="等级",
            aliases=(),
        )
        level_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(
                            dataset.schema.columns[0],
                            level,
                            dataset.schema.columns[2],
                        ),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: level_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        for question in ("等级为最高的订单金额平均值", "等级为最高标准的订单金额平均值"):
            repository.send_message(conversation_id, question, "source")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(
            [parameters["filter_0"] for _, parameters in gateway.calls],
            ["最高", "最高标准"],
        )

    def test_ambiguous_cross_dataset_equality_values_keep_legacy_agent_path(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        level = replace(
            dataset.schema.columns[1],
            physical_name="level",
            original_name="等级",
            display_name="等级",
            aliases=(),
        )
        first = replace(
            dataset,
            schema=replace(
                dataset.schema,
                columns=(
                    dataset.schema.columns[0],
                    level,
                    dataset.schema.columns[2],
                ),
            ),
        )
        second_level = replace(level, physical_name="grade_level")
        second = replace(
            first,
            schema=replace(
                first.schema,
                dataset_id="ds-sales-2",
                source_id="kb-sales-2",
                columns=(
                    first.schema.columns[0],
                    second_level,
                    first.schema.columns[2],
                ),
            ),
            source_name="sales-2.xlsx",
            active_publication=replace(
                first.active_publication,
                publication_id="pub-sales-2",
                dataset_id="ds-sales-2",
                physical_table_name="structured_ds_sales_2_v1",
            ),
        )
        ambiguous_catalog = replace(catalog, datasets=(first, second))

        for question in ("等级为最高", "等级为最高的记录", "sales.xlsx的等级为最高"):
            with self.subTest(question=question):
                provider = RecordingLLMProvider()
                gateway = RecordingClickHouseGateway()
                repository = InMemoryChatRepository(
                    empty_state(),
                    llm_provider=provider,
                    structured_service=StructuredAnswerService(
                        lambda: ambiguous_catalog,
                        gateway,
                    ),
                )
                _, conversation_id, _ = repository.create_conversation()

                repository.send_message(conversation_id, question, "source")

                self.assertEqual(provider.calls, 1)
                self.assertEqual(gateway.calls, [])

    def test_ambiguous_same_dataset_alias_equality_value_keeps_legacy_agent_path(
        self,
    ) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        region = replace(dataset.schema.columns[1], aliases=("等级",))
        order_date = replace(dataset.schema.columns[2], aliases=("等级",))
        ambiguous_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(dataset.schema.columns[0], region, order_date),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: ambiguous_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "等级为最高", "source")

        self.assertEqual(provider.calls, 1)
        self.assertEqual(gateway.calls, [])

    def test_filter_field_name_inside_equality_value_does_not_fake_an_aggregate(
        self,
    ) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        level = replace(
            dataset.schema.columns[1],
            physical_name="level",
            original_name="等级",
            display_name="等级",
            aliases=(),
        )
        standard = replace(
            dataset.schema.columns[2],
            physical_name="standard",
            original_name="标准",
            display_name="标准",
            aliases=(),
        )
        filter_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(dataset.schema.columns[0], level, standard),
                    ),
                ),
            ),
        )

        for question in ("等级为最高标准", "等级为最高标准的记录", "等级为标准最高"):
            with self.subTest(question=question):
                provider = RecordingLLMProvider()
                gateway = RecordingClickHouseGateway()
                repository = InMemoryChatRepository(
                    empty_state(),
                    llm_provider=provider,
                    structured_service=StructuredAnswerService(lambda: filter_catalog, gateway),
                )
                _, conversation_id, _ = repository.create_conversation()

                repository.send_message(conversation_id, question, "source")

                self.assertEqual(provider.calls, 1)
                self.assertEqual(gateway.calls, [])

        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: filter_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(
            conversation_id,
            "等级为最高标准的订单金额平均值",
            "source",
        )

        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0][1]["filter_0"], "最高标准")

    def test_multiple_equality_only_filters_keep_legacy_agent_path(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        level = replace(
            dataset.schema.columns[1],
            physical_name="level",
            original_name="等级",
            display_name="等级",
            aliases=(),
        )
        standard = replace(
            dataset.schema.columns[1],
            physical_name="standard",
            original_name="标准",
            display_name="标准",
            aliases=(),
        )
        filter_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(
                            dataset.schema.columns[0],
                            level,
                            dataset.schema.columns[2],
                            standard,
                        ),
                    ),
                ),
            ),
        )
        equality_only_questions = (
            "等级为最高且订单日期为最低",
            "等级为最高，订单日期为2026-01-01",
            "等级为最高且订单日期为最低且标准为平均值",
        )
        for question in equality_only_questions:
            with self.subTest(question=question):
                provider = RecordingLLMProvider()
                gateway = RecordingClickHouseGateway()
                repository = InMemoryChatRepository(
                    empty_state(),
                    llm_provider=provider,
                    structured_service=StructuredAnswerService(lambda: filter_catalog, gateway),
                )
                _, conversation_id, _ = repository.create_conversation()

                repository.send_message(conversation_id, question, "source")

                self.assertEqual(provider.calls, 1)
                self.assertEqual(gateway.calls, [])

        aggregate_questions = (
            "等级为最高且订单日期为最低的订单金额平均值",
            "等级为最高，订单日期为2026-01-01，订单金额平均值",
            "等级为最高且订单日期为最低且标准为平均值的订单金额平均值",
        )
        for question in aggregate_questions:
            with self.subTest(question=question):
                provider = RecordingLLMProvider()
                gateway = RecordingClickHouseGateway()
                repository = InMemoryChatRepository(
                    empty_state(),
                    llm_provider=provider,
                    structured_service=StructuredAnswerService(lambda: filter_catalog, gateway),
                )
                _, conversation_id, _ = repository.create_conversation()

                repository.send_message(conversation_id, question, "source")

                self.assertEqual(provider.calls, 0)

    def test_ambiguous_cross_dataset_equality_with_real_aggregate_stays_structured(
        self,
    ) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        level = replace(
            dataset.schema.columns[1],
            physical_name="level",
            original_name="等级",
            display_name="等级",
            aliases=(),
        )
        first = replace(
            dataset,
            schema=replace(
                dataset.schema,
                columns=(
                    dataset.schema.columns[0],
                    level,
                    dataset.schema.columns[2],
                ),
            ),
        )
        second_level = replace(level, physical_name="grade_level")
        second = replace(
            first,
            schema=replace(
                first.schema,
                dataset_id="ds-sales-2",
                source_id="kb-sales-2",
                columns=(
                    first.schema.columns[0],
                    second_level,
                    first.schema.columns[2],
                ),
            ),
            source_name="sales-2.xlsx",
            active_publication=replace(
                first.active_publication,
                publication_id="pub-sales-2",
                dataset_id="ds-sales-2",
                physical_table_name="structured_ds_sales_2_v1",
            ),
        )
        ambiguous_catalog = replace(catalog, datasets=(first, second))
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: ambiguous_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(
            conversation_id,
            "等级为最高的订单金额平均值",
            "source",
        )
        repository.send_message(
            conversation_id,
            "sales.xlsx的等级为最高的订单金额平均值",
            "source",
        )

        self.assertEqual(provider.calls, 0)
        self.assertIn("澄清", messages[-1].paragraphs[0].text)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0][1]["filter_0"], "最高")

    def test_equality_filter_and_aggregate_orderings_remain_structured_candidates(
        self,
    ) -> None:
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: sample_catalog(), gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        for question in ("地区为华东订单金额平均值", "订单金额平均值，地区为华东"):
            repository.send_message(conversation_id, question, "source")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)

    def test_catalog_failure_natural_aggregate_tails_are_strong_candidates(self) -> None:
        questions = (
            "订单金额平均值是多少",
            "请问订单金额平均值是多少",
            "订单金额总和有多少",
            "订单金额的最大值呢",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_is_strong_candidate(question)

    def test_catalog_failure_combined_natural_tails_are_strong_candidates(self) -> None:
        questions = (
            "订单金额平均值是多少呢",
            "请问订单金额的最大值是多少呢",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_is_strong_candidate(question)

    def test_catalog_failure_chinese_equality_preserves_punctuation_delimiters(self) -> None:
        questions = (
            "订单金额平均值，地区为华东",
            "订单金额平均值,地区为华东",
            "订单金额平均值。地区为华东",
            "订单金额平均值；地区为华东",
            "订单金额平均值;地区为华东",
        )

        for question in questions:
            with self.subTest(question=question):
                self._assert_catalog_failure_is_strong_candidate(question)

    def test_catalog_failure_keeps_arbitrary_prefix_what_is_question_on_legacy_path(
        self,
    ) -> None:
        self._assert_catalog_failure_uses_legacy_path("能否说明什么是平均值")

    def test_catalog_failure_keeps_told_what_is_question_on_legacy_path(self) -> None:
        self._assert_catalog_failure_uses_legacy_path("请告诉我什么是平均值")

    def test_catalog_failure_keeps_what_called_question_on_legacy_path(self) -> None:
        self._assert_catalog_failure_uses_legacy_path("请问什么叫平均值")

    def test_catalog_failure_keeps_explanation_prefix_question_on_legacy_path(self) -> None:
        self._assert_catalog_failure_uses_legacy_path("麻烦说明一下平均值")

    def test_catalog_failure_keeps_explicit_equality_query_as_strong_candidate(self) -> None:
        def failing_catalog():
            raise RuntimeError("catalog down")

        provider = RecordingLLMProvider()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(
                failing_catalog,
                RecordingClickHouseGateway(),
            ),
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(
            conversation_id,
            "请说明订单金额平均值且地区为华东",
            "source",
        )

        self.assertEqual(provider.calls, 0)
        self.assertIn("不可用", messages[-1].paragraphs[0].text)

    def test_short_alias_contained_by_long_field_does_not_anchor_candidate(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        quantity = replace(
            dataset.schema.columns[0],
            original_name="销售数量",
            display_name="销售数量",
            aliases=("销售",),
        )
        quantity_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(quantity, *dataset.schema.columns[1:]),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: quantity_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "销售数量", "source")

        self.assertEqual(provider.calls, 1)
        self.assertEqual(gateway.calls, [])

    def test_long_field_span_still_allows_following_aggregate(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        quantity = replace(
            dataset.schema.columns[0],
            original_name="销售数量",
            display_name="销售数量",
            aliases=("销售",),
        )
        quantity_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(quantity, *dataset.schema.columns[1:]),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: quantity_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "销售数量平均值", "source")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)

    def test_catalog_failure_keeps_polite_what_is_question_on_legacy_path(self) -> None:
        provider = RecordingLLMProvider()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(
                lambda: (_ for _ in ()).throw(RuntimeError("catalog down")),
                RecordingClickHouseGateway(),
            ),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "请问什么是平均值", "source")

        self.assertEqual(provider.calls, 1)

    def test_catalog_failure_keeps_polite_explanation_question_on_legacy_path(self) -> None:
        provider = RecordingLLMProvider()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(
                lambda: (_ for _ in ()).throw(RuntimeError("catalog down")),
                RecordingClickHouseGateway(),
            ),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "请解释何为总和", "source")

        self.assertEqual(provider.calls, 1)

    def test_catalog_failure_keeps_polite_meaning_question_on_legacy_path(self) -> None:
        provider = RecordingLLMProvider()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(
                lambda: (_ for _ in ()).throw(RuntimeError("catalog down")),
                RecordingClickHouseGateway(),
            ),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "请说明平均值是什么意思", "source")

        self.assertEqual(provider.calls, 1)

    def test_aggregate_named_field_keeps_independent_aggregate_span(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        average = replace(
            dataset.schema.columns[0],
            original_name="平均",
            display_name="平均",
            aliases=(),
        )
        average_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(average, *dataset.schema.columns[1:]),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: average_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "平均的平均值", "source")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)

    def test_masked_field_span_cannot_create_an_aggregate_across_its_boundary(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        average = replace(
            dataset.schema.columns[0],
            original_name="平均",
            display_name="平均",
            aliases=(),
        )
        average_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(average, *dataset.schema.columns[1:]),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: average_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "平平均均", "source")

        self.assertEqual(provider.calls, 1)
        self.assertEqual(gateway.calls, [])

    def test_single_character_alias_keeps_independent_aggregate_span(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        amount = replace(dataset.schema.columns[0], aliases=("均",))
        alias_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(amount, *dataset.schema.columns[1:]),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: alias_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "均平均值", "source")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)

    def test_single_character_alias_can_anchor_structured_query(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        amount = replace(dataset.schema.columns[0], aliases=("量",))
        single_alias_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(amount, *dataset.schema.columns[1:]),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: single_alias_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "量平均值", "source")

        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(gateway.calls), 1)

    def test_aggregate_word_inside_field_name_requires_separate_aggregate_intent(self) -> None:
        catalog = sample_catalog()
        dataset = catalog.datasets[0]
        quantity = replace(
            dataset.schema.columns[0],
            original_name="销售数量",
            display_name="销售数量",
            aliases=(),
        )
        quantity_catalog = replace(
            catalog,
            datasets=(
                replace(
                    dataset,
                    schema=replace(
                        dataset.schema,
                        columns=(quantity, *dataset.schema.columns[1:]),
                    ),
                ),
            ),
        )
        provider = RecordingLLMProvider()
        gateway = RecordingClickHouseGateway()
        repository = InMemoryChatRepository(
            empty_state(),
            llm_provider=provider,
            structured_service=StructuredAnswerService(lambda: quantity_catalog, gateway),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "销售数量", "source")
        repository.send_message(conversation_id, "销售数量平均值", "source")

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(gateway.calls), 1)

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
