from __future__ import annotations

import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.database import Database
from app.embedding_contracts import EmbeddingModelMetadata
from app.hybrid_retriever import HybridRetriever
from app.infra.health import DependencyHealthRegistry
from app.knowledge_route_models import KnowledgeRouteType
from app.knowledge_router import KnowledgeAnswerRouter, LegacyKnowledgeAnswerRouter
from app.llm import LLMProviderError
from app.main import (
    _DefaultRetrievalResourceFactory,
    create_app,
    create_default_repository,
    create_production_app,
)
from app.models import (
    ChatMessageModel,
    ChatState,
    KnowledgeChunkModel,
    KnowledgeSourceModel,
    ResponseParagraphModel,
)
from app.repository import STATUS_INDEXED, InMemoryChatRepository
from app.retrieval_models import RetrievalCandidate, RetrievalMode, RetrievalScope
from app.retrieval_router import RetrievalRouter
from app.retrieval_settings import RerankerModelSettings, RetrievalSettings
from app.sparse_embedding import SparseVector
from app.structured_answer import StructuredAnswerService
from app.word_fact_answer import WordFactAnswerService
from app.word_facts import KnowledgeFactModel
from tests.support.structured_fakes import sample_multi_metric_catalog


EMBEDDING_METADATA = EmbeddingModelMetadata(
    name="Qwen/Qwen3-Embedding-0.6B",
    version="acceptance-v1",
    sha256="a" * 64,
    dimensions=3,
    normalized=True,
    encoding_profile_sha256="b" * 64,
    protocol_version="v1",
)
RERANKER_METADATA = RerankerModelSettings(
    name="BAAI/bge-reranker-v2-m3",
    version="acceptance-v1",
    sha256="c" * 64,
    prompt_profile_sha256="d" * 64,
    protocol_version="v1",
)


class AcceptanceEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts, *, purpose, expected, timeout_seconds=None):
        del purpose, expected, timeout_seconds
        self.calls += len(texts)
        return [[0.1, 0.2, 0.3] for _text in texts]


class AcceptanceSparse:
    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, query: str) -> SparseVector:
        del query
        self.calls += 1
        return SparseVector(indices=(1,), values=(1.0,))


class AcceptanceRetrievalGateway:
    def __init__(self, candidate: RetrievalCandidate) -> None:
        self.candidate = candidate
        self.search_calls = 0

    def search_dense(self, _vector, *, scope, limit, timeout_seconds=None):
        del scope, timeout_seconds
        self.search_calls += 1
        return (self.candidate,)[:limit]

    def search_sparse(self, _vector, *, scope, limit, timeout_seconds=None):
        del scope, timeout_seconds
        self.search_calls += 1
        return (self.candidate,)[:limit]

    def retrieve_points(self, _point_ids, *, scope, timeout_seconds=None):
        del scope, timeout_seconds
        return ()

    def resolve_alias(self) -> str:
        return "knowledge_chunks_qwen3_v17"


class AcceptanceReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, _query, passages, *, expected, timeout_seconds=None):
        del expected, timeout_seconds
        self.calls += 1
        return [1.0 - index / 100 for index, _passage in enumerate(passages)]


class AcceptanceLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.fail_with: Exception | None = None

    def generate_reply(self, _request) -> ChatMessageModel:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return ChatMessageModel(
            id=f"msg-acceptance-{self.calls}",
            role="assistant",
            time="2026-08-12 09:00:00",
            paragraphs=[ResponseParagraphModel(text="张三是工程师，相关资料已经过重排序。")],
        )


class AcceptanceClickHouse:
    def __init__(self) -> None:
        self.calls = 0

    def query(self, statement: str, _parameters: object) -> object:
        self.calls += 1
        if "metric_1_value" in statement:
            return {
                "total_count": 4,
                "metric_0_value": Decimal("350.50"),
                "metric_0_valid_count": 4,
                "metric_0_null_count": 0,
                "metric_1_value": Decimal("200"),
                "metric_1_valid_count": 4,
                "metric_1_null_count": 0,
            }
        return {
            "aggregate_value": Decimal("350.50"),
            "total_count": 4,
            "valid_count": 4,
            "null_count": 0,
        }


class UnifiedRoutingAcceptanceHarness:
    def __init__(self) -> None:
        people_text = "姓名：张三，年龄：28岁，性别：女，职务：工程师"
        people_source = KnowledgeSourceModel(
            id="kb-people",
            name="people.docx",
            source_type="DOCX",
            records=1,
            status=STATUS_INDEXED,
            updated_at="2026-08-12 09:00:00",
            classification="internal",
        )
        irrelevant_source = KnowledgeSourceModel(
            id="kb-irrelevant",
            name="irrelevant.docx",
            source_type="DOCX",
            records=1,
            status=STATUS_INDEXED,
            updated_at="2026-08-12 09:00:00",
            classification="internal",
        )
        people_chunk = KnowledgeChunkModel(
            id="chunk-people",
            source_id=people_source.id,
            chunk_index=0,
            text=people_text,
            token_count=len(people_text),
        )
        irrelevant_text = "报销制度与本问题无关。"
        irrelevant_chunk = KnowledgeChunkModel(
            id="chunk-irrelevant",
            source_id=irrelevant_source.id,
            chunk_index=0,
            text=irrelevant_text,
            token_count=len(irrelevant_text),
        )
        facts = [
            KnowledgeFactModel.create(
                id=f"fact-{field}",
                source_id=people_source.id,
                chunk_id=people_chunk.id,
                entity="张三",
                field=field,
                value=value,
                confidence=0.99,
                locator={"paragraph": 1},
            )
            for field, value in (("年龄", "28岁"), ("性别", "女"), ("职务", "工程师"))
        ]
        state = ChatState(
            conversations=[],
            messages_by_conversation={},
            knowledge_sources=[people_source, irrelevant_source],
            knowledge_chunks_by_source={
                people_source.id: [people_chunk],
                irrelevant_source.id: [irrelevant_chunk],
            },
            knowledge_facts_by_source={people_source.id: facts},
        )

        candidate = RetrievalCandidate(
            source_id=people_source.id,
            source_name=people_source.name,
            source_type=people_source.source_type,
            classification=people_source.classification,
            chunk_id=people_chunk.id,
            chunk_index=people_chunk.chunk_index,
            text=people_chunk.text,
            point_id="point-people",
        )
        self.embedding = AcceptanceEmbedding()
        self.sparse = AcceptanceSparse()
        self.retrieval_gateway = AcceptanceRetrievalGateway(candidate)
        self.reranker = AcceptanceReranker()
        self.hybrid = HybridRetriever(
            embedding=self.embedding,
            sparse=self.sparse,
            gateway=self.retrieval_gateway,
            reranker=self.reranker,
            embedding_metadata=EMBEDDING_METADATA,
            reranker_metadata=RERANKER_METADATA,
            dense_top_k=8,
            sparse_top_k=8,
            rerank_top_k=8,
            degraded_rerank_top_k=5,
            final_top_k=5,
            total_timeout_seconds=2.0,
        )
        self.retrieval_router = RetrievalRouter(
            mode=RetrievalMode.QWEN3,
            legacy_search=lambda _query, _limit: (),
            hybrid=self.hybrid,
            canary_percent=100.0,
            embedding_model_version=EMBEDDING_METADATA.version,
            reranker_model_version=RERANKER_METADATA.version,
            qdrant_alias="knowledge_chunks_qwen3",
        )
        self.llm = AcceptanceLLM()
        self.clickhouse = AcceptanceClickHouse()
        self.repository = InMemoryChatRepository(
            state,
            llm_provider=self.llm,
            structured_service=StructuredAnswerService(
                lambda: sample_multi_metric_catalog(),
                self.clickhouse,
            ),
            retrieval_router=self.retrieval_router,
            retrieval_scope=RetrievalScope("default", ("internal",), "v1"),
        )
        self.repository.configure_answer_services(
            word_fact_service=WordFactAnswerService(
                self.repository,
                permission_tags=("internal",),
            ),
        )
        self.repository.configure_knowledge_routing(
            unified_enabled=True,
            word_factual_enabled=True,
        )
        _, self.conversation_id, _ = self.repository.create_conversation()

    def close(self) -> None:
        self.retrieval_router.close()
        self.hybrid.close()

    def ask(self, question: str):
        _, _, messages = self.repository.send_message(self.conversation_id, question, "quick")
        return self.repository.list_agent_runs(1)[0], messages[-1]


class RagAcceptanceTest(unittest.TestCase):
    def test_clickhouse_timeout_has_no_word_citations_rag_or_llm_calls(self) -> None:
        class FailingClickHouseGateway:
            def __init__(self) -> None:
                self.query_calls = 0
                self.returned_row_count = 0
                self.fail_with = TimeoutError("timed out")

            def query(self, _statement: str, _parameters: object) -> object:
                self.query_calls += 1
                raise self.fail_with

        class RecordingRagSearch:
            def __init__(self) -> None:
                self.calls = 0

            def search(self, request: object) -> object:
                self.calls += 1
                raise AssertionError(f"structured query reached Word/PDF RAG: {request}")

        class WordAnswerLLM:
            def __init__(self) -> None:
                self.generation_calls = 0

            def generate_reply(self, _request: object) -> ChatMessageModel:
                self.generation_calls += 1
                return ChatMessageModel(
                    id="msg-word-fallback",
                    role="assistant",
                    time="2026-08-11 12:00:00",
                    paragraphs=[ResponseParagraphModel(text="word-policy.docx fallback")],
                )

        gateway = FailingClickHouseGateway()
        rag_search = RecordingRagSearch()
        llm = WordAnswerLLM()
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[]),
            llm_provider=llm,
            structured_service=StructuredAnswerService(
                lambda: sample_multi_metric_catalog(),
                gateway,
            ),
            retrieval_router=rag_search,
            retrieval_scope=RetrievalScope("default", ("internal",), "v1"),
        )
        client = TestClient(
            create_app(repository=repository, upload_dir=Path(self.temp_dir.name))
        )
        conversation_id = client.post("/api/conversations").json()["activeConversationId"]

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "地区为华东的销售额、成本、利润汇总", "mode": "source"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        assistant = response.json()["messages"][-1]
        paragraph = assistant["paragraphs"][0]
        self.assertIn("结构化查询超时", paragraph["text"])
        self.assertEqual(paragraph["citations"], [])
        self.assertEqual(assistant["artifacts"], [])
        self.assertNotIn("word-policy.docx", paragraph["text"])
        self.assertEqual(gateway.query_calls, 1)
        self.assertEqual(gateway.returned_row_count, 0)
        self.assertEqual(rag_search.calls, 0)
        self.assertEqual(llm.generation_calls, 0)

    def test_model_failure_returns_explicit_unavailable_without_raw_retrieval_chunks(self) -> None:
        raw_chunk = "CONFIDENTIAL raw retrieved payroll passage"
        internal_url = "http://physoc.internal.example/private"

        class FailingPhysocProvider:
            def generate_reply(self, request):
                raise LLMProviderError("大模型服务暂时不可用，请稍后重试。")

        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[]),
            llm_provider=FailingPhysocProvider(),
        )
        client = TestClient(create_app(repository=repository, upload_dir=Path(self.temp_dir.name)))
        upload = client.post(
            "/api/knowledge/uploads",
            data={"classification": "internal"},
            files={"file": ("policy.txt", raw_chunk.encode(), "text/plain")},
        )
        self.assertEqual(upload.status_code, 200)
        conversation_id = client.post("/api/conversations").json()["activeConversationId"]

        with patch("app.routes.logger"):
            response = client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"content": "payroll policy", "mode": "source"},
            )

        self.assertEqual(response.status_code, 502)
        serialized = response.text
        self.assertIn("大模型服务暂时不可用", serialized)
        self.assertNotIn(raw_chunk, serialized)
        self.assertNotIn(internal_url, serialized)

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        self.client = TestClient(
            create_app(repository=repository, upload_dir=Path(self.temp_dir.name))
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_upload_index_and_user_question_returns_grounded_answer_without_source_internals(
        self,
    ) -> None:
        policy_text = (
            "差旅报销制度规定：员工出差前必须先提交审批单。"
            "返程后需要在五个工作日内上传发票、行程单和审批记录。"
        )
        upload_response = self.client.post(
            "/api/knowledge/uploads",
            data={"classification": "内部·机密"},
            files={"file": ("travel-policy.txt", policy_text.encode("utf-8"), "text/plain")},
        )
        self.assertEqual(upload_response.status_code, 200)
        source = upload_response.json()[0]
        self.assertEqual(source["status"], "已索引")
        self.assertGreater(source["records"], 0)

        indexed_sources = self.client.get("/api/knowledge/sources").json()
        indexed = next(item for item in indexed_sources if item["id"] == source["id"])
        self.assertEqual(indexed["status"], "已索引")
        self.assertGreater(indexed["records"], 0)

        conversation_id = self.client.post("/api/conversations").json()["activeConversationId"]
        answer_response = self.client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "差旅票据材料", "mode": "source"},
        )

        self.assertEqual(answer_response.status_code, 200)
        assistant = answer_response.json()["messages"][-1]
        answer_payload = str(assistant)
        self.assertIn("差旅报销", assistant["paragraphs"][0]["text"])
        self.assertIn("发票", assistant["paragraphs"][0]["text"])
        self.assertEqual(assistant["paragraphs"][0]["citations"], [])
        self.assertNotIn("sourceId", answer_payload)
        self.assertNotIn("chunkId", answer_payload)
        self.assertNotIn("travel-policy.txt", answer_payload)

        audit_response = self.client.get("/api/admin/agent/runs")
        self.assertEqual(audit_response.status_code, 200)
        audit_run = audit_response.json()[0]
        self.assertEqual(audit_run["conversationId"], conversation_id)
        self.assertEqual(audit_run["query"], "差旅票据材料")
        self.assertEqual(audit_run["status"], "completed")
        self.assertGreater(audit_run["evidenceCount"], 0)
        tool_names = [step["toolName"] for step in audit_run["steps"]]
        self.assertIn("search_knowledge", tool_names)
        self.assertNotIn("inspect_document", tool_names)
        self.assertIn("compare_evidence", tool_names)
        self.assertIn("compose_answer", tool_names)
        self.assertTrue(all(step["readOnly"] for step in audit_run["steps"]))


class UnifiedKnowledgeRoutingAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = UnifiedRoutingAcceptanceHarness()

    def tearDown(self) -> None:
        self.harness.close()

    def test_startup_selects_legacy_unified_and_word_enabled_routers(self) -> None:
        cases = (
            (False, False, LegacyKnowledgeAnswerRouter, False),
            (True, False, KnowledgeAnswerRouter, False),
            (True, True, KnowledgeAnswerRouter, True),
        )

        for unified_enabled, word_enabled, router_type, expects_word in cases:
            with self.subTest(unified=unified_enabled, word=word_enabled):
                repository = create_default_repository(
                    environ={
                        "OFFLINE_MODE": "false",
                        "DATABASE_URL": "sqlite+pysqlite:///:memory:",
                        "UNIFIED_KNOWLEDGE_ROUTING_ENABLED": str(unified_enabled).lower(),
                        "WORD_FACTUAL_QA_ENABLED": str(word_enabled).lower(),
                    },
                    database_factory=Database,
                )
                try:
                    self.assertIsInstance(repository._answer_router, router_type)
                    if isinstance(repository._answer_router, KnowledgeAnswerRouter):
                        self.assertEqual(
                            repository._answer_router._word_fact_service is not None,
                            expects_word,
                        )
                finally:
                    repository.close()

    def test_production_factory_preserves_answer_service_signature_for_all_routes(self) -> None:
        class LegacyAnswerServiceSignatureRepository(InMemoryChatRepository):
            def configure_answer_services(
                self,
                *,
                structured_service=None,
                word_fact_service=None,
            ) -> None:
                super().configure_answer_services(
                    structured_service=structured_service,
                    word_fact_service=word_fact_service,
                )

        cases = (
            (False, False, LegacyKnowledgeAnswerRouter, False),
            (True, False, KnowledgeAnswerRouter, False),
            (True, True, KnowledgeAnswerRouter, True),
        )

        for unified_enabled, word_enabled, router_type, expects_word in cases:
            with self.subTest(unified=unified_enabled, word=word_enabled):
                repository = LegacyAnswerServiceSignatureRepository(
                    ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
                )
                application = self._create_production_app(
                    repository,
                    unified_enabled=unified_enabled,
                    word_enabled=word_enabled,
                )

                with TestClient(application):
                    self.assertIsInstance(repository._answer_router, router_type)
                    if isinstance(repository._answer_router, KnowledgeAnswerRouter):
                        self.assertEqual(
                            repository._answer_router._word_fact_service is not None,
                            expects_word,
                        )

    def test_production_factory_allows_legacy_chat_repository_when_flags_are_off(
        self,
    ) -> None:
        class MissingRoutingConfigurationRepository(InMemoryChatRepository):
            configure_knowledge_routing = None

        repository = MissingRoutingConfigurationRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[]),
            unified_knowledge_routing_enabled=False,
            word_factual_qa_enabled=False,
        )
        application = self._create_production_app(
            repository,
            unified_enabled=False,
            word_enabled=False,
        )

        with TestClient(application):
            self.assertIsInstance(repository._answer_router, LegacyKnowledgeAnswerRouter)

    def test_production_factory_allows_routing_only_repository_without_answer_services(
        self,
    ) -> None:
        class RoutingOnlyRepository(InMemoryChatRepository):
            configure_answer_services = None

        repository = RoutingOnlyRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        application = self._create_production_app(
            repository,
            unified_enabled=True,
            word_enabled=False,
        )

        with TestClient(application):
            self.assertIsInstance(repository._answer_router, KnowledgeAnswerRouter)
            self.assertIsNone(repository._answer_router._word_fact_service)

    def test_production_factory_requires_word_answer_service_capability(self) -> None:
        class RoutingOnlyRepository(InMemoryChatRepository):
            configure_answer_services = None

        repository = RoutingOnlyRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        application = self._create_production_app(
            repository,
            unified_enabled=True,
            word_enabled=True,
        )

        with self.assertRaisesRegex(TypeError, "Word factual QA.*configure_answer_services"):
            with TestClient(application):
                pass

    def test_development_app_keeps_legacy_defaults_despite_ambient_routing_flags(self) -> None:
        environ = {
            "OFFLINE_MODE": "false",
            "DATABASE_URL": "sqlite+pysqlite:///:memory:",
            "UNIFIED_KNOWLEDGE_ROUTING_ENABLED": "true",
            "WORD_FACTUAL_QA_ENABLED": "true",
        }

        with patch.dict(os.environ, environ, clear=True), patch(
            "app.main.load_runtime_environment"
        ):
            app = create_app()

        try:
            self.assertIsInstance(app.state.repository._answer_router, LegacyKnowledgeAnswerRouter)
        finally:
            app.state.repository.close()

    def test_excel_question_never_returns_seeded_word_answer_or_searches_rag(self) -> None:
        run, reply = self.harness.ask("地区为华东的销售额、成本汇总")

        self.assertEqual(run.route_type, KnowledgeRouteType.EXCEL_MULTI_AGGREGATE)
        self.assertEqual(self.harness.retrieval_gateway.search_calls, 0)
        self.assertEqual(self.harness.reranker.calls, 0)
        self.assertEqual(self.harness.llm.calls, 0)
        self.assertEqual(reply.paragraphs[0].citations, [])
        self.assertNotIn("报销制度", reply.paragraphs[0].text)

    def test_age_question_returns_only_age_without_reranker(self) -> None:
        run, reply = self.harness.ask("张三几岁")
        text = reply.paragraphs[0].text

        self.assertEqual(run.route_type, KnowledgeRouteType.WORD_FACTUAL)
        self.assertEqual(text, "张三的年龄是28岁。")
        self.assertNotIn("女", text)
        self.assertNotIn("工程师", text)
        self.assertEqual(self.harness.embedding.calls, 0)
        self.assertEqual(self.harness.retrieval_gateway.search_calls, 0)
        self.assertEqual(self.harness.reranker.calls, 0)
        self.assertEqual(self.harness.llm.calls, 0)

    def test_open_introduction_uses_production_reranker_and_llm(self) -> None:
        run, _reply = self.harness.ask("介绍张三")

        self.assertEqual(run.route_type, KnowledgeRouteType.SUMMARY_COMPARE)
        self.assertGreater(self.harness.retrieval_gateway.search_calls, 0)
        self.assertGreater(self.harness.reranker.calls, 0)
        self.assertEqual(self.harness.llm.calls, 1)

    def test_production_topology_uses_bge_reranker_and_llm_for_document_routes(self) -> None:
        source = self.harness.repository.list_knowledge_sources()[0]
        chunk = self.harness.repository.list_knowledge_chunks(source.id)[0]
        candidate = RetrievalCandidate(
            source_id=source.id,
            source_name=source.name,
            source_type=source.source_type,
            classification=source.classification,
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            point_id="point-production-routing",
        )
        environment = self._qwen_environment()
        retrieval_settings = RetrievalSettings.from_environ(environment)
        gateway = AcceptanceRetrievalGateway(candidate)
        llm = AcceptanceLLM()
        repository = InMemoryChatRepository(
            self.harness.repository._state,
            llm_provider=llm,
        )

        class ProductionRoutingResources(_DefaultRetrievalResourceFactory):
            def __init__(self) -> None:
                self.embedding = AcceptanceEmbedding()
                self.sparse = AcceptanceSparse()
                self.gateway = gateway

            def create_qdrant_client(self, _settings):
                return SimpleNamespace(close=lambda: None)

            def create_gateway(self, _client, _settings):
                return self.gateway

            def create_embedding_client(self, _settings):
                return self.embedding

            def create_sparse_encoder(self, _environ):
                return self.sparse

            def create_audit(self, _database):
                publication = SimpleNamespace(
                    id="publication-production-routing",
                    collection_name="knowledge_chunks_qwen3_v17",
                    embedding_fingerprint=retrieval_settings.embedding_fingerprint,
                )
                return SimpleNamespace(active_publication=lambda _alias: publication)

            def create_index_lifecycle(self, **_dependencies):
                return SimpleNamespace(close=lambda: None)

        class RerankerHttpResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "modelName": RERANKER_METADATA.name,
                    "modelVersion": RERANKER_METADATA.version,
                    "modelChecksum": RERANKER_METADATA.sha256,
                    "promptProfileSha256": RERANKER_METADATA.prompt_profile_sha256,
                    "protocolVersion": RERANKER_METADATA.protocol_version,
                    "passageCount": 1,
                    "scores": [0.99],
                }

        class RerankerHttpClient:
            def __init__(self) -> None:
                self.posts: list[tuple[str, dict[str, object]]] = []
                self.close_calls = 0

            def post(self, url: str, *, json: dict[str, object], **_kwargs):
                self.posts.append((url, json))
                return RerankerHttpResponse()

            def close(self) -> None:
                self.close_calls += 1

        resources = ProductionRoutingResources()
        reranker_http = RerankerHttpClient()
        application = create_production_app(
            environ=environment,
            repository_factory=lambda: repository,
            retrieval_resource_factory=resources,
            database_factory=lambda _url: Database("sqlite+pysqlite:///:memory:"),
            llm_provider_factory=lambda _environment: llm,
            health_registry_factory=lambda: DependencyHealthRegistry([]),
            ingestion_queue_factory=lambda **_kwargs: SimpleNamespace(close=lambda: None),
            storage_factory=lambda _root: SimpleNamespace(close=lambda: None),
            evaluation_import_service_factory=lambda: SimpleNamespace(close=lambda: None),
        )

        with patch("app.reranker_client.httpx.Client", return_value=reranker_http):
            with TestClient(application):
                _, conversation_id, _ = repository.create_conversation()
                summary_run = self._send(repository, conversation_id, "\u4ecb\u7ecd\u5f20\u4e09")
                document_run = self._send(
                    repository,
                    conversation_id,
                    "\u62a5\u9500\u6d41\u7a0b\u662f\u4ec0\u4e48",
                )

        self.assertEqual(summary_run.route_type, KnowledgeRouteType.SUMMARY_COMPARE)
        self.assertEqual(document_run.route_type, KnowledgeRouteType.DOCUMENT_QA)
        self.assertEqual(len(reranker_http.posts), 2)
        self.assertTrue(
            all(url == "http://127.0.0.1:8082/v1/rerank" for url, _ in reranker_http.posts)
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(reranker_http.close_calls, 1)

    def _create_production_app(
        self,
        repository: InMemoryChatRepository,
        *,
        unified_enabled: bool,
        word_enabled: bool,
    ):
        return create_production_app(
            environ={
                "OFFLINE_MODE": "false",
                "DATABASE_URL": "sqlite+pysqlite:///:memory:",
                "LLM_PROVIDER": "physoc_deepseek",
                "UNIFIED_KNOWLEDGE_ROUTING_ENABLED": str(unified_enabled).lower(),
                "WORD_FACTUAL_QA_ENABLED": str(word_enabled).lower(),
            },
            repository_factory=lambda: repository,
            database_factory=lambda _url: Database("sqlite+pysqlite:///:memory:"),
            llm_provider_factory=lambda _environment: AcceptanceLLM(),
            health_registry_factory=lambda: DependencyHealthRegistry([]),
            ingestion_queue_factory=lambda _repository: SimpleNamespace(close=lambda: None),
            storage_factory=lambda _root: SimpleNamespace(close=lambda: None),
            evaluation_import_service_factory=lambda: SimpleNamespace(close=lambda: None),
        )

    @staticmethod
    def _qwen_environment() -> dict[str, str]:
        return {
            "OFFLINE_MODE": "false",
            "DATABASE_URL": "sqlite+pysqlite:///:memory:",
            "LLM_PROVIDER": "physoc_deepseek",
            "UNIFIED_KNOWLEDGE_ROUTING_ENABLED": "true",
            "WORD_FACTUAL_QA_ENABLED": "false",
            "RETRIEVAL_MODE": "qwen3",
            "RETRIEVAL_PERMISSION_TAGS": "internal",
            "QDRANT_URL": "http://127.0.0.1:6333",
            "EMBEDDING_SERVICE_URL": "http://127.0.0.1:8081",
            "RERANKER_SERVICE_URL": "http://127.0.0.1:8082",
            "EMBEDDING_MODEL_NAME": EMBEDDING_METADATA.name,
            "EMBEDDING_MODEL_VERSION": EMBEDDING_METADATA.version,
            "EMBEDDING_MODEL_SHA256": EMBEDDING_METADATA.sha256,
            "EMBEDDING_MODEL_DIMENSIONS": str(EMBEDDING_METADATA.dimensions),
            "EMBEDDING_MODEL_NORMALIZED": "true",
            "EMBEDDING_ENCODING_PROFILE_SHA256": EMBEDDING_METADATA.encoding_profile_sha256,
            "EMBEDDING_PROTOCOL_VERSION": EMBEDDING_METADATA.protocol_version,
            "RERANKER_MODEL_NAME": RERANKER_METADATA.name,
            "RERANKER_MODEL_VERSION": RERANKER_METADATA.version,
            "RERANKER_MODEL_SHA256": RERANKER_METADATA.sha256,
            "RERANKER_PROMPT_PROFILE_SHA256": RERANKER_METADATA.prompt_profile_sha256,
            "RERANKER_PROTOCOL_VERSION": RERANKER_METADATA.protocol_version,
            "SPARSE_MODEL_ROOT": "C:/models/bm25",
            "SPARSE_PROFILE_SHA256": "e" * 64,
        }

    @staticmethod
    def _send(repository: InMemoryChatRepository, conversation_id: str, question: str):
        repository.send_message(conversation_id, question, "quick")
        return repository.list_agent_runs(1)[0]


if __name__ == "__main__":
    unittest.main()
