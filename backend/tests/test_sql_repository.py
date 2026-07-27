from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.agent import AgentRunResult
from app.database import Database
from app.models import ChatMessageModel, KnowledgeChunkModel
from app.repository import InMemoryChatRepository
from app.retrieval_models import RetrievalMode, RetrievalScope
from app.retrieval_router import RoutedRetrievalOutcome
from app.seed import build_seed_state
from app.sql_repository import SqlChatRepository


class SqlRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.database = Database("sqlite+pysqlite:///:memory:")
        self.database.create_schema()
        self.repository = SqlChatRepository(self.database)
        self.repository.seed_if_empty(build_seed_state())

    def test_reads_seed_conversations_and_messages(self) -> None:
        conversations = self.repository.list_conversations()

        self.assertGreaterEqual(len(conversations), 1)
        self.assertEqual(conversations[0].id, "conv-q4")
        messages = self.repository.get_messages("conv-q4")
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1].paragraphs[0].citations[0].source_id, "ARC-FIN-Q4")
        self.assertEqual(messages[1].artifacts[0].type, "summary")

    def test_persists_created_conversation_and_first_exchange(self) -> None:
        conversations, active_id, messages = self.repository.create_conversation()

        self.assertEqual(conversations[0].id, active_id)
        self.assertEqual(messages, [])

        self.repository.send_message(active_id, "跟进现金流压力测试", "source")

        second_repository = SqlChatRepository(self.database)
        persisted_messages = second_repository.get_messages(active_id)
        persisted_conversations = second_repository.list_conversations()
        self.assertEqual([message.role for message in persisted_messages], ["user", "assistant"])
        self.assertEqual(persisted_messages[0].content, "跟进现金流压力测试")
        self.assertEqual(persisted_conversations[0].title, "跟进现金流压力测试")

    def test_persists_agent_run_and_read_only_steps(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-agent-audit",
            name="agent-policy.txt",
            source_type="文档",
            classification="内部",
            records=0,
            file_path="agent-policy.txt",
            file_size=128,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "kb-agent-audit",
            [
                KnowledgeChunkModel(
                    id="chunk-agent-audit-0",
                    source_id="kb-agent-audit",
                    chunk_index=0,
                    text="差旅报销必须提交发票、行程单和审批记录。",
                    token_count=24,
                )
            ],
        )
        _, conversation_id, _ = self.repository.create_conversation()

        self.repository.send_message(conversation_id, "差旅票据材料", "deep")

        second_repository = SqlChatRepository(self.database)
        runs = second_repository.list_agent_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].conversation_id, conversation_id)
        self.assertEqual(runs[0].query, "差旅票据材料")
        self.assertGreater(runs[0].evidence_count, 0)
        self.assertTrue(all(step.read_only for step in runs[0].steps))
        self.assertIn("search_knowledge", [step.tool_name for step in runs[0].steps])
        self.assertIn("compose_answer", [step.tool_name for step in runs[0].steps])

    def test_adds_knowledge_source(self) -> None:
        sources = self.repository.add_knowledge_source("董事会纪要.pdf", "PDF", "内部·机密")

        self.assertEqual(sources[0].name, "董事会纪要.pdf")
        self.assertEqual(sources[0].source_type, "PDF")
        self.assertEqual(sources[0].status, "解析中")

    def test_deletes_uploaded_knowledge_source_and_chunks(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-delete",
            name="delete-me.txt",
            source_type="文档",
            classification="内部",
            records=0,
            file_path="delete-me.txt",
            file_size=128,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "kb-delete",
            [
                KnowledgeChunkModel(
                    id="chunk-delete-0",
                    source_id="kb-delete",
                    chunk_index=0,
                    text="待删除资料片段",
                    token_count=12,
                )
            ],
        )

        sources, deleted = self.repository.delete_knowledge_source("kb-delete")

        self.assertEqual(deleted.file_path, "delete-me.txt")
        self.assertNotIn("kb-delete", [source.id for source in sources])
        with self.assertRaises(Exception):
            self.repository.list_knowledge_chunks("kb-delete")

    def test_send_message_uses_indexed_knowledge_chunks_for_citations(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-sql-cashflow",
            name="cashflow-note.txt",
            source_type="文档",
            classification="内部·机密",
            records=0,
            file_path="cashflow-note.txt",
            file_size=128,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "kb-sql-cashflow",
            [
                KnowledgeChunkModel(
                    id="chunk-sql-cashflow-0",
                    source_id="kb-sql-cashflow",
                    chunk_index=0,
                    text="现金流风险与回款周期直接相关。",
                    token_count=24,
                )
            ],
        )
        _, conversation_id, _ = self.repository.create_conversation()

        _, _, messages = self.repository.send_message(
            conversation_id,
            "请分析现金流风险",
            "source",
        )

        assistant = messages[-1]
        source_ids = [
            citation.source_id
            for paragraph in assistant.paragraphs
            for citation in paragraph.citations
        ]
        self.assertIn("kb-sql-cashflow", source_ids)
        citation = next(
            citation
            for paragraph in assistant.paragraphs
            for citation in paragraph.citations
            if citation.source_id == "kb-sql-cashflow"
        )
        self.assertEqual(citation.source_name, "cashflow-note.txt")
        self.assertEqual(citation.chunk_id, "chunk-sql-cashflow-0")
        self.assertEqual(citation.chunk_index, 0)
        self.assertEqual(citation.rank, 1)
        self.assertGreater(citation.score or 0, 0)
        self.assertIn("现金", citation.matched_terms)
        self.assertIn("现金流风险", citation.excerpt)
        self.assertIn("现金流", assistant.paragraphs[0].text)

    def test_indexes_chunk_embeddings_and_ranks_semantic_matches(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-vector-cashflow",
            name="cashflow-risk.txt",
            source_type="文档",
            classification="内部·机密",
            records=0,
            file_path="cashflow-risk.txt",
            file_size=256,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "kb-vector-cashflow",
            [
                KnowledgeChunkModel(
                    id="chunk-policy-risk",
                    source_id="kb-vector-cashflow",
                    chunk_index=0,
                    text="风险评级制度需要按月复核。",
                    token_count=16,
                ),
                KnowledgeChunkModel(
                    id="chunk-cash-collection",
                    source_id="kb-vector-cashflow",
                    chunk_index=1,
                    text="应收账款增加，回款周期拉长，造成现金流压力。",
                    token_count=27,
                ),
            ],
        )

        chunks = self.repository.list_knowledge_chunks("kb-vector-cashflow")
        self.assertTrue(all(chunk.embedding for chunk in chunks))

        hits = self.repository.search_knowledge_chunks("回款风险", limit=1)

        self.assertEqual(hits[0].chunk.id, "chunk-cash-collection")
        self.assertEqual(hits[0].rank, 1)
        self.assertIn("回款", hits[0].matched_terms)
        self.assertGreater(hits[0].keyword_score, 0)
        self.assertGreaterEqual(hits[0].vector_score, 0)

        with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": "100"}):
            filtered_hits = self.repository.search_knowledge_chunks("回款风险", limit=1)

        self.assertEqual(filtered_hits, [])

    def test_chunk_metadata_round_trips_through_sql_repository(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="source-metadata",
            name="policy.txt",
            source_type="TXT",
            classification="internal",
            records=0,
            file_path="policy.txt",
            file_size=64,
            mime_type="text/plain",
        )
        chunk = KnowledgeChunkModel(
            id="chunk-metadata-1",
            source_id="source-metadata",
            chunk_index=0,
            text="body",
            token_count=1,
            metadata={"section_title": "Policy", "page_number": 3},
        )

        self.repository.complete_knowledge_source_indexing("source-metadata", [chunk])

        stored = SqlChatRepository(self.database).list_knowledge_chunks("source-metadata")[0]
        self.assertEqual(stored.metadata, {"section_title": "Policy", "page_number": 3})

    def test_chunk_metadata_defaults_and_round_trips_through_memory_repository(self) -> None:
        repository = InMemoryChatRepository(build_seed_state())
        repository.add_uploaded_knowledge_source(
            source_id="source-memory-metadata",
            name="policy.txt",
            source_type="TXT",
            classification="internal",
            records=0,
            file_path="policy.txt",
            file_size=64,
            mime_type="text/plain",
        )
        default_chunk = KnowledgeChunkModel(
            id="chunk-default-metadata",
            source_id="source-memory-metadata",
            chunk_index=0,
            text="default",
            token_count=1,
        )
        metadata_chunk = KnowledgeChunkModel(
            id="chunk-memory-metadata",
            source_id="source-memory-metadata",
            chunk_index=1,
            text="body",
            token_count=1,
            metadata={"slide_number": 4},
        )

        repository.complete_knowledge_source_indexing(
            "source-memory-metadata", [default_chunk, metadata_chunk]
        )

        stored = repository.list_knowledge_chunks("source-memory-metadata")
        self.assertEqual(stored[0].metadata, {})
        self.assertEqual(stored[1].metadata, {"slide_number": 4})

    def test_ranks_travel_receipt_materials_for_business_synonyms(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-travel-materials",
            name="travel-policy.txt",
            source_type="文档",
            classification="内部",
            records=0,
            file_path="travel-policy.txt",
            file_size=256,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "kb-travel-materials",
            [
                KnowledgeChunkModel(
                    id="chunk-travel-standard",
                    source_id="kb-travel-materials",
                    chunk_index=0,
                    text="差旅住宿标准按照城市等级执行，住宿费不得超过公司限额。",
                    token_count=28,
                ),
                KnowledgeChunkModel(
                    id="chunk-travel-receipts",
                    source_id="kb-travel-materials",
                    chunk_index=1,
                    text="返程后需要在五个工作日内上传发票、行程单和审批记录。",
                    token_count=30,
                ),
            ],
        )

        hits = self.repository.search_knowledge_chunks("差旅票据材料", limit=1)

        self.assertEqual(hits[0].chunk.id, "chunk-travel-receipts")
        self.assertEqual(hits[0].rank, 1)

    def test_legacy_search_filters_sources_to_configured_permission_classifications(self) -> None:
        for source_id, classification in (
            ("kb-allowed", "internal"),
            ("kb-denied", "executive"),
        ):
            self.repository.add_uploaded_knowledge_source(
                source_id=source_id,
                name=f"{source_id}.txt",
                source_type="TXT",
                classification=classification,
                records=0,
                file_path=f"{source_id}.txt",
                file_size=64,
                mime_type="text/plain",
            )
            self.repository.complete_knowledge_source_indexing(
                source_id,
                [
                    KnowledgeChunkModel(
                        id=f"chunk-{source_id}",
                        source_id=source_id,
                        chunk_index=0,
                        text="unique fallback policy evidence",
                        token_count=4,
                    )
                ],
            )

        scoped = SqlChatRepository(self.database, retrieval_permission_tags=("internal",))

        hits = scoped.search_knowledge_chunks("unique fallback policy evidence", limit=10)

        self.assertEqual([hit.source.id for hit in hits], ["kb-allowed"])
        self.assertTrue(hasattr(scoped, "_search_legacy_knowledge_chunks"))

    def test_agent_uses_conversation_id_as_retrieval_routing_key(self) -> None:
        class RecordingRouter:
            def __init__(self) -> None:
                self.requests = []

            def search(self, request):
                self.requests.append(request)
                return RoutedRetrievalOutcome(
                    mode=RetrievalMode.LEGACY,
                    hits=(),
                    stage_ms={},
                )

        router = RecordingRouter()
        repository = SqlChatRepository(
            self.database,
            retrieval_router=router,
            retrieval_scope=RetrievalScope("default", ("internal",), "v1"),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "policy question", "quick")

        self.assertEqual(len(router.requests), 1)
        self.assertEqual(router.requests[0].routing_key, conversation_id)
        self.assertEqual(router.requests[0].query, "policy question")

    def test_structured_answer_path_remains_before_retrieval_router(self) -> None:
        class NeverRouter:
            def __init__(self) -> None:
                self.calls = 0

            def search(self, request):
                self.calls += 1
                raise AssertionError(f"router must not receive structured request {request}")

        class StructuredService:
            def __init__(self) -> None:
                self.calls = 0

            def try_answer(self, *, conversation_id, content, mode, previous_messages):
                del content, previous_messages
                self.calls += 1
                return AgentRunResult(
                    id="structured-run-1",
                    conversation_id=conversation_id,
                    query="redacted structured query",
                    mode=mode,
                    status="completed",
                    started_at="2026-07-28 10:00:00",
                    completed_at="2026-07-28 10:00:01",
                    reply=ChatMessageModel(
                        id="structured-reply-1",
                        role="assistant",
                        time="2026-07-28 10:00:01",
                        content="42",
                    ),
                    steps=[],
                    evidence_count=0,
                    source_count=0,
                )

        router = NeverRouter()
        structured = StructuredService()
        repository = SqlChatRepository(
            self.database,
            structured_service=structured,
            retrieval_router=router,
            retrieval_scope=RetrievalScope("default", ("internal",), "v1"),
        )
        _, conversation_id, _ = repository.create_conversation()

        _, _, messages = repository.send_message(conversation_id, "average sales", "quick")

        self.assertEqual(structured.calls, 1)
        self.assertEqual(router.calls, 0)
        self.assertEqual(messages[-1].content, "42")

    def test_repositories_reject_partial_retrieval_router_configuration(self) -> None:
        scope = RetrievalScope("default", ("internal",), "v1")
        cases = (
            (
                "sql-router-only",
                lambda: SqlChatRepository(self.database, retrieval_router=object()),
            ),
            (
                "sql-scope-only",
                lambda: SqlChatRepository(self.database, retrieval_scope=scope),
            ),
            (
                "memory-router-only",
                lambda: InMemoryChatRepository(build_seed_state(), retrieval_router=object()),
            ),
            (
                "memory-scope-only",
                lambda: InMemoryChatRepository(build_seed_state(), retrieval_scope=scope),
            ),
        )

        for name, construct in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError,
                    "retrieval_router and retrieval_scope must be configured together",
                ):
                    construct()

    def test_repositories_preserve_both_none_legacy_configuration(self) -> None:
        memory = InMemoryChatRepository(build_seed_state())
        sql = SqlChatRepository(self.database)

        self.assertIsNone(memory.retrieval_router)
        self.assertIsNone(sql.retrieval_router)
        self.assertIsInstance(memory.search_knowledge_chunks("policy"), list)
        self.assertIsInstance(sql.search_knowledge_chunks("policy"), list)

    def test_sql_repository_allows_one_startup_retrieval_configuration(self) -> None:
        repository = SqlChatRepository(self.database)
        router = object()
        scope = RetrievalScope("default", ("internal",), "publication-v1")

        repository.configure_retrieval(router, scope)

        self.assertIs(repository.retrieval_router, router)
        self.assertIs(repository._retrieval_scope, scope)
        with self.assertRaisesRegex(RuntimeError, "retrieval is already configured"):
            repository.configure_retrieval(object(), scope)


if __name__ == "__main__":
    unittest.main()
