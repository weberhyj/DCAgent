from __future__ import annotations

import io
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from loguru import logger as loguru_logger
from sqlalchemy import select

from app.agent import GREETING_REPLY, AgentRunResult
from app.database import Database, KnowledgeFactRecord, KnowledgeSourceRecord
from app.embedding_fingerprint import EmbeddingFingerprint
from app.models import ChatMessageModel, KnowledgeChunkModel
from app.repository import InMemoryChatRepository
from app.retrieval_models import RetrievalMode, RetrievalRequest, RetrievalScope
from app.retrieval_router import (
    RetrievalFallbackReason,
    RetrievalRouter,
    RoutedRetrievalOutcome,
)
from app.retrieval_scope import DynamicRetrievalScopeProvider
from app.seed import build_seed_state
from app.sql_repository import SqlChatRepository
from app.word_facts import KnowledgeFactModel, WordFactualIntent, normalize_fact_key

TEST_EMBEDDING_FINGERPRINT = EmbeddingFingerprint(
    model_name="qwen2.5:0.5b",
    model_version="test-v1",
    model_sha256="a" * 64,
    dimensions=896,
    normalized=True,
    encoding_profile_sha256="b" * 64,
    protocol_version="v1",
)


class SqlRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.database = Database("sqlite+pysqlite:///:memory:")
        self.database.create_schema()
        self.repository = SqlChatRepository(self.database)
        self.repository.seed_if_empty(build_seed_state())

    def _add_fact_source(
        self,
        repository,
        *,
        source_id: str,
        source_name: str,
        classification: str = "公开",
        fact_id: str | None = None,
        entity: str = "张三",
        value: str = "27岁",
    ) -> KnowledgeFactModel:
        repository.add_uploaded_knowledge_source(
            source_id=source_id,
            name=source_name,
            source_type="文档",
            classification=classification,
            records=0,
            file_path=source_name,
            file_size=128,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        chunk_id = f"chunk-{source_id}"
        fact = KnowledgeFactModel.create(
            id=fact_id or f"fact-{source_id}",
            source_id=source_id,
            chunk_id=chunk_id,
            entity=entity,
            field="年龄",
            value=value,
            confidence=0.97,
            locator={"paragraph": 1},
        )
        repository.complete_knowledge_source_indexing(
            source_id,
            [
                KnowledgeChunkModel(
                    id=chunk_id,
                    source_id=source_id,
                    chunk_index=0,
                    text=f"姓名：{entity}，年龄：{value}",
                    token_count=10,
                )
            ],
            facts=[fact],
        )
        return fact

    def _fact_intent(self) -> WordFactualIntent:
        return WordFactualIntent(
            entity="张 三",
            entity_normalized=normalize_fact_key("张三"),
            field="岁数",
            field_normalized=normalize_fact_key("年龄"),
        )

    def _memory_repository_with_fact(self, source_id: str) -> InMemoryChatRepository:
        repository = InMemoryChatRepository(build_seed_state())
        self._add_fact_source(
            repository,
            source_id=source_id,
            source_name=f"{source_id}.docx",
        )
        return repository

    def _stored_fact_ids(self, source_id: str) -> list[str]:
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(KnowledgeFactRecord.id)
                    .where(KnowledgeFactRecord.source_id == source_id)
                    .order_by(KnowledgeFactRecord.id)
                ).all()
            )

    def test_complete_indexing_replaces_old_source_facts(self) -> None:
        first = self._add_fact_source(
            self.repository,
            source_id="kb-people",
            source_name="people.docx",
            fact_id="fact-old",
        )
        second = replace(
            first,
            id="fact-new",
            chunk_id="chunk-new",
            value="28岁",
        )

        self.repository.complete_knowledge_source_indexing(
            "kb-people",
            [
                KnowledgeChunkModel(
                    id="chunk-new",
                    source_id="kb-people",
                    chunk_index=0,
                    text="姓名：张三，年龄：28岁",
                    token_count=10,
                )
            ],
            facts=[second],
        )

        matches = self.repository.find_knowledge_facts(self._fact_intent())
        self.assertEqual([item.fact.id for item in matches], ["fact-new"])
        self.assertEqual([item.fact.value for item in matches], ["28岁"])

    def test_query_requires_indexed_permitted_exact_sources_in_stable_order(self) -> None:
        self._add_fact_source(
            self.repository,
            source_id="kb-public-z",
            source_name="b-people.docx",
            fact_id="fact-z",
        )
        self._add_fact_source(
            self.repository,
            source_id="kb-public-a",
            source_name="a-people.docx",
            fact_id="fact-a",
        )
        self._add_fact_source(
            self.repository,
            source_id="kb-private",
            source_name="private.docx",
            classification="内部·机密",
        )
        self._add_fact_source(
            self.repository,
            source_id="kb-parsing",
            source_name="parsing.docx",
        )
        self._add_fact_source(
            self.repository,
            source_id="kb-near-match",
            source_name="near-match.docx",
            entity="张三丰",
        )
        with self.database.session() as session:
            session.get(KnowledgeSourceRecord, "kb-parsing").status = "解析中"

        matches = self.repository.find_knowledge_facts(
            self._fact_intent(),
            permission_tags=("公开",),
        )

        self.assertEqual(
            [item.fact.source_id for item in matches],
            ["kb-public-a", "kb-public-z"],
        )
        self.assertEqual(
            [(item.source_name, item.classification) for item in matches],
            [("a-people.docx", "公开"), ("b-people.docx", "公开")],
        )

    def test_complete_indexing_rejects_facts_outside_incoming_bundle_atomically(self) -> None:
        old_fact = self._add_fact_source(
            self.repository,
            source_id="kb-atomic",
            source_name="atomic.docx",
            fact_id="fact-old",
        )
        invalid_facts = (
            replace(
                old_fact,
                id="fact-wrong-source",
                source_id="kb-other",
                chunk_id="chunk-new",
            ),
            replace(old_fact, id="fact-missing-chunk", chunk_id="chunk-missing"),
        )

        for invalid_fact in invalid_facts:
            with self.subTest(fact_id=invalid_fact.id):
                with self.assertRaises(ValueError):
                    self.repository.complete_knowledge_source_indexing(
                        "kb-atomic",
                        [
                            KnowledgeChunkModel(
                                id="chunk-new",
                                source_id="kb-atomic",
                                chunk_index=0,
                                text="姓名：张三，年龄：28岁",
                                token_count=10,
                            )
                        ],
                        facts=[invalid_fact],
                    )

                matches = self.repository.find_knowledge_facts(self._fact_intent())
                self.assertEqual([item.fact.id for item in matches], ["fact-old"])
                self.assertEqual(
                    [chunk.id for chunk in self.repository.list_knowledge_chunks("kb-atomic")],
                    ["chunk-kb-atomic"],
                )
                source = next(
                    item
                    for item in self.repository.list_knowledge_sources()
                    if item.id == "kb-atomic"
                )
                self.assertEqual(source.status, "已索引")

    def test_memory_complete_indexing_replaces_old_source_facts(self) -> None:
        repository = self._memory_repository_with_fact("kb-memory-replace")
        replacement = KnowledgeFactModel.create(
            id="fact-memory-new",
            source_id="kb-memory-replace",
            chunk_id="chunk-memory-new",
            entity="张三",
            field="年龄",
            value="28岁",
            confidence=0.97,
            locator={"paragraph": 2},
        )

        repository.complete_knowledge_source_indexing(
            "kb-memory-replace",
            [
                KnowledgeChunkModel(
                    id="chunk-memory-new",
                    source_id="kb-memory-replace",
                    chunk_index=0,
                    text="姓名：张三，年龄：28岁",
                    token_count=10,
                )
            ],
            facts=[replacement],
        )

        matches = repository.find_knowledge_facts(self._fact_intent())
        self.assertEqual([item.fact.id for item in matches], ["fact-memory-new"])

    def test_replace_knowledge_facts_uses_existing_source_chunks(self) -> None:
        first = self._add_fact_source(
            self.repository,
            source_id="kb-replace-only",
            source_name="replace-only.docx",
            fact_id="fact-replace-old",
        )
        replacement = replace(first, id="fact-replace-new", value="29岁")

        self.repository.replace_knowledge_facts("kb-replace-only", [replacement])

        matches = self.repository.find_knowledge_facts(self._fact_intent())
        self.assertEqual([item.fact.id for item in matches], ["fact-replace-new"])
        self.assertEqual([item.fact.value for item in matches], ["29岁"])

    def test_memory_replace_knowledge_facts_uses_existing_source_chunks(self) -> None:
        repository = self._memory_repository_with_fact("kb-memory-replace-only")
        replacement = KnowledgeFactModel.create(
            id="fact-memory-replace-new",
            source_id="kb-memory-replace-only",
            chunk_id="chunk-kb-memory-replace-only",
            entity="张三",
            field="年龄",
            value="29岁",
            confidence=0.97,
            locator={"paragraph": 2},
        )

        repository.replace_knowledge_facts("kb-memory-replace-only", [replacement])

        matches = repository.find_knowledge_facts(self._fact_intent())
        self.assertEqual([item.fact.id for item in matches], ["fact-memory-replace-new"])

    def test_memory_reindex_clears_source_facts(self) -> None:
        repository = self._memory_repository_with_fact("kb-memory-reindex")

        repository.reindex_knowledge_source("kb-memory-reindex")

        self.assertNotIn("kb-memory-reindex", repository._state.knowledge_facts_by_source)

    def test_memory_failure_clears_source_facts(self) -> None:
        repository = self._memory_repository_with_fact("kb-memory-failure")

        repository.fail_knowledge_source_indexing("kb-memory-failure", "parse failed")

        self.assertNotIn("kb-memory-failure", repository._state.knowledge_facts_by_source)

    def test_memory_delete_clears_source_facts(self) -> None:
        repository = self._memory_repository_with_fact("kb-memory-delete")

        repository.delete_knowledge_source("kb-memory-delete")

        self.assertNotIn("kb-memory-delete", repository._state.knowledge_facts_by_source)

    def test_sql_reindex_physically_clears_source_facts(self) -> None:
        self._add_fact_source(
            self.repository,
            source_id="kb-sql-reindex",
            source_name="sql-reindex.docx",
        )

        self.repository.reindex_knowledge_source("kb-sql-reindex")

        self.assertEqual(self._stored_fact_ids("kb-sql-reindex"), [])

    def test_sql_failure_physically_clears_source_facts(self) -> None:
        self._add_fact_source(
            self.repository,
            source_id="kb-sql-failure",
            source_name="sql-failure.docx",
        )

        self.repository.fail_knowledge_source_indexing("kb-sql-failure", "parse failed")

        self.assertEqual(self._stored_fact_ids("kb-sql-failure"), [])

    def test_sql_delete_physically_clears_source_facts(self) -> None:
        self._add_fact_source(
            self.repository,
            source_id="kb-sql-delete",
            source_name="sql-delete.docx",
        )

        self.repository.delete_knowledge_source("kb-sql-delete")

        self.assertEqual(self._stored_fact_ids("kb-sql-delete"), [])

    def test_reads_seed_conversations_and_messages(self) -> None:
        conversations = self.repository.list_conversations()

        self.assertGreaterEqual(len(conversations), 1)
        self.assertEqual(conversations[0].id, "conv-q4")
        messages = self.repository.get_messages("conv-q4")
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1].paragraphs[0].citations[0].source_id, "ARC-FIN-Q4")
        self.assertEqual(messages[1].artifacts[0].type, "summary")

    def test_evaluation_run_routes_explicit_case_and_relevant_chunk_labels(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="source-evaluation-labels",
            name="evaluation-labels.txt",
            source_type="TXT",
            classification="internal",
            records=0,
            file_path="evaluation-labels.txt",
            file_size=128,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "source-evaluation-labels",
            [
                KnowledgeChunkModel(
                    id="chunk-evaluation-a",
                    source_id="source-evaluation-labels",
                    chunk_index=0,
                    text="evaluation evidence a",
                    token_count=3,
                ),
                KnowledgeChunkModel(
                    id="chunk-evaluation-b",
                    source_id="source-evaluation-labels",
                    chunk_index=1,
                    text="evaluation evidence b",
                    token_count=3,
                ),
            ],
        )
        case = self.repository.create_evaluation_case(
            question="evaluation evidence",
            expected_source_ids=["source-evaluation-labels"],
            expected_terms=[],
            top_k=8,
        )

        class RecordingRouter:
            def __init__(self) -> None:
                self.requests: list[RetrievalRequest] = []

            def search(self, request: RetrievalRequest) -> RoutedRetrievalOutcome:
                self.requests.append(request)
                return RoutedRetrievalOutcome(
                    mode=RetrievalMode.QWEN3,
                    hits=(),
                    stage_ms={},
                )

        router = RecordingRouter()
        provider = SimpleNamespace(
            resolve=lambda: SimpleNamespace(
                scope=RetrievalScope("default", ("internal",), "v1"),
                detail="ready",
            )
        )
        self.repository.configure_retrieval(router, provider)

        self.repository.run_evaluation_cases([case.id])

        self.assertEqual(len(router.requests), 1)
        request = router.requests[0]
        self.assertEqual(request.evaluation_case_id, case.id)
        self.assertEqual(
            request.relevant_chunk_ids,
            ("chunk-evaluation-a", "chunk-evaluation-b"),
        )

    def test_memory_evaluation_run_routes_the_same_explicit_labels(self) -> None:
        repository = InMemoryChatRepository(build_seed_state())
        repository.add_uploaded_knowledge_source(
            source_id="source-memory-evaluation",
            name="memory-evaluation.txt",
            source_type="TXT",
            classification="internal",
            records=0,
            file_path="memory-evaluation.txt",
            file_size=128,
            mime_type="text/plain",
        )
        repository.complete_knowledge_source_indexing(
            "source-memory-evaluation",
            [
                KnowledgeChunkModel(
                    id="chunk-memory-evaluation",
                    source_id="source-memory-evaluation",
                    chunk_index=0,
                    text="memory evaluation evidence",
                    token_count=3,
                )
            ],
        )
        case = repository.create_evaluation_case(
            question="memory evaluation evidence",
            expected_source_ids=["source-memory-evaluation"],
            expected_terms=[],
            top_k=8,
        )

        class RecordingRouter:
            def __init__(self) -> None:
                self.requests: list[RetrievalRequest] = []

            def search(self, request: RetrievalRequest) -> RoutedRetrievalOutcome:
                self.requests.append(request)
                return RoutedRetrievalOutcome(
                    mode=RetrievalMode.QWEN3,
                    hits=(),
                    stage_ms={},
                )

        router = RecordingRouter()
        provider = SimpleNamespace(
            resolve=lambda: SimpleNamespace(
                scope=RetrievalScope("default", ("internal",), "v1"),
                detail="ready",
            )
        )
        repository.configure_retrieval(router, provider)

        repository.run_evaluation_cases([case.id])

        self.assertEqual(len(router.requests), 1)
        self.assertEqual(router.requests[0].evaluation_case_id, case.id)
        self.assertEqual(
            router.requests[0].relevant_chunk_ids,
            ("chunk-memory-evaluation",),
        )

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
        with self.assertRaises(HTTPException):
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

    def test_default_public_scope_can_retrieve_public_source(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-public",
            name="public-policy.txt",
            source_type="TXT",
            classification="公开",
            records=0,
            file_path="public-policy.txt",
            file_size=64,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "kb-public",
            [
                KnowledgeChunkModel(
                    id="chunk-kb-public",
                    source_id="kb-public",
                    chunk_index=0,
                    text="公开制度规定访客需要在前台登记。",
                    token_count=18,
                )
            ],
        )
        scoped = SqlChatRepository(self.database, retrieval_permission_tags=("公开",))

        hits = scoped.search_knowledge_chunks("访客前台登记", limit=5)

        self.assertEqual([hit.source.id for hit in hits], ["kb-public"])

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

    def test_repositories_route_greeting_before_structured_retrieval_and_llm(self) -> None:
        class NeverStructuredService:
            def try_answer(self, **kwargs):
                raise AssertionError(f"structured service must not receive greeting {kwargs}")

        class NeverRouter:
            def search(self, request):
                raise AssertionError(f"router must not receive greeting {request}")

        class NeverProvider:
            def generate_reply(self, request):
                raise AssertionError(f"LLM must not receive greeting {request}")

        scope = RetrievalScope("default", ("internal",), "v1")
        repositories = (
            (
                "sql",
                SqlChatRepository(
                    self.database,
                    llm_provider=NeverProvider(),
                    structured_service=NeverStructuredService(),
                    retrieval_router=NeverRouter(),
                    retrieval_scope=scope,
                ),
            ),
            (
                "memory",
                InMemoryChatRepository(
                    build_seed_state(),
                    llm_provider=NeverProvider(),
                    structured_service=NeverStructuredService(),
                    retrieval_router=NeverRouter(),
                    retrieval_scope=scope,
                ),
            ),
        )

        for name, repository in repositories:
            with self.subTest(repository=name):
                _, conversation_id, _ = repository.create_conversation()

                _, _, messages = repository.send_message(conversation_id, "你好", "quick")

                self.assertEqual(messages[-1].paragraphs[-1].text, GREETING_REPLY)
                run = repository.list_agent_runs(limit=1)[0]
                self.assertEqual(run.evidence_count, 0)
                self.assertEqual(run.source_count, 0)
                self.assertEqual(len(run.steps), 1)
                self.assertEqual(run.steps[0].tool_name, "respond_greeting")

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
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValueError,
                    "retrieval_router and retrieval_scope must be configured together",
                ),
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
        scope = RetrievalScope("default", ("internal",), "v1")
        provider = SimpleNamespace(resolve=lambda: SimpleNamespace(scope=scope, detail="ready"))

        repository.configure_retrieval(router, provider)

        self.assertIs(repository.retrieval_router, router)
        self.assertIs(repository._retrieval_scope_provider, provider)
        with self.assertRaisesRegex(RuntimeError, "retrieval is already configured"):
            repository.configure_retrieval(object(), provider)

    def test_repository_resolves_scope_for_every_request_and_falls_back_when_unavailable(
        self,
    ) -> None:
        class RecordingRouter:
            def __init__(self) -> None:
                self.requests = []
                self.fallbacks = []

            def search(self, request):
                self.requests.append(request)
                return RoutedRetrievalOutcome(
                    mode=RetrievalMode.LEGACY,
                    hits=(),
                    stage_ms={},
                )

            def fallback_to_legacy(self, **values):
                self.fallbacks.append(values)
                return RoutedRetrievalOutcome(
                    mode=RetrievalMode.LEGACY,
                    hits=(),
                    stage_ms={"legacy": 0.0},
                    fallback_reason=values["fallback_reason"].value,
                )

        class MutableProvider:
            def __init__(self) -> None:
                self.scope = RetrievalScope("default", ("internal",), "v1")

            def resolve(self):
                return SimpleNamespace(
                    scope=self.scope,
                    detail="ready" if self.scope is not None else "unavailable",
                )

        router = RecordingRouter()
        provider = MutableProvider()
        repository = SqlChatRepository(
            self.database,
            retrieval_router=router,
            retrieval_scope_provider=provider,
        )

        repository._search_routed_knowledge_chunks("first", 8, "conversation-1")
        provider.scope = RetrievalScope("default", ("internal",), "v2")
        repository._search_routed_knowledge_chunks("second", 8, "conversation-1")
        provider.scope = None
        fallback = repository._search_routed_knowledge_chunks(
            "third",
            8,
            "conversation-1",
        )

        self.assertEqual(
            [item.scope.publication_version for item in router.requests],
            ["v1", "v2"],
        )
        self.assertEqual(fallback, [])
        self.assertEqual(
            router.fallbacks,
            [
                {
                    "query": "third",
                    "limit": 8,
                    "routing_key": "conversation-1",
                    "fallback_reason": RetrievalFallbackReason.RETRIEVAL_SCOPE_UNAVAILABLE,
                }
            ],
        )

    def test_sql_no_active_scope_uses_router_legacy_completion_without_leaking(self) -> None:
        repository = SqlChatRepository(self.database)
        provider = DynamicRetrievalScopeProvider(
            audit=SimpleNamespace(active_publication=lambda _alias: None),
            gateway=SimpleNamespace(
                resolve_alias=lambda: "private-alias http://qdrant-internal:6333"
            ),
            alias_name="knowledge_chunks_current",
            knowledge_base_id="default",
            permission_tags=("internal",),
            embedding_fingerprint=TEST_EMBEDDING_FINGERPRINT,
        )
        hybrid = SimpleNamespace(calls=0)

        def never_retrieve(_request):
            hybrid.calls += 1
            raise AssertionError("hybrid must not run without a trusted scope")

        hybrid.retrieve = never_retrieve
        router = RetrievalRouter(
            mode="qwen3",
            legacy_search=repository.search_knowledge_chunks,
            hybrid=hybrid,
            canary_percent=100,
            request_id_factory=lambda: "scope-fallback-sql",
        )
        self.addCleanup(router.close)
        repository.configure_retrieval(router, provider)

        self._assert_scope_unavailable_completion(
            repository,
            router,
            hybrid,
            query="private no-active query sentinel",
            routing_key="conversation-sql",
            forbidden=("private-alias", "http://qdrant-internal:6333"),
        )

    def test_memory_divergent_scope_uses_router_legacy_completion_without_leaking(
        self,
    ) -> None:
        repository = InMemoryChatRepository(build_seed_state())
        active_collection = "knowledge_chunks_qwen3_v41"
        alias_collection = "knowledge_chunks_qwen3_v42"
        provider = DynamicRetrievalScopeProvider(
            audit=SimpleNamespace(
                active_publication=lambda _alias: SimpleNamespace(collection_name=active_collection)
            ),
            gateway=SimpleNamespace(resolve_alias=lambda: alias_collection),
            alias_name="knowledge_chunks_current",
            knowledge_base_id="default",
            permission_tags=("internal",),
            embedding_fingerprint=TEST_EMBEDDING_FINGERPRINT,
        )
        hybrid = SimpleNamespace(calls=0)

        def never_retrieve(_request):
            hybrid.calls += 1
            raise AssertionError("hybrid must not run for divergent publication state")

        hybrid.retrieve = never_retrieve
        router = RetrievalRouter(
            mode="qwen3",
            legacy_search=repository.search_knowledge_chunks,
            hybrid=hybrid,
            canary_percent=100,
            request_id_factory=lambda: "scope-fallback-memory",
        )
        self.addCleanup(router.close)
        repository.configure_retrieval(router, provider)

        self._assert_scope_unavailable_completion(
            repository,
            router,
            hybrid,
            query="private divergence query sentinel",
            routing_key="conversation-memory",
            forbidden=(active_collection, alias_collection),
        )

    def _assert_scope_unavailable_completion(
        self,
        repository,
        router,
        hybrid,
        *,
        query: str,
        routing_key: str,
        forbidden: tuple[str, ...],
    ) -> None:
        expected = repository.search_knowledge_chunks(query, 8)
        records: list[dict[str, object]] = []
        rendered = io.StringIO()
        record_sink = loguru_logger.add(
            lambda message: records.append(dict(message.record)),
            level="INFO",
        )
        rendered_sink = loguru_logger.add(
            rendered,
            format="{message} | {extra}",
            level="INFO",
            backtrace=True,
            diagnose=True,
            colorize=False,
        )
        try:
            with patch.object(router, "search", wraps=router.search) as routed_search:
                actual = repository._search_routed_knowledge_chunks(query, 8, routing_key)
        finally:
            loguru_logger.remove(record_sink)
            loguru_logger.remove(rendered_sink)

        self.assertEqual(actual, expected)
        routed_search.assert_not_called()
        self.assertEqual(hybrid.calls, 0)
        completions = [record for record in records if record["message"] == "retrieval completed"]
        self.assertEqual(len(completions), 1)
        extra = completions[0]["extra"]
        self.assertEqual(extra["mode"], "qwen3")
        self.assertEqual(extra["fallback_code"], "retrieval_scope_unavailable")
        self.assertEqual(extra["fallback_reason"], "retrieval_scope_unavailable")
        self.assertEqual(extra["candidate_counts"], {"qwen": 0, "legacy": len(expected)})
        self.assertEqual(extra["result_count"], len(expected))
        self.assertEqual(extra["stage_timings"], extra["stage_timings_ms"])
        self.assertIn("legacy", extra["stage_timings"])
        output = rendered.getvalue()
        self.assertNotIn(query, output)
        for detail in forbidden:
            self.assertNotIn(detail, output)


if __name__ == "__main__":
    unittest.main()
