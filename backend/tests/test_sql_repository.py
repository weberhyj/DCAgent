from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.database import Database
from app.models import KnowledgeChunkModel
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


if __name__ == "__main__":
    unittest.main()
