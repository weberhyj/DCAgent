from __future__ import annotations

import unittest

from app.models import ChatState, KnowledgeChunkModel
from app.repository import (
    InMemoryChatRepository,
    is_reliable_knowledge_score,
    resolve_retrieval_min_score,
)


class RetrievalThresholdTest(unittest.TestCase):
    def test_resolves_configurable_minimum_retrieval_score(self) -> None:
        self.assertEqual(resolve_retrieval_min_score({}), 2.2)
        self.assertEqual(resolve_retrieval_min_score({"RETRIEVAL_MIN_SCORE": "3.75"}), 3.75)
        self.assertEqual(resolve_retrieval_min_score({"RETRIEVAL_MIN_SCORE": "-3"}), 0.0)
        self.assertEqual(resolve_retrieval_min_score({"RETRIEVAL_MIN_SCORE": "invalid"}), 2.2)
        for raw_value in ("nan", "inf", "-inf", "1e999"):
            with self.subTest(raw_value=raw_value):
                self.assertEqual(
                    resolve_retrieval_min_score({"RETRIEVAL_MIN_SCORE": raw_value}),
                    2.2,
                )

    def test_rejects_weak_vector_only_evidence(self) -> None:
        self.assertFalse(
            is_reliable_knowledge_score(
                keyword_score=0,
                vector_score=0.25,
                total_score=1.0,
                minimum_score=2.2,
            )
        )
        self.assertTrue(
            is_reliable_knowledge_score(
                keyword_score=2,
                vector_score=0.1,
                total_score=2.4,
                minimum_score=2.2,
            )
        )

    def test_generic_question_words_do_not_create_false_positive_hits(self) -> None:
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        repository.add_uploaded_knowledge_source(
            source_id="kb-security",
            name="系统安全技术白皮书.pdf",
            source_type="PDF",
            classification="内部",
            records=0,
            file_path="security.pdf",
            file_size=128,
            mime_type="application/pdf",
        )
        repository.complete_knowledge_source_indexing(
            "kb-security",
            [
                KnowledgeChunkModel(
                    id="chunk-security-0",
                    source_id="kb-security",
                    chunk_index=0,
                    text="云服务为客户提供安全隔离能力和访问控制策略。",
                    token_count=24,
                )
            ],
        )

        hits = repository.search_knowledge_chunks("公司是否提供火星基地住房补贴", limit=5)

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
