from __future__ import annotations

import unittest

from app.database import Database
from app.retrieval_audit import RetrievalAuditRepository, RetrievalAuditValidationError


class RetrievalAuditRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.database = Database("sqlite+pysqlite:///:memory:")
        self.database.create_schema()
        self.repository = RetrievalAuditRepository(self.database)

    def test_records_publication_and_redacted_shadow_comparison(self) -> None:
        publication = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(publication.id, point_count=12)
        self.repository.mark_publication_active(publication.id, point_count=12)
        self.repository.record_shadow(
            request_id="request-1",
            routing_key_hash="b" * 64,
            query_hash="c" * 64,
            legacy_chunk_ids=("legacy-1",),
            qwen_chunk_ids=("qwen-1",),
            legacy_ms=40.0,
            qwen_ms=220.0,
            status="completed",
        )

        stored = self.repository.list_shadow(limit=1)[0]

        self.assertEqual(stored.query_hash, "c" * 64)
        self.assertFalse(hasattr(stored, "query"))
        self.assertFalse(hasattr(stored, "text"))
        self.assertEqual(stored.legacy_chunk_ids, ("legacy-1",))

    def test_rejects_illegal_publication_transitions(self) -> None:
        publication = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )

        with self.assertRaises(RetrievalAuditValidationError):
            self.repository.mark_publication_active(publication.id, point_count=1)

        self.repository.mark_publication_validated(publication.id, point_count=1)
        with self.assertRaises(RetrievalAuditValidationError):
            self.repository.mark_publication_retired(publication.id)

        self.repository.mark_publication_active(publication.id, point_count=1)
        with self.assertRaises(RetrievalAuditValidationError):
            self.repository.mark_publication_failed(publication.id, "late failure")

    def test_activation_atomically_preserves_active_publication_on_invalid_target(self) -> None:
        first = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        self.repository.mark_publication_validated(first.id, point_count=12)
        self.repository.mark_publication_active(first.id, point_count=12)
        second = self.repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v2",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-2",
            sparse_profile_sha256="b" * 64,
            dimensions=1024,
        )

        with self.assertRaises(RetrievalAuditValidationError):
            self.repository.mark_publication_active(second.id, point_count=13)

        self.assertEqual(self.repository.get_publication(first.id).status, "active")
        self.assertEqual(self.repository.get_publication(second.id).status, "building")

        self.repository.mark_publication_validated(second.id, point_count=13)
        self.repository.mark_publication_active(second.id, point_count=13)
        self.assertEqual(self.repository.get_publication(first.id).status, "retired")
        self.assertEqual(self.repository.get_publication(second.id).status, "active")
        self.assertEqual(
            self.repository.active_publication("knowledge_chunks_current").id, second.id
        )

    def test_rejects_raw_or_malformed_shadow_audit_values(self) -> None:
        valid = {
            "request_id": "request-1",
            "routing_key_hash": "b" * 64,
            "query_hash": "c" * 64,
            "legacy_chunk_ids": ("legacy-1",),
            "qwen_chunk_ids": ("qwen-1",),
            "legacy_ms": 40.0,
            "qwen_ms": 220.0,
            "status": "completed",
        }
        invalid_cases = (
            {"query_hash": "secret passage"},
            {"legacy_ms": -1.0},
            {"fallback_reason": "http://internal-model/error"},
            {"legacy_chunk_ids": ("",)},
            {"status": "secret passage"},
        )

        for override in invalid_cases:
            with self.subTest(override=override):
                with self.assertRaises(RetrievalAuditValidationError):
                    self.repository.record_shadow(**(valid | override))


if __name__ == "__main__":
    unittest.main()
