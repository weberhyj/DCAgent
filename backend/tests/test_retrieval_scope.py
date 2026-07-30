from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from app.embedding_fingerprint import EmbeddingFingerprint
from app.retrieval_scope import DynamicRetrievalScopeProvider

FINGERPRINT = EmbeddingFingerprint(
    model_name="qwen2.5:0.5b",
    model_version="qwen25-embedding-v1",
    model_sha256="a" * 64,
    dimensions=896,
    normalized=True,
    encoding_profile_sha256="b" * 64,
    protocol_version="v1",
)


class RetrievalScopeTest(unittest.TestCase):
    def _resolve(self, publication_fingerprint: object):
        collection = "knowledge_chunks_qwen3_v7"
        provider = DynamicRetrievalScopeProvider(
            audit=SimpleNamespace(
                active_publication=lambda _alias: SimpleNamespace(
                    collection_name=collection,
                    embedding_fingerprint=publication_fingerprint,
                )
            ),
            gateway=SimpleNamespace(resolve_alias=lambda: collection),
            alias_name="knowledge_chunks_current",
            knowledge_base_id="default",
            permission_tags=("internal",),
            embedding_fingerprint=FINGERPRINT,
        )
        return provider.resolve()

    def test_matching_embedding_fingerprint_is_ready(self) -> None:
        resolution = self._resolve(FINGERPRINT)

        self.assertEqual(resolution.detail, "ready")
        self.assertEqual(resolution.collection_name, "knowledge_chunks_qwen3_v7")
        self.assertIsNotNone(resolution.scope)

    def test_same_dimensions_with_any_embedding_identity_mismatch_fails_closed(self) -> None:
        mismatches = (
            replace(FINGERPRINT, model_name="different-model"),
            replace(FINGERPRINT, model_version="old-qwen3-v1"),
            replace(FINGERPRINT, model_sha256="c" * 64),
            replace(FINGERPRINT, normalized=False),
            replace(FINGERPRINT, encoding_profile_sha256="d" * 64),
            replace(FINGERPRINT, protocol_version="v2"),
        )

        for fingerprint in mismatches:
            with self.subTest(fingerprint=fingerprint):
                resolution = self._resolve(fingerprint)
                self.assertIsNone(resolution.scope)
                self.assertIsNone(resolution.collection_name)
                self.assertEqual(resolution.detail, "embedding_fingerprint_mismatch")

    def test_legacy_publication_without_complete_fingerprint_fails_closed(self) -> None:
        resolution = self._resolve(None)

        self.assertIsNone(resolution.scope)
        self.assertEqual(resolution.detail, "embedding_fingerprint_unavailable")


if __name__ == "__main__":
    unittest.main()
