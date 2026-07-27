from __future__ import annotations

import unittest

from app.retrieval_models import RetrievalCandidate, RetrievalMode, RetrievalOutcome, RetrievalScope
from app.retrieval_settings import RetrievalSettings


def private_qwen_environment() -> dict[str, str]:
    checksum = "a" * 64
    return {
        "RETRIEVAL_MODE": "qwen3",
        "RETRIEVAL_PERMISSION_TAGS": "internal, restricted",
        "EMBEDDING_SERVICE_URL": "http://embedding-service:8081",
        "RERANKER_SERVICE_URL": "http://reranker-service:8082",
        "QDRANT_URL": "http://qdrant:6333",
        "EMBEDDING_MODEL_NAME": "Qwen/Qwen3-Embedding-0.6B",
        "EMBEDDING_MODEL_VERSION": "1.0.0",
        "EMBEDDING_MODEL_SHA256": checksum,
        "EMBEDDING_MODEL_DIMENSIONS": "1024",
        "EMBEDDING_MODEL_NORMALIZED": "true",
        "EMBEDDING_ENCODING_PROFILE_SHA256": checksum,
        "EMBEDDING_PROTOCOL_VERSION": "v1",
        "RERANKER_MODEL_NAME": "Qwen/Qwen3-Reranker-0.6B",
        "RERANKER_MODEL_VERSION": "1.0.0",
        "RERANKER_MODEL_SHA256": checksum,
        "RERANKER_PROMPT_PROFILE_SHA256": checksum,
        "RERANKER_PROTOCOL_VERSION": "v1",
    }


class RetrievalSettingsTest(unittest.TestCase):
    def test_qwen3_defaults_match_approved_design(self) -> None:
        settings = RetrievalSettings.from_environ(private_qwen_environment())

        self.assertEqual(settings.mode, RetrievalMode.QWEN3)
        self.assertEqual(settings.dense_top_k, 50)
        self.assertEqual(settings.sparse_top_k, 50)
        self.assertEqual(settings.rerank_top_k, 24)
        self.assertEqual(settings.degraded_rerank_top_k, 12)
        self.assertEqual(settings.final_top_k, 8)
        self.assertEqual(settings.rrf_k, 60)
        self.assertEqual(settings.total_timeout_seconds, 5.0)
        self.assertEqual(settings.embedding.name, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(settings.embedding.dimensions, 1024)
        self.assertEqual(settings.reranker.name, "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(settings.reranker.prompt_profile_sha256, "a" * 64)
        self.assertEqual(settings.knowledge_base_id, "default")
        self.assertEqual(settings.permission_tags, ("internal", "restricted"))
        self.assertEqual(settings.qdrant_collection_alias, "knowledge_chunks_current")

    def test_legacy_does_not_require_qwen_services(self) -> None:
        settings = RetrievalSettings.from_environ({"RETRIEVAL_MODE": "legacy"})

        self.assertEqual(settings.mode, RetrievalMode.LEGACY)
        self.assertIsNone(settings.embedding)
        self.assertIsNone(settings.reranker)
        self.assertEqual(settings.permission_tags, ())

    def test_shadow_and_qwen3_require_private_services_and_permission_tags(self) -> None:
        for mode in ("shadow", "qwen3"):
            with self.subTest(mode=mode):
                environ = private_qwen_environment()
                environ["RETRIEVAL_MODE"] = mode
                self.assertEqual(RetrievalSettings.from_environ(environ).mode, RetrievalMode(mode))
                environ["RETRIEVAL_PERMISSION_TAGS"] = ""
                with self.assertRaisesRegex(ValueError, "RETRIEVAL_PERMISSION_TAGS"):
                    RetrievalSettings.from_environ(environ)

    def test_rejects_public_reranker_url_and_invalid_percentages(self) -> None:
        for key in ("QDRANT_URL", "EMBEDDING_SERVICE_URL", "RERANKER_SERVICE_URL"):
            with self.subTest(key=key):
                environ = private_qwen_environment()
                environ[key] = "https://public.example"
                with self.assertRaises(ValueError):
                    RetrievalSettings.from_environ(environ)

        for value in ("-1", "101", "nan"):
            with self.subTest(value=value):
                environ = private_qwen_environment()
                environ["RETRIEVAL_CANARY_PERCENT"] = value
                with self.assertRaises(ValueError):
                    RetrievalSettings.from_environ(environ)

    def test_rejects_invalid_shadow_percent_and_fixed_model_identities(self) -> None:
        for value in ("-1", "101", "nan"):
            with self.subTest(value=value):
                environ = private_qwen_environment()
                environ["RETRIEVAL_SHADOW_PERCENT"] = value
                with self.assertRaises(ValueError):
                    RetrievalSettings.from_environ(environ)

        for key, value in (
            ("EMBEDDING_MODEL_NAME", "other-model"),
            ("RERANKER_MODEL_NAME", "other-model"),
            ("EMBEDDING_MODEL_DIMENSIONS", "768"),
        ):
            with self.subTest(key=key):
                environ = private_qwen_environment()
                environ[key] = value
                with self.assertRaises(ValueError):
                    RetrievalSettings.from_environ(environ)

    def test_requires_all_pinned_model_metadata_for_qwen_modes(self) -> None:
        for key in (
            "EMBEDDING_MODEL_NAME",
            "EMBEDDING_MODEL_VERSION",
            "EMBEDDING_MODEL_SHA256",
            "EMBEDDING_MODEL_DIMENSIONS",
            "EMBEDDING_MODEL_NORMALIZED",
            "EMBEDDING_ENCODING_PROFILE_SHA256",
            "EMBEDDING_PROTOCOL_VERSION",
            "RERANKER_MODEL_VERSION",
            "RERANKER_MODEL_NAME",
            "RERANKER_MODEL_SHA256",
            "RERANKER_PROMPT_PROFILE_SHA256",
            "RERANKER_PROTOCOL_VERSION",
        ):
            with self.subTest(key=key):
                environ = private_qwen_environment()
                del environ[key]
                with self.assertRaisesRegex(ValueError, key):
                    RetrievalSettings.from_environ(environ)

    def test_reranker_does_not_require_embedding_only_metadata(self) -> None:
        settings = RetrievalSettings.from_environ(private_qwen_environment())

        self.assertFalse(hasattr(settings.reranker, "dimensions"))
        self.assertFalse(hasattr(settings.reranker, "normalized"))
        self.assertFalse(hasattr(settings.reranker, "encoding_profile_sha256"))


class RetrievalModelsTest(unittest.TestCase):
    def test_scope_is_stable_and_fails_closed_without_permissions(self) -> None:
        scope = RetrievalScope(
            knowledge_base_id="finance",
            permission_tags=("internal",),
            publication_version="knowledge_chunks_qwen3_v1",
        )

        self.assertEqual(scope.publication_version, "knowledge_chunks_qwen3_v1")
        with self.assertRaisesRegex(ValueError, "knowledge_base_id"):
            RetrievalScope(" ", ("internal",), "v1")
        with self.assertRaisesRegex(ValueError, "permission_tags"):
            RetrievalScope("finance", (), "v1")

    def test_outcome_keeps_retrieval_diagnostics_internal(self) -> None:
        candidate = RetrievalCandidate(
            source_id="source-1",
            source_name="policy.docx",
            source_type="docx",
            classification="internal",
            chunk_id="chunk-1",
            chunk_index=0,
            text="Policy text",
        )
        outcome = RetrievalOutcome(
            mode=RetrievalMode.QWEN3,
            candidates=(candidate,),
            stage_ms={"dense": 12.5},
        )

        self.assertEqual(outcome.candidates[0].chunk_id, "chunk-1")
        self.assertEqual(outcome.stage_ms["dense"], 12.5)


if __name__ == "__main__":
    unittest.main()
