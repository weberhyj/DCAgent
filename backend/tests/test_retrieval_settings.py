from __future__ import annotations

import unittest

from app.embedding_fingerprint import EmbeddingFingerprint
from app.retrieval_models import RetrievalCandidate, RetrievalMode, RetrievalOutcome, RetrievalScope
from app.retrieval_settings import RetrievalSettings


def private_hybrid_environment() -> dict[str, str]:
    checksum = "a" * 64
    return {
        "RETRIEVAL_MODE": "qwen3",
        "RETRIEVAL_PERMISSION_TAGS": "internal, restricted",
        "EMBEDDING_SERVICE_URL": "http://embedding-service:8081",
        "RERANKER_SERVICE_URL": "http://reranker-service:8082",
        "QDRANT_URL": "http://qdrant:6333",
        "EMBEDDING_MODEL_NAME": "qwen2.5:0.5b",
        "EMBEDDING_MODEL_VERSION": "1.0.0",
        "EMBEDDING_MODEL_SHA256": checksum,
        "EMBEDDING_MODEL_DIMENSIONS": "896",
        "EMBEDDING_MODEL_NORMALIZED": "true",
        "EMBEDDING_ENCODING_PROFILE_SHA256": checksum,
        "EMBEDDING_PROTOCOL_VERSION": "v1",
        "RERANKER_MODEL_NAME": "qwen2.5:3b",
        "RERANKER_MODEL_VERSION": "1.0.0",
        "RERANKER_MODEL_SHA256": checksum,
        "RERANKER_PROMPT_PROFILE_SHA256": checksum,
        "RERANKER_PROTOCOL_VERSION": "v1",
    }


class RetrievalSettingsTest(unittest.TestCase):
    def test_exposes_immutable_embedding_fingerprint_for_configured_embedding(self) -> None:
        settings = RetrievalSettings.from_environ(private_hybrid_environment())

        self.assertEqual(
            settings.embedding_fingerprint,
            EmbeddingFingerprint.from_metadata(settings.embedding),
        )

    def test_hybrid_route_defaults_accept_qwen25_models(self) -> None:
        settings = RetrievalSettings.from_environ(private_hybrid_environment())

        self.assertEqual(settings.mode, RetrievalMode.QWEN3)
        self.assertEqual(settings.dense_top_k, 50)
        self.assertEqual(settings.sparse_top_k, 50)
        self.assertEqual(settings.rerank_top_k, 24)
        self.assertEqual(settings.degraded_rerank_top_k, 12)
        self.assertEqual(settings.final_top_k, 8)
        self.assertEqual(settings.rrf_k, 60)
        self.assertEqual(settings.total_timeout_seconds, 5.0)
        self.assertEqual(settings.embedding.name, "qwen2.5:0.5b")
        self.assertEqual(settings.embedding.dimensions, 896)
        self.assertEqual(settings.reranker.name, "qwen2.5:3b")
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
                environ = private_hybrid_environment()
                environ["RETRIEVAL_MODE"] = mode
                self.assertEqual(RetrievalSettings.from_environ(environ).mode, RetrievalMode(mode))
                environ["RETRIEVAL_PERMISSION_TAGS"] = ""
                with self.assertRaisesRegex(ValueError, "RETRIEVAL_PERMISSION_TAGS"):
                    RetrievalSettings.from_environ(environ)

    def test_rejects_public_reranker_url_and_invalid_percentages(self) -> None:
        for key in ("QDRANT_URL", "EMBEDDING_SERVICE_URL", "RERANKER_SERVICE_URL"):
            with self.subTest(key=key):
                environ = private_hybrid_environment()
                environ[key] = "https://public.example"
                with self.assertRaises(ValueError):
                    RetrievalSettings.from_environ(environ)

        for value in ("-1", "101", "nan"):
            with self.subTest(value=value):
                environ = private_hybrid_environment()
                environ["RETRIEVAL_CANARY_PERCENT"] = value
                with self.assertRaises(ValueError):
                    RetrievalSettings.from_environ(environ)

    def test_rejects_invalid_shadow_percent(self) -> None:
        for value in ("-1", "101", "nan"):
            with self.subTest(value=value):
                environ = private_hybrid_environment()
                environ["RETRIEVAL_SHADOW_PERCENT"] = value
                with self.assertRaises(ValueError):
                    RetrievalSettings.from_environ(environ)

    def test_accepts_arbitrary_pinned_model_names_and_positive_dimension(self) -> None:
        environ = private_hybrid_environment()
        environ["EMBEDDING_MODEL_NAME"] = "acme/private-embedding-v4"
        environ["RERANKER_MODEL_NAME"] = "acme/private-reranker-v7"
        environ["EMBEDDING_MODEL_DIMENSIONS"] = "37"

        settings = RetrievalSettings.from_environ(environ)

        self.assertEqual(settings.embedding.name, "acme/private-embedding-v4")
        self.assertEqual(settings.embedding.dimensions, 37)
        self.assertEqual(settings.reranker.name, "acme/private-reranker-v7")

    def test_rejects_blank_model_names(self) -> None:
        for key in ("EMBEDDING_MODEL_NAME", "RERANKER_MODEL_NAME"):
            with self.subTest(key=key):
                environ = private_hybrid_environment()
                environ[key] = "   "
                with self.assertRaisesRegex(ValueError, key):
                    RetrievalSettings.from_environ(environ)

    def test_rejects_non_positive_bool_like_and_noninteger_dimensions(self) -> None:
        for value in ("0", "-1", "true", "false", "1.5", "896.0"):
            with self.subTest(value=value):
                environ = private_hybrid_environment()
                environ["EMBEDDING_MODEL_DIMENSIONS"] = value
                with self.assertRaisesRegex(ValueError, "EMBEDDING_MODEL_DIMENSIONS"):
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
                environ = private_hybrid_environment()
                del environ[key]
                with self.assertRaisesRegex(ValueError, key):
                    RetrievalSettings.from_environ(environ)

    def test_reranker_does_not_require_embedding_only_metadata(self) -> None:
        settings = RetrievalSettings.from_environ(private_hybrid_environment())

        self.assertFalse(hasattr(settings.reranker, "dimensions"))
        self.assertFalse(hasattr(settings.reranker, "normalized"))
        self.assertFalse(hasattr(settings.reranker, "encoding_profile_sha256"))

    def test_rejects_final_top_k_above_degraded_rerank_top_k_in_every_mode(self) -> None:
        for mode in RetrievalMode:
            with self.subTest(mode=mode):
                environ = {"RETRIEVAL_MODE": mode.value}
                if mode is not RetrievalMode.LEGACY:
                    environ = private_hybrid_environment()
                    environ["RETRIEVAL_MODE"] = mode.value
                environ["RETRIEVAL_FINAL_TOP_K"] = "13"
                environ["RETRIEVAL_DEGRADED_RERANK_TOP_K"] = "12"
                with self.assertRaisesRegex(ValueError, "RETRIEVAL_FINAL_TOP_K"):
                    RetrievalSettings.from_environ(environ)

    def test_rejects_degraded_rerank_top_k_above_rerank_top_k_in_every_mode(self) -> None:
        for mode in RetrievalMode:
            with self.subTest(mode=mode):
                environ = {"RETRIEVAL_MODE": mode.value}
                if mode is not RetrievalMode.LEGACY:
                    environ = private_hybrid_environment()
                    environ["RETRIEVAL_MODE"] = mode.value
                environ["RETRIEVAL_DEGRADED_RERANK_TOP_K"] = "25"
                environ["RETRIEVAL_RERANK_TOP_K"] = "24"
                with self.assertRaisesRegex(ValueError, "RETRIEVAL_DEGRADED_RERANK_TOP_K"):
                    RetrievalSettings.from_environ(environ)


class RetrievalModelsTest(unittest.TestCase):
    def test_scope_is_stable_and_fails_closed_without_permissions(self) -> None:
        scope = RetrievalScope(
            knowledge_base_id=" finance ",
            permission_tags=(" internal ",),
            publication_version=" knowledge_chunks_qwen3_v1 ",
        )

        self.assertEqual(scope.knowledge_base_id, "finance")
        self.assertEqual(scope.permission_tags, ("internal",))
        self.assertEqual(scope.publication_version, "knowledge_chunks_qwen3_v1")
        with self.assertRaisesRegex(ValueError, "knowledge_base_id"):
            RetrievalScope(" ", ("internal",), "v1")
        with self.assertRaisesRegex(ValueError, "permission_tags"):
            RetrievalScope("finance", (), "v1")
        with self.assertRaisesRegex(ValueError, "permission_tags"):
            RetrievalScope("finance", (" ",), "v1")
        with self.assertRaisesRegex(ValueError, "publication_version"):
            RetrievalScope("finance", ("internal",), " ")

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
