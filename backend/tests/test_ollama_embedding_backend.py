from __future__ import annotations

import hashlib
import math
import unittest
from collections.abc import Callable, Mapping

from app import ollama_embedding_backend
from app.ollama_client import OllamaResponseError, OllamaServiceError
from app.ollama_embedding_backend import OllamaEmbeddingBackend

RAW_PREFIX_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
BGE_PREFIX_SHA256 = "2bb658b7e092d6b4b1dbde4c3fc5f281f9ed9f1ace5b49566fb8b10f57836e48"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def expected_encoding_profile(path: str, query_profile: str) -> str:
    prefix_sha256 = RAW_PREFIX_SHA256 if query_profile == "raw" else BGE_PREFIX_SHA256
    query_mode = "raw_text" if query_profile == "raw" else "prefixed_text"
    endpoint_lines = (
        ("input=transformed_text_batch", "truncate=true", "output.count=one_per_input")
        if path == "/api/embed"
        else ("prompt=single_transformed_text", "output.count=one_per_input")
    )
    return "\n".join(
        (
            "profile=dc-agent.ollama.embedding",
            "protocol=dc-agent.ollama.embedding.v2",
            f"purpose.query={query_mode}",
            f"purpose.query.profile={query_profile}",
            f"purpose.query.prefix_sha256={prefix_sha256}",
            "purpose.document=raw_text",
            f"path={path}",
            *endpoint_lines,
            "output.dimensions=configured_exact",
            "output.coordinates=finite_numeric",
            "output.vector=nonzero",
            "normalization.algorithm=max_abs_scaled_l2",
            "normalization.output=unit_l2",
        )
    )


class RecordingOllamaClient:
    def __init__(
        self,
        responses: list[Mapping[str, object]],
        *,
        error: Exception | None = None,
    ) -> None:
        self.responses = list(responses)
        self.error = error
        self.calls: list[tuple[str, object]] = []
        self.close_attempts = 0
        self.close_calls = 0

    def post_json(self, path: str, payload: object) -> Mapping[str, object]:
        self.calls.append((path, payload))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)

    def close(self) -> None:
        self.close_attempts += 1
        if self.close_calls:
            return
        self.close_calls += 1


class OllamaEmbeddingBackendTest(unittest.TestCase):
    def test_endpoint_and_query_profiles_are_canonical_distinct_and_hash_derived(
        self,
    ) -> None:
        hashes = set()
        for path in ("/api/embed", "/api/embeddings"):
            for query_profile in ("raw", "bge-large-zh-v1.5"):
                with self.subTest(path=path, query_profile=query_profile):
                    expected_profile = expected_encoding_profile(path, query_profile)
                    profile = ollama_embedding_backend.ollama_embedding_encoding_profile(
                        path, query_profile
                    )
                    profile_hash = (
                        ollama_embedding_backend.ollama_embedding_encoding_profile_sha256(
                            path, query_profile
                        )
                    )
                    self.assertEqual(profile, expected_profile)
                    self.assertTrue(profile.isascii())
                    self.assertNotIn("\r", profile)
                    self.assertFalse(profile.endswith("\n"))
                    self.assertTrue(
                        all(
                            line.count("=") == 1 and line.split("=", 1)[0]
                            for line in profile.split("\n")
                        )
                    )
                    self.assertEqual(
                        profile_hash,
                        hashlib.sha256(expected_profile.encode("utf-8")).hexdigest(),
                    )
                    hashes.add(profile_hash)
        self.assertEqual(len(hashes), 4)

    def test_constructor_rejects_unknown_query_profiles(self) -> None:
        for query_profile in ("", "   ", "BGE-LARGE-ZH-V1.5", "unknown"):
            with self.subTest(query_profile=query_profile):
                client = RecordingOllamaClient([])
                with self.assertRaisesRegex(ValueError, "query profile"):
                    OllamaEmbeddingBackend(
                        client,
                        model="bge-large-zh-v1.5:latest",
                        path="/api/embed",
                        dimensions=2,
                        keep_alive="10m",
                        query_profile=query_profile,
                    )
                self.assertEqual(client.calls, [])

    def assert_response_error(self, operation: Callable[[], object]) -> None:
        try:
            operation()
        except Exception as error:
            self.assertIsInstance(error, OllamaResponseError)
        else:
            self.fail("OllamaResponseError not raised")

    def test_constructor_rejects_empty_model_and_keep_alive(self) -> None:
        client = RecordingOllamaClient([])
        for field, value in (
            ("model", ""),
            ("model", "   "),
            ("model", None),
            ("keep_alive", ""),
            ("keep_alive", "\t"),
            ("keep_alive", 10),
        ):
            kwargs: dict[str, object] = {
                "model": "qwen2.5",
                "path": "/api/embed",
                "dimensions": 2,
                "keep_alive": "10m",
                "query_profile": "raw",
            }
            kwargs[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                OllamaEmbeddingBackend(client, **kwargs)  # type: ignore[arg-type]

    def test_close_delegates_to_the_idempotent_client(self) -> None:
        client = RecordingOllamaClient([])
        backend = OllamaEmbeddingBackend(
            client,
            model="qwen2.5",
            path="/api/embed",
            dimensions=2,
            keep_alive="10m",
            query_profile="raw",
        )

        self.assertTrue(hasattr(backend, "close"), "backend must expose close()")
        backend.close()
        backend.close()

        self.assertEqual(client.close_attempts, 2)
        self.assertEqual(client.close_calls, 1)

    def test_propagates_sanitized_client_errors_unchanged(self) -> None:
        expected = OllamaServiceError("Ollama service request failed")
        client = RecordingOllamaClient([], error=expected)
        backend = OllamaEmbeddingBackend(
            client,
            model="qwen2.5",
            path="/api/embed",
            dimensions=2,
            keep_alive="10m",
            query_profile="raw",
        )

        with self.assertRaises(OllamaServiceError) as raised:
            backend.embed(["text"], purpose="query")

        self.assertIs(raised.exception, expected)

    def test_constructor_rejects_invalid_dimensions(self) -> None:
        client = RecordingOllamaClient([])
        for dimensions in (0, -1, True, False, 1.5, "2", None):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=dimensions,  # type: ignore[arg-type]
                    keep_alive="10m",
                    query_profile="raw",
                )

    def test_constructor_accepts_only_supported_paths(self) -> None:
        client = RecordingOllamaClient([])
        for path in ("", "/api/embed/", "/api/other", "api/embed", None, []):
            with self.subTest(path=path), self.assertRaises(ValueError):
                OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path=path,  # type: ignore[arg-type]
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

    def test_embed_rejects_invalid_text_sequences_before_calling_client(self) -> None:
        for texts in ("text", b"text", [], (), [""], ["   "], [1], ["ok", None]):
            with self.subTest(texts=texts):
                client = RecordingOllamaClient([{"embeddings": [[1, 0]]}])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                with self.assertRaises(ValueError):
                    backend.embed(texts, purpose="document")  # type: ignore[arg-type]

                self.assertEqual(client.calls, [])

    def test_embed_rejects_invalid_purpose_before_calling_client(self) -> None:
        for purpose in ("", "search", None, True, []):
            with self.subTest(purpose=purpose):
                client = RecordingOllamaClient([{"embeddings": [[1, 0]]}])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                with self.assertRaises(ValueError):
                    backend.embed(["text"], purpose=purpose)  # type: ignore[arg-type]

                self.assertEqual(client.calls, [])

    def test_modern_endpoint_batches_texts_and_normalizes_vectors(self) -> None:
        client = RecordingOllamaClient([{"embeddings": [[3, 4], [0, 2]]}])
        backend = OllamaEmbeddingBackend(
            client,
            model="qwen2.5",
            path="/api/embed",
            dimensions=2,
            keep_alive="10m",
            query_profile="raw",
        )

        vectors = backend.embed(["alpha", "beta"], purpose="document")

        self.assertEqual(vectors, [[0.6, 0.8], [0.0, 1.0]])
        self.assertEqual(
            client.calls,
            [
                (
                    "/api/embed",
                    {
                        "model": "qwen2.5",
                        "input": ["alpha", "beta"],
                        "truncate": True,
                        "keep_alive": "10m",
                    },
                )
            ],
        )

    def test_modern_endpoint_requires_one_vector_per_text(self) -> None:
        for embeddings in ([[1, 0]], [[1, 0], [0, 1], [1, 1]]):
            with self.subTest(vector_count=len(embeddings)):
                client = RecordingOllamaClient([{"embeddings": embeddings}])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                self.assert_response_error(
                    lambda: backend.embed(["first", "second"], purpose="document")
                )

    def test_modern_endpoint_rejects_missing_or_malformed_embedding_containers(self) -> None:
        responses: tuple[Mapping[str, object], ...] = (
            {},
            {"embeddings": None},
            {"embeddings": {"row": [1, 0]}},
            {"embeddings": "x"},
            {"embeddings": [None]},
            {"embeddings": [{"coordinate": 1}]},
            {"embeddings": ["10"]},
        )
        for response in responses:
            with self.subTest(response=response):
                client = RecordingOllamaClient([response])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                self.assert_response_error(lambda: backend.embed(["text"], purpose="document"))

    def test_legacy_endpoint_rejects_missing_or_malformed_embedding_containers(self) -> None:
        responses: tuple[Mapping[str, object], ...] = (
            {},
            {"embedding": None},
            {"embedding": {"coordinate": 1}},
            {"embedding": "10"},
        )
        for response in responses:
            with self.subTest(response=response):
                client = RecordingOllamaClient([response])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embeddings",
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                self.assert_response_error(lambda: backend.embed(["text"], purpose="query"))

    def test_rejects_vectors_with_wrong_dimensions(self) -> None:
        for path, response in (
            ("/api/embed", {"embeddings": [[1, 2, 3]]}),
            ("/api/embeddings", {"embedding": [1]}),
        ):
            with self.subTest(path=path):
                client = RecordingOllamaClient([response])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path=path,
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                self.assert_response_error(lambda: backend.embed(["text"], purpose="document"))

    def test_rejects_boolean_non_numeric_and_non_finite_coordinates(self) -> None:
        vectors = (
            [True, 0],
            [False, 1],
            ["1", 0],
            [None, 1],
            [float("nan"), 1],
            [float("inf"), 1],
            [float("-inf"), 1],
        )
        for vector in vectors:
            with self.subTest(vector=vector):
                client = RecordingOllamaClient([{"embeddings": [vector]}])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                self.assert_response_error(lambda: backend.embed(["text"], purpose="document"))

    def test_rejects_zero_vector_norms(self) -> None:
        for vector in ([0, 0], [-0.0, 0.0]):
            with self.subTest(vector=vector):
                client = RecordingOllamaClient([{"embeddings": [vector]}])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                self.assert_response_error(lambda: backend.embed(["text"], purpose="document"))

    def test_normalizes_large_and_tiny_finite_vectors_stably(self) -> None:
        cases = (
            ([1e308, 1e308], [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]),
            ([1e-308, 0.0], [1.0, 0.0]),
        )
        for vector, expected in cases:
            with self.subTest(vector=vector):
                client = RecordingOllamaClient([{"embeddings": [vector]}])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
                    query_profile="raw",
                )

                try:
                    actual = backend.embed(["text"], purpose="document")[0]
                except OllamaResponseError as error:
                    self.fail(f"finite nonzero vector was rejected: {error}")

                for coordinate, expected_coordinate in zip(actual, expected, strict=True):
                    self.assertAlmostEqual(coordinate, expected_coordinate, places=15)

    def test_translates_huge_integer_conversion_overflow_to_response_error(self) -> None:
        huge_integer = 10**400
        client = RecordingOllamaClient([{"embeddings": [[huge_integer, 1]]}])
        backend = OllamaEmbeddingBackend(
            client,
            model="qwen2.5",
            path="/api/embed",
            dimensions=2,
            keep_alive="10m",
            query_profile="raw",
        )

        self.assert_response_error(lambda: backend.embed(["text"], purpose="document"))

    def test_legacy_endpoint_sends_one_request_per_text(self) -> None:
        client = RecordingOllamaClient(
            [
                {"embedding": [1, 0]},
                {"embedding": [0, 5]},
            ]
        )
        backend = OllamaEmbeddingBackend(
            client,
            model="qwen2.5",
            path="/api/embeddings",
            dimensions=2,
            keep_alive="30m",
            query_profile="raw",
        )

        vectors = backend.embed(["first", "second"], purpose="query")

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(
            client.calls,
            [
                (
                    "/api/embeddings",
                    {"model": "qwen2.5", "prompt": "first", "keep_alive": "30m"},
                ),
                (
                    "/api/embeddings",
                    {"model": "qwen2.5", "prompt": "second", "keep_alive": "30m"},
                ),
            ],
        )

    def test_query_and_document_use_the_same_raw_text_for_both_endpoints(self) -> None:
        for path, response, payload_field in (
            ("/api/embed", {"embeddings": [[1, 0]]}, "input"),
            ("/api/embeddings", {"embedding": [1, 0]}, "prompt"),
        ):
            for purpose in ("query", "document"):
                with self.subTest(path=path, purpose=purpose):
                    client = RecordingOllamaClient([response])
                    backend = OllamaEmbeddingBackend(
                        client,
                        model="qwen2.5",
                        path=path,
                        dimensions=2,
                        keep_alive="10m",
                        query_profile="raw",
                    )

                    backend.embed(["raw text"], purpose=purpose)

                    expected_text: object = ["raw text"] if path == "/api/embed" else "raw text"
                    self.assertEqual(client.calls[0][1][payload_field], expected_text)  # type: ignore[index]

    def test_bge_profile_prefixes_only_query_text_for_both_endpoints(self) -> None:
        for path, response, payload_field in (
            ("/api/embed", {"embeddings": [[1, 0]]}, "input"),
            ("/api/embeddings", {"embedding": [1, 0]}, "prompt"),
        ):
            for purpose, expected_text in (
                ("query", f"{BGE_QUERY_PREFIX}原始查询"),
                ("document", "原始查询"),
            ):
                with self.subTest(path=path, purpose=purpose):
                    client = RecordingOllamaClient([response])
                    backend = OllamaEmbeddingBackend(
                        client,
                        model="bge-large-zh-v1.5:latest",
                        path=path,
                        dimensions=2,
                        keep_alive="10m",
                        query_profile="bge-large-zh-v1.5",
                    )
                    backend.embed(["原始查询"], purpose=purpose)
                    payload = client.calls[0][1]
                    assert isinstance(payload, Mapping)
                    expected_payload_value: object = (
                        [expected_text] if path == "/api/embed" else expected_text
                    )
                    self.assertEqual(payload[payload_field], expected_payload_value)


if __name__ == "__main__":
    unittest.main()
