from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping

from app.ollama_client import OllamaResponseError, OllamaServiceError
from app.ollama_embedding_backend import OllamaEmbeddingBackend


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
                )

                self.assert_response_error(lambda: backend.embed(["text"], purpose="document"))

    def test_rejects_zero_and_non_finite_vector_norms(self) -> None:
        for vector in ([0, 0], [1e308, 1e308]):
            with self.subTest(vector=vector):
                client = RecordingOllamaClient([{"embeddings": [vector]}])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
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

    def test_query_and_document_use_the_same_raw_text(self) -> None:
        for purpose in ("query", "document"):
            with self.subTest(purpose=purpose):
                client = RecordingOllamaClient([{"embeddings": [[1, 0]]}])
                backend = OllamaEmbeddingBackend(
                    client,
                    model="qwen2.5",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="10m",
                )

                backend.embed(["raw text"], purpose=purpose)

                self.assertEqual(client.calls[0][1]["input"], ["raw text"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
