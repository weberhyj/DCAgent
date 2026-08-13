from __future__ import annotations

import unittest

from app.llama_cpp_embedding_backend import LlamaCppEmbeddingBackend


class _Transport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, object]] = []

    def post_json(self, path: str, payload: object) -> object:
        self.calls.append((path, payload))
        return self.responses.pop(0)

    def close(self) -> None:
        return None


class LlamaCppEmbeddingBackendTests(unittest.TestCase):
    def test_sends_openai_embedding_request_and_restores_indexes(self) -> None:
        transport = _Transport(
            [
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 2.0]},
                        {"index": 0, "embedding": [3.0, 4.0]},
                    ]
                }
            ]
        )
        backend = LlamaCppEmbeddingBackend(
            transport,
            base_path="/v1/embeddings",
            model="bge-m3-Q4_K_M.gguf",
            dimensions=2,
            normalized=True,
        )

        vectors = backend.embed(["first", "second"], purpose="document")

        self.assertEqual(vectors, [[0.6, 0.8], [0.0, 1.0]])
        self.assertEqual(
            transport.calls,
            [
                (
                    "/v1/embeddings",
                    {
                        "model": "bge-m3-Q4_K_M.gguf",
                        "input": ["first", "second"],
                    },
                )
            ],
        )

    def test_batches_requests_at_configured_limit(self) -> None:
        transport = _Transport(
            [
                {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
                {"data": [{"index": 0, "embedding": [0.0, 1.0]}]},
            ]
        )
        backend = LlamaCppEmbeddingBackend(
            transport,
            base_path="/v1/embeddings",
            model="bge-m3",
            dimensions=2,
            normalized=False,
            batch_max_items=1,
        )

        self.assertEqual(
            backend.embed(["first", "second"], purpose="query"),
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertEqual(len(transport.calls), 2)

    def test_rejects_malformed_or_inconsistent_results(self) -> None:
        responses = (
            {},
            {"data": []},
            {"data": [{"index": 0, "embedding": [1.0]}]},
            {"data": [{"index": 0, "embedding": [0.0, 0.0]}]},
            {"data": [{"index": 2, "embedding": [1.0, 0.0]}]},
        )
        for response in responses:
            with self.subTest(response=response):
                backend = LlamaCppEmbeddingBackend(
                    _Transport([response]),
                    base_path="/v1/embeddings",
                    model="bge-m3",
                    dimensions=2,
                    normalized=True,
                )
                with self.assertRaises(ValueError):
                    backend.embed(["text"], purpose="query")

    def test_rejects_invalid_arguments_before_transport_call(self) -> None:
        transport = _Transport([])
        for path in ("", "v1/embeddings", "/v1/embeddings/"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                LlamaCppEmbeddingBackend(
                    transport,
                    base_path=path,
                    model="bge-m3",
                    dimensions=2,
                    normalized=True,
                )
        backend = LlamaCppEmbeddingBackend(
            transport,
            base_path="/v1/embeddings",
            model="bge-m3",
            dimensions=2,
            normalized=True,
        )
        for texts, purpose in (([], "query"), ([""], "query"), (["text"], "other")):
            with self.subTest(texts=texts, purpose=purpose), self.assertRaises(ValueError):
                backend.embed(texts, purpose=purpose)  # type: ignore[arg-type]
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
