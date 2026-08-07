from __future__ import annotations

import unittest

from app.llama_cpp_reranker_backend import LlamaCppRerankerBackend


class _Transport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, object]] = []

    def post_json(self, path: str, payload: object) -> object:
        self.calls.append((path, payload))
        return self.response

    def close(self) -> None:
        return None


class LlamaCppRerankerBackendTests(unittest.TestCase):
    def test_converts_documents_and_restores_scores_by_index(self) -> None:
        transport = _Transport(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.2},
                    {"index": 0, "relevance_score": 0.9},
                ]
            }
        )
        backend = LlamaCppRerankerBackend(
            transport,
            base_path="/v1/rerank",
            model="bge-reranker-v2-m3",
        )

        self.assertEqual(backend.rerank("where", ["first", "second"]), [0.9, 0.2])
        self.assertEqual(transport.calls[0][0], "/v1/rerank")
        self.assertEqual(
            transport.calls[0][1],
            {
                "model": "bge-reranker-v2-m3",
                "query": "where",
                "documents": ["first", "second"],
            },
        )

    def test_rejects_duplicate_or_missing_indexes(self) -> None:
        for results in (
            [{"index": 0, "relevance_score": 0.4}, {"index": 0, "relevance_score": 0.3}],
            [{"index": 0, "relevance_score": 0.4}],
        ):
            with self.subTest(results=results):
                backend = LlamaCppRerankerBackend(
                    _Transport({"results": results}),
                    base_path="/v1/rerank",
                    model="bge-reranker-v2-m3",
                )
                with self.assertRaises(ValueError):
                    backend.rerank("where", ["first", "second"])


if __name__ == "__main__":
    unittest.main()
