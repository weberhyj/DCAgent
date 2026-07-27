from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.reranker_contracts import RerankerModelMetadata
from app.reranker_service import create_reranker_app


class Backend:
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [1.0 if "good" in p else 0.0 for p in passages]


class FailingBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        raise ValueError("secret model failure")


class MalformedBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [float("nan") for _ in passages]


class RerankerServiceTest(unittest.TestCase):
    def test_rerank_and_metadata(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        with TestClient(create_reranker_app(Backend(), metadata)) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)
            response = client.post("/v1/rerank", json={"query": "q", "passages": ["good", "bad"]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["scores"], [1.0, 0.0])

    def test_backend_failures_are_sanitized_and_malformed_scores_are_500(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        cases = ((FailingBackend(), 503), (MalformedBackend(), 500))
        for backend, status in cases:
            with (
                self.subTest(status=status),
                TestClient(
                    create_reranker_app(backend, metadata), raise_server_exceptions=False
                ) as client,
            ):
                response = client.post("/v1/rerank", json={"query": "q", "passages": ["passage"]})
                self.assertEqual(response.status_code, status)
                self.assertNotIn("secret", response.text)
