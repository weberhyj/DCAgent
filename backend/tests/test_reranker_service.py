from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.qwen3_reranker_runtime import Qwen3RerankerMalformedOutput
from app.reranker_contracts import RerankerModelMetadata
from app.reranker_service import create_reranker_app


class Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [1.0 if "good" in p else 0.0 for p in passages]


class FailingBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if query == "q":
            raise RuntimeError("secret model failure")
        return super().rerank(query, passages)


class MalformedBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if query == "q":
            return [float("nan") for _ in passages]
        return super().rerank(query, passages)


class AlwaysFailingBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        raise RuntimeError("secret model failure")


class AlwaysMalformedBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [float("nan") for _ in passages]


class MalformedRuntimeBackend(Backend):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if pairs[0][0] == "q":
            raise Qwen3RerankerMalformedOutput("secret malformed model logits")
        return [0.5 for _ in pairs]


class GenericValueErrorBackend(Backend):
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if query == "q":
            raise ValueError("secret tokenizer/session failure")
        return super().rerank(query, passages)


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
        cases = (
            (FailingBackend(), 503),
            (MalformedBackend(), 500),
            (MalformedRuntimeBackend(), 500),
            (GenericValueErrorBackend(), 503),
        )
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

    def test_readiness_is_advertised_only_after_startup_self_test(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        backend = Backend()
        app = create_reranker_app(backend, metadata)
        self.assertFalse(app.state.reranker_ready)

        with TestClient(app) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)
            self.assertEqual(len(backend.calls), 2)

        self.assertFalse(app.state.reranker_ready)

    def test_startup_self_test_failure_prevents_readiness(self) -> None:
        metadata = RerankerModelMetadata("qwen", "1", "a" * 64, "b" * 64, "1")
        for backend in (AlwaysFailingBackend(), AlwaysMalformedBackend()):
            with self.subTest(backend=type(backend).__name__):
                app = create_reranker_app(backend, metadata)
                with self.assertRaisesRegex(
                    RuntimeError, "reranker backend startup self-test failed"
                ):
                    with TestClient(app):
                        pass
                self.assertFalse(app.state.reranker_ready)
