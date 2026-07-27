from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import patch

import httpx

from app.reranker_client import (
    RerankerBusy,
    RerankerModelMismatch,
    RerankerResponseError,
    RerankerServiceError,
    SyncHttpRerankerClient,
)
from app.reranker_contracts import MAX_RERANK_REQUEST_BYTES, RerankerModelMetadata


def metadata() -> RerankerModelMetadata:
    return RerankerModelMetadata("qwen-reranker", "1", "a" * 64, "b" * 64, "1")


class FakeSyncTransport:
    def __init__(
        self,
        *,
        scores: list[float] | None = None,
        overrides: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.scores = scores
        self.overrides = overrides or {}
        self.error = error
        self.calls: list[tuple[str, dict[str, Any], float | None]] = []
        self.close_calls = 0

    def post_json(
        self, url: str, payload: dict[str, Any], *, timeout_seconds: float | None = None
    ) -> dict[str, Any]:
        self.calls.append((url, payload, timeout_seconds))
        if self.error is not None:
            raise self.error
        scores = self.scores
        if scores is None or len(scores) != len(payload["passages"]):
            scores = [1.0 - index / 100 for index in range(len(payload["passages"]))]
        response: dict[str, Any] = {
            "modelName": "qwen-reranker",
            "modelVersion": "1",
            "modelChecksum": "a" * 64,
            "promptProfileSha256": "b" * 64,
            "protocolVersion": "1",
            "passageCount": len(payload["passages"]),
            "scores": scores,
        }
        response.update(self.overrides)
        return response

    def close(self) -> None:
        self.close_calls += 1


class StatusTransport(FakeSyncTransport):
    def __init__(self, status_code: int) -> None:
        request = httpx.Request("POST", "http://reranker-service:8082/v1/rerank")
        response = httpx.Response(status_code, request=request, text="secret passage")
        super().__init__(error=httpx.HTTPStatusError("failure", request=request, response=response))


class RecordingHttpxClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.close_calls = 0

    def post(self, url: str, json: dict[str, Any], **kwargs: Any) -> httpx.Response:
        self.requests.append((url, json))
        response = FakeSyncTransport().post_json(url, json)
        return httpx.Response(200, request=httpx.Request("POST", url), json=response)

    def close(self) -> None:
        self.close_calls += 1


class RerankerClientTest(unittest.TestCase):
    def test_scores_pairs_and_preserves_order(self) -> None:
        transport = FakeSyncTransport(scores=[0.8, 0.2])
        client = SyncHttpRerankerClient("http://reranker-service:8082", transport=transport)

        scores = client.rerank("policy", ["required policy", "unrelated"], expected=metadata())

        self.assertEqual(scores, [0.8, 0.2])
        self.assertEqual(transport.calls[0][0], "http://reranker-service:8082/v1/rerank")

    def test_splits_by_count_and_payload_bounds(self) -> None:
        transport = FakeSyncTransport()
        client = SyncHttpRerankerClient("http://reranker-service:8082", transport=transport)

        passages = ["x" * 16000 for _ in range(33)]
        scores = client.rerank("q", passages, expected=metadata())

        self.assertEqual(len(scores), 33)
        self.assertGreater(len(transport.calls), 1)
        for _, payload, _ in transport.calls:
            self.assertLessEqual(len(payload["passages"]), 32)
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            self.assertLessEqual(len(encoded), MAX_RERANK_REQUEST_BYTES)

    def test_rejects_public_urls_and_invalid_input_without_calls(self) -> None:
        for url in (
            "https://public.example",
            "http://postgres:8082",
            "http://0.0.0.0:8082",
            "http://reranker-service:8082/unexpected",
            "http://user:secret@reranker-service:8082",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                SyncHttpRerankerClient(url)

        transport = FakeSyncTransport()
        client = SyncHttpRerankerClient("http://reranker-service:8082", transport=transport)
        for query, passages in (("", ["p"]), ("q", []), ("q", [""])):
            with self.subTest(query=query, passages=passages), self.assertRaises(ValueError):
                client.rerank(query, passages, expected=metadata())
        self.assertEqual(transport.calls, [])

    def test_rejects_each_metadata_mismatch_without_retry(self) -> None:
        overrides = (
            {"modelName": "other"},
            {"modelVersion": "2"},
            {"modelChecksum": "c" * 64},
            {"promptProfileSha256": "d" * 64},
            {"protocolVersion": "2"},
        )
        for override in overrides:
            with self.subTest(override=override):
                transport = FakeSyncTransport(overrides=override)
                client = SyncHttpRerankerClient("http://reranker-service:8082", transport=transport)
                with self.assertRaises(RerankerModelMismatch):
                    client.rerank("q", ["p"], expected=metadata())
                self.assertEqual(len(transport.calls), 1)

    def test_maps_429_to_busy_and_transport_failures_to_unavailable(self) -> None:
        cases = (
            (StatusTransport(429), RerankerBusy),
            (StatusTransport(503), RerankerServiceError),
            (
                FakeSyncTransport(error=httpx.ConnectError("private", request=None)),
                RerankerServiceError,
            ),
            (
                FakeSyncTransport(error=httpx.ReadTimeout("private", request=None)),
                RerankerServiceError,
            ),
        )
        for transport, error_type in cases:
            with self.subTest(error_type=error_type.__name__):
                client = SyncHttpRerankerClient("http://reranker-service:8082", transport=transport)
                with self.assertRaises(error_type) as caught:
                    client.rerank("q", ["p"], expected=metadata())
                self.assertNotIn("private", str(caught.exception))
                self.assertNotIn("secret passage", str(caught.exception))

    def test_rejects_malformed_response_and_score_count(self) -> None:
        for override in (
            {"scores": [float("inf")]},
            {"passageCount": 2},
            {"extra": "invalid"},
        ):
            with self.subTest(override=override):
                client = SyncHttpRerankerClient(
                    "http://reranker-service:8082",
                    transport=FakeSyncTransport(overrides=override),
                )
                with self.assertRaises(RerankerResponseError):
                    client.rerank("q", ["p"], expected=metadata())

    def test_persistent_httpx_client_config_and_idempotent_close(self) -> None:
        http_client = RecordingHttpxClient()
        with patch("app.reranker_client.httpx.Client", return_value=http_client) as factory:
            client = SyncHttpRerankerClient("http://reranker-service:8082", timeout_seconds=4)
            self.assertEqual(client.rerank("q", ["p"], expected=metadata()), [1.0])
            self.assertEqual(client.rerank("q2", ["p2"], expected=metadata()), [1.0])
            client.close()
            client.close()

        self.assertEqual(factory.call_count, 1)
        self.assertEqual(len(http_client.requests), 2)
        self.assertEqual(http_client.close_calls, 1)
        kwargs = factory.call_args.kwargs
        self.assertFalse(kwargs["follow_redirects"])
        self.assertFalse(kwargs["trust_env"])
        self.assertEqual(kwargs["limits"].max_connections, 32)
        self.assertEqual(kwargs["limits"].max_keepalive_connections, 16)

    def test_client_passes_per_request_timeout_to_transport(self) -> None:
        transport = FakeSyncTransport()
        client = SyncHttpRerankerClient(
            "http://reranker-service:8082",
            transport=transport,
        )

        client.rerank("q", ["p"], expected=metadata(), timeout_seconds=0.25)

        self.assertEqual(transport.calls[0][2], 0.25)


if __name__ == "__main__":
    unittest.main()
