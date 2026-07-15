from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import patch

import httpx

from app.embedding_client import (
    EmbeddingModelMismatch,
    EmbeddingResponseError,
    EmbeddingServiceError,
    HttpEmbeddingClient,
)
from app.embedding_contracts import EmbeddingModelMetadata


def metadata(*, checksum: str = "a" * 64, dimensions: int = 4) -> EmbeddingModelMetadata:
    return EmbeddingModelMetadata(
        "bge-test",
        "1",
        checksum,
        dimensions,
        True,
        "e" * 64,
        "1",
    )


class FakeTransport:
    def __init__(
        self,
        *,
        checksum: str = "a" * 64,
        encoding_profile_sha256: str = "e" * 64,
        protocol_version: str = "1",
        dimensions: int = 4,
        vector_dimensions: int | None = None,
        vector_count_delta: int = 0,
        error: Exception | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.checksum = checksum
        self.encoding_profile_sha256 = encoding_profile_sha256
        self.protocol_version = protocol_version
        self.dimensions = dimensions
        self.vector_dimensions = vector_dimensions or dimensions
        self.vector_count_delta = vector_count_delta
        self.error = error
        self.overrides = overrides or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def post_json(
        self, url: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append((url, payload))
        if self.error is not None:
            raise self.error
        vector_count = max(0, len(payload["texts"]) + self.vector_count_delta)
        response: dict[str, Any] = {
            "modelName": "bge-test",
            "modelVersion": "1",
            "modelChecksum": self.checksum,
            "dimensions": self.dimensions,
            "normalized": True,
            "encodingProfileSha256": self.encoding_profile_sha256,
            "protocolVersion": self.protocol_version,
            "purpose": payload["purpose"],
            "vectors": [
                [float(index * 10 + coordinate) for coordinate in range(self.vector_dimensions)]
                for index in range(vector_count)
            ],
        }
        response.update(self.overrides)
        return response


class FakeHttpxResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://embedding-service:8081/v1/embeddings")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "upstream failure", request=self.request, response=self
            )

    def json(self) -> dict[str, Any]:
        return self._payload


class RecordingAsyncClient:
    def __init__(self, response: FakeHttpxResponse, **kwargs: Any) -> None:
        self.response = response
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> "RecordingAsyncClient":
        self.entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.exited = True

    async def post(self, url: str, json: dict[str, Any]) -> FakeHttpxResponse:
        self.requests.append((url, json))
        return self.response


class EmbeddingClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_non_embedding_hosts_and_ambiguous_url_parts(self) -> None:
        invalid_urls = (
            "http://postgres:8081",
            "ftp://embedding-service:8081",
            "http://embedding-service:0",
            "http://embedding-service:99999",
            "http://user:secret@embedding-service:8081",
            "http://embedding-service:8081/v1?redirect=public",
            "http://embedding-service:8081/v1#fragment",
            "http://embedding-service:8081/unexpected",
            "http://0.0.0.0:8081",
            "http://169.254.1.1:8081",
            "http://100.64.0.1:8081",
            "http://[fe80::1]:8081",
            "http://[::ffff:10.20.30.40]:8081",
            "http://[fe80::1%25Ethernet]:8081",
        )

        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    HttpEmbeddingClient(url)

        private_client = HttpEmbeddingClient(
            "http://10.20.30.40:8081", transport=FakeTransport()
        )
        vectors = await private_client.embed(
            ["one"], expected=metadata(), purpose="query"
        )
        self.assertEqual(len(vectors), 1)
        for url in ("http://127.0.0.1:8081", "http://[fd00::1]:8081"):
            with self.subTest(private_url=url):
                client = HttpEmbeddingClient(url, transport=FakeTransport())
                self.assertEqual(
                    len(
                        await client.embed(
                            ["one"], expected=metadata(), purpose="query"
                        )
                    ),
                    1,
                )

    async def test_rejects_public_endpoint_and_metadata_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            HttpEmbeddingClient("https://public.example/v1")
        transport = FakeTransport(
            checksum="b" * 64,
            encoding_profile_sha256="e" * 64,
            protocol_version="1",
        )
        client = HttpEmbeddingClient(
            "http://embedding-service:8081", transport=transport
        )

        with self.assertRaises(EmbeddingModelMismatch):
            await client.embed(
                ["one"],
                expected=EmbeddingModelMetadata(
                    "bge-test", "1", "a" * 64, 4, True, "e" * 64, "1"
                ),
                purpose="query",
            )
        self.assertEqual(len(transport.calls), 1)

    async def test_accepts_private_endpoint_and_preserves_flat_vector_order(self) -> None:
        transport = FakeTransport()
        client = HttpEmbeddingClient(
            "http://embedding-service:8081/", transport=transport
        )

        vectors = await client.embed(
            ["one", "two"], expected=metadata(), purpose="document"
        )

        self.assertEqual(vectors[0][0], 0.0)
        self.assertEqual(vectors[1][0], 10.0)
        self.assertEqual(
            transport.calls[0][0],
            "http://embedding-service:8081/v1/embeddings",
        )
        self.assertEqual(transport.calls[0][1]["purpose"], "document")

    async def test_supports_private_base_url_that_already_has_v1_path(self) -> None:
        transport = FakeTransport()
        client = HttpEmbeddingClient(
            "http://embedding-service:8081/v1", transport=transport
        )

        await client.embed(["one"], expected=metadata(), purpose="query")

        self.assertEqual(
            transport.calls[0][0],
            "http://embedding-service:8081/v1/embeddings",
        )

    async def test_splits_requests_by_count_limit(self) -> None:
        transport = FakeTransport()
        client = HttpEmbeddingClient(
            "http://embedding-service:8081", transport=transport
        )

        vectors = await client.embed(
            [f"text-{index}" for index in range(65)],
            expected=metadata(),
            purpose="document",
        )

        self.assertEqual([len(call[1]["texts"]) for call in transport.calls], [64, 1])
        self.assertEqual(len(vectors), 65)

    async def test_splits_requests_by_utf8_payload_limit(self) -> None:
        transport = FakeTransport()
        client = HttpEmbeddingClient(
            "http://embedding-service:8081", transport=transport
        )

        vectors = await client.embed(
            ["x" * 16000 for _ in range(17)],
            expected=metadata(),
            purpose="document",
        )

        self.assertEqual([len(call[1]["texts"]) for call in transport.calls], [16, 1])
        self.assertEqual(len(vectors), 17)
        for _, payload in transport.calls:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self.assertLessEqual(len(encoded), 256 * 1024)

    async def test_rejects_invalid_inputs_without_transport_call(self) -> None:
        invalid_cases: tuple[tuple[Any, Any], ...] = (
            ([], "query"),
            ([""], "query"),
            (["   "], "query"),
            (["x" * 16385], "document"),
            (["one"], "invalid"),
        )

        for texts, purpose in invalid_cases:
            with self.subTest(texts_length=len(texts), purpose=purpose):
                transport = FakeTransport()
                client = HttpEmbeddingClient(
                    "http://embedding-service:8081", transport=transport
                )
                with self.assertRaises(ValueError):
                    await client.embed(
                        texts,
                        expected=metadata(),
                        purpose=purpose,
                    )
                self.assertEqual(transport.calls, [])

    async def test_rejects_vector_count_dimension_and_purpose_mismatches(self) -> None:
        transports = (
            FakeTransport(vector_count_delta=-1),
            FakeTransport(vector_dimensions=3),
            FakeTransport(overrides={"purpose": "document"}),
        )

        for transport in transports:
            with self.subTest(transport=transport):
                client = HttpEmbeddingClient(
                    "http://embedding-service:8081", transport=transport
                )
                with self.assertRaises(EmbeddingResponseError):
                    await client.embed(
                        ["one", "two"], expected=metadata(), purpose="query"
                    )

    async def test_rejects_every_pinned_metadata_field_mismatch_without_retry(self) -> None:
        mismatch_overrides = (
            {"modelName": "other"},
            {"modelVersion": "2"},
            {"modelChecksum": "b" * 64},
            {"dimensions": 3, "vectors": [[0.0, 1.0, 2.0]]},
            {"normalized": False},
            {"encodingProfileSha256": "f" * 64},
            {"protocolVersion": "2"},
        )

        for overrides in mismatch_overrides:
            with self.subTest(overrides=overrides):
                transport = FakeTransport(overrides=overrides)
                client = HttpEmbeddingClient(
                    "http://embedding-service:8081", transport=transport
                )
                with self.assertRaises(EmbeddingModelMismatch):
                    await client.embed(
                        ["one"], expected=metadata(), purpose="query"
                    )
                self.assertEqual(len(transport.calls), 1)

    async def test_rejects_malformed_metadata_and_non_finite_vectors(self) -> None:
        transports = (
            FakeTransport(overrides={"modelChecksum": "not-a-checksum"}),
            FakeTransport(overrides={"vectors": [[0.0, 1.0, 2.0, float("inf")]]}),
        )

        for transport in transports:
            with self.subTest(transport=transport):
                client = HttpEmbeddingClient(
                    "http://embedding-service:8081", transport=transport
                )
                with self.assertRaises(EmbeddingResponseError):
                    await client.embed(
                        ["one"], expected=metadata(), purpose="query"
                    )

    async def test_wraps_transport_failures_without_fallback(self) -> None:
        transport = FakeTransport(error=ConnectionError("private detail"))
        client = HttpEmbeddingClient(
            "http://embedding-service:8081", transport=transport
        )

        with self.assertRaises(EmbeddingServiceError) as error:
            await client.embed(["one"], expected=metadata(), purpose="query")

        self.assertNotIn("private detail", str(error.exception))
        self.assertEqual(len(transport.calls), 1)

    async def test_default_transport_closes_http_client_and_wraps_5xx(self) -> None:
        success_payload = await FakeTransport().post_json(
            "unused", {"texts": ["one"], "purpose": "query"}
        )
        success_client = RecordingAsyncClient(FakeHttpxResponse(success_payload))

        with patch(
            "app.embedding_client.httpx.AsyncClient", return_value=success_client
        ) as async_client_factory:
            vectors = await HttpEmbeddingClient(
                "http://embedding-service:8081"
            ).embed(["one"], expected=metadata(), purpose="query")

        self.assertEqual(len(vectors), 1)
        self.assertTrue(success_client.entered)
        self.assertTrue(success_client.exited)
        self.assertIn("timeout", async_client_factory.call_args.kwargs)
        self.assertFalse(async_client_factory.call_args.kwargs["trust_env"])
        self.assertFalse(async_client_factory.call_args.kwargs["follow_redirects"])

        failure_client = RecordingAsyncClient(
            FakeHttpxResponse({}, status_code=503)
        )
        with patch("app.embedding_client.httpx.AsyncClient", return_value=failure_client):
            with self.assertRaises(EmbeddingServiceError):
                await HttpEmbeddingClient(
                    "http://embedding-service:8081"
                ).embed(["one"], expected=metadata(), purpose="query")
        self.assertTrue(failure_client.exited)

    async def test_default_transport_reuses_one_async_client_for_all_batches(self) -> None:
        class BatchAwareAsyncClient(RecordingAsyncClient):
            async def post(
                self, url: str, json: dict[str, Any]
            ) -> FakeHttpxResponse:
                self.requests.append((url, json))
                payload = await FakeTransport().post_json(url, json)
                return FakeHttpxResponse(payload)

        async_client = BatchAwareAsyncClient(FakeHttpxResponse({}))
        with patch(
            "app.embedding_client.httpx.AsyncClient", return_value=async_client
        ) as async_client_factory:
            vectors = await HttpEmbeddingClient(
                "http://embedding-service:8081"
            ).embed(
                [f"text-{index}" for index in range(65)],
                expected=metadata(),
                purpose="query",
            )

        self.assertEqual(len(vectors), 65)
        self.assertEqual(async_client_factory.call_count, 1)
        self.assertEqual(len(async_client.requests), 2)
        self.assertTrue(async_client.entered)
        self.assertTrue(async_client.exited)


if __name__ == "__main__":
    unittest.main()
