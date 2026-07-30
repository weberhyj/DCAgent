from __future__ import annotations

import gzip
import unittest
import zlib
from collections.abc import Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx

from app import ollama_client as ollama_client_module
from app.embedding_contracts import MAX_EMBEDDING_TEXTS
from app.ollama_client import (
    DEFAULT_OLLAMA_MAX_RESPONSE_BYTES,
    MAX_OLLAMA_MAX_RESPONSE_BYTES,
    OllamaBusy,
    OllamaResponseError,
    OllamaServiceError,
    SyncOllamaClient,
)

_DEFAULT_RESPONSE = object()


class RecordingTransport:
    def __init__(
        self,
        *,
        response: object = _DEFAULT_RESPONSE,
        error: Exception | None = None,
    ) -> None:
        self.response = {} if response is _DEFAULT_RESPONSE else response
        self.error = error
        self.calls: list[tuple[str, object, float | None]] = []
        self.get_calls: list[tuple[str, float | None]] = []
        self.close_calls = 0

    def post_json(
        self,
        url: str,
        payload: object,
        *,
        timeout_seconds: float | None = None,
    ) -> object:
        self.calls.append((url, payload, timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.response

    def get_json(
        self,
        url: str,
        *,
        timeout_seconds: float | None = None,
    ) -> object:
        self.get_calls.append((url, timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.close_calls += 1


class FalseyRecordingTransport(RecordingTransport):
    def __bool__(self) -> bool:
        return False


class RecordingHttpxClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, str, object, object]] = []
        self.close_calls = 0

    @contextmanager
    def stream(
        self,
        method: str,
        url: str,
        *,
        json: object = None,
        timeout: object = None,
    ):
        self.calls.append((method, url, json, timeout))
        yield httpx.Response(
            200,
            request=httpx.Request(method, url),
            stream=httpx.ByteStream(b'{"model":"qwen2.5"}'),
        )

    def close(self) -> None:
        self.close_calls += 1


class TrackingByteStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.iterated_chunks = 0
        self.close_calls = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.iterated_chunks += 1
            yield chunk

    def close(self) -> None:
        self.close_calls += 1


class RecordingZlibDecoder:
    def __init__(self, decoder: Any) -> None:
        self.decoder = decoder
        self.decompress_calls: list[tuple[int, int, int]] = []
        self.flush_calls: list[tuple[int, int]] = []

    @property
    def eof(self) -> bool:
        return self.decoder.eof

    @property
    def unconsumed_tail(self) -> bytes:
        return self.decoder.unconsumed_tail

    @property
    def unused_data(self) -> bytes:
        return self.decoder.unused_data

    def decompress(self, data: bytes, max_length: int) -> bytes:
        output = self.decoder.decompress(data, max_length)
        self.decompress_calls.append((len(data), max_length, len(output)))
        return output

    def flush(self, length: int) -> bytes:
        output = self.decoder.flush(length)
        self.flush_calls.append((length, len(output)))
        return output


class OllamaClientTest(unittest.TestCase):
    def _controlled_zlib(
        self,
    ) -> tuple[SimpleNamespace, list[RecordingZlibDecoder]]:
        real_decompressobj = zlib.decompressobj
        decoders: list[RecordingZlibDecoder] = []

        def decompressobj(wbits: int) -> RecordingZlibDecoder:
            decoder = RecordingZlibDecoder(real_decompressobj(wbits))
            decoders.append(decoder)
            return decoder

        return (
            SimpleNamespace(
                MAX_WBITS=zlib.MAX_WBITS,
                error=zlib.error,
                decompressobj=decompressobj,
            ),
            decoders,
        )

    def _default_client_with_handler(
        self,
        handler: Any,
        *,
        max_response_bytes: int,
        timeout_seconds: float = 15.0,
    ) -> tuple[SyncOllamaClient, Any]:
        real_client_type = httpx.Client

        def client_factory(**kwargs: Any) -> httpx.Client:
            return real_client_type(transport=httpx.MockTransport(handler), **kwargs)

        factory_patch = patch("app.ollama_client.httpx.Client", side_effect=client_factory)
        factory = factory_patch.start()
        try:
            client = SyncOllamaClient(
                "http://10.20.30.40:11434",
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        finally:
            factory_patch.stop()
        self.addCleanup(client.close)
        return client, factory

    def test_default_response_limit_has_headroom_for_max_embedding_batch(self) -> None:
        configured_dimensions = 896
        conservative_json_bytes_per_float = 32
        conservative_response_bytes = (
            MAX_EMBEDDING_TEXTS * configured_dimensions * conservative_json_bytes_per_float
            + MAX_EMBEDDING_TEXTS * 2
            + len('{"embeddings":[]}')
        )

        self.assertLess(conservative_response_bytes, DEFAULT_OLLAMA_MAX_RESPONSE_BYTES // 4)

    def test_rejects_invalid_response_limits(self) -> None:
        invalid_limits = (0, -1, True, 1.5, "1024", MAX_OLLAMA_MAX_RESPONSE_BYTES + 1)
        for max_response_bytes in invalid_limits:
            with self.subTest(max_response_bytes=max_response_bytes), self.assertRaises(ValueError):
                SyncOllamaClient(
                    "http://ollama:11434",
                    transport=RecordingTransport(),
                    max_response_bytes=max_response_bytes,  # type: ignore[arg-type]
                )

    def test_content_length_over_limit_is_rejected_before_body_read_and_response_closes(
        self,
    ) -> None:
        stream = TrackingByteStream([b"KEEP-THIS-CANARY-PRIVATE"])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Length": "13"},
                stream=stream,
                request=request,
            )

        client, _ = self._default_client_with_handler(handler, max_response_bytes=12)

        with self.assertRaises(OllamaResponseError) as caught:
            client.post_json("/api/generate", {})

        self.assertEqual(stream.iterated_chunks, 0)
        self.assertEqual(stream.close_calls, 1)
        self.assertNotIn("KEEP-THIS-CANARY-PRIVATE", str(caught.exception))
        self.assertNotIn("10.20.30.40", str(caught.exception))

    def test_content_length_equal_to_limit_succeeds(self) -> None:
        body = b'{"ok":true}'
        stream = TrackingByteStream([body])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(body))},
                stream=stream,
                request=request,
            )

        client, _ = self._default_client_with_handler(handler, max_response_bytes=len(body))

        self.assertEqual(client.post_json("/api/generate", {}), {"ok": True})
        self.assertEqual(stream.close_calls, 1)

    def test_chunked_response_without_content_length_is_bounded(self) -> None:
        cases = (
            ([b'{"ok":', b"true}"], len(b'{"ok":true}'), {"ok": True}),
            ([b'{"secret":"', b"CANARY", b'"}'], 15, OllamaResponseError),
        )
        for chunks, limit, expected in cases:
            with self.subTest(expected=expected):
                stream = TrackingByteStream(chunks)

                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, stream=stream, request=request)

                client, _ = self._default_client_with_handler(
                    handler,
                    max_response_bytes=limit,
                )
                if isinstance(expected, dict):
                    self.assertEqual(client.get_json("/api/tags"), expected)
                else:
                    with self.assertRaises(expected) as caught:
                        client.get_json("/api/tags")
                    self.assertEqual(stream.iterated_chunks, 2)
                    self.assertNotIn("CANARY", str(caught.exception))
                    self.assertNotIn("10.20.30.40", str(caught.exception))
                self.assertEqual(stream.close_calls, 1)

    def test_small_and_exact_limit_gzip_json_succeed(self) -> None:
        exact_limit = 64
        bodies = (
            (b'{"ok":true}', exact_limit),
            (b'{"ok":true}', len(b'{"ok":true}')),
            (b'{"ok":true}' + b" " * (exact_limit - len(b'{"ok":true}')), exact_limit),
        )
        for body, limit in bodies:
            with self.subTest(body_length=len(body), limit=limit):
                compressed = gzip.compress(body)
                stream = TrackingByteStream([compressed])
                controlled_zlib, decoders = self._controlled_zlib()

                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        200,
                        headers={
                            "Content-Encoding": "gzip",
                            "Content-Length": str(len(compressed)),
                        },
                        stream=stream,
                        request=request,
                    )

                client, _ = self._default_client_with_handler(
                    handler,
                    max_response_bytes=limit,
                )

                with patch.object(ollama_client_module, "zlib", controlled_zlib):
                    self.assertEqual(client.get_json("/api/tags"), {"ok": True})
                self.assertEqual(len(decoders), 1)
                decoder = decoders[0]
                self.assertEqual(decoder.flush_calls, [])
                generated_sizes = [generated for _, _, generated in decoder.decompress_calls]
                requested_sizes = [requested for _, requested, _ in decoder.decompress_calls]
                self.assertTrue(generated_sizes)
                self.assertLessEqual(max(generated_sizes), limit + 1)
                self.assertLessEqual(sum(generated_sizes), limit)
                for generated, requested in zip(generated_sizes, requested_sizes, strict=True):
                    self.assertLessEqual(generated, requested)
                self.assertEqual(stream.close_calls, 1)

    def test_chunked_gzip_bomb_limits_every_decoder_allocation_and_closes(self) -> None:
        limit = 64
        decoded = b'{"payload":"' + b"x" * (16 * 1024 * 1024) + b'"}'
        compressed = gzip.compress(decoded, compresslevel=9)
        self.assertLess(len(compressed), 20 * 1024)
        stream = TrackingByteStream(
            [compressed[index : index + 257] for index in range(0, len(compressed), 257)]
        )
        controlled_zlib, decoders = self._controlled_zlib()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=stream,
                request=request,
            )

        client, _ = self._default_client_with_handler(
            handler,
            max_response_bytes=limit,
        )

        with (
            patch.object(ollama_client_module, "zlib", controlled_zlib),
            self.assertRaises(OllamaResponseError) as caught,
        ):
            client.post_json("/api/embed", {"prompt": "PRIVATE-PROMPT"})

        self.assertEqual(len(decoders), 1)
        decoder = decoders[0]
        self.assertTrue(decoder.decompress_calls)
        for _, requested_bytes, generated_bytes in decoder.decompress_calls:
            self.assertLessEqual(requested_bytes, limit + 1)
            self.assertLessEqual(generated_bytes, requested_bytes)
        self.assertLessEqual(
            sum(generated_bytes for _, _, generated_bytes in decoder.decompress_calls),
            limit + 1,
        )
        self.assertEqual(stream.close_calls, 1)
        self.assertNotIn("PRIVATE-PROMPT", str(caught.exception))
        self.assertNotIn("10.20.30.40", str(caught.exception))

    def test_chunked_gzip_raw_bytes_are_bounded_independently(self) -> None:
        body = b'{"ok":true}'
        compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        deflated = compressor.compress(body) + compressor.flush()
        oversized_header = (
            b"\x1f\x8b\x08\x08\x00\x00\x00\x00\x00\xff" + b"a" * (64 * 1024) + b"\x00"
        )
        trailer = (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "little") + len(body).to_bytes(
            4, "little"
        )
        compressed = oversized_header + deflated + trailer
        stream = TrackingByteStream([compressed])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=stream,
                request=request,
            )

        client, _ = self._default_client_with_handler(handler, max_response_bytes=64)

        with self.assertRaises(OllamaResponseError):
            client.get_json("/api/tags")

        self.assertEqual(stream.close_calls, 1)

    def test_invalid_truncated_and_trailing_gzip_fail_closed(self) -> None:
        valid = gzip.compress(b'{"ok":true}')
        cases = (
            b"PRIVATE-NOT-GZIP",
            valid[:-4],
            valid + b"PRIVATE-TRAILING-DATA",
        )
        for compressed in cases:
            with self.subTest(compressed_length=len(compressed)):
                stream = TrackingByteStream([compressed])

                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        200,
                        headers={"Content-Encoding": "gzip"},
                        stream=stream,
                        request=request,
                    )

                client, _ = self._default_client_with_handler(
                    handler,
                    max_response_bytes=64,
                )
                with self.assertRaises(OllamaResponseError) as caught:
                    client.get_json("/api/tags")
                self.assertEqual(stream.close_calls, 1)
                self.assertNotIn("PRIVATE", str(caught.exception))
                self.assertNotIn("10.20.30.40", str(caught.exception))

    def test_unsupported_or_multiple_content_encoding_fails_before_body_read(self) -> None:
        for content_encoding in ("br", "gzip, identity", "gzip, br"):
            with self.subTest(content_encoding=content_encoding):
                stream = TrackingByteStream([b"PRIVATE-ENCODED-BODY"])

                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        200,
                        headers={"Content-Encoding": content_encoding},
                        stream=stream,
                        request=request,
                    )

                client, _ = self._default_client_with_handler(
                    handler,
                    max_response_bytes=64,
                )
                with self.assertRaises(OllamaResponseError) as caught:
                    client.get_json("/api/tags")
                self.assertEqual(stream.iterated_chunks, 0)
                self.assertEqual(stream.close_calls, 1)
                self.assertNotIn("PRIVATE", str(caught.exception))

    def test_invalid_content_length_json_utf8_and_non_object_are_sanitized(self) -> None:
        cases = (
            ({"Content-Length": "-1"}, [b"{}"]),
            ({"Content-Length": "not-a-number"}, [b"{}"]),
            ({}, [b""]),
            ({}, [b'{"unterminated"']),
            ({}, [b"\xff"]),
            ({}, [b"[]"]),
        )
        for headers, chunks in cases:
            with self.subTest(headers=headers, chunks=chunks):
                stream = TrackingByteStream(chunks)

                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, headers=headers, stream=stream, request=request)

                client, _ = self._default_client_with_handler(handler, max_response_bytes=64)
                with self.assertRaises(OllamaResponseError) as caught:
                    client.post_json("/api/generate", {"prompt": "PRIVATE-PROMPT"})
                message = str(caught.exception)
                self.assertNotIn("PRIVATE-PROMPT", message)
                self.assertNotIn("10.20.30.40", message)
                self.assertNotIn("unterminated", message)

    def test_deeply_nested_identity_get_and_gzip_post_map_recursion_safely(self) -> None:
        canary = "PRIVATE-DEEP-JSON-CANARY"
        body = ("[" * 10_000 + f'"{canary}"' + "]" * 10_000).encode("utf-8")
        cases = (("GET", None), ("POST", "gzip"))
        for method, content_encoding in cases:
            with self.subTest(method=method, content_encoding=content_encoding):
                wire_body = body if content_encoding is None else gzip.compress(body)
                stream = TrackingByteStream([wire_body])
                headers = {"Content-Length": str(len(wire_body))}
                if content_encoding is not None:
                    headers["Content-Encoding"] = content_encoding

                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        200,
                        headers=headers,
                        stream=stream,
                        request=request,
                    )

                client, _ = self._default_client_with_handler(
                    handler,
                    max_response_bytes=len(body),
                )
                with self.assertRaises(OllamaResponseError) as caught:
                    if method == "GET":
                        client.get_json("/api/tags")
                    else:
                        client.post_json("/api/generate", {"prompt": "PRIVATE-PROMPT"})

                message = str(caught.exception)
                self.assertNotIn(canary, message)
                self.assertNotIn("PRIVATE-PROMPT", message)
                self.assertNotIn("10.20.30.40", message)
                self.assertEqual(stream.close_calls, 1)

    def test_http_status_fails_before_oversized_content_length_validation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                503,
                headers={"Content-Length": "999999"},
                content=b"PRIVATE-STATUS-BODY",
                request=request,
            )

        client, _ = self._default_client_with_handler(handler, max_response_bytes=8)

        with self.assertRaises(OllamaServiceError) as caught:
            client.get_json("/api/tags")

        self.assertNotIsInstance(caught.exception, OllamaResponseError)
        self.assertNotIn("PRIVATE-STATUS-BODY", str(caught.exception))

    def test_real_httpx_client_receives_get_post_urls_and_timeout_overrides(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                stream=httpx.ByteStream(b'{"ok":true}'),
                request=request,
            )

        client, factory = self._default_client_with_handler(
            handler,
            max_response_bytes=64,
            timeout_seconds=4.0,
        )

        self.assertEqual(client.get_json("/api/tags"), {"ok": True})
        self.assertEqual(
            client.post_json("/api/generate", {"prompt": "hi"}, timeout_seconds=5.0),
            {"ok": True},
        )

        self.assertEqual(
            [(request.method, str(request.url)) for request in requests],
            [
                ("GET", "http://10.20.30.40:11434/api/tags"),
                ("POST", "http://10.20.30.40:11434/api/generate"),
            ],
        )
        self.assertEqual(requests[0].extensions["timeout"]["read"], 4.0)
        self.assertEqual(requests[1].extensions["timeout"]["read"], 5.0)
        self.assertEqual(factory.call_count, 1)

    def test_context_manager_closes_default_transport_once(self) -> None:
        http_client = RecordingHttpxClient()
        with patch("app.ollama_client.httpx.Client", return_value=http_client):
            with SyncOllamaClient("http://localhost:11434") as client:
                self.assertEqual(client.get_json("/api/tags"), {"model": "qwen2.5"})

        self.assertEqual(http_client.close_calls, 1)

    def test_resolves_unique_model_digest_and_normalizes_optional_prefix(self) -> None:
        digest = "a" * 64
        for returned_digest in (digest, f"sha256:{digest}"):
            with self.subTest(returned_digest=returned_digest):
                transport = RecordingTransport(
                    response={
                        "models": [
                            {
                                "name": "unrelated:latest",
                                "model": "unrelated:latest",
                                "digest": "b" * 64,
                            },
                            {
                                "name": "qwen2.5:0.5b",
                                "model": "qwen2.5:0.5b",
                                "digest": returned_digest,
                            },
                        ]
                    }
                )
                client = SyncOllamaClient("http://ollama:11434", transport=transport)
                resolver = getattr(client, "model_digest", None)
                self.assertTrue(callable(resolver))

                self.assertEqual(resolver("qwen2.5:0.5b"), digest)
                self.assertEqual(
                    transport.get_calls,
                    [("http://ollama:11434/api/tags", None)],
                )
                self.assertEqual(transport.calls, [])

    def test_model_digest_rejects_missing_duplicate_and_malformed_inventory(self) -> None:
        digest = "a" * 64
        cases = (
            ({"models": []}, "unavailable"),
            (
                {
                    "models": [
                        {"name": "qwen2.5:0.5b", "digest": digest},
                        {"model": "qwen2.5:0.5b", "digest": digest},
                    ]
                },
                "ambiguous",
            ),
            ({}, "invalid"),
            ({"models": "not-a-list"}, "invalid"),
            ({"models": ["not-an-object"]}, "invalid"),
            (
                {"models": [{"name": "qwen2.5:0.5b", "digest": "sha256:bad"}]},
                "invalid",
            ),
            (
                {
                    "models": [
                        {
                            "name": "qwen2.5:0.5b",
                            "model": "different:latest",
                            "digest": digest,
                        }
                    ]
                },
                "invalid",
            ),
        )
        for response, message in cases:
            with self.subTest(response=response):
                client = SyncOllamaClient(
                    "http://ollama:11434",
                    transport=RecordingTransport(response=response),
                )
                resolver = getattr(client, "model_digest", None)
                self.assertTrue(callable(resolver))
                with self.assertRaisesRegex(OllamaResponseError, message):
                    resolver("qwen2.5:0.5b")

    def test_model_digest_maps_transport_failure_without_leaking_details(self) -> None:
        request = httpx.Request("GET", "http://ollama:11434/api/tags")
        client = SyncOllamaClient(
            "http://ollama:11434",
            transport=RecordingTransport(
                error=httpx.ConnectError("private inventory host details", request=request)
            ),
        )
        resolver = getattr(client, "model_digest", None)
        self.assertTrue(callable(resolver))

        with self.assertRaises(OllamaServiceError) as caught:
            resolver("qwen2.5:0.5b")

        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("qwen2.5", str(caught.exception))

    def test_accepts_only_local_or_private_base_urls_and_normalizes_slash(self) -> None:
        cases = (
            ("http://localhost:11434/", "http://localhost:11434/api/generate"),
            ("https://ollama", "https://ollama/api/generate"),
            ("http://127.0.0.1:11434", "http://127.0.0.1:11434/api/generate"),
            ("http://10.20.30.40", "http://10.20.30.40/api/generate"),
            ("http://172.16.0.1", "http://172.16.0.1/api/generate"),
            ("http://192.168.1.2", "http://192.168.1.2/api/generate"),
            ("http://[::1]:11434", "http://[::1]:11434/api/generate"),
            ("http://[fd00::1]:11434/", "http://[fd00::1]:11434/api/generate"),
        )
        for base_url, expected_url in cases:
            with self.subTest(base_url=base_url):
                transport = RecordingTransport(response={"ok": True})
                client = SyncOllamaClient(base_url, transport=transport)

                self.assertEqual(client.post_json("/api/generate", {"prompt": "hi"}), {"ok": True})
                self.assertEqual(transport.calls[0][0], expected_url)

    def test_rejects_non_private_or_malformed_base_urls(self) -> None:
        rejected = (
            "",
            "localhost:11434",
            "ftp://localhost:11434",
            "http://",
            "http://example.com",
            "http://8.8.8.8",
            "http://0.0.0.0",
            "http://169.254.1.2",
            "http://192.0.2.1",
            "http://198.18.0.1",
            "http://224.0.0.1",
            "http://[::]",
            "http://[fe80::1]",
            "http://[ff02::1]",
            "http://[fd00::1%25ethernet]",
            "http://[::ffff:127.0.0.1]",
            "http://user:secret@localhost:11434",
            "http://localhost:11434/extra",
            "http://localhost:11434?",
            "http://localhost:11434?secret=1",
            "http://localhost:11434#",
            "http://localhost:11434#fragment",
            "http://localhost:0",
            "http://localhost:65536",
            "http://localhost:not-a-port",
        )
        for base_url in rejected:
            with self.subTest(base_url=base_url), self.assertRaises(ValueError):
                SyncOllamaClient(base_url, transport=RecordingTransport())

    def test_rejects_ascii_control_characters_in_base_url(self) -> None:
        for base_url in (
            "http://local\thost:11434",
            "http://local\nhost:11434",
            "http://local\rhost:11434",
        ):
            with self.subTest(base_url=repr(base_url)), self.assertRaises(ValueError):
                SyncOllamaClient(base_url, transport=RecordingTransport())

    def test_rejects_paths_outside_absolute_api_namespace(self) -> None:
        transport = RecordingTransport()
        client = SyncOllamaClient("http://localhost:11434", transport=transport)

        for path in (
            "",
            "api/generate",
            "/api",
            "//api/generate",
            "/v1/generate",
            "/api/generate?secret=1",
            "/api/generate#fragment",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                client.post_json(path, {})

        self.assertEqual(transport.calls, [])

    def test_rejects_percent_encoded_path_ambiguity_without_calls(self) -> None:
        transport = RecordingTransport()
        client = SyncOllamaClient("http://localhost:11434", transport=transport)

        for path in ("/api/%2e%2e/admin", "/api/foo%2fbar", "/api/foo%5cbar"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                client.post_json(path, {})

        self.assertEqual(transport.calls, [])

    def test_rejects_ascii_control_characters_in_path_without_calls(self) -> None:
        transport = RecordingTransport()
        client = SyncOllamaClient("http://localhost:11434", transport=transport)

        for code_point in (*range(32), 127):
            path = f"/api/foo{chr(code_point)}bar"
            with self.subTest(code_point=code_point), self.assertRaises(ValueError):
                client.post_json(path, {})

        self.assertEqual(transport.calls, [])

    def test_requires_mapping_response(self) -> None:
        for response in (None, [], "json", 42):
            with self.subTest(response=response):
                client = SyncOllamaClient(
                    "http://ollama:11434",
                    transport=RecordingTransport(response=response),
                )
                with self.assertRaises(OllamaResponseError):
                    client.post_json("/api/generate", {})

    def test_returns_mapping_and_propagates_timeout_override(self) -> None:
        response: Mapping[str, object] = {"response": "answer"}
        transport = RecordingTransport(response=response)
        client = SyncOllamaClient("http://ollama:11434", transport=transport)

        result = client.post_json(
            "/api/generate",
            {"model": "qwen2.5"},
            timeout_seconds=0.25,
        )

        self.assertIs(result, response)
        self.assertEqual(transport.calls[0][2], 0.25)

    def test_maps_malformed_json_and_unicode_errors_to_response_error(self) -> None:
        errors = (
            ValueError("private malformed JSON details"),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "private decoding details"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                client = SyncOllamaClient(
                    "http://ollama:11434",
                    transport=RecordingTransport(error=error),
                )
                with self.assertRaises(OllamaResponseError) as caught:
                    client.post_json("/api/generate", {})
                self.assertNotIn("private", str(caught.exception))

    def test_rejects_invalid_timeout_overrides_without_calls(self) -> None:
        transport = RecordingTransport()
        client = SyncOllamaClient("http://ollama:11434", transport=transport)

        invalid_timeouts = (
            0,
            -1,
            float("inf"),
            float("-inf"),
            float("nan"),
            True,
            False,
            "1",
            object(),
        )
        for timeout_seconds in invalid_timeouts:
            with self.subTest(timeout_seconds=timeout_seconds), self.assertRaises(ValueError):
                client.post_json(
                    "/api/generate",
                    {},
                    timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
                )

        self.assertEqual(transport.calls, [])

    def test_uses_an_explicitly_injected_falsey_transport(self) -> None:
        transport = FalseyRecordingTransport(response={"ok": True})
        with patch("app.ollama_client.httpx.Client") as factory:
            client = SyncOllamaClient("http://ollama:11434", transport=transport)

            self.assertEqual(client.post_json("/api/generate", {}), {"ok": True})
            client.close()

        factory.assert_not_called()
        self.assertEqual(transport.close_calls, 1)

    def test_maps_429_to_busy_without_leaking_upstream_details(self) -> None:
        request = httpx.Request("POST", "http://ollama:11434/api/generate")
        response = httpx.Response(429, request=request, text="secret queue details")
        transport = RecordingTransport(
            error=httpx.HTTPStatusError("private failure", request=request, response=response)
        )
        client = SyncOllamaClient("http://ollama:11434", transport=transport)

        with self.assertRaises(OllamaBusy) as caught:
            client.post_json("/api/generate", {})

        self.assertNotIn("private failure", str(caught.exception))
        self.assertNotIn("secret queue details", str(caught.exception))

    def test_maps_status_network_and_timeout_failures_to_sanitized_service_error(self) -> None:
        request = httpx.Request("POST", "http://ollama:11434/api/generate")
        response = httpx.Response(503, request=request, text="secret service details")
        errors = (
            httpx.HTTPStatusError("private failure", request=request, response=response),
            httpx.ConnectError("private host details", request=request),
            httpx.ReadTimeout("private timeout details", request=request),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                client = SyncOllamaClient(
                    "http://ollama:11434",
                    transport=RecordingTransport(error=error),
                )
                with self.assertRaises(OllamaServiceError) as caught:
                    client.post_json("/api/generate", {})
                self.assertNotIn("private", str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))

    def test_close_is_idempotent_and_closed_client_fails_safely(self) -> None:
        transport = RecordingTransport(response={"ok": True})
        client = SyncOllamaClient("http://ollama:11434", transport=transport)

        client.close()
        client.close()

        self.assertEqual(transport.close_calls, 1)
        with self.assertRaises(OllamaServiceError) as caught:
            client.post_json("/api/generate", {"secret": "do not leak"})
        self.assertNotIn("do not leak", str(caught.exception))
        self.assertEqual(transport.calls, [])

    def test_default_transport_reuses_bounded_httpx_client(self) -> None:
        http_client = RecordingHttpxClient()
        with patch("app.ollama_client.httpx.Client", return_value=http_client) as factory:
            client = SyncOllamaClient("http://localhost:11434", timeout_seconds=4.0)
            client.get_json("/api/tags")
            client.post_json("/api/generate", {"prompt": "one"})
            client.post_json("/api/chat", {"prompt": "two"}, timeout_seconds=5.0)
            client.close()
            client.close()

        self.assertEqual(factory.call_count, 1)
        self.assertEqual(len(http_client.calls), 3)
        self.assertEqual(http_client.calls[0][:3], ("GET", "http://localhost:11434/api/tags", None))
        self.assertEqual(
            http_client.calls[1][:3],
            ("POST", "http://localhost:11434/api/generate", {"prompt": "one"}),
        )
        self.assertIsNone(http_client.calls[0][3])
        self.assertIsNone(http_client.calls[1][3])
        override = http_client.calls[2][3]
        self.assertIsInstance(override, httpx.Timeout)
        self.assertEqual(override.connect, 2.0)
        self.assertEqual(override.read, 5.0)
        self.assertEqual(http_client.close_calls, 1)
        kwargs = factory.call_args.kwargs
        self.assertFalse(kwargs["follow_redirects"])
        self.assertFalse(kwargs["trust_env"])
        self.assertEqual(kwargs["headers"], {"Accept-Encoding": "identity"})
        self.assertEqual(kwargs["limits"].max_connections, 16)
        self.assertEqual(kwargs["limits"].max_keepalive_connections, 8)
        self.assertEqual(kwargs["timeout"].connect, 2.0)
        self.assertEqual(kwargs["timeout"].read, 4.0)


if __name__ == "__main__":
    unittest.main()
