from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import httpx

from app.ollama_client import (
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

    def close(self) -> None:
        self.close_calls += 1


class FalseyRecordingTransport(RecordingTransport):
    def __bool__(self) -> bool:
        return False


class RecordingHttpxClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, object, object]] = []
        self.close_calls = 0

    def post(self, url: str, *, json: object, timeout: object = None) -> httpx.Response:
        self.calls.append((url, json, timeout))
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"model": "qwen2.5"},
        )

    def close(self) -> None:
        self.close_calls += 1


class OllamaClientTest(unittest.TestCase):
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
            client.post_json("/api/generate", {"prompt": "one"})
            client.post_json("/api/chat", {"prompt": "two"}, timeout_seconds=5.0)
            client.close()
            client.close()

        self.assertEqual(factory.call_count, 1)
        self.assertEqual(len(http_client.calls), 2)
        self.assertIsNone(http_client.calls[0][2])
        override = http_client.calls[1][2]
        self.assertIsInstance(override, httpx.Timeout)
        self.assertEqual(override.connect, 2.0)
        self.assertEqual(override.read, 5.0)
        self.assertEqual(http_client.close_calls, 1)
        kwargs = factory.call_args.kwargs
        self.assertFalse(kwargs["follow_redirects"])
        self.assertFalse(kwargs["trust_env"])
        self.assertEqual(kwargs["limits"].max_connections, 16)
        self.assertEqual(kwargs["limits"].max_keepalive_connections, 8)
        self.assertEqual(kwargs["timeout"].connect, 2.0)
        self.assertEqual(kwargs["timeout"].read, 4.0)


if __name__ == "__main__":
    unittest.main()
