"""Async client for the private shared Embedding service."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from ipaddress import IPv4Network, IPv6Network, ip_address
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from .embedding_contracts import (
    EmbeddingMetadataExpectation,
    EmbeddingModelMetadata,
    EmbeddingPurpose,
    EmbeddingResponse,
    MAX_EMBEDDING_REQUEST_BYTES,
    MAX_EMBEDDING_TEXT_BYTES,
    MAX_EMBEDDING_TEXTS,
    embedding_request_json_size,
)
from .offline_settings import require_private_url


class EmbeddingClientError(RuntimeError):
    """Base error for private Embedding client failures."""


class EmbeddingServiceError(EmbeddingClientError):
    """Connection, timeout, or upstream HTTP failure."""


class EmbeddingResponseError(EmbeddingClientError):
    """Malformed or internally inconsistent upstream response."""


class EmbeddingModelMismatch(EmbeddingClientError):
    """The service returned a model identity different from the pinned expectation."""


class EmbeddingTransport(Protocol):
    async def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]: ...


class _ScopedHttpxTransport:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        try:
            response = await self._client.post(url, json=dict(payload))
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise EmbeddingServiceError("embedding service request timed out") from error
        except httpx.HTTPStatusError as error:
            raise EmbeddingServiceError(
                "embedding service returned an unsuccessful status"
            ) from error
        except httpx.RequestError as error:
            raise EmbeddingServiceError("embedding service request failed") from error

        try:
            result = response.json()
        except (TypeError, ValueError) as error:
            raise EmbeddingResponseError("embedding service returned invalid JSON") from error
        if not isinstance(result, Mapping):
            raise EmbeddingResponseError("embedding service returned a non-object JSON payload")
        return result


class HttpxEmbeddingTransport:
    """Default network transport; every scope owns and closes one AsyncClient."""

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def request_scope(self):
        timeout = httpx.Timeout(
            self.timeout_seconds,
            connect=min(self.timeout_seconds, 5.0),
        )
        limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            yield _ScopedHttpxTransport(client)

    async def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        async with self.request_scope() as transport:
            return await transport.post_json(url, payload)


class HttpEmbeddingClient:
    """The sole production Embedding client implementation."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: EmbeddingTransport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = _validate_base_url(base_url)
        self._transport: EmbeddingTransport = (
            HttpxEmbeddingTransport(timeout_seconds=timeout_seconds)
            if transport is None
            else transport
        )

    async def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
        expected: EmbeddingMetadataExpectation,
    ) -> list[list[float]]:
        values = _validate_texts(texts)
        if purpose not in {"query", "document"}:
            raise ValueError("purpose must be 'query' or 'document'")
        expected_metadata = _metadata_from_expectation(expected)
        batches = _split_batches(values, purpose)

        if isinstance(self._transport, HttpxEmbeddingTransport):
            async with self._transport.request_scope() as scoped_transport:
                return await self._embed_batches(
                    scoped_transport,
                    batches,
                    purpose,
                    expected_metadata,
                )
        return await self._embed_batches(
            self._transport,
            batches,
            purpose,
            expected_metadata,
        )

    async def _embed_batches(
        self,
        transport: EmbeddingTransport,
        batches: Sequence[Sequence[str]],
        purpose: EmbeddingPurpose,
        expected: EmbeddingModelMetadata,
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        endpoint = f"{self.base_url}/embeddings"
        for batch in batches:
            payload = {"texts": list(batch), "purpose": purpose}
            try:
                raw_response = await transport.post_json(endpoint, payload)
            except EmbeddingClientError:
                raise
            except Exception as error:
                raise EmbeddingServiceError(
                    "embedding service request failed"
                ) from error

            try:
                response = EmbeddingResponse.model_validate(raw_response)
            except Exception as error:
                raise EmbeddingResponseError(
                    "embedding service returned malformed embedding data"
                ) from error
            if response.purpose != purpose:
                raise EmbeddingResponseError("embedding response purpose mismatch")

            actual = response.to_metadata()
            mismatches = _metadata_mismatches(expected, actual)
            if mismatches:
                raise EmbeddingModelMismatch(
                    "embedding model metadata mismatch: " + ", ".join(mismatches)
                )
            if len(response.vectors) != len(batch):
                raise EmbeddingResponseError("embedding response vector count mismatch")
            vectors.extend([list(vector) for vector in response.vectors])
        return vectors


def _validate_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a nonempty URL")
    candidate = require_private_url(base_url, "EMBEDDING_SERVICE_URL")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("EMBEDDING_SERVICE_URL must use HTTP or HTTPS")
    if not parsed.hostname:
        raise ValueError("EMBEDDING_SERVICE_URL must include a host")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("EMBEDDING_SERVICE_URL must use a valid port") from error
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("EMBEDDING_SERVICE_URL must use a valid port")
    if parsed.hostname not in {"embedding-service", "localhost"}:
        if "%" in parsed.hostname:
            raise ValueError("EMBEDDING_SERVICE_URL must not use a scoped IP address")
        try:
            address = ip_address(parsed.hostname)
        except ValueError as error:
            raise ValueError(
                "EMBEDDING_SERVICE_URL must use embedding-service or a private/loopback IP"
            ) from error
        if address.version == 4:
            allowed = address.is_loopback or address in IPv4Network("10.0.0.0/8")
            allowed = allowed or address in IPv4Network("172.16.0.0/12")
            allowed = allowed or address in IPv4Network("192.168.0.0/16")
        else:
            allowed = address.is_loopback or address in IPv6Network("fc00::/7")
            allowed = allowed and address.ipv4_mapped is None
        if not allowed:
            raise ValueError(
                "EMBEDDING_SERVICE_URL must use embedding-service or a private/loopback IP"
            )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("EMBEDDING_SERVICE_URL must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("EMBEDDING_SERVICE_URL must not include a query or fragment")
    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise ValueError("EMBEDDING_SERVICE_URL path must be empty or /v1")
    if path == "/v1":
        return candidate.rstrip("/")
    return candidate.rstrip("/") + "/v1"


def _metadata_from_expectation(
    expected: EmbeddingMetadataExpectation,
) -> EmbeddingModelMetadata:
    try:
        return EmbeddingModelMetadata(
            expected.name,
            expected.version,
            expected.sha256,
            expected.dimensions,
            expected.normalized,
            expected.encoding_profile_sha256,
            expected.protocol_version,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("expected must expose valid Embedding model metadata") from error


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes, bytearray)):
        raise ValueError("texts must be a nonempty sequence of strings")
    try:
        values = list(texts)
    except TypeError as error:
        raise ValueError("texts must be a nonempty sequence of strings") from error
    if not values:
        raise ValueError("texts must not be empty")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"texts[{index}] must be a nonempty string")
        if len(value.encode("utf-8")) > MAX_EMBEDDING_TEXT_BYTES:
            raise ValueError(
                f"texts[{index}] must not exceed {MAX_EMBEDDING_TEXT_BYTES} UTF-8 bytes"
            )
    return values


def _split_batches(
    texts: Sequence[str], purpose: EmbeddingPurpose
) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    for text in texts:
        candidate = current + [text]
        if current and (
            len(candidate) > MAX_EMBEDDING_TEXTS
            or embedding_request_json_size(candidate, purpose)
            > MAX_EMBEDDING_REQUEST_BYTES
        ):
            batches.append(current)
            current = [text]
        else:
            current = candidate
        if embedding_request_json_size(current, purpose) > MAX_EMBEDDING_REQUEST_BYTES:
            raise ValueError("a single embedding request exceeds 256 KiB")
    if current:
        batches.append(current)
    return batches


def _metadata_mismatches(
    expected: EmbeddingModelMetadata,
    actual: EmbeddingModelMetadata,
) -> list[str]:
    fields = (
        "name",
        "version",
        "sha256",
        "dimensions",
        "normalized",
        "encoding_profile_sha256",
        "protocol_version",
    )
    return [field for field in fields if getattr(expected, field) != getattr(actual, field)]


__all__ = [
    "EmbeddingClientError",
    "EmbeddingModelMismatch",
    "EmbeddingResponseError",
    "EmbeddingServiceError",
    "EmbeddingTransport",
    "HttpEmbeddingClient",
    "HttpxEmbeddingTransport",
]
