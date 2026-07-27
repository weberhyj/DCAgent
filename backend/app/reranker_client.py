"""Synchronous client for the checksum-pinned private Reranker service."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from ipaddress import IPv4Network, IPv6Network, ip_address
from typing import Protocol
from urllib.parse import urlparse

import httpx

from .offline_settings import require_private_url
from .reranker_contracts import (
    MAX_RERANK_PASSAGES,
    MAX_RERANK_REQUEST_BYTES,
    MAX_RERANK_TEXT_BYTES,
    RerankerMetadataExpectation,
    RerankerModelMetadata,
    RerankerResponse,
    reranker_request_json_size,
)


class RerankerClientError(RuntimeError):
    """Base error for private Reranker client failures."""


class RerankerBusy(RerankerClientError):
    """The bounded Reranker queue rejected the request."""


class RerankerServiceError(RerankerClientError):
    """Connection, timeout, or upstream HTTP failure."""


class RerankerResponseError(RerankerClientError):
    """Malformed or internally inconsistent upstream response."""


class RerankerModelMismatch(RerankerClientError):
    """The service returned metadata different from the pinned expectation."""


class RerankerTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class _HttpxRerankerTransport:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, object]:
        request_kwargs: dict[str, object] = {}
        if timeout_seconds is not None:
            request_kwargs["timeout"] = httpx.Timeout(
                timeout_seconds,
                connect=min(timeout_seconds, 2.0),
            )
        response = self._client.post(url, json=dict(payload), **request_kwargs)
        response.raise_for_status()
        try:
            result = response.json()
        except (TypeError, ValueError) as error:
            raise RerankerResponseError("reranker service returned invalid JSON") from error
        if not isinstance(result, Mapping):
            raise RerankerResponseError("reranker service returned a non-object JSON payload")
        return result

    def close(self) -> None:
        self._client.close()


class SyncHttpRerankerClient:
    """Persistent synchronous client used by retrieval worker threads."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: RerankerTransport | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.base_url = _validate_base_url(base_url)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if transport is None:
            client = httpx.Client(
                timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                follow_redirects=False,
                trust_env=False,
            )
            self._transport: RerankerTransport = _HttpxRerankerTransport(client)
        else:
            self._transport = transport
        self._closed = False

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        expected: RerankerMetadataExpectation,
        timeout_seconds: float | None = None,
    ) -> list[float]:
        if self._closed:
            raise RerankerServiceError("reranker client is closed")
        query_value = _validate_text(query, "query")
        passage_values = _validate_passages(passages)
        expected_metadata = _metadata_from_expectation(expected)
        request_timeout = _optional_timeout(timeout_seconds)

        scores: list[float] = []
        endpoint = f"{self.base_url}/rerank"
        for batch in _split_batches(query_value, passage_values):
            payload = {"query": query_value, "passages": batch}
            try:
                raw_response = self._transport.post_json(
                    endpoint,
                    payload,
                    timeout_seconds=request_timeout,
                )
            except RerankerClientError:
                raise
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 429:
                    raise RerankerBusy("reranker service is busy") from error
                raise RerankerServiceError(
                    "reranker service returned an unsuccessful status"
                ) from error
            except (httpx.TimeoutException, httpx.RequestError) as error:
                raise RerankerServiceError("reranker service request failed") from error
            except Exception as error:
                raise RerankerServiceError("reranker service request failed") from error

            try:
                response = RerankerResponse.model_validate(raw_response)
            except Exception as error:
                raise RerankerResponseError(
                    "reranker service returned malformed reranking data"
                ) from error
            actual = response.to_metadata()
            mismatches = _metadata_mismatches(expected_metadata, actual)
            if mismatches:
                raise RerankerModelMismatch(
                    "reranker model metadata mismatch: " + ", ".join(mismatches)
                )
            if response.passage_count != len(batch):
                raise RerankerResponseError("reranker response passage count mismatch")
            scores.extend(response.scores)
        return scores

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()


def _validate_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a nonempty URL")
    candidate = base_url.strip()
    if "?" in candidate or "#" in candidate:
        raise ValueError("RERANKER_SERVICE_URL must not include a query or fragment")
    candidate = require_private_url(candidate, "RERANKER_SERVICE_URL")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("RERANKER_SERVICE_URL must use HTTP or HTTPS")
    if not parsed.hostname:
        raise ValueError("RERANKER_SERVICE_URL must include a host")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("RERANKER_SERVICE_URL must use a valid port") from error
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("RERANKER_SERVICE_URL must use a valid port")
    if parsed.hostname not in {"reranker-service", "localhost"}:
        if "%" in parsed.hostname:
            raise ValueError("RERANKER_SERVICE_URL must not use a scoped IP address")
        try:
            address = ip_address(parsed.hostname)
        except ValueError as error:
            raise ValueError(
                "RERANKER_SERVICE_URL must use reranker-service or a private/loopback IP"
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
                "RERANKER_SERVICE_URL must use reranker-service or a private/loopback IP"
            )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("RERANKER_SERVICE_URL must not include credentials")
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("RERANKER_SERVICE_URL must not include query, fragment, or parameters")
    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise ValueError("RERANKER_SERVICE_URL path must be empty or /v1")
    host = parsed.hostname
    assert host is not None
    authority_host = f"[{host}]" if ":" in host else host
    authority = authority_host
    if parsed.port is not None:
        authority += f":{parsed.port}"
    return f"{parsed.scheme.lower()}://{authority}/v1"


def _optional_timeout(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be positive and finite")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    return timeout


def _validate_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    if len(value.encode("utf-8")) > MAX_RERANK_TEXT_BYTES:
        raise ValueError(f"{field} must not exceed {MAX_RERANK_TEXT_BYTES} UTF-8 bytes")
    return value


def _validate_passages(passages: Sequence[str]) -> list[str]:
    if isinstance(passages, (str, bytes, bytearray)):
        raise ValueError("passages must be a nonempty sequence of strings")
    try:
        values = list(passages)
    except TypeError as error:
        raise ValueError("passages must be a nonempty sequence of strings") from error
    if not values:
        raise ValueError("passages must not be empty")
    return [_validate_text(value, f"passages[{index}]") for index, value in enumerate(values)]


def _split_batches(query: str, passages: Sequence[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    for passage in passages:
        candidate = current + [passage]
        if current and (
            len(candidate) > MAX_RERANK_PASSAGES
            or reranker_request_json_size(query, candidate) > MAX_RERANK_REQUEST_BYTES
        ):
            batches.append(current)
            current = [passage]
        else:
            current = candidate
        if reranker_request_json_size(query, current) > MAX_RERANK_REQUEST_BYTES:
            raise ValueError("a single reranker request exceeds 384 KiB")
    if current:
        batches.append(current)
    return batches


def _metadata_from_expectation(
    expected: RerankerMetadataExpectation,
) -> RerankerModelMetadata:
    try:
        return RerankerModelMetadata(
            expected.name,
            expected.version,
            expected.sha256,
            expected.prompt_profile_sha256,
            expected.protocol_version,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("expected must expose valid Reranker model metadata") from error


def _metadata_mismatches(
    expected: RerankerModelMetadata,
    actual: RerankerModelMetadata,
) -> list[str]:
    fields = ("name", "version", "sha256", "prompt_profile_sha256", "protocol_version")
    return [field for field in fields if getattr(expected, field) != getattr(actual, field)]


__all__ = [
    "RerankerBusy",
    "RerankerClientError",
    "RerankerModelMismatch",
    "RerankerResponseError",
    "RerankerServiceError",
    "RerankerTransport",
    "SyncHttpRerankerClient",
]
