from __future__ import annotations

import ipaddress
import json
import math
import re
import zlib
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx

_PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.IPv4Network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_PRIVATE_IPV6_NETWORK = ipaddress.IPv6Network("fc00::/7")
_MODEL_DIGEST_PATTERN = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")
DEFAULT_OLLAMA_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_OLLAMA_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_RESPONSE_CHUNK_BYTES = 64 * 1024
_GZIP_WINDOW_BITS = zlib.MAX_WBITS | 16


class OllamaError(RuntimeError):
    """Base error raised by the Ollama client."""


class OllamaServiceError(OllamaError):
    """The local Ollama service could not complete a request."""


class OllamaBusy(OllamaServiceError):
    """The local Ollama service is temporarily busy."""


class OllamaResponseError(OllamaError):
    """The local Ollama service returned an invalid response."""


class OllamaTransport(Protocol):
    def get_json(
        self,
        url: str,
        *,
        timeout_seconds: float | None = None,
    ) -> object: ...

    def post_json(
        self,
        url: str,
        payload: object,
        *,
        timeout_seconds: float | None = None,
    ) -> object: ...

    def close(self) -> None: ...


class _HttpxOllamaTransport:
    def __init__(self, timeout_seconds: float, max_response_bytes: int) -> None:
        self._max_response_bytes = max_response_bytes
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            headers={"Accept-Encoding": "identity"},
            follow_redirects=False,
            trust_env=False,
        )

    def post_json(
        self,
        url: str,
        payload: object,
        *,
        timeout_seconds: float | None = None,
    ) -> object:
        if timeout_seconds is None:
            request = self._client.stream("POST", url, json=payload)
        else:
            request = self._client.stream(
                "POST",
                url,
                json=payload,
                timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
            )
        with request as response:
            response.raise_for_status()
            return _read_bounded_json(response, self._max_response_bytes)

    def get_json(
        self,
        url: str,
        *,
        timeout_seconds: float | None = None,
    ) -> object:
        if timeout_seconds is None:
            request = self._client.stream("GET", url)
        else:
            request = self._client.stream(
                "GET",
                url,
                timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
            )
        with request as response:
            response.raise_for_status()
            return _read_bounded_json(response, self._max_response_bytes)

    def close(self) -> None:
        self._client.close()


class SyncOllamaClient:
    def __init__(
        self,
        base_url: str,
        transport: OllamaTransport | None = None,
        timeout_seconds: float = 15.0,
        max_response_bytes: int = DEFAULT_OLLAMA_MAX_RESPONSE_BYTES,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._timeout_seconds = _validate_timeout(timeout_seconds)
        self._max_response_bytes = _validate_max_response_bytes(max_response_bytes)
        self._transport = (
            _HttpxOllamaTransport(self._timeout_seconds, self._max_response_bytes)
            if transport is None
            else transport
        )
        self._closed = False

    def __enter__(self) -> SyncOllamaClient:
        if self._closed:
            raise OllamaServiceError("Ollama client is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def post_json(
        self,
        path: str,
        payload: object,
        *,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, object]:
        if self._closed:
            raise OllamaServiceError("Ollama client is closed")
        normalized_path = _validate_api_path(path)
        request_timeout = None if timeout_seconds is None else _validate_timeout(timeout_seconds)
        try:
            response = self._transport.post_json(
                f"{self._base_url}{normalized_path}",
                payload,
                timeout_seconds=request_timeout,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                raise OllamaBusy("Ollama service is busy") from None
            raise OllamaServiceError("Ollama service request failed") from None
        except httpx.HTTPError:
            raise OllamaServiceError("Ollama service request failed") from None
        except (UnicodeError, ValueError):
            raise OllamaResponseError("Ollama service returned invalid JSON") from None

        if not isinstance(response, Mapping):
            raise OllamaResponseError("Ollama service returned a non-object response")
        return response

    def get_json(
        self,
        path: str,
        *,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, object]:
        if self._closed:
            raise OllamaServiceError("Ollama client is closed")
        normalized_path = _validate_api_path(path)
        request_timeout = None if timeout_seconds is None else _validate_timeout(timeout_seconds)
        try:
            response = self._transport.get_json(
                f"{self._base_url}{normalized_path}",
                timeout_seconds=request_timeout,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                raise OllamaBusy("Ollama service is busy") from None
            raise OllamaServiceError("Ollama service request failed") from None
        except httpx.HTTPError:
            raise OllamaServiceError("Ollama service request failed") from None
        except (UnicodeError, ValueError):
            raise OllamaResponseError("Ollama service returned invalid JSON") from None

        if not isinstance(response, Mapping):
            raise OllamaResponseError("Ollama service returned a non-object response")
        return response

    def model_digest(self, model: str) -> str:
        if (
            not isinstance(model, str)
            or not model
            or model != model.strip()
            or _contains_ascii_control(model)
        ):
            raise ValueError("Ollama model name must be a non-empty string")

        inventory = self.get_json("/api/tags")
        models = inventory.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama model inventory response is invalid")

        matches: list[str] = []
        for entry in models:
            if not isinstance(entry, Mapping):
                raise OllamaResponseError("Ollama model inventory response is invalid")
            name = _inventory_identifier(entry, "name")
            model_name = _inventory_identifier(entry, "model")
            if name is None and model_name is None:
                raise OllamaResponseError("Ollama model inventory response is invalid")
            if name is not None and model_name is not None and name != model_name:
                raise OllamaResponseError("Ollama model inventory response is invalid")
            digest = entry.get("digest")
            if not isinstance(digest, str):
                raise OllamaResponseError("Ollama model inventory response is invalid")
            digest_match = _MODEL_DIGEST_PATTERN.fullmatch(digest)
            if digest_match is None:
                raise OllamaResponseError("Ollama model inventory response is invalid")
            if model in (name, model_name):
                matches.append(digest_match.group(1))

        if not matches:
            raise OllamaResponseError("Configured Ollama model is unavailable")
        if len(matches) != 1:
            raise OllamaResponseError("Ollama model inventory is ambiguous")
        return matches[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._transport.close()


def _validate_base_url(base_url: str) -> str:
    if (
        not isinstance(base_url, str)
        or not base_url
        or base_url != base_url.strip()
        or _contains_ascii_control(base_url)
    ):
        raise ValueError("Ollama base URL must be a non-empty HTTP(S) URL")
    if "?" in base_url or "#" in base_url:
        raise ValueError("Ollama base URL must not contain a query or fragment")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("Ollama base URL is invalid") from error

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Ollama base URL must be a non-empty HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama base URL must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Ollama base URL must not contain a path, query, or fragment")
    _validate_port(parsed, port)
    _validate_private_host(parsed.hostname)

    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _validate_port(parsed: SplitResult, port: int | None) -> None:
    netloc = parsed.netloc
    if netloc.startswith("["):
        closing_bracket = netloc.find("]")
        suffix = netloc[closing_bracket + 1 :]
        if suffix and (not suffix.startswith(":") or not suffix[1:]):
            raise ValueError("Ollama base URL has an invalid port")
    elif ":" in netloc and not netloc.rsplit(":", 1)[1]:
        raise ValueError("Ollama base URL has an invalid port")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Ollama base URL has an invalid port")


def _validate_private_host(host: str) -> None:
    normalized_host = host.lower()
    if normalized_host in {"localhost", "ollama"}:
        return
    if "%" in normalized_host:
        raise ValueError("Ollama host must not use a scoped IP address")
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError as error:
        raise ValueError("Ollama host must be localhost, ollama, or a private IP") from error

    if isinstance(address, ipaddress.IPv4Address):
        allowed = address.is_loopback or any(
            address in network for network in _PRIVATE_IPV4_NETWORKS
        )
    else:
        allowed = address.ipv4_mapped is None and (
            address.is_loopback or address in _PRIVATE_IPV6_NETWORK
        )
    if not allowed:
        raise ValueError("Ollama host must be a private or loopback IP")


def _validate_api_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/api/"):
        raise ValueError("Ollama request path must be an absolute /api/ path")
    if _contains_ascii_control(path):
        raise ValueError("Ollama request path must not contain ASCII control characters")
    if "%" in path:
        raise ValueError("Ollama request path must not contain percent encoding")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or parsed.path != path:
        raise ValueError("Ollama request path must not contain a query or fragment")
    if "\\" in path or any(segment in {".", ".."} for segment in parsed.path.split("/")):
        raise ValueError("Ollama request path is invalid")
    return parsed.path


def _validate_timeout(timeout_seconds: float) -> float:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("Ollama timeout must be a positive finite number")
    return float(timeout_seconds)


def _validate_max_response_bytes(max_response_bytes: int) -> int:
    if (
        type(max_response_bytes) is not int
        or max_response_bytes <= 0
        or max_response_bytes > MAX_OLLAMA_MAX_RESPONSE_BYTES
    ):
        raise ValueError(
            "Ollama maximum response bytes must be a positive integer no greater than "
            f"{MAX_OLLAMA_MAX_RESPONSE_BYTES}"
        )
    return max_response_bytes


def _read_bounded_json(response: httpx.Response, max_response_bytes: int) -> object:
    content_encoding = response.headers.get("Content-Encoding")
    normalized_encoding = "identity" if content_encoding is None else content_encoding.lower()
    if normalized_encoding == "identity":
        raw_limit = max_response_bytes
    elif normalized_encoding == "gzip":
        raw_limit = min(
            MAX_OLLAMA_MAX_RESPONSE_BYTES,
            max(max_response_bytes, _RESPONSE_CHUNK_BYTES),
        )
    else:
        raise OllamaResponseError("Ollama service returned an invalid response")

    declared_length = response.headers.get("Content-Length")
    if declared_length is not None:
        if re.fullmatch(r"[0-9]+", declared_length) is None:
            raise OllamaResponseError("Ollama service returned an invalid response")
        if int(declared_length) > raw_limit:
            raise OllamaResponseError("Ollama service response exceeded size limit")

    if normalized_encoding == "gzip":
        body = _read_bounded_gzip(response, max_response_bytes, raw_limit)
    else:
        body = _read_bounded_identity(response, max_response_bytes)
    decoded_body = body.decode("utf-8")
    try:
        return json.loads(decoded_body)
    except RecursionError:
        raise OllamaResponseError("Ollama service returned invalid JSON") from None


def _read_bounded_identity(response: httpx.Response, max_response_bytes: int) -> bytearray:
    body = bytearray()
    chunk_size = min(_RESPONSE_CHUNK_BYTES, max_response_bytes + 1)
    for chunk in response.iter_raw(chunk_size=chunk_size):
        if len(body) + len(chunk) > max_response_bytes:
            raise OllamaResponseError("Ollama service response exceeded size limit")
        body.extend(chunk)
    return body


def _read_bounded_gzip(
    response: httpx.Response,
    max_response_bytes: int,
    raw_limit: int,
) -> bytearray:
    decoder = zlib.decompressobj(_GZIP_WINDOW_BITS)
    body = bytearray()
    raw_size = 0
    finished = False
    raw_chunk_size = min(_RESPONSE_CHUNK_BYTES, raw_limit + 1)
    try:
        for raw_chunk in response.iter_raw(chunk_size=raw_chunk_size):
            raw_size += len(raw_chunk)
            if raw_size > raw_limit:
                raise OllamaResponseError("Ollama service response exceeded size limit")
            if finished:
                if raw_chunk:
                    raise OllamaResponseError("Ollama service returned an invalid response")
                continue

            pending = raw_chunk
            while pending:
                previous_pending_size = len(pending)
                remaining = max_response_bytes - len(body)
                decoded = decoder.decompress(pending, remaining + 1)
                _append_bounded_decoded(body, decoded, max_response_bytes)
                if decoder.unused_data:
                    raise OllamaResponseError("Ollama service returned an invalid response")

                pending = decoder.unconsumed_tail
                if decoder.eof:
                    if pending:
                        raise OllamaResponseError("Ollama service returned an invalid response")
                    finished = True
                    break
                if pending and len(pending) >= previous_pending_size and not decoded:
                    raise OllamaResponseError("Ollama service returned an invalid response")

        if not finished or not decoder.eof:
            raise OllamaResponseError("Ollama service returned an invalid response")
        remaining = max_response_bytes - len(body)
        # ``Decompress.flush(length)`` treats length as an initial allocation, not a hard cap.
        # A bounded empty-input decompress safely drains any final decoder output after EOF.
        final_decoded = decoder.decompress(b"", remaining + 1)
        _append_bounded_decoded(body, final_decoded, max_response_bytes)
    except zlib.error:
        raise OllamaResponseError("Ollama service returned an invalid response") from None
    return body


def _append_bounded_decoded(
    body: bytearray,
    decoded: bytes,
    max_response_bytes: int,
) -> None:
    if len(decoded) > max_response_bytes - len(body):
        raise OllamaResponseError("Ollama service response exceeded size limit")
    body.extend(decoded)


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _inventory_identifier(entry: Mapping[str, object], field: str) -> str | None:
    value = entry.get(field)
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _contains_ascii_control(value)
    ):
        raise OllamaResponseError("Ollama model inventory response is invalid")
    return value
