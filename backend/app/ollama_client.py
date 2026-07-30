from __future__ import annotations

import ipaddress
import math
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx

_PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.IPv4Network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_PRIVATE_IPV6_NETWORK = ipaddress.IPv6Network("fc00::/7")


class OllamaError(RuntimeError):
    """Base error raised by the Ollama client."""


class OllamaServiceError(OllamaError):
    """The local Ollama service could not complete a request."""


class OllamaBusy(OllamaServiceError):
    """The local Ollama service is temporarily busy."""


class OllamaResponseError(OllamaError):
    """The local Ollama service returned an invalid response."""


class OllamaTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: object,
        *,
        timeout_seconds: float | None = None,
    ) -> object: ...

    def close(self) -> None: ...


class _HttpxOllamaTransport:
    def __init__(self, timeout_seconds: float) -> None:
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
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
            response = self._client.post(url, json=payload)
        else:
            response = self._client.post(
                url,
                json=payload,
                timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
            )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()


class SyncOllamaClient:
    def __init__(
        self,
        base_url: str,
        transport: OllamaTransport | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._timeout_seconds = _validate_timeout(timeout_seconds)
        self._transport = (
            _HttpxOllamaTransport(self._timeout_seconds) if transport is None else transport
        )
        self._closed = False

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
        or any(ord(character) < 32 or ord(character) == 127 for character in base_url)
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
