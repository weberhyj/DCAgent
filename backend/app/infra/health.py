from __future__ import annotations

import math
import re
import socket
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..offline_settings import OfflineSettings, parse_bool


DependencyCheckCallable = Callable[[], tuple[bool, str]]
DependencyReport = dict[str, dict[str, bool | str]]
HttpClientFactory = Callable[..., Any]

_MAX_DETAIL_LENGTH = 160
_MAX_DEPENDENCY_TIMEOUT_SECONDS = 10.0
_MAX_CLAMAV_RESPONSE_BYTES = 64
_ALLOWED_PRIVATE_SERVICE_HOSTS = frozenset({"clamav", "localhost"})
_PRIVATE_NETWORKS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("fc00::/7"),
)
_SENSITIVE_DETAIL_PATTERN = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://|"
    r"\b(?:api[_-]?key|authorization|password|secret|token)\s*[:=])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    name: str
    check: DependencyCheckCallable


class DependencyHealthRegistry:
    def __init__(self, checks: list[DependencyCheck] | None = None) -> None:
        materialized_checks = tuple(checks or ())
        names = [item.name for item in materialized_checks]
        if len(names) != len(set(names)):
            raise ValueError("duplicate dependency check name")
        self._checks = materialized_checks

    def report(self) -> DependencyReport:
        report: DependencyReport = {}
        for item in self._checks:
            report[item.name] = _evaluate_check(item.check)
        return report

    def ready(self) -> bool:
        return all(bool(item["ok"]) for item in self.report().values())


def _evaluate_check(check: DependencyCheckCallable) -> dict[str, bool | str]:
    try:
        result = check()
    except Exception:
        return {"ok": False, "detail": "check failed"}

    if not isinstance(result, tuple) or len(result) != 2:
        return {"ok": False, "detail": "invalid check result"}
    ok, detail = result
    if type(ok) is not bool or not isinstance(detail, str):
        return {"ok": False, "detail": "invalid check result"}
    return {"ok": ok, "detail": _sanitize_detail(detail, ok=ok)}


def _sanitize_detail(detail: str, *, ok: bool) -> str:
    normalized = " ".join(detail.split())
    if not normalized:
        return "ready" if ok else "unavailable"
    if _SENSITIVE_DETAIL_PATTERN.search(normalized):
        return "ready" if ok else "unavailable"
    return normalized[:_MAX_DETAIL_LENGTH]


def _bounded_timeout(value: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 2.0
    if not math.isfinite(timeout) or timeout <= 0:
        return 2.0
    return min(timeout, _MAX_DEPENDENCY_TIMEOUT_SECONDS)


def _root_endpoint(service_url: str, path: str) -> str:
    parsed = urlsplit(service_url)
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _http_client(
    timeout_seconds: float,
    client_factory: HttpClientFactory | None,
) -> Any:
    return create_http_health_client(
        timeout_seconds,
        client_factory=client_factory,
    )


def create_http_health_client(
    timeout_seconds: float,
    *,
    client_factory: HttpClientFactory | None = None,
) -> Any:
    timeout_seconds = _bounded_timeout(timeout_seconds)
    if client_factory is not None:
        return client_factory(
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    import httpx

    return httpx.Client(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    )


def _http_status_check(
    service_url: str,
    path: str,
    timeout_seconds: float,
    *,
    client: Any | None = None,
    client_factory: HttpClientFactory | None = None,
    require_ready_json: bool = False,
    expected_text: frozenset[str] | None = None,
) -> DependencyCheckCallable:
    endpoint = _root_endpoint(service_url, path)

    def check() -> tuple[bool, str]:
        if client is None:
            with _http_client(timeout_seconds, client_factory) as one_shot_client:
                response = one_shot_client.get(endpoint)
        else:
            response = client.get(endpoint)
        if response.status_code != 200:
            return False, "unavailable"
        if expected_text is not None:
            response_text = str(getattr(response, "text", "")).strip().lower()
            if response_text not in expected_text:
                return False, "invalid readiness response"
        if require_ready_json:
            try:
                payload = response.json()
            except (TypeError, ValueError):
                return False, "invalid readiness response"
            if not isinstance(payload, dict) or payload.get("status") not in {
                "ok",
                "ready",
            }:
                return False, "not ready"
        return True, "ready"

    return check


def postgres_schema_revision_check(
    database: object,
    *,
    config_path: str | Path | None = None,
    timeout_seconds: float = 2.0,
) -> DependencyCheckCallable:
    resolved_config_path = Path(
        config_path
        or Path(__file__).resolve().parents[2] / "alembic.ini"
    ).resolve()

    def check() -> tuple[bool, str]:
        from alembic.config import Config
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory

        config = Config(str(resolved_config_path))
        config.set_main_option(
            "script_location",
            str(resolved_config_path.parent / "alembic"),
        )
        expected_heads = frozenset(
            ScriptDirectory.from_config(config).get_heads()
        )
        engine = getattr(database, "engine")
        with engine.connect() as connection:
            if (
                getattr(getattr(connection, "dialect", None), "name", None)
                == "postgresql"
            ):
                timeout_milliseconds = max(
                    1,
                    math.ceil(_bounded_timeout(timeout_seconds) * 1000),
                )
                connection.exec_driver_sql(
                    f"SET LOCAL statement_timeout = {timeout_milliseconds}"
                )
            current_heads = frozenset(
                MigrationContext.configure(connection).get_current_heads()
            )
        if (
            len(expected_heads) == 1
            and len(current_heads) == 1
            and current_heads == expected_heads
        ):
            return True, "schema current"
        return False, "schema revision mismatch"

    return check


def _redis_ping_check(
    redis_url: str,
    timeout_seconds: float,
    *,
    client: Any | None = None,
    client_factory: HttpClientFactory | None = None,
) -> DependencyCheckCallable:
    def check() -> tuple[bool, str]:
        if client is not None:
            ok = bool(client.ping())
            return ok, "ready" if ok else "unavailable"

        one_shot_client = create_redis_health_client(
            redis_url,
            timeout_seconds,
            client_factory=client_factory,
        )
        try:
            ok = bool(one_shot_client.ping())
        finally:
            close = getattr(one_shot_client, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
            else:
                pool = getattr(one_shot_client, "connection_pool", None)
                disconnect = getattr(pool, "disconnect", None)
                if callable(disconnect):
                    with suppress(Exception):
                        disconnect()
        return ok, "ready" if ok else "unavailable"

    return check


def create_redis_health_client(
    redis_url: str,
    timeout_seconds: float,
    *,
    client_factory: HttpClientFactory | None = None,
) -> Any:
    timeout_seconds = _bounded_timeout(timeout_seconds)
    keyword_arguments = {
        "socket_connect_timeout": timeout_seconds,
        "socket_timeout": timeout_seconds,
        "health_check_interval": 0,
    }
    if client_factory is not None:
        return client_factory(redis_url, **keyword_arguments)

    from redis import Redis

    return Redis.from_url(redis_url, **keyword_arguments)


def _clamav_ping_check(
    host: str,
    port_value: str,
    timeout_seconds: float,
) -> DependencyCheckCallable:
    def check() -> tuple[bool, str]:
        if not _is_private_service_host(host):
            return False, "invalid configuration"
        try:
            port = int(port_value)
        except (TypeError, ValueError):
            return False, "invalid configuration"
        if not 1 <= port <= 65535:
            return False, "invalid configuration"

        with socket.create_connection(
            (host, port),
            timeout=timeout_seconds,
        ) as connection:
            connection.settimeout(timeout_seconds)
            connection.sendall(b"zPING\0")
            response = bytearray()
            while len(response) < _MAX_CLAMAV_RESPONSE_BYTES:
                chunk = connection.recv(
                    min(16, _MAX_CLAMAV_RESPONSE_BYTES - len(response))
                )
                if not chunk:
                    break
                response.extend(chunk)
                if b"\0" in chunk or b"\n" in chunk:
                    break
        response_bytes = bytes(response)
        terminator_indexes = [
            index
            for marker in (b"\0", b"\n")
            if (index := response_bytes.find(marker)) >= 0
        ]
        if terminator_indexes:
            response_bytes = response_bytes[: min(terminator_indexes)]
        ok = response_bytes.rstrip(b"\r") == b"PONG"
        return ok, "ready" if ok else "unavailable"

    return check


def _is_private_service_host(host: str) -> bool:
    candidate = host.strip().lower().rstrip(".")
    if not candidate:
        return False
    try:
        address = ip_address(candidate)
    except ValueError:
        return candidate in _ALLOWED_PRIVATE_SERVICE_HOSTS
    return address.is_loopback or any(
        address.version == network.version and address in network
        for network in _PRIVATE_NETWORKS
    )


def _generation_enabled(environ: Mapping[str, str]) -> bool:
    provider = (
        environ.get("LLM_PROVIDER", "template")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if provider == "openai_compatible":
        return True
    return any(
        parse_bool(environ.get(name), default=False)
        for name in ("GENERATION_ENABLED", "LLM_GENERATION_ENABLED")
        if name in environ
    )


def build_dependency_checks(
    settings: OfflineSettings,
    *,
    database: object,
    environ: Mapping[str, str],
    http_client: Any | None = None,
    redis_client: Any | None = None,
    http_client_factory: HttpClientFactory | None = None,
) -> list[DependencyCheck]:
    timeout_seconds = _bounded_timeout(settings.dependency_timeout_seconds)
    checks = [
        DependencyCheck(
            "postgresql",
            postgres_schema_revision_check(
                database,
                timeout_seconds=timeout_seconds,
            ),
        ),
        DependencyCheck(
            "clickhouse",
            _http_status_check(
                settings.clickhouse_url,
                "/ping",
                timeout_seconds,
                client=http_client,
                client_factory=http_client_factory,
                expected_text=frozenset({"ok", "ok."}),
            ),
        ),
        DependencyCheck(
            "qdrant",
            _http_status_check(
                settings.qdrant_url,
                "/readyz",
                timeout_seconds,
                client=http_client,
                client_factory=http_client_factory,
            ),
        ),
        DependencyCheck(
            "redis",
            _redis_ping_check(
                settings.redis_url,
                timeout_seconds,
                client=redis_client,
            ),
        ),
        DependencyCheck(
            "clamav",
            _clamav_ping_check(
                settings.clamav_host,
                environ.get("CLAMAV_PORT", "3310"),
                timeout_seconds,
            ),
        ),
        DependencyCheck(
            "embedding",
            _http_status_check(
                settings.embedding_service_url,
                "/readyz",
                timeout_seconds,
                client=http_client,
                client_factory=http_client_factory,
                require_ready_json=True,
            ),
        ),
    ]
    if _generation_enabled(environ):
        checks.append(
            DependencyCheck(
                "llama",
                _http_status_check(
                    settings.llama_server_url,
                    "/health",
                    timeout_seconds,
                    client=http_client,
                    client_factory=http_client_factory,
                ),
            )
        )
    return checks


__all__ = [
    "DependencyCheck",
    "DependencyHealthRegistry",
    "build_dependency_checks",
    "create_http_health_client",
    "create_redis_health_client",
    "postgres_schema_revision_check",
]
