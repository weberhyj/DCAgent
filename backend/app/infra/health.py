from __future__ import annotations

import atexit
import json
import math
import re
import socket
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock, Thread
from time import monotonic
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..offline_settings import OfflineSettings, parse_bool

DependencyCheckCallable = Callable[[], tuple[bool, str]]
DependencyReport = dict[str, dict[str, bool | str]]
HttpClientFactory = Callable[..., Any]

_MAX_DETAIL_LENGTH = 160
_MAX_DEPENDENCY_TIMEOUT_SECONDS = 10.0
_MAX_CLAMAV_RESPONSE_BYTES = 64
_MAX_HTTP_HEALTH_BODY_BYTES = 1024
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
_REDIS_UNSAFE_QUERY_KEYS = frozenset(
    {
        "socket_timeout",
        "socket_connect_timeout",
        "health_check_interval",
        "timeout",
        "retry_on_timeout",
        "retry_on_error",
        "retry",
        "max_connections",
        "host",
        "port",
        "username",
        "password",
    }
)
_SHARED_EXECUTOR: ThreadPoolExecutor | None = None
_SHARED_EXECUTOR_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    name: str
    check: DependencyCheckCallable


class DependencyHealthRegistry:
    def __init__(
        self,
        checks: list[DependencyCheck] | None = None,
        *,
        cache_ttl_seconds: float = 0.0,
        max_stale_seconds: float = 0.0,
    ) -> None:
        materialized_checks = tuple(checks or ())
        names = [item.name for item in materialized_checks]
        if len(names) != len(set(names)):
            raise ValueError("duplicate dependency check name")
        self._checks = materialized_checks
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._max_stale_seconds = max(0.0, float(max_stale_seconds))
        self._state_lock = Lock()
        self._inflight = False
        self._last_report: DependencyReport | None = None
        self._last_report_at: float | None = None

    def report(self) -> DependencyReport:
        if not self._checks:
            return {}
        now = monotonic()
        with self._state_lock:
            if (
                self._last_report is not None
                and self._last_report_at is not None
                and self._cache_ttl_seconds > 0
                and now - self._last_report_at <= self._cache_ttl_seconds
            ):
                return _copy_report(self._last_report)
            if self._inflight:
                if (
                    self._last_report is not None
                    and self._last_report_at is not None
                    and self._max_stale_seconds > 0
                    and now - self._last_report_at <= self._max_stale_seconds
                ):
                    return _copy_report(self._last_report)
                return {
                    item.name: {
                        "ok": False,
                        "detail": "check in progress",
                    }
                    for item in self._checks
                }
            self._inflight = True

        report: DependencyReport | None = None
        try:
            try:
                report = dict(
                    _shared_dependency_executor().map(
                        _evaluate_named_check,
                        self._checks,
                    )
                )
            except Exception:
                report = {
                    item.name: {"ok": False, "detail": "check failed"} for item in self._checks
                }
            return _copy_report(report)
        finally:
            with self._state_lock:
                if report is not None:
                    self._last_report = _copy_report(report)
                    self._last_report_at = monotonic()
                self._inflight = False

    def ready(self) -> bool:
        return all(bool(item["ok"]) for item in self.report().values())

    def close(self) -> None:
        _shutdown_shared_executor()


def _copy_report(report: DependencyReport) -> DependencyReport:
    return {name: dict(result) for name, result in report.items()}


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


def _evaluate_named_check(
    item: DependencyCheck,
) -> tuple[str, dict[str, bool | str]]:
    return item.name, _evaluate_check(item.check)


def _shared_dependency_executor() -> ThreadPoolExecutor:
    global _SHARED_EXECUTOR
    with _SHARED_EXECUTOR_LOCK:
        if _SHARED_EXECUTOR is None:
            _SHARED_EXECUTOR = ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix="dependency-health",
            )
        return _SHARED_EXECUTOR


def _shutdown_shared_executor() -> None:
    global _SHARED_EXECUTOR
    with _SHARED_EXECUTOR_LOCK:
        executor = _SHARED_EXECUTOR
        _SHARED_EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


atexit.register(_shutdown_shared_executor)


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


def _hard_timeout_check(
    check: DependencyCheckCallable,
    timeout_seconds: float,
    *,
    dependency_name: str,
) -> DependencyCheckCallable:
    timeout_seconds = _bounded_timeout(timeout_seconds)
    gate = BoundedSemaphore(value=1)

    def bounded_check() -> tuple[bool, str]:
        if not gate.acquire(blocking=False):
            return False, "unavailable"

        done = Event()
        outcome: list[tuple[bool, str]] = []

        def run_check() -> None:
            try:
                outcome.append(check())
            except BaseException:
                outcome.append((False, "check failed"))
            finally:
                gate.release()
                done.set()

        worker = Thread(
            target=run_check,
            name=f"dependency-probe-{dependency_name}",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException:
            gate.release()
            return False, "check failed"

        if not done.wait(timeout_seconds):
            return False, "unavailable"
        if not outcome:
            return False, "check failed"
        return outcome[0]

    return bounded_check


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


def _http_response_result(
    client: Any,
    endpoint: str,
    timeout_seconds: float,
    *,
    require_ready_json: bool,
    expected_text: frozenset[str] | None,
) -> tuple[bool, str]:
    deadline = monotonic() + timeout_seconds
    stream = getattr(client, "stream", None)
    if callable(stream):
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False, "unavailable"
        try:
            stream_context = stream(
                "GET",
                endpoint,
                headers={"accept-encoding": "identity"},
                timeout=max(0.001, remaining),
            )
            enter = getattr(stream_context, "__enter__", None)
            if callable(enter):
                with stream_context as response:
                    return _inspect_http_response(
                        response,
                        deadline,
                        require_ready_json=require_ready_json,
                        expected_text=expected_text,
                    )
            try:
                return _inspect_http_response(
                    stream_context,
                    deadline,
                    require_ready_json=require_ready_json,
                    expected_text=expected_text,
                )
            finally:
                close = getattr(stream_context, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()
        except Exception:
            return False, "unavailable"

    # Keep compatibility with injected test/dummy clients that only expose
    # ``get``. Production clients are httpx clients and always take the
    # bounded streaming path above.
    try:
        response = client.get(endpoint)
        return _inspect_buffered_http_response(
            response,
            deadline,
            require_ready_json=require_ready_json,
            expected_text=expected_text,
        )
    except Exception:
        return False, "unavailable"


def _response_content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("content-length")
    if value is None:
        return None
    try:
        length = int(value)
    except (TypeError, ValueError):
        return -1
    return length


def _response_uses_unsupported_encoding(response: Any) -> bool:
    headers = getattr(response, "headers", None)
    if headers is None:
        return False
    value = headers.get("content-encoding")
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "identity"}


def _inspect_http_response(
    response: Any,
    deadline: float,
    *,
    require_ready_json: bool,
    expected_text: frozenset[str] | None,
) -> tuple[bool, str]:
    if getattr(response, "status_code", None) != 200:
        return False, "unavailable"
    if _response_uses_unsupported_encoding(response):
        return False, "invalid readiness response"
    content_length = _response_content_length(response)
    if content_length is not None and not (0 <= content_length <= _MAX_HTTP_HEALTH_BODY_BYTES):
        return False, "invalid readiness response"
    if deadline - monotonic() <= 0:
        return False, "unavailable"
    if expected_text is None and not require_ready_json:
        return True, "ready"

    body = bytearray()
    raw_iterator = getattr(response, "iter_raw", None)
    if callable(raw_iterator):
        iterator = iter(raw_iterator())
    else:
        iterator = iter(response.iter_bytes())
    while True:
        if deadline - monotonic() <= 0:
            return False, "unavailable"
        try:
            chunk = next(iterator)
        except StopIteration:
            break
        if not isinstance(chunk, (bytes, bytearray)):
            return False, "invalid readiness response"
        if len(body) + len(chunk) > _MAX_HTTP_HEALTH_BODY_BYTES:
            return False, "invalid readiness response"
        body.extend(chunk)
        if deadline - monotonic() <= 0:
            return False, "unavailable"

    if expected_text is not None:
        try:
            response_text = bytes(body).decode("utf-8").strip().lower()
        except UnicodeDecodeError:
            return False, "invalid readiness response"
        if response_text not in expected_text:
            return False, "invalid readiness response"
    if require_ready_json:
        try:
            payload = json.loads(bytes(body).decode("utf-8"))
        except (UnicodeDecodeError, TypeError, ValueError):
            return False, "invalid readiness response"
        if not isinstance(payload, dict) or payload.get("status") not in {
            "ok",
            "ready",
        }:
            return False, "not ready"
    return True, "ready"


def _inspect_buffered_http_response(
    response: Any,
    deadline: float,
    *,
    require_ready_json: bool,
    expected_text: frozenset[str] | None,
) -> tuple[bool, str]:
    if getattr(response, "status_code", None) != 200:
        return False, "unavailable"
    if _response_uses_unsupported_encoding(response):
        return False, "invalid readiness response"
    content_length = _response_content_length(response)
    if content_length is not None and not (0 <= content_length <= _MAX_HTTP_HEALTH_BODY_BYTES):
        return False, "invalid readiness response"
    if deadline - monotonic() <= 0:
        return False, "unavailable"
    if expected_text is None and not require_ready_json:
        return True, "ready"
    raw_content = getattr(response, "content", None)
    if raw_content is None and require_ready_json and expected_text is None:
        if deadline - monotonic() <= 0:
            return False, "unavailable"
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
    if raw_content is None:
        raw_content = str(getattr(response, "text", "")).encode("utf-8")
    if not isinstance(raw_content, (bytes, bytearray)):
        return False, "invalid readiness response"
    if len(raw_content) > _MAX_HTTP_HEALTH_BODY_BYTES:
        return False, "invalid readiness response"
    if deadline - monotonic() <= 0:
        return False, "unavailable"
    body = bytes(raw_content)
    if expected_text is not None:
        try:
            response_text = body.decode("utf-8").strip().lower()
        except UnicodeDecodeError:
            return False, "invalid readiness response"
        if response_text not in expected_text:
            return False, "invalid readiness response"
    if require_ready_json:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, TypeError, ValueError):
            return False, "invalid readiness response"
        if not isinstance(payload, dict) or payload.get("status") not in {
            "ok",
            "ready",
        }:
            return False, "not ready"
    return True, "ready"


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
                return _http_response_result(
                    one_shot_client,
                    endpoint,
                    timeout_seconds,
                    require_ready_json=require_ready_json,
                    expected_text=expected_text,
                )
        return _http_response_result(
            client,
            endpoint,
            timeout_seconds,
            require_ready_json=require_ready_json,
            expected_text=expected_text,
        )

    return check


def postgres_schema_revision_check(
    database: object,
    *,
    config_path: str | Path | None = None,
    timeout_seconds: float = 2.0,
) -> DependencyCheckCallable:
    resolved_config_path = Path(
        config_path or Path(__file__).resolve().parents[2] / "alembic.ini"
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
        expected_heads = frozenset(ScriptDirectory.from_config(config).get_heads())
        engine = getattr(database, "engine", database)
        with engine.connect() as connection:
            if getattr(getattr(connection, "dialect", None), "name", None) == "postgresql":
                timeout_milliseconds = max(
                    1,
                    math.ceil(_bounded_timeout(timeout_seconds) * 1000),
                )
                connection.exec_driver_sql(f"SET LOCAL statement_timeout = {timeout_milliseconds}")
            current_heads = frozenset(MigrationContext.configure(connection).get_current_heads())
        if len(expected_heads) == 1 and len(current_heads) == 1 and current_heads == expected_heads:
            return True, "schema current"
        return False, "schema revision mismatch"

    return check


def create_postgres_health_engine(
    database_url: str,
    *,
    engine_factory: Callable[..., Any] | None = None,
) -> Any:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    factory = engine_factory or create_engine
    return factory(
        database_url,
        future=True,
        pool_pre_ping=False,
        poolclass=NullPool,
    )


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
    parsed_url = urlsplit(redis_url)
    safe_query = [
        (key, value)
        for key, value in parse_qsl(parsed_url.query, keep_blank_values=True)
        if key.lower() not in _REDIS_UNSAFE_QUERY_KEYS
    ]
    health_redis_url = urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            urlencode(safe_query, doseq=True),
            "",
        )
    )
    keyword_arguments = {
        "socket_connect_timeout": timeout_seconds,
        "socket_timeout": timeout_seconds,
        "health_check_interval": 0,
        "max_connections": 8,
    }
    if client_factory is not None:
        return client_factory(health_redis_url, **keyword_arguments)

    from redis import Redis

    return Redis.from_url(health_redis_url, **keyword_arguments)


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

        deadline = monotonic() + timeout_seconds
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False, "unavailable"
        with socket.create_connection(
            (host, port),
            timeout=max(0.001, remaining),
        ) as connection:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False, "unavailable"
            connection.settimeout(max(0.001, remaining))
            connection.sendall(b"zPING\0")
            response = bytearray()
            while len(response) < _MAX_CLAMAV_RESPONSE_BYTES:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                connection.settimeout(max(0.001, remaining))
                chunk = connection.recv(min(16, _MAX_CLAMAV_RESPONSE_BYTES - len(response)))
                if not chunk:
                    break
                response.extend(chunk)
                if b"\0" in chunk or b"\n" in chunk:
                    break
        response_bytes = bytes(response)
        terminator_indexes = [
            index for marker in (b"\0", b"\n") if (index := response_bytes.find(marker)) >= 0
        ]
        if terminator_indexes:
            response_bytes = response_bytes[: min(terminator_indexes)]
        ok = response_bytes.rstrip(b"\r") == b"PONG"
        return ok, "ready" if ok else "unavailable"

    return check


def _is_private_service_host(host: str) -> bool:
    return _is_private_or_allowed_host(host, _ALLOWED_PRIVATE_SERVICE_HOSTS)


def _is_private_or_allowed_host(
    host: str,
    allowed_hosts: frozenset[str],
) -> bool:
    candidate = host.strip().lower().rstrip(".")
    if not candidate:
        return False
    try:
        address = ip_address(candidate)
    except ValueError:
        return candidate in allowed_hosts
    return address.is_loopback or any(
        address.version == network.version and address in network for network in _PRIVATE_NETWORKS
    )


def _generation_enabled(environ: Mapping[str, str]) -> bool:
    provider = environ.get("LLM_PROVIDER", "template").strip().lower().replace("-", "_")
    if provider == "openai_compatible":
        return True
    return any(
        parse_bool(environ.get(name), default=False)
        for name in ("GENERATION_ENABLED", "LLM_GENERATION_ENABLED")
        if name in environ
    )


def validate_health_service_urls(
    settings: OfflineSettings,
    environ: Mapping[str, str],
) -> None:
    services = [
        (
            "DATABASE_URL",
            settings.database_url,
            frozenset({"postgresql", "postgresql+psycopg"}),
            frozenset({"postgres", "localhost"}),
        ),
        (
            "CLICKHOUSE_URL",
            settings.clickhouse_url,
            frozenset({"http", "https"}),
            frozenset({"clickhouse", "localhost"}),
        ),
        (
            "QDRANT_URL",
            settings.qdrant_url,
            frozenset({"http", "https"}),
            frozenset({"qdrant", "localhost"}),
        ),
        (
            "REDIS_URL",
            settings.redis_url,
            frozenset({"redis", "rediss"}),
            frozenset({"redis", "localhost"}),
        ),
        (
            "EMBEDDING_SERVICE_URL",
            settings.embedding_service_url,
            frozenset({"http", "https"}),
            frozenset({"embedding-service", "localhost"}),
        ),
    ]
    if _generation_enabled(environ):
        services.append(
            (
                "LLAMA_SERVER_URL",
                settings.llama_server_url,
                frozenset({"http", "https"}),
                frozenset({"llama", "localhost"}),
            )
        )

    for field, value, allowed_schemes, allowed_hosts in services:
        scheme = ""
        port_valid = True
        port: int | None = None
        try:
            parsed = urlsplit(value)
            scheme = parsed.scheme.lower()
            host = parsed.hostname or ""
            port = parsed.port
        except ValueError:
            host = ""
            port_valid = False
        if (
            scheme not in allowed_schemes
            or not host
            or not port_valid
            or (port is not None and not 1 <= port <= 65535)
            or (settings.offline_mode and not _is_private_or_allowed_host(host, allowed_hosts))
        ):
            raise ValueError(f"{field} health endpoint must use a private or loopback host")


def build_dependency_checks(
    settings: OfflineSettings,
    *,
    database: object,
    environ: Mapping[str, str],
    http_client: Any | None = None,
    redis_client: Any | None = None,
    http_client_factory: HttpClientFactory | None = None,
) -> list[DependencyCheck]:
    validate_health_service_urls(settings, environ)
    timeout_seconds = _bounded_timeout(settings.dependency_timeout_seconds)
    raw_checks = [
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
        raw_checks.append(
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
    return [
        DependencyCheck(
            item.name,
            _hard_timeout_check(
                item.check,
                timeout_seconds,
                dependency_name=item.name,
            ),
        )
        for item in raw_checks
    ]


__all__ = [
    "DependencyCheck",
    "DependencyHealthRegistry",
    "build_dependency_checks",
    "create_http_health_client",
    "create_postgres_health_engine",
    "create_redis_health_client",
    "postgres_schema_revision_check",
    "validate_health_service_urls",
]
