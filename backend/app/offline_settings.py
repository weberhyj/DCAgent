from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

from .database import resolve_database_url


class OfflineSettingsError(ValueError):
    pass


_SECRET_MAX_BYTES = 4096


def read_secret_file(path: str | Path, field_name: str) -> str:
    """Read a small UTF-8 secret without accepting links, directories, or plaintext env values."""
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            raise OfflineSettingsError(f"{field_name} must reference a regular non-link file")
        if candidate.stat().st_size > _SECRET_MAX_BYTES:
            raise OfflineSettingsError(f"{field_name} must be at most {_SECRET_MAX_BYTES} bytes")
        raw = candidate.read_bytes()
    except OfflineSettingsError:
        raise
    except OSError as error:
        raise OfflineSettingsError(f"{field_name} could not be read") from error
    try:
        value = raw.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as error:
        raise OfflineSettingsError(f"{field_name} must be UTF-8") from error
    if not value or "\x00" in value or "\r" in value or "\n" in value:
        raise OfflineSettingsError(f"{field_name} must contain one non-empty secret value")
    return value


def require_secret_file(path: Path | None, field_name: str) -> str:
    if path is None:
        raise OfflineSettingsError(f"{field_name} is required when STRUCTURED_QUERY_ENABLED=true")
    return read_secret_file(path, field_name)


def _password_file_from_environ(
    environ: Mapping[str, str],
    *,
    path_key: str,
) -> Path | None:
    direct_key = path_key.removesuffix("_FILE")
    if environ.get(direct_key):
        raise OfflineSettingsError(f"{direct_key} is not supported; use {path_key}")
    configured = environ.get(path_key, "").strip()
    if not configured:
        return None
    return Path(configured)


def _clickhouse_user(environ: Mapping[str, str], key: str, default: str) -> str:
    value = environ.get(key, default).strip()
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in value
    ):
        raise OfflineSettingsError(f"{key} must contain only letters, digits, '.', '-', or '_'")
    return value


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise OfflineSettingsError("Boolean value must be one of: 1, true, yes, on, 0, false, no, off")


def require_private_url(value: str, field: str) -> str:
    candidate = value.strip()
    parsed = urlparse(candidate)
    if field.lower() == "database_url":
        routing_keys = {key.lower() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
        forbidden_keys = routing_keys & {"host", "hostaddr", "service", "servicefile"}
        if forbidden_keys:
            keys = ", ".join(sorted(forbidden_keys))
            raise OfflineSettingsError(
                f"{field} offline mode forbids PostgreSQL connection-routing query parameters: {keys}"
            )
    host = parsed.hostname or ""
    if host in {
        "localhost",
        "postgres",
        "clickhouse",
        "qdrant",
        "redis",
        "embedding-service",
        "llama",
    }:
        return candidate.rstrip("/")
    try:
        address = ip_address(host)
    except ValueError as error:
        raise OfflineSettingsError(f"{field} must use a private or loopback host") from error
    if not (address.is_private or address.is_loopback):
        raise OfflineSettingsError(f"{field} must use a private or loopback host")
    return candidate.rstrip("/")


@dataclass(frozen=True, slots=True)
class OfflineSettings:
    offline_mode: bool
    structured_query_enabled: bool
    database_url: str
    clickhouse_url: str
    clickhouse_query_user: str
    clickhouse_query_password_file: Path | None
    clickhouse_ingest_user: str
    clickhouse_ingest_password_file: Path | None
    structured_query_timeout_seconds: int
    qdrant_url: str
    redis_url: str
    clamav_host: str
    embedding_service_url: str
    llama_server_url: str
    raw_data_root: Path
    parquet_root: Path
    structured_ingest_batch_rows: int
    model_root: Path
    model_slots: int
    dependency_timeout_seconds: float

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> OfflineSettings:
        offline_mode = parse_bool(environ.get("OFFLINE_MODE"), default=True)
        structured_query_enabled = parse_bool(
            environ.get("STRUCTURED_QUERY_ENABLED"), default=False
        )
        query_password_file = _password_file_from_environ(
            environ,
            path_key="CLICKHOUSE_QUERY_PASSWORD_FILE",
        )
        ingest_password_file = _password_file_from_environ(
            environ,
            path_key="CLICKHOUSE_INGEST_PASSWORD_FILE",
        )
        values = {
            "database_url": resolve_database_url(environ),
            "clickhouse_url": environ.get("CLICKHOUSE_URL", "http://127.0.0.1:8123"),
            "qdrant_url": environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
            "redis_url": environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
            "embedding_service_url": environ.get("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8081"),
            "llama_server_url": environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080"),
        }
        if offline_mode:
            values = {key: require_private_url(value, key) for key, value in values.items()}

        try:
            model_slots = int(environ.get("MODEL_SLOTS", "2"))
        except ValueError as error:
            raise OfflineSettingsError("MODEL_SLOTS must be between 1 and 4") from error
        if model_slots not in {1, 2, 3, 4}:
            raise OfflineSettingsError("MODEL_SLOTS must be between 1 and 4")

        try:
            structured_ingest_batch_rows = int(environ.get("STRUCTURED_INGEST_BATCH_ROWS", "50000"))
        except ValueError as error:
            raise OfflineSettingsError(
                "STRUCTURED_INGEST_BATCH_ROWS must be between 1 and 50000"
            ) from error
        if not 1 <= structured_ingest_batch_rows <= 50_000:
            raise OfflineSettingsError("STRUCTURED_INGEST_BATCH_ROWS must be between 1 and 50000")

        try:
            structured_query_timeout_seconds = int(
                environ.get("STRUCTURED_QUERY_TIMEOUT_SECONDS", "4")
            )
        except ValueError as error:
            raise OfflineSettingsError(
                "STRUCTURED_QUERY_TIMEOUT_SECONDS must be between 1 and 60 seconds"
            ) from error
        if not 1 <= structured_query_timeout_seconds <= 60:
            raise OfflineSettingsError(
                "STRUCTURED_QUERY_TIMEOUT_SECONDS must be between 1 and 60 seconds"
            )

        return cls(
            offline_mode=offline_mode,
            structured_query_enabled=structured_query_enabled,
            clamav_host=environ.get("CLAMAV_HOST", "127.0.0.1"),
            clickhouse_query_user=_clickhouse_user(
                environ, "CLICKHOUSE_QUERY_USER", "dc_agent_query"
            ),
            clickhouse_query_password_file=query_password_file,
            clickhouse_ingest_user=_clickhouse_user(
                environ, "CLICKHOUSE_INGEST_USER", "dc_agent_ingest"
            ),
            clickhouse_ingest_password_file=ingest_password_file,
            structured_query_timeout_seconds=structured_query_timeout_seconds,
            raw_data_root=Path(environ.get("RAW_DATA_ROOT", "./data/raw")),
            parquet_root=Path(environ.get("PARQUET_ROOT", "./data/parquet")),
            structured_ingest_batch_rows=structured_ingest_batch_rows,
            model_root=Path(environ.get("MODEL_ROOT", "./models")),
            model_slots=model_slots,
            dependency_timeout_seconds=float(environ.get("DEPENDENCY_TIMEOUT_SECONDS", "2.0")),
            **values,
        )
