from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

from .database import resolve_database_url


class OfflineSettingsError(ValueError):
    pass


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
    database_url: str
    clickhouse_url: str
    qdrant_url: str
    redis_url: str
    clamav_host: str
    embedding_service_url: str
    llama_server_url: str
    raw_data_root: Path
    parquet_root: Path
    model_root: Path
    model_slots: int
    dependency_timeout_seconds: float

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> OfflineSettings:
        offline_mode = parse_bool(environ.get("OFFLINE_MODE"), default=True)
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

        return cls(
            offline_mode=offline_mode,
            clamav_host=environ.get("CLAMAV_HOST", "127.0.0.1"),
            raw_data_root=Path(environ.get("RAW_DATA_ROOT", "./data/raw")),
            parquet_root=Path(environ.get("PARQUET_ROOT", "./data/parquet")),
            model_root=Path(environ.get("MODEL_ROOT", "./models")),
            model_slots=model_slots,
            dependency_timeout_seconds=float(environ.get("DEPENDENCY_TIMEOUT_SECONDS", "2.0")),
            **values,
        )
