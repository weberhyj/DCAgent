"""Opt-in acceptance coverage for a real private ClickHouse 18.16.1 target.

Run on the Ubuntu target only:

RUN_CLICKHOUSE_18_16=1 CLICKHOUSE_COMPATIBILITY_MODE=legacy_18_16 \
uv run --project backend --group dev --group offline pytest \
    backend/tests/integration/test_clickhouse_legacy_18_16.py -v
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import re
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import pytest

from app.offline_settings import OfflineSettingsError, read_secret_file, require_private_url


_LEGACY_MODE = "legacy_18_16"
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_TARGET_SERVER_VERSION = re.compile(r"^18\.16\.1$")
_CREATE_TABLE = re.compile(r"^CREATE TABLE ([a-z0-9_]+)\s*\(")
_RENAME_TABLE = re.compile(r"^RENAME TABLE ([a-z0-9_]+) TO ([a-z0-9_]+)$")


class TargetConfigurationError(ValueError):
    """The test target is not explicitly safe and fully configured."""


@dataclass(frozen=True, slots=True)
class _TargetEnvironment:
    host: str
    port: int
    secure: bool
    database: str | None
    ingest_user: str
    ingest_password: str = field(repr=False)
    query_user: str
    query_password: str = field(repr=False)


def _validate_target_environment(environ: Mapping[str, str]) -> _TargetEnvironment:
    """Validate every opt-in boundary before importing or connecting a client."""
    if environ.get("RUN_CLICKHOUSE_18_16", "").strip() != "1":
        raise TargetConfigurationError(
            "set RUN_CLICKHOUSE_18_16=1 for an explicit ClickHouse 18.16.1 target"
        )
    if environ.get("CLICKHOUSE_COMPATIBILITY_MODE", "").strip() != _LEGACY_MODE:
        raise TargetConfigurationError(
            "CLICKHOUSE_COMPATIBILITY_MODE must be exactly legacy_18_16"
        )

    raw_url = environ.get("CLICKHOUSE_URL", "").strip()
    if not raw_url:
        raise TargetConfigurationError("CLICKHOUSE_URL is required")
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise TargetConfigurationError("CLICKHOUSE_URL must be an http(s) private target URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise TargetConfigurationError(
            "CLICKHOUSE_URL must not contain credentials, query parameters, or fragments"
        )
    try:
        require_private_url(raw_url, "CLICKHOUSE_URL")
        port = parsed.port or (8443 if parsed.scheme == "https" else 8123)
    except (OfflineSettingsError, ValueError) as error:
        raise TargetConfigurationError(
            "CLICKHOUSE_URL must use a private or loopback host"
        ) from error

    database = parsed.path.strip("/") or None
    if database is not None and not _SAFE_IDENTIFIER.fullmatch(database):
        raise TargetConfigurationError("CLICKHOUSE_URL database path must be a safe identifier")

    ingest_user = _required_user(environ, "CLICKHOUSE_INGEST_USER")
    query_user = _required_user(environ, "CLICKHOUSE_QUERY_USER")
    if ingest_user == query_user:
        raise TargetConfigurationError(
            "CLICKHOUSE_INGEST_USER and CLICKHOUSE_QUERY_USER must be distinct"
        )
    for direct_key in ("CLICKHOUSE_INGEST_PASSWORD", "CLICKHOUSE_QUERY_PASSWORD"):
        if environ.get(direct_key, "").strip():
            raise TargetConfigurationError(f"{direct_key} is not supported; use its _FILE variable")
    return _TargetEnvironment(
        host=parsed.hostname,
        port=port,
        secure=parsed.scheme == "https",
        database=database,
        ingest_user=ingest_user,
        ingest_password=_password_from_file(environ, "CLICKHOUSE_INGEST_PASSWORD_FILE"),
        query_user=query_user,
        query_password=_password_from_file(environ, "CLICKHOUSE_QUERY_PASSWORD_FILE"),
    )


def _required_user(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise TargetConfigurationError(f"{name} is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise TargetConfigurationError(f"{name} contains unsupported characters")
    return value


def _password_from_file(environ: Mapping[str, str], name: str) -> str:
    configured = environ.get(name, "").strip()
    if not configured:
        raise TargetConfigurationError(f"{name} is required")
    try:
        return read_secret_file(Path(configured), name)
    except OfflineSettingsError as error:
        raise TargetConfigurationError(f"{name} must reference a readable secret file") from error


def _target_skip_reason(environ: Mapping[str, str]) -> str | None:
    if environ.get("RUN_CLICKHOUSE_18_16", "").strip() != "1":
        return (
            "ClickHouse 18.16.1 target guard: "
            "set RUN_CLICKHOUSE_18_16=1 for an explicit ClickHouse 18.16.1 target"
        )
    _validate_target_environment(environ)
    return None


def _require_18_16_1_version(version: str) -> str:
    normalized = version.strip()
    if not _TARGET_SERVER_VERSION.fullmatch(normalized):
        raise AssertionError("ClickHouse target must be exactly 18.16.1")
    return normalized


class TestLegacy18EnvironmentValidation:
    def test_requires_explicit_opt_in(self) -> None:
        with pytest.raises(TargetConfigurationError, match="RUN_CLICKHOUSE_18_16=1"):
            _validate_target_environment({})

    def test_requires_exact_legacy_mode_before_any_connection(self) -> None:
        with pytest.raises(TargetConfigurationError, match="exactly legacy_18_16"):
            _validate_target_environment({"RUN_CLICKHOUSE_18_16": "1"})

    def test_rejects_non_private_target_without_revealing_it(self) -> None:
        with pytest.raises(TargetConfigurationError, match="private or loopback") as caught:
            _validate_target_environment(
                {
                    "RUN_CLICKHOUSE_18_16": "1",
                    "CLICKHOUSE_COMPATIBILITY_MODE": _LEGACY_MODE,
                    "CLICKHOUSE_URL": "https://public.example.invalid:8443",
                }
            )
        assert "public.example.invalid" not in str(caught.value)

    def test_rejects_duplicate_service_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            password = root / "password"
            password.write_text("not-a-real-secret", encoding="utf-8")
            with pytest.raises(TargetConfigurationError, match="distinct"):
                _validate_target_environment(
                    {
                        "RUN_CLICKHOUSE_18_16": "1",
                        "CLICKHOUSE_COMPATIBILITY_MODE": _LEGACY_MODE,
                        "CLICKHOUSE_URL": "http://127.0.0.1:8123",
                        "CLICKHOUSE_INGEST_USER": "same-user",
                        "CLICKHOUSE_QUERY_USER": "same-user",
                        "CLICKHOUSE_INGEST_PASSWORD_FILE": str(password),
                        "CLICKHOUSE_QUERY_PASSWORD_FILE": str(password),
                    }
                )

    def test_requires_regular_readable_password_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            password = root / "password"
            password.write_text("ingest-password", encoding="utf-8")
            with pytest.raises(TargetConfigurationError, match="QUERY_PASSWORD_FILE"):
                _validate_target_environment(
                    {
                        "RUN_CLICKHOUSE_18_16": "1",
                        "CLICKHOUSE_COMPATIBILITY_MODE": _LEGACY_MODE,
                        "CLICKHOUSE_URL": "http://127.0.0.1:8123",
                        "CLICKHOUSE_INGEST_USER": "ingest-user",
                        "CLICKHOUSE_QUERY_USER": "query-user",
                        "CLICKHOUSE_INGEST_PASSWORD_FILE": str(password),
                        "CLICKHOUSE_QUERY_PASSWORD_FILE": str(root / "missing-password"),
                    }
                )

    def test_builds_redacted_private_target_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ingest_password = root / "ingest-password"
            query_password = root / "query-password"
            ingest_password.write_text("ingest-secret", encoding="utf-8")
            query_password.write_text("query-secret", encoding="utf-8")

            target = _validate_target_environment(
                {
                    "RUN_CLICKHOUSE_18_16": "1",
                    "CLICKHOUSE_COMPATIBILITY_MODE": _LEGACY_MODE,
                    "CLICKHOUSE_URL": "https://127.0.0.1:8443/legacy",
                    "CLICKHOUSE_INGEST_USER": "ingest-user",
                    "CLICKHOUSE_QUERY_USER": "query-user",
                    "CLICKHOUSE_INGEST_PASSWORD_FILE": str(ingest_password),
                    "CLICKHOUSE_QUERY_PASSWORD_FILE": str(query_password),
                }
            )

        assert (target.host, target.port, target.secure, target.database) == (
            "127.0.0.1",
            8443,
            True,
            "legacy",
        )
        assert "ingest-secret" not in repr(target)
        assert "query-secret" not in repr(target)


class TestLegacy1816AcceptanceGuards:
    def test_non_opted_in_complete_target_skips_without_importing_dependencies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ingest_password = root / "ingest-password"
            query_password = root / "query-password"
            ingest_password.write_text("ingest-secret", encoding="utf-8")
            query_password.write_text("query-secret", encoding="utf-8")
            environ = {
                "CLICKHOUSE_COMPATIBILITY_MODE": _LEGACY_MODE,
                "CLICKHOUSE_URL": "http://127.0.0.1:8123",
                "CLICKHOUSE_INGEST_USER": "ingest-user",
                "CLICKHOUSE_QUERY_USER": "query-user",
                "CLICKHOUSE_INGEST_PASSWORD_FILE": str(ingest_password),
                "CLICKHOUSE_QUERY_PASSWORD_FILE": str(query_password),
            }

            def dependencies_must_not_be_imported(name: str) -> object:
                raise AssertionError(f"unexpected dependency import: {name}")

            def target_validation_must_not_run(environ: Mapping[str, str]) -> _TargetEnvironment:
                raise AssertionError("unexpected target validation")

            monkeypatch.setitem(
                globals(), "_validate_target_environment", target_validation_must_not_run
            )

            assert _acceptance_collection_guard(
                environ, dependencies_must_not_be_imported
            ) == (
                "ClickHouse 18.16.1 target guard: "
                "set RUN_CLICKHOUSE_18_16=1 for an explicit ClickHouse 18.16.1 target"
            )

    def test_opted_in_invalid_target_fails_collection_guard_instead_of_skipping(self) -> None:
        with pytest.raises(TargetConfigurationError, match="CLICKHOUSE_COMPATIBILITY_MODE"):
            _acceptance_collection_guard({"RUN_CLICKHOUSE_18_16": "1"})

    def test_opted_in_missing_dependency_fails_collection_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ingest_password = root / "ingest-password"
            query_password = root / "query-password"
            ingest_password.write_text("ingest-secret", encoding="utf-8")
            query_password.write_text("query-secret", encoding="utf-8")
            environ = {
                "RUN_CLICKHOUSE_18_16": "1",
                "CLICKHOUSE_COMPATIBILITY_MODE": _LEGACY_MODE,
                "CLICKHOUSE_URL": "http://127.0.0.1:8123",
                "CLICKHOUSE_INGEST_USER": "ingest-user",
                "CLICKHOUSE_QUERY_USER": "query-user",
                "CLICKHOUSE_INGEST_PASSWORD_FILE": str(ingest_password),
                "CLICKHOUSE_QUERY_PASSWORD_FILE": str(query_password),
            }

            def missing_dependency(name: str) -> object:
                raise ModuleNotFoundError(name)

            with pytest.raises(TargetConfigurationError, match="pyarrow"):
                _acceptance_collection_guard(environ, missing_dependency)

    @pytest.mark.parametrize(
        "version",
        ("18.16.1",),
    )
    def test_accepts_only_exact_18_16_1(self, version: str) -> None:
        assert _require_18_16_1_version(version) == version

    @pytest.mark.parametrize(
        "version",
        ("18.16.2", "18.16.10", "18.16.1.42", "18.16.1-stable", "18.16.1-", "18.16.1+build", "v18.16.1"),
    )
    def test_rejects_non_target_server_versions(self, version: str) -> None:
        with pytest.raises(AssertionError, match="18.16.1"):
            _require_18_16_1_version(version)

    def test_tracks_generated_staging_table_from_production_gateway_ddl(self) -> None:
        from app.clickhouse_compatibility import (
            ClickHouseCompatibilityMode,
            ClickHouseCompatibilityProfile,
        )
        from app.clickhouse_gateway import ClickHouseGateway

        client = _DdlClient()
        tracking = _RecordingIngestClient(client)
        gateway = ClickHouseGateway(
            tracking,
            compatibility=ClickHouseCompatibilityProfile.for_mode(
                ClickHouseCompatibilityMode.LEGACY_18_16
            ),
        )
        schema = _legacy_schema(
            "it18_16_tracking_gateway",
            (("amount", "decimal", True),),
        )

        target = gateway.prepare_publication(schema, "tracking-publication", "a" * 64)

        assert target.staging_table in tracking.created_tables
        assert any(
            statement.startswith(f"CREATE TABLE {target.staging_table} (")
            for statement in client.commands
        )

    def test_retains_create_candidate_when_server_executes_then_raises(self) -> None:
        from app.clickhouse_compatibility import (
            ClickHouseCompatibilityMode,
            ClickHouseCompatibilityProfile,
        )
        from app.clickhouse_gateway import ClickHouseGateway

        client = _DdlClient(raise_after_create=True)
        tracking = _RecordingIngestClient(client)
        gateway = ClickHouseGateway(
            tracking,
            compatibility=ClickHouseCompatibilityProfile.for_mode(
                ClickHouseCompatibilityMode.LEGACY_18_16
            ),
        )
        schema = _legacy_schema(
            "it18_16_tracking_failure",
            (("amount", "decimal", True),),
        )

        with pytest.raises(ConnectionError, match="after execution"):
            gateway.prepare_publication(schema, "tracking-publication", "b" * 64)

        created_statement = next(
            statement for statement in client.commands if statement.startswith("CREATE TABLE ")
        )
        created_table = created_statement.split()[2]
        assert tracking.created_tables == {created_table}
        assert _cleanup_target_clients(tracking, None) is None
        assert f"DROP TABLE IF EXISTS {created_table}" in client.commands

    def test_retains_rename_source_and_target_when_server_executes_then_raises(self) -> None:
        source = "structured_it18_16_rename_source"
        target = "structured_it18_16_rename_target"
        client = _DdlClient(raise_after_rename=True)
        tracking = _RecordingIngestClient(client)
        tracking.created_tables.add(source)

        with pytest.raises(ConnectionError, match="after execution"):
            tracking.command(f"RENAME TABLE {source} TO {target}")

        assert tracking.created_tables == {source, target}
        assert _cleanup_target_clients(tracking, None) is None
        assert f"DROP TABLE IF EXISTS {source}" in client.commands
        assert f"DROP TABLE IF EXISTS {target}" in client.commands


class _DdlClient:
    def __init__(
        self,
        *,
        raise_after_create: bool = False,
        raise_after_rename: bool = False,
    ) -> None:
        self.commands: list[str] = []
        self.raise_after_create = raise_after_create
        self.raise_after_rename = raise_after_rename

    def command(self, statement: str, **_kwargs: object) -> None:
        self.commands.append(statement)
        if statement.startswith("CREATE TABLE ") and self.raise_after_create:
            raise ConnectionError("server executed CREATE before connection failed")
        if statement.startswith("RENAME TABLE ") and self.raise_after_rename:
            raise ConnectionError("server executed RENAME before connection failed")

    def close(self) -> None:
        return None


class _RecordingIngestClient:
    """Delegate every production client operation while tracking only tables this test creates."""

    def __init__(self, client: object) -> None:
        self._client = client
        self.batch_sizes: list[int] = []
        self.created_tables: set[str] = set()

    def command(self, statement: str, **kwargs: object) -> object:
        self._reserve_owned_ddl_candidates(statement)
        return self._client.command(statement, **kwargs)

    def _reserve_owned_ddl_candidates(self, statement: str) -> None:
        if match := _CREATE_TABLE.match(statement):
            table_name = match.group(1)
            if _is_owned_test_table(table_name):
                self.created_tables.add(table_name)
        elif match := _RENAME_TABLE.fullmatch(statement):
            source, target = match.groups()
            if source in self.created_tables and _is_owned_test_table(target):
                self.created_tables.update((source, target))

    def insert_arrow(self, table: str, batch: object, **kwargs: object) -> object:
        self.batch_sizes.append(int(getattr(batch, "num_rows")))
        return self._client.insert_arrow(table, batch, **kwargs)

    def close(self) -> object:
        return self._client.close()


def _require_acceptance_dependencies(
    import_module: Callable[[str], object] = importlib.import_module,
) -> object:
    for dependency in ("pyarrow", "openpyxl"):
        try:
            import_module(dependency)
        except ModuleNotFoundError as error:
            raise TargetConfigurationError(
                f"{dependency} is required when RUN_CLICKHOUSE_18_16=1"
            ) from error
    try:
        return import_module("clickhouse_connect")
    except ModuleNotFoundError as error:
        raise TargetConfigurationError(
            "clickhouse_connect is required when RUN_CLICKHOUSE_18_16=1"
        ) from error


def _acceptance_collection_guard(
    environ: Mapping[str, str],
    import_module: Callable[[str], object] = importlib.import_module,
) -> str | None:
    reason = _target_skip_reason(environ)
    if reason is None:
        _require_acceptance_dependencies(import_module)
    return reason


_TARGET_SKIP_REASON = _acceptance_collection_guard(os.environ)


@contextlib.contextmanager
def _legacy_gateway() -> Iterator[tuple[object, _RecordingIngestClient]]:
    """Build production clients only after all target guards and dependency checks pass."""
    target = _validate_target_environment(os.environ)
    clickhouse_connect = _require_acceptance_dependencies()
    common = {
        "host": target.host,
        "port": target.port,
        "secure": target.secure,
        "database": target.database,
    }
    ingest_client = clickhouse_connect.get_client(
        **common,
        username=target.ingest_user,
        password=target.ingest_password,
    )
    tracking_ingest = _RecordingIngestClient(ingest_client)
    query_client = None
    primary_error: BaseException | None = None
    try:
        query_client = clickhouse_connect.get_client(
            **common,
            username=target.query_user,
            password=target.query_password,
            autogenerate_session_id=False,
        )
        from app.clickhouse_compatibility import (
            ClickHouseCompatibilityMode,
            ClickHouseCompatibilityProfile,
        )
        from app.clickhouse_gateway import ClickHouseGateway

        profile = ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16)
        yield ClickHouseGateway(
            tracking_ingest,
            query_client=query_client,
            compatibility=profile,
        ), tracking_ingest
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error = _cleanup_target_clients(tracking_ingest, query_client)
        if cleanup_error is not None:
            if primary_error is None:
                raise cleanup_error
            primary_error.add_note(f"ClickHouse 18.16 test cleanup failed: {cleanup_error}")


def _cleanup_target_clients(
    ingest: _RecordingIngestClient,
    query: object | None,
) -> BaseException | None:
    """Drop only exact generated names observed after this test successfully created them."""
    failures: list[BaseException] = []
    for table_name in sorted(ingest.created_tables):
        if not _is_owned_test_table(table_name):
            failures.append(RuntimeError("refusing to clean an unowned ClickHouse test table"))
            continue
        try:
            ingest.command(f"DROP TABLE IF EXISTS {table_name}")
        except BaseException as error:
            failures.append(error)
    for client in (query, ingest):
        if client is None:
            continue
        try:
            client.close()
        except BaseException as error:
            failures.append(error)
    return failures[0] if failures else None


def _is_owned_test_table(table_name: str) -> bool:
    return bool(
        _SAFE_IDENTIFIER.fullmatch(table_name) and table_name.startswith("structured_it18_16_")
    )


def _legacy_schema(dataset_id: str, columns: tuple[tuple[str, str, bool], ...]) -> object:
    from app.structured_models import (
        StructuredColumnSchema,
        StructuredColumnType,
        StructuredDatasetSchema,
    )

    resolved_columns = tuple(
        StructuredColumnSchema(
            physical_name=name,
            original_name=name,
            display_name=name,
            data_type=StructuredColumnType(data_type),
            aliases=(),
            allow_aggregate=allow_aggregate,
            allow_filter=True,
        )
        for name, data_type, allow_aggregate in columns
    )
    return StructuredDatasetSchema(
        dataset_id=dataset_id,
        source_id=f"source_{dataset_id}",
        worksheet_name="Legacy",
        schema_version=1,
        columns=resolved_columns,
        schema_hash=hashlib.sha256(dataset_id.encode("utf-8")).hexdigest(),
    )


def _write_workbook(
    path: Path,
    headers: tuple[str, ...],
    rows: Iterator[tuple[object, ...]],
) -> None:
    from openpyxl import Workbook

    workbook = Workbook(write_only=True)
    worksheet = workbook.create_sheet("Legacy")
    worksheet.append(headers)
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    workbook.close()


def _catalog_for_publication(schema: object, result: object) -> tuple[object, object]:
    from app.structured_models import (
        StructuredCatalog,
        StructuredDatasetCatalog,
        StructuredPublication,
    )

    publication = StructuredPublication(
        publication_id=result.publication_id,
        dataset_id=schema.dataset_id,
        schema_version=schema.schema_version,
        physical_table_name=result.physical_table_name,
        row_count=result.row_count,
        content_hash=result.content_hash,
    )
    return StructuredCatalog(
        datasets=(
            StructuredDatasetCatalog(
                schema=schema,
                source_name="ClickHouse 18.16 acceptance",
                active_publication=publication,
            ),
        )
    ), publication


def _result_rows(result: object) -> list[tuple[object, ...]]:
    return [tuple(row) for row in result.result_rows]


@pytest.mark.skipif(
    os.environ.get("RUN_CLICKHOUSE_18_16", "").strip() != "1",
    reason="set RUN_CLICKHOUSE_18_16=1 for an explicit ClickHouse 18.16.1 target",
)
class TestClickHouseLegacy1816Acceptance:
    def test_server_is_18_16_and_legacy_preflight_passes(self) -> None:
        with _legacy_gateway() as (gateway, _tracking_ingest):
            assert _require_18_16_1_version(gateway.preflight())

    def test_publishes_supported_types_and_queries_datetime_and_decimals(self) -> None:
        _require_acceptance_dependencies()
        from app.clickhouse_compatibility import (
            ClickHouseCompatibilityMode,
            ClickHouseCompatibilityProfile,
        )
        from app.structured_ingestion import ArrowParquetSink, SpreadsheetPublisher
        from app.structured_models import (
            StructuredFilter,
            StructuredIntent,
            StructuredMetricIntent,
            StructuredMultiAggregateIntent,
        )
        from app.structured_query import StructuredQueryExecutor, StructuredQueryPlanner

        dataset_id = f"it18_16_small_{uuid.uuid4().hex}"
        schema = _legacy_schema(
            dataset_id,
            (
                ("label", "string", False),
                ("units", "integer", True),
                ("amount", "decimal", True),
                ("business_date", "date", False),
                ("occurred_at", "datetime", False),
                ("enabled", "boolean", False),
            ),
        )
        rows = (
            (
                "中文一",
                7,
                "1",
                date(2026, 1, 1),
                datetime(2026, 1, 1, 9, 30, 45, 654321),
                True,
            ),
            (
                "中文二",
                3,
                "1.2",
                date(2026, 1, 2),
                datetime(2026, 1, 2, 9, 30, 45, 123456),
                False,
            ),
            ("空值", None, None, None, None, None),
        )
        profile = ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16)

        with tempfile.TemporaryDirectory() as directory, _legacy_gateway() as (gateway, _tracking):
            root = Path(directory)
            workbook_path = root / "legacy-small.xlsx"
            _write_workbook(
                workbook_path,
                tuple(column[0] for column in schema.columns),
                iter(rows),
            )
            assert _require_18_16_1_version(gateway.preflight())
            result = SpreadsheetPublisher(
                sink=ArrowParquetSink(root / "parquet"),
                clickhouse=gateway,
                compatibility=profile,
                batch_rows=2,
            ).publish(workbook_path, schema, f"publication_{uuid.uuid4().hex}")

            assert result.row_count == 3
            assert result.null_counts == {
                "label": 0,
                "units": 1,
                "amount": 1,
                "business_date": 1,
                "occurred_at": 1,
                "enabled": 1,
            }
            described = _result_rows(gateway.query(f"DESCRIBE TABLE {result.physical_table_name}"))
            described_types = {row[0]: row[1] for row in described}
            assert described_types == {
                "label": "Nullable(String)",
                "units": "Nullable(Int64)",
                "amount": "Nullable(Decimal(38, 9))",
                "business_date": "Nullable(Date)",
                "occurred_at": "Nullable(DateTime)",
                "enabled": "Nullable(UInt8)",
                "_source_id": "String",
                "_dataset_id": "String",
                "_schema_version": "UInt64",
                "_worksheet": "String",
                "_row_number": "UInt64",
                "_content_hash": "String",
            }
            assert "DateTime64" not in described_types["occurred_at"]
            digest = _result_rows(
                gateway.query(
                    f"SELECT uniqExact(_content_hash), any(_content_hash) "
                    f"FROM {result.physical_table_name}"
                )
            )
            assert digest == [(1, result.content_hash)]
            stored = _result_rows(
                gateway.query(
                    f"SELECT label, amount, toString(amount), occurred_at "
                    f"FROM {result.physical_table_name} "
                    "ORDER BY _row_number"
                )
            )
            assert stored[0][0] == "中文一"
            assert stored[1][0] == "中文二"
            assert stored[0][1:3] == (Decimal("1.000000000"), "1.000000000")
            assert stored[1][1:3] == (Decimal("1.200000000"), "1.200000000")
            assert stored[0][3] == datetime(2026, 1, 1, 9, 30, 45)

            catalog, publication = _catalog_for_publication(schema, result)
            planner = StructuredQueryPlanner(catalog, compatibility=profile)
            executor = StructuredQueryExecutor(catalog, gateway, compatibility=profile)
            datetime_range = planner.plan(
                StructuredIntent(
                    dataset_id,
                    "count",
                    None,
                    (StructuredFilter("occurred_at", "between", "2026-01-01", "2026-01-01"),),
                ),
                publication,
            )
            range_result = executor.execute(datetime_range)
            assert range_result.value == 1
            assert (
                range_result.total_count,
                range_result.valid_count,
                range_result.null_count,
            ) == (1, 1, 0)

            equality_result = executor.execute(
                planner.plan(
                    StructuredIntent(
                        dataset_id,
                        "count",
                        None,
                        (StructuredFilter("label", "eq", "中文二"),),
                    ),
                    publication,
                )
            )
            assert equality_result.value == 1

            comparison_cases = (
                ("gt", "1", 1),
                ("gte", "1.2", 1),
                ("lt", "1.2", 1),
                ("lte", "1", 1),
            )
            for operator, threshold, expected_count in comparison_cases:
                comparison_result = executor.execute(
                    planner.plan(
                        StructuredIntent(
                            dataset_id,
                            "count",
                            None,
                            (StructuredFilter("amount", operator, threshold),),
                        ),
                        publication,
                    )
                )
                assert comparison_result.value == expected_count

            sum_result = executor.execute(
                planner.plan(StructuredIntent(dataset_id, "sum", "amount", ()), publication)
            )
            average_result = executor.execute(
                planner.plan(StructuredIntent(dataset_id, "avg", "amount", ()), publication)
            )
            minimum_result = executor.execute(
                planner.plan(StructuredIntent(dataset_id, "min", "amount", ()), publication)
            )
            maximum_result = executor.execute(
                planner.plan(StructuredIntent(dataset_id, "max", "amount", ()), publication)
            )
            assert sum_result.value == Decimal("2.200000000")
            assert average_result.value == Decimal("1.100000000")
            assert minimum_result.value == Decimal("1.000000000")
            assert maximum_result.value == Decimal("1.200000000")
            assert (
                sum_result.total_count,
                sum_result.valid_count,
                sum_result.null_count,
            ) == (3, 2, 1)

            multi_plan = planner.plan_multi(
                StructuredMultiAggregateIntent(
                    dataset_id,
                    (
                        StructuredMetricIntent("sum", "amount"),
                        StructuredMetricIntent("sum", "units"),
                    ),
                    (),
                    False,
                ),
                publication,
            )
            assert multi_plan.sql.count("SELECT") == 1
            assert "toDecimalString" not in multi_plan.sql
            assert " WITH " not in f" {multi_plan.sql.upper()} "
            assert " OVER " not in f" {multi_plan.sql.upper()} "
            assert " JSON" not in multi_plan.sql.upper()

            multi_result = executor.execute_multi(multi_plan)

            assert multi_result.total_count == 3
            assert [metric.value for metric in multi_result.metrics] == [
                Decimal("2.200000000"),
                Decimal("10"),
            ]
            assert [
                (metric.valid_count, metric.null_count)
                for metric in multi_result.metrics
            ] == [(2, 1), (2, 1)]

    def test_publishes_100000_rows_with_bounded_insert_batches(self) -> None:
        _require_acceptance_dependencies()
        from app.clickhouse_compatibility import (
            ClickHouseCompatibilityMode,
            ClickHouseCompatibilityProfile,
        )
        from app.structured_ingestion import ArrowParquetSink, SpreadsheetPublisher

        row_count = 100_000
        batch_rows = 4_096
        dataset_id = f"it18_16_large_{uuid.uuid4().hex}"
        schema = _legacy_schema(
            dataset_id,
            (("label", "string", False), ("amount", "decimal", True)),
        )
        profile = ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16)

        def generated_rows() -> Iterator[tuple[object, ...]]:
            for index in range(row_count):
                yield (f"第{index}行", f"{index % 1000}.123456789")

        with (
            tempfile.TemporaryDirectory() as directory,
            _legacy_gateway() as (gateway, tracking_ingest),
        ):
            root = Path(directory)
            workbook_path = root / "legacy-100k.xlsx"
            _write_workbook(
                workbook_path,
                tuple(column[0] for column in schema.columns),
                generated_rows(),
            )
            assert _require_18_16_1_version(gateway.preflight())
            result = SpreadsheetPublisher(
                sink=ArrowParquetSink(root / "parquet"),
                clickhouse=gateway,
                compatibility=profile,
                batch_rows=batch_rows,
            ).publish(workbook_path, schema, f"publication_{uuid.uuid4().hex}")

            assert result.row_count == row_count
            assert tracking_ingest.batch_sizes
            assert sum(tracking_ingest.batch_sizes) == row_count
            assert max(tracking_ingest.batch_sizes) <= batch_rows
            assert len(tracking_ingest.batch_sizes) > 1
            row_count_result = _result_rows(
                gateway.query(f"SELECT count() FROM {result.physical_table_name}")
            )
            assert row_count_result == [(row_count,)]
