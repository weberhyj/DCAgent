from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .structured_models import (
    StructuredColumnSchema,
    StructuredColumnType,
    StructuredDatasetSchema,
)

_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]+$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_UINT64_MASK = (1 << 64) - 1
_METADATA_COLUMNS = (
    ("_source_id", "String"),
    ("_dataset_id", "String"),
    ("_schema_version", "UInt64"),
    ("_worksheet", "String"),
    ("_row_number", "UInt64"),
    ("_content_hash", "String"),
)


class StructuredStorageError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ClickHousePublicationTarget:
    schema: StructuredDatasetSchema
    staging_table: str
    physical_table_name: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class StructuredValidationStatistics:
    row_count: int
    column_count: int
    null_counts: Mapping[str, int]
    numeric_ranges: Mapping[str, tuple[Decimal | float | int | None, Decimal | float | int | None]]
    content_hash: str


class ClickHouseGateway:
    def __init__(
        self,
        ingest_client: Any,
        *,
        query_client: Any | None = None,
        max_execution_time: int = 30,
        max_memory_usage: int = 512 * 1024 * 1024,
        max_result_rows: int = 10_000,
    ) -> None:
        if query_client is ingest_client and query_client is not None:
            raise StructuredStorageError(
                "ClickHouse ingest and query clients must be separate read-only identities"
            )
        self._ingest_client = ingest_client
        self._query_client = query_client
        self._settings = {
            "max_execution_time": max_execution_time,
            "max_memory_usage": max_memory_usage,
            "max_result_rows": max_result_rows,
            "overflow_mode": "break",
        }
        self._query_settings = {**self._settings, "readonly": 1}

    def create_table(
        self,
        table_name: str,
        columns: Sequence[StructuredColumnSchema],
    ) -> None:
        _require_identifier(table_name)
        definitions = []
        for column in columns:
            _require_identifier(column.physical_name)
            definitions.append(f"{column.physical_name} {_clickhouse_type(column.data_type)}")
        definitions.extend(f"{name} {column_type}" for name, column_type in _METADATA_COLUMNS)
        statement = (
            f"CREATE TABLE {table_name} ("
            + ", ".join(definitions)
            + ") ENGINE = MergeTree ORDER BY (_dataset_id, _row_number)"
        )
        self._command(statement)

    def prepare_publication(
        self,
        schema: StructuredDatasetSchema,
        publication_id: str,
        content_hash: str,
        *,
        staging_generation: int | None = None,
        staging_token: str | None = None,
    ) -> ClickHousePublicationTarget:
        if not _HASH_RE.fullmatch(content_hash):
            raise StructuredStorageError("content hash must be a lowercase SHA-256 value")
        if staging_generation is not None and not (
            1 <= staging_generation <= 99_999_999_999_999_999_999
        ):
            raise StructuredStorageError(
                "ClickHouse staging generation must be a positive 20-digit integer"
            )
        dataset_component = _safe_component(schema.dataset_id)
        publication_suffix = hashlib.sha256(publication_id.encode("utf-8")).hexdigest()[:12]
        physical_table_name = (
            f"structured_{dataset_component}_v{schema.schema_version}_{publication_suffix}"
        )
        self._discard_stale_staging_tables(physical_table_name, staging_generation)
        if staging_generation is None:
            staging_table = f"{physical_table_name}_staging_{secrets.token_hex(6)}"
        else:
            owner = (
                secrets.token_hex(6)
                if staging_token is None
                else hashlib.sha256(staging_token.encode("utf-8")).hexdigest()[:12]
            )
            staging_table = f"{physical_table_name}_staging_g{staging_generation:020d}_{owner}"
        _require_identifier(physical_table_name)
        _require_identifier(staging_table)
        self._command(f"DROP TABLE IF EXISTS {staging_table}")
        self.create_table(staging_table, schema.columns)
        return ClickHousePublicationTarget(
            schema=schema,
            staging_table=staging_table,
            physical_table_name=physical_table_name,
            content_hash=content_hash,
        )

    def _discard_stale_staging_tables(
        self,
        physical_table_name: str,
        staging_generation: int | None,
    ) -> None:
        if self._query_client is None:
            return
        _require_identifier(physical_table_name)
        prefix = f"{physical_table_name}_staging_"
        result = self._query(
            "SELECT name FROM system.tables "
            "WHERE database = currentDatabase() "
            f"AND startsWith(name, '{prefix}')"
        )
        legacy_name = re.compile(rf"^{re.escape(prefix)}[0-9a-f]{{12}}$")
        generated_name = re.compile(rf"^{re.escape(prefix)}g([0-9]{{20}})_[0-9a-f]{{12}}$")
        for table_name in _as_single_column_values(result):
            if staging_generation is None:
                should_drop = legacy_name.fullmatch(table_name) is not None
            else:
                match = generated_name.fullmatch(table_name)
                should_drop = match is not None and int(match.group(1)) < staging_generation
            if should_drop:
                self._command(f"DROP TABLE IF EXISTS {table_name}")

    def insert_batch(self, target: ClickHousePublicationTarget, batch: Any) -> None:
        _require_identifier(target.staging_table)
        insert_arrow = getattr(self._ingest_client, "insert_arrow", None)
        if insert_arrow is not None:
            insert_arrow(target.staging_table, batch, settings=dict(self._settings))
            return
        insert = getattr(self._ingest_client, "insert", None)
        if insert is None:
            raise StructuredStorageError("ClickHouse ingest client cannot insert Arrow batches")
        insert(target.staging_table, batch, settings=dict(self._settings))

    def validate_and_promote(
        self,
        target: ClickHousePublicationTarget,
        *,
        statistics: StructuredValidationStatistics | None = None,
        **raw_statistics: Any,
    ) -> str:
        _require_identifier(target.staging_table)
        _require_identifier(target.physical_table_name)
        if statistics is None:
            statistics = StructuredValidationStatistics(**raw_statistics)
        if statistics.content_hash != target.content_hash:
            raise StructuredStorageError("publication content hash changed before validation")

        self._validate_table(target, target.staging_table, statistics)
        try:
            self._command(f"RENAME TABLE {target.staging_table} TO {target.physical_table_name}")
        except Exception as promotion_error:
            try:
                self._validate_table(target, target.physical_table_name, statistics)
            except Exception as recovery_error:
                raise promotion_error from recovery_error
            try:
                self.discard_publication(target)
            except Exception:
                pass
        return target.physical_table_name

    def _validate_table(
        self,
        target: ClickHousePublicationTarget,
        table_name: str,
        statistics: StructuredValidationStatistics,
    ) -> None:
        _require_identifier(table_name)
        described = self._query(f"DESCRIBE TABLE {table_name}")
        observed = self._query(_validation_query(target, table_name))
        observed_values = _as_mapping(observed)

        if int(observed_values.get("row_count", -1)) != statistics.row_count:
            raise StructuredStorageError("ClickHouse row count validation failed")
        content_hash_versions = int(observed_values.get("content_hash_versions", -1))
        observed_stored_hash = observed_values.get("content_hash")
        if statistics.row_count == 0:
            if content_hash_versions != 0 or observed_stored_hash not in {None, ""}:
                raise StructuredStorageError("ClickHouse empty content hash validation failed")
        else:
            if content_hash_versions != 1:
                raise StructuredStorageError("ClickHouse content hash is not uniform")
            if observed_stored_hash != statistics.content_hash:
                raise StructuredStorageError("ClickHouse content hash validation failed")
        observed_content_hash = _content_hash_from_observation(
            row_count=statistics.row_count,
            sums=tuple(int(observed_values.get(f"content_sum_{index}", -1)) for index in range(4)),
            xors=tuple(int(observed_values.get(f"content_xor_{index}", -1)) for index in range(4)),
        )
        if observed_content_hash != statistics.content_hash:
            raise StructuredStorageError("ClickHouse actual row content hash validation failed")

        expected_schema = [
            (column.physical_name, _clickhouse_type(column.data_type))
            for column in target.schema.columns
        ] + list(_METADATA_COLUMNS)
        described_schema = _as_described_schema(described)
        if described_schema != expected_schema:
            raise StructuredStorageError("ClickHouse schema validation failed")
        if statistics.column_count != len(target.schema.columns):
            raise StructuredStorageError("structured column count validation failed")

        for column, expected_nulls in statistics.null_counts.items():
            _require_identifier(column)
            if int(observed_values.get(f"null_{column}", -1)) != expected_nulls:
                raise StructuredStorageError(
                    f"ClickHouse null count validation failed for {column}"
                )

        for column, (expected_minimum, expected_maximum) in statistics.numeric_ranges.items():
            _require_identifier(column)
            expected_count = statistics.row_count - statistics.null_counts[column]
            if int(observed_values.get(f"count_{column}", -1)) != expected_count:
                raise StructuredStorageError(
                    f"ClickHouse numeric count validation failed for {column}"
                )
            if not _same_number(observed_values.get(f"min_{column}"), expected_minimum):
                raise StructuredStorageError(f"ClickHouse minimum validation failed for {column}")
            if not _same_number(observed_values.get(f"max_{column}"), expected_maximum):
                raise StructuredStorageError(f"ClickHouse maximum validation failed for {column}")

    def discard_publication(self, target: ClickHousePublicationTarget) -> None:
        _require_identifier(target.staging_table)
        self._command(f"DROP TABLE IF EXISTS {target.staging_table}")

    def query(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> Any:
        if self._query_client is None:
            raise StructuredStorageError(
                "ClickHouse structured queries require a separate read-only query client"
            )
        query = getattr(self._query_client, "query", None)
        if query is None:
            raise StructuredStorageError(
                "ClickHouse query client cannot execute structured queries"
            )
        kwargs: dict[str, object] = {"settings": dict(self._query_settings)}
        if parameters is not None:
            kwargs["parameters"] = dict(parameters)
        return query(statement, **kwargs)

    def _command(self, statement: str) -> None:
        command = getattr(self._ingest_client, "command", None)
        if command is not None:
            command(statement, settings=dict(self._settings))
            return
        execute_ddl = getattr(self._ingest_client, "execute_ddl", None)
        if execute_ddl is None:
            raise StructuredStorageError("ClickHouse ingest client cannot execute DDL")
        execute_ddl(statement, settings=dict(self._settings))

    def _query(self, statement: str) -> Any:
        if self._query_client is None:
            raise StructuredStorageError(
                "ClickHouse validation requires a separate read-only query client"
            )
        query = getattr(self._query_client, "query", None)
        if query is None:
            raise StructuredStorageError("ClickHouse query client cannot validate staging tables")
        return query(statement, settings=dict(self._query_settings))


def _validation_query(target: ClickHousePublicationTarget, table_name: str) -> str:
    _require_identifier(table_name)
    projections = [
        "count() AS row_count",
        "uniqExact(_content_hash) AS content_hash_versions",
        "any(_content_hash) AS content_hash",
    ]
    for column in target.schema.columns:
        name = column.physical_name
        _require_identifier(name)
        projections.append(f"countIf({name} IS NULL) AS null_{name}")
        if column.data_type in {StructuredColumnType.INTEGER, StructuredColumnType.DECIMAL}:
            projections.append(f"count({name}) AS count_{name}")
            projections.append(f"min({name}) AS min_{name}")
            projections.append(f"max({name}) AS max_{name}")
    row_digest = f"SHA256({_canonical_row_expression(target.schema.columns)})"
    for index in range(4):
        lane = f"reinterpretAsUInt64(substring({row_digest}, {index * 8 + 1}, 8))"
        projections.append(f"sumWithOverflow({lane}) AS content_sum_{index}")
        projections.append(f"groupBitXor({lane}) AS content_xor_{index}")
    return f"SELECT {', '.join(projections)} FROM {table_name}"


def _canonical_row_expression(columns: Sequence[StructuredColumnSchema]) -> str:
    names_and_types = [(column.physical_name, column.data_type) for column in columns]
    names_and_types.extend(
        (
            name,
            {
                "String": StructuredColumnType.STRING,
                "UInt64": StructuredColumnType.INTEGER,
            }[column_type],
        )
        for name, column_type in _METADATA_COLUMNS
        if name != "_content_hash"
    )
    segments = []
    for name, column_type in names_and_types:
        _require_identifier(name)
        rendered = (
            f"toDecimalString({name}, 9)"
            if column_type is StructuredColumnType.DECIMAL
            else f"toString({name})"
        )
        segments.append(
            f"if(isNull({name}), 'N;', "
            f"concat('V', toString(length({rendered})), ':', {rendered}, ';'))"
        )
    return f"concat({', '.join(segments)})"


def _row_digest_lanes(payload: bytes) -> tuple[int, int, int, int]:
    digest = hashlib.sha256(payload).digest()
    return tuple(
        int.from_bytes(digest[offset : offset + 8], "little") for offset in range(0, 32, 8)
    )


def _content_hash_from_observation(
    *,
    row_count: int,
    sums: Sequence[int],
    xors: Sequence[int],
) -> str:
    if len(sums) != 4 or len(xors) != 4:
        raise StructuredStorageError("content digest must contain four sum and xor lanes")
    payload = json.dumps(
        {
            "rowCount": int(row_count),
            "sums": tuple(int(value) & _UINT64_MASK for value in sums),
            "xors": tuple(int(value) & _UINT64_MASK for value in xors),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    column_names = getattr(result, "column_names", None)
    result_rows = getattr(result, "result_rows", None)
    if column_names and result_rows:
        return dict(zip(column_names, result_rows[0], strict=True))
    named_results = getattr(result, "named_results", None)
    if named_results is not None:
        rows = list(named_results())
        if rows:
            return rows[0]
    raise StructuredStorageError("ClickHouse validation result has an unsupported shape")


def _as_described_schema(result: Any) -> list[tuple[str, str]]:
    rows = getattr(result, "result_rows", result)
    try:
        return [(str(row[0]), str(row[1])) for row in rows]
    except (TypeError, IndexError) as error:
        raise StructuredStorageError(
            "ClickHouse DESCRIBE result has an unsupported shape"
        ) from error


def _as_single_column_values(result: Any) -> list[str]:
    rows = getattr(result, "result_rows", None)
    if rows is None:
        named_results = getattr(result, "named_results", None)
        if named_results is not None:
            return [str(row["name"]) for row in named_results()]
        rows = result
    try:
        values = []
        for row in rows:
            if isinstance(row, Mapping):
                values.append(str(row["name"]))
            else:
                values.append(str(row[0]))
        return values
    except (KeyError, TypeError, IndexError) as error:
        raise StructuredStorageError(
            "ClickHouse staging table listing has an unsupported shape"
        ) from error


def _clickhouse_type(column_type: StructuredColumnType) -> str:
    return {
        StructuredColumnType.STRING: "Nullable(String)",
        StructuredColumnType.INTEGER: "Nullable(Int64)",
        StructuredColumnType.DECIMAL: "Nullable(Decimal(38, 9))",
        StructuredColumnType.DATE: "Nullable(Date)",
        StructuredColumnType.DATETIME: "Nullable(DateTime64(3))",
        StructuredColumnType.BOOLEAN: "Nullable(UInt8)",
    }[column_type]


def _require_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise StructuredStorageError(f"Untrusted ClickHouse identifier: {value!r}")
    return value


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        normalized = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return normalized[:48]


def _same_number(observed: Any, expected: Any) -> bool:
    if observed is None or expected is None:
        return observed is expected
    try:
        return Decimal(str(observed)) == Decimal(str(expected))
    except (TypeError, ValueError, ArithmeticError):
        return observed == expected
