"""Target-host acceptance benchmark for structured spreadsheet aggregation.

The executable path requires an explicit ClickHouse integration target. Unit tests
inject a runner and exercise the reporting/threshold contract without ClickHouse.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid


DEFAULT_ROWS = 1_000_000
DEFAULT_CONCURRENCY = 15
DEFAULT_REQUESTS = 150
DEFAULT_P95_SECONDS = 5.0
DEFAULT_MAX_RSS_GROWTH_MB = 512.0
INGEST_BATCH_ROWS = 10_000
MEBIBYTE = 1024 * 1024

AGGREGATE_QUERIES = (
    "SELECT avg(order_amount) FROM {table}",
    "SELECT sum(order_amount) FROM {table} WHERE region = '华东'",
    "SELECT count(order_amount) FROM {table} WHERE order_amount >= 100",
    "SELECT min(order_amount) FROM {table} WHERE order_date >= toDate('2025-04-01')",
    "SELECT max(order_amount) FROM {table} WHERE order_date <= toDate('2025-09-30')",
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    rows: int = DEFAULT_ROWS
    concurrency: int = DEFAULT_CONCURRENCY
    requests: int = DEFAULT_REQUESTS
    p95_seconds: float = DEFAULT_P95_SECONDS
    max_rss_growth_mb: float = DEFAULT_MAX_RSS_GROWTH_MB

    def __post_init__(self) -> None:
        for name in ("rows", "concurrency", "requests"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("p95_seconds", "max_rss_growth_mb"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return the nearest-rank percentile used by the acceptance gate."""

    if (
        not samples
        or isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not 0 < quantile <= 1
    ):
        raise ValueError("percentile requires samples and a quantile in (0, 1]")
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency samples must be finite and non-negative")
    values.sort()
    rank = math.ceil(float(quantile) * len(values))
    return values[rank - 1]


def build_report(
    *,
    row_count: int,
    peak_rss_growth_mb: float,
    latencies: Sequence[float],
    success_count: int,
    error_count: int,
) -> dict[str, int | float]:
    if min(row_count, success_count, error_count) < 0:
        raise ValueError("report counts must be non-negative")
    growth = float(peak_rss_growth_mb)
    if not math.isfinite(growth) or growth < 0:
        raise ValueError("peak RSS growth must be finite and non-negative")
    p50 = percentile(latencies, 0.50) if latencies else 0.0
    p95 = percentile(latencies, 0.95) if latencies else 0.0
    return {
        "rowCount": row_count,
        "peakRssGrowthMb": round(growth, 6),
        "successCount": success_count,
        "errorCount": error_count,
        "p50Seconds": round(p50, 6),
        "p95Seconds": round(p95, 6),
    }


def report_passes(report: Mapping[str, object], config: BenchmarkConfig) -> bool:
    try:
        return (
            int(report["rowCount"]) == config.rows
            and int(report["successCount"]) == config.requests
            and int(report["errorCount"]) == 0
            and float(report["p95Seconds"]) <= config.p95_seconds
            and float(report["peakRssGrowthMb"]) <= config.max_rss_growth_mb
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _secret_from_file(environ: Mapping[str, str], name: str) -> str:
    raw_path = environ.get(name, "").strip()
    if not raw_path:
        raise ValueError(f"{name} must reference a readable secret file")
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError(f"{name} must reference a readable secret file")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"{name} must reference a readable secret file") from error
    if not value:
        raise ValueError(f"{name} secret file must not be empty")
    return value


def build_clickhouse_clients(
    environ: Mapping[str, str],
    *,
    client_factory: Callable[..., Any],
) -> tuple[Any, Any]:
    ingest_password = _secret_from_file(environ, "CLICKHOUSE_INGEST_PASSWORD_FILE")
    query_password = _secret_from_file(environ, "CLICKHOUSE_QUERY_PASSWORD_FILE")
    host = environ.get("CLICKHOUSE_HOST", "").strip()
    if not host:
        raise ValueError("CLICKHOUSE_HOST is required")
    try:
        port = int(environ.get("CLICKHOUSE_PORT", "8123"))
    except ValueError as error:
        raise ValueError("CLICKHOUSE_PORT must be an integer") from error
    if not 1 <= port <= 65_535:
        raise ValueError("CLICKHOUSE_PORT must be between 1 and 65535")

    ingest = client_factory(
        host=host,
        port=port,
        username=environ.get("CLICKHOUSE_INGEST_USER", "structured_ingest"),
        password=ingest_password,
    )
    try:
        query = client_factory(
            host=host,
            port=port,
            username=environ.get("CLICKHOUSE_QUERY_USER", "structured_query"),
            password=query_password,
            autogenerate_session_id=False,
        )
        if query is ingest:
            raise RuntimeError("benchmark requires separate ingest and query clients")
    except Exception:
        close = getattr(ingest, "close", None)
        if callable(close):
            close()
        raise
    return ingest, query


def _iter_rows(
    total: int, batch_rows: int = INGEST_BATCH_ROWS
) -> Iterable[list[list[object]]]:
    regions = ("华东", "华南", "华北", "西部")
    epoch = date(2025, 1, 1)
    for start in range(0, total, batch_rows):
        stop = min(total, start + batch_rows)
        batch: list[list[object]] = []
        for index in range(start, stop):
            amount = (
                None if index % 97 == 0 else Decimal(index % 100_000) / Decimal(100)
            )
            batch.append(
                [
                    index,
                    amount,
                    regions[index % len(regions)],
                    epoch + timedelta(days=index % 365),
                ]
            )
        yield batch


def expected_aggregate_values(
    total: int,
) -> tuple[Decimal, Decimal, int, Decimal, Decimal]:
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("total must be a positive integer")
    regions = ("华东", "华南", "华北", "西部")
    epoch = date(2025, 1, 1)
    average_sum = Decimal(0)
    average_count = 0
    east_sum = Decimal(0)
    count_at_least_100 = 0
    date_min: Decimal | None = None
    date_max: Decimal | None = None
    for index in range(total):
        if index % 97 == 0:
            continue
        amount = Decimal(index % 100_000) / Decimal(100)
        region = regions[index % len(regions)]
        order_date = epoch + timedelta(days=index % 365)
        average_sum += amount
        average_count += 1
        if region == "华东":
            east_sum += amount
        if amount >= Decimal(100):
            count_at_least_100 += 1
        if order_date >= date(2025, 4, 1):
            date_min = amount if date_min is None else min(date_min, amount)
        if order_date <= date(2025, 9, 30):
            date_max = amount if date_max is None else max(date_max, amount)
    if average_count == 0 or date_min is None or date_max is None:
        raise ValueError("fixture does not cover every fixed aggregate query")
    return (
        average_sum / average_count,
        east_sum,
        count_at_least_100,
        date_min,
        date_max,
    )


def execute_workload(
    config: BenchmarkConfig,
    *,
    ingestion_batches: Iterable[object],
    publish_batch: Callable[[object], None],
    finish_publication: Callable[[], None],
    execute_query: Callable[[int], None],
    rss_reader: Callable[[], int] = _rss_bytes,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, int | float]:
    baseline = int(rss_reader())
    rss_samples: list[int] = [baseline]
    published_rows = 0
    for batch in ingestion_batches:
        rss_samples.append(int(rss_reader()))
        publish_batch(batch)
        batch_rows = getattr(batch, "num_rows", None)
        if batch_rows is None:
            try:
                batch_rows = len(batch)  # type: ignore[arg-type]
            except TypeError as error:
                raise ValueError(
                    "ingestion batches must expose a bounded row count"
                ) from error
        if (
            isinstance(batch_rows, bool)
            or not isinstance(batch_rows, int)
            or batch_rows <= 0
        ):
            raise ValueError("ingestion batch row counts must be positive integers")
        published_rows += batch_rows
        rss_samples.append(int(rss_reader()))
    if published_rows == 0:
        raise ValueError("ingestion produced no batches")
    finish_publication()
    rss_samples.append(int(rss_reader()))

    latencies: list[float] = []
    errors = 0

    def timed_query(request_index: int) -> float:
        started = clock()
        execute_query(request_index)
        return max(0.0, clock() - started)

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [
            executor.submit(timed_query, index) for index in range(config.requests)
        ]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception:
                errors += 1

    peak_growth_mb = max(0, max(rss_samples) - baseline) / MEBIBYTE
    return build_report(
        row_count=published_rows,
        peak_rss_growth_mb=peak_growth_mb,
        latencies=latencies,
        success_count=len(latencies),
        error_count=errors,
    )


class _ClickHouseTarget:
    def __init__(
        self,
        ingest_client: Any,
        query_client: Any,
        table: str,
        row_count: int,
    ) -> None:
        self.ingest_client = ingest_client
        self.query_client = query_client
        self.table = table
        self.staging_table = f"{table}_staging"
        self.expected_values = expected_aggregate_values(row_count)

    def prepare(self) -> None:
        self.ingest_client.command(f"DROP TABLE IF EXISTS {self.staging_table}")
        self.ingest_client.command(f"DROP TABLE IF EXISTS {self.table}")
        self.ingest_client.command(
            f"CREATE TABLE {self.staging_table} ("
            "row_id UInt64, "
            "order_amount Nullable(Decimal(38, 9)), "
            "region LowCardinality(String), "
            "order_date Date"
            ") ENGINE = MergeTree ORDER BY row_id"
        )

    def insert_batch(self, batch: object) -> None:
        self.ingest_client.insert(
            self.staging_table,
            batch,
            column_names=("row_id", "order_amount", "region", "order_date"),
        )

    def publish(self) -> None:
        self.ingest_client.command(f"RENAME TABLE {self.staging_table} TO {self.table}")

    def query(self, request_index: int) -> None:
        query_index = request_index % len(AGGREGATE_QUERIES)
        statement = AGGREGATE_QUERIES[query_index].format(table=self.table)
        result = self.query_client.query(statement)
        rows = getattr(result, "result_rows", None)
        if not rows or not rows[0] or len(rows[0]) != 1:
            raise RuntimeError(
                "structured benchmark aggregate returned an invalid result"
            )
        actual = rows[0][0]
        expected = self.expected_values[query_index]
        if isinstance(expected, int):
            matches = not isinstance(actual, bool) and int(actual) == expected
        else:
            try:
                matches = abs(Decimal(str(actual)) - expected) <= Decimal("0.000000001")
            except Exception:
                matches = False
        if not matches:
            raise RuntimeError(
                f"structured benchmark aggregate mismatch for query {query_index}"
            )

    def close(self) -> None:
        try:
            self.ingest_client.command(f"DROP TABLE IF EXISTS {self.staging_table}")
            self.ingest_client.command(f"DROP TABLE IF EXISTS {self.table}")
        finally:
            self.query_client.close()
            self.ingest_client.close()


def run_benchmark(config: BenchmarkConfig) -> dict[str, int | float]:
    if os.getenv("RUN_OFFLINE_INTEGRATION") != "1" or not os.getenv("CLICKHOUSE_HOST"):
        raise RuntimeError(
            "target-host gate requires RUN_OFFLINE_INTEGRATION=1 and CLICKHOUSE_HOST"
        )
    import clickhouse_connect

    ingest_client, query_client = build_clickhouse_clients(
        os.environ,
        client_factory=clickhouse_connect.get_client,
    )
    target = _ClickHouseTarget(
        ingest_client,
        query_client,
        f"structured_benchmark_{uuid.uuid4().hex[:16]}",
        config.rows,
    )
    try:
        target.prepare()
        return execute_workload(
            config,
            ingestion_batches=_iter_rows(config.rows),
            publish_batch=target.insert_batch,
            finish_publication=target.publish,
            execute_query=target.query,
        )
    finally:
        target.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--p95-seconds", type=float, default=DEFAULT_P95_SECONDS)
    parser.add_argument(
        "--max-rss-growth-mb", type=float, default=DEFAULT_MAX_RSS_GROWTH_MB
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[BenchmarkConfig], dict[str, int | float]] = run_benchmark,
) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = BenchmarkConfig(
            rows=arguments.rows,
            concurrency=arguments.concurrency,
            requests=arguments.requests,
            p95_seconds=arguments.p95_seconds,
            max_rss_growth_mb=arguments.max_rss_growth_mb,
        )
    except ValueError as error:
        print(f"structured aggregation benchmark failed: {error}", file=sys.stderr)
        return 2
    try:
        report = runner(config)
    except Exception as error:
        report = build_report(
            row_count=0,
            peak_rss_growth_mb=0,
            latencies=(),
            success_count=0,
            error_count=config.requests,
        )
        print(f"structured aggregation benchmark failed: {error}", file=sys.stderr)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report_passes(report, config) else 1


if __name__ == "__main__":
    raise SystemExit(main())
