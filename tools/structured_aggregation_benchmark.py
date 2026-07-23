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
    rss_samples: list[int] = []
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
    if not rss_samples:
        raise ValueError("ingestion produced no batches")
    finish_publication()

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

    baseline = rss_samples[0]
    peak_growth_mb = max(0, max(rss_samples) - baseline) / MEBIBYTE
    return build_report(
        row_count=published_rows,
        peak_rss_growth_mb=peak_growth_mb,
        latencies=latencies,
        success_count=len(latencies),
        error_count=errors,
    )


class _ClickHouseTarget:
    def __init__(self, client: Any, table: str) -> None:
        self.client = client
        self.table = table
        self.staging_table = f"{table}_staging"

    def prepare(self) -> None:
        self.client.command(f"DROP TABLE IF EXISTS {self.staging_table}")
        self.client.command(f"DROP TABLE IF EXISTS {self.table}")
        self.client.command(
            f"CREATE TABLE {self.staging_table} ("
            "row_id UInt64, "
            "order_amount Nullable(Decimal(38, 9)), "
            "region LowCardinality(String), "
            "order_date Date"
            ") ENGINE = MergeTree ORDER BY row_id"
        )

    def insert_batch(self, batch: object) -> None:
        self.client.insert(
            self.staging_table,
            batch,
            column_names=("row_id", "order_amount", "region", "order_date"),
        )

    def publish(self) -> None:
        self.client.command(f"RENAME TABLE {self.staging_table} TO {self.table}")

    def query(self, request_index: int) -> None:
        statement = AGGREGATE_QUERIES[request_index % len(AGGREGATE_QUERIES)].format(
            table=self.table
        )
        self.client.query(statement)

    def close(self) -> None:
        try:
            self.client.command(f"DROP TABLE IF EXISTS {self.staging_table}")
            self.client.command(f"DROP TABLE IF EXISTS {self.table}")
        finally:
            self.client.close()


def run_benchmark(config: BenchmarkConfig) -> dict[str, int | float]:
    if os.getenv("RUN_OFFLINE_INTEGRATION") != "1" or not os.getenv("CLICKHOUSE_HOST"):
        raise RuntimeError(
            "target-host gate requires RUN_OFFLINE_INTEGRATION=1 and CLICKHOUSE_HOST"
        )
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_INGEST_USER", "default"),
        password=os.getenv("CLICKHOUSE_INGEST_PASSWORD", ""),
        autogenerate_session_id=False,
    )
    target = _ClickHouseTarget(client, f"structured_benchmark_{uuid.uuid4().hex[:16]}")
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
