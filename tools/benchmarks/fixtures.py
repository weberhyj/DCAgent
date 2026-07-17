"""Bounded deterministic fixture streams for offline capacity tests."""

from __future__ import annotations

from collections.abc import Iterator
import random
import re


_CLICKHOUSE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?")


def iter_qdrant_points(
    total: int,
    dimensions: int,
    batch_size: int,
    seed: int,
) -> Iterator[list[dict[str, object]]]:
    """Return a deterministic point stream that holds at most one batch in memory."""

    _require_non_negative_integer("total", total)
    _require_positive_integer("dimensions", dimensions)
    _require_positive_integer("batch_size", batch_size)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    def generate() -> Iterator[list[dict[str, object]]]:
        randomizer = random.Random(seed)
        batch: list[dict[str, object]] = []
        for point_id in range(total):
            batch.append(
                {
                    "id": point_id,
                    "vector": {
                        "dense": [randomizer.uniform(-1.0, 1.0) for _ in range(dimensions)],
                        "bm25": {
                            "indices": [point_id % 257, (point_id + 17) % 257],
                            "values": [1.0, 0.5],
                        },
                    },
                    "payload": {
                        "tenant_id": point_id % 100,
                        "department_id": point_id % 20,
                        "classification": point_id % 3,
                    },
                }
            )
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    return generate()


def clickhouse_numbers_sql(total: int, table: str = "benchmark_rows") -> str:
    """Build a server-side ClickHouse fixture statement without Python row objects."""

    _require_non_negative_integer("total", total)
    if not isinstance(table, str) or _CLICKHOUSE_IDENTIFIER.fullmatch(table) is None:
        raise ValueError("table must be a simple ClickHouse identifier")
    return (
        f"INSERT INTO {table} "
        "(id, tenant_id, department_id, classification, numeric_value, event_date) "
        "SELECT number AS id, number % 100 AS tenant_id, "
        "number % 20 AS department_id, number % 3 AS classification, "
        "toFloat64(number % 100000) / 100 AS numeric_value, "
        "toDate('2025-01-01') + toIntervalDay(number % 365) AS event_date "
        f"FROM numbers({total})"
    )


def _require_non_negative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
