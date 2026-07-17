"""Offline 40/40/20 Phase 4 query workload.

The module remains importable when the optional benchmark dependency is absent,
so ordinary tools tests do not require Locust.
"""

from __future__ import annotations

from collections import deque
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from tools.benchmarks.manifest import BenchmarkManifest


try:
    from locust import HttpUser, between, events, task
except ModuleNotFoundError:  # pragma: no cover - exercised by environments without Locust
    class _Hook:
        def add_listener(self, function):
            return function

    class _Events:
        request = _Hook()
        test_stop = _Hook()

    events = _Events()

    class HttpUser:  # type: ignore[no-redef]
        pass

    def task(weight):  # type: ignore[no-redef]
        def decorate(function):
            function.locust_task_weight = weight
            return function

        return decorate

    def between(minimum, maximum):  # type: ignore[no-redef]
        def wait_time(_instance):
            return minimum if minimum == maximum else (minimum + maximum) / 2

        return wait_time


REQUEST_WEIGHTS = {"structured": 4, "document": 4, "mixed": 2}
DEFAULT_MANIFEST = Path(__file__).with_name("manifests") / "acceptance-30m-5m.json"
MANIFEST_PATH = Path(os.environ.get("BENCHMARK_MANIFEST", str(DEFAULT_MANIFEST)))
MANIFEST = BenchmarkManifest.load(MANIFEST_PATH)
WAIT_SECONDS = MANIFEST.think_time_seconds
TRUSTED_PRINCIPAL = os.environ.get("BENCHMARK_PRINCIPAL", "benchmark-user")
CONVERSATION_ID = os.environ.get("BENCHMARK_CONVERSATION_ID", "benchmark-conversation")
QUERY_PATH_TEMPLATE = "/api/conversations/{conversation_id}/messages/stream"
QUERY_PATH = QUERY_PATH_TEMPLATE.format(conversation_id=CONVERSATION_ID)
IDEMPOTENCY_PREFIX = os.environ.get("BENCHMARK_IDEMPOTENCY_PREFIX", "capacity")
WORKER_ID = os.environ.get("BENCHMARK_WORKER_ID", "local")
REQUEST_MODES = {"structured": "quick", "document": "source", "mixed": "deep"}
_REQUEST_SEQUENCE = itertools.count()
RECORDED_TIMINGS: deque[dict[str, float | str]] = deque(maxlen=100_000)
RECORDED_REQUESTS: deque[dict[str, object]] = deque(maxlen=100_000)
METRICS_PATH = Path(
    os.environ.get(
        "BENCHMARK_METRICS_PATH", "artifacts/benchmarks/locust-metrics.json"
    )
)


@events.request.add_listener
def record_stream_timings(
    *,
    name: str | None = None,
    context: dict[str, object] | None = None,
    response_time: float | None = None,
    exception: object | None = None,
    **_kwargs: object,
) -> None:
    """Capture the two streaming responsiveness signals from request context."""

    if context is not None and context.get("_capacity_recorded") is True:
        return
    if context is not None:
        context["_capacity_recorded"] = True
    request_kind = None
    if context is not None and context.get("request_kind") in REQUEST_WEIGHTS:
        request_kind = context["request_kind"]
    elif isinstance(name, str) and name.startswith("query:"):
        request_kind = name.removeprefix("query:")
    if request_kind not in REQUEST_WEIGHTS:
        return

    timing: dict[str, float | str] = {"request": name or "unknown"}
    for metric in ("queue_feedback_ms", "first_token_ms"):
        value = context.get(metric) if context is not None else None
        if isinstance(value, (int, float)):
            timing[metric] = float(value)
    if len(timing) > 1:
        RECORDED_TIMINGS.append(timing)

    full_stream_ms = context.get("full_stream_ms") if context is not None else None
    has_full_stream_ms = (
        isinstance(full_stream_ms, (int, float))
        and math.isfinite(float(full_stream_ms))
    )
    if not has_full_stream_ms:
        full_stream_ms = response_time
    if not isinstance(full_stream_ms, (int, float)) or not math.isfinite(float(full_stream_ms)):
        full_stream_ms = 0.0
    failed = (
        exception is not None
        or (context is not None and context.get("failed") is True)
        or not has_full_stream_ms
    )
    if (
        isinstance(full_stream_ms, (int, float)) and math.isfinite(float(full_stream_ms))
    ):
        record: dict[str, object] = {
            "request_kind": request_kind,
            "full_stream_ms": float(full_stream_ms),
            "failed": failed,
        }
        for metric in ("queue_feedback_ms", "first_token_ms"):
            value = context.get(metric) if context is not None else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                record[metric] = float(value)
        if context is not None and isinstance(context.get("cache_hit"), bool):
            record["cache_hit"] = context["cache_hit"]
        RECORDED_REQUESTS.append(record)


def parse_sse_events(lines) -> Any:
    """Parse independent event/data lines, multi-line data, and final events."""

    event_name = "message"
    data_lines: list[str] = []

    def build_event() -> dict[str, Any] | None:
        if not data_lines:
            return None
        raw_data = "\n".join(data_lines)
        try:
            data: Any = json.loads(raw_data)
        except json.JSONDecodeError:
            data = raw_data
        return {"event": event_name, "data": data}

    for raw_line in lines:
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", errors="replace")
        line = raw_line.rstrip("\r\n")
        if not line:
            event = build_event()
            if event is not None:
                yield event
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value or "message"
        elif field == "data":
            data_lines.append(value)

    event = build_event()
    if event is not None:
        yield event


def record_sse_timings(
    stream_events,
    context: dict[str, object],
    *,
    elapsed_ms,
) -> None:
    """Record first response feedback and first generated token once."""

    for stream_event, elapsed in zip(stream_events, elapsed_ms):
        event_name = stream_event.get("event")
        if event_name in ("accepted", "queued"):
            context.setdefault("queue_feedback_ms", float(elapsed))
        if event_name == "delta":
            context.setdefault("first_token_ms", float(elapsed))


def stream_failure_reason(stream_events, status_code: int) -> str | None:
    event_names = [event.get("event") for event in stream_events]
    if status_code >= 400:
        return f"http_{status_code}"
    if not event_names:
        return "empty_stream"
    if "error" in event_names:
        return "sse_error"
    if not any(name in ("completed", "degraded") for name in event_names):
        return "missing_terminal_event"
    return None


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def build_metrics(records) -> dict[str, float]:
    samples = list(records)
    successful = [record for record in samples if record.get("failed") is not True]
    metrics: dict[str, float] = {}
    for request_kind in REQUEST_WEIGHTS:
        value = _percentile_95(
            [
                float(record["full_stream_ms"])
                for record in successful
                if record.get("request_kind") == request_kind
                and isinstance(record.get("full_stream_ms"), (int, float))
            ]
        )
        if value is not None:
            metrics[f"{request_kind}_p95_ms"] = value

    if successful:
        for name in ("queue_feedback_ms", "first_token_ms"):
            if all(
                isinstance(record.get(name), (int, float))
                and math.isfinite(float(record[name]))
                for record in successful
            ):
                value = _percentile_95([float(record[name]) for record in successful])
                if value is not None:
                    metrics[f"{name.removesuffix('_ms')}_p95_ms"] = value

    if samples:
        metrics["error_rate"] = sum(
            1 for record in samples if record.get("failed") is True
        ) / len(samples)
    if successful and all(isinstance(record.get("cache_hit"), bool) for record in successful):
        cache_samples = [record["cache_hit"] for record in successful]
        metrics["warm_cache_hit_rate"] = sum(cache_samples) / len(cache_samples)
    return metrics


def _atomic_write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(metrics, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@events.test_stop.add_listener
def persist_benchmark_metrics(
    environment=None,
    *,
    path: Path | None = None,
    **_kwargs: object,
) -> None:
    del environment
    _atomic_write_metrics(path or METRICS_PATH, build_metrics(RECORDED_REQUESTS))


class CapacityUser(HttpUser):
    wait_time = between(WAIT_SECONDS, WAIT_SECONDS)

    @task(4)
    def structured(self) -> None:
        self._stream_query("structured")

    @task(4)
    def document(self) -> None:
        self._stream_query("document")

    @task(2)
    def mixed(self) -> None:
        self._stream_query("mixed")

    def _stream_query(self, query_kind: str) -> None:
        started = time.perf_counter()
        timing_context: dict[str, object] = {
            "request_kind": query_kind,
            "failed": False,
        }
        idempotency_key = (
            f"{IDEMPOTENCY_PREFIX}-{WORKER_ID}-{next(_REQUEST_SEQUENCE)}"
        )
        headers = {
            "X-Identity": TRUSTED_PRINCIPAL,
            "Idempotency-Key": idempotency_key,
        }
        body = {
            "content": f"offline capacity {query_kind}",
            "mode": REQUEST_MODES[query_kind],
        }
        with self.client.post(
            QUERY_PATH,
            json=body,
            headers=headers,
            stream=True,
            catch_response=True,
            context=timing_context,
            name=f"query:{query_kind}",
        ) as response:
            stream_events: list[dict[str, Any]] = []
            try:
                for stream_event in parse_sse_events(response.iter_lines()):
                    stream_events.append(stream_event)
                    elapsed_ms = (time.perf_counter() - started) * 1_000
                    record_sse_timings(
                        (stream_event,), timing_context, elapsed_ms=(elapsed_ms,)
                    )
                    data = stream_event.get("data")
                    if isinstance(data, dict):
                        cache_hit = data.get("cacheHit", data.get("cache_hit"))
                        if isinstance(cache_hit, bool):
                            timing_context["cache_hit"] = cache_hit
                cache_header = response.headers.get("X-Cache", "").strip().casefold()
                if cache_header in ("hit", "miss"):
                    timing_context["cache_hit"] = cache_header == "hit"
                failure = stream_failure_reason(stream_events, response.status_code)
                if failure is not None:
                    timing_context["failed"] = True
                    response.failure(failure)
            except Exception as exc:
                timing_context["failed"] = True
                response.failure(f"stream_error:{type(exc).__name__}")
            finally:
                timing_context["full_stream_ms"] = (
                    time.perf_counter() - started
                ) * 1_000
