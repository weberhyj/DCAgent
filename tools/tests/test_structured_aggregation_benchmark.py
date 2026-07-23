from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from threading import Event, Lock
import tempfile
import unittest


class StructuredAggregationBenchmarkTest(unittest.TestCase):
    def test_target_builds_separate_file_secret_ingest_and_query_clients(self) -> None:
        from tools.structured_aggregation_benchmark import build_clickhouse_clients

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ingest_secret = root / "ingest-password"
            query_secret = root / "query-password"
            ingest_secret.write_text("ingest-file-secret\n", encoding="utf-8")
            query_secret.write_text("query-file-secret\n", encoding="utf-8")
            calls = []

            def factory(**kwargs):
                calls.append(kwargs)
                return object()

            ingest, query = build_clickhouse_clients(
                {
                    "CLICKHOUSE_HOST": "clickhouse.local",
                    "CLICKHOUSE_PORT": "8123",
                    "CLICKHOUSE_INGEST_USER": "benchmark-ingest",
                    "CLICKHOUSE_QUERY_USER": "benchmark-query",
                    "CLICKHOUSE_INGEST_PASSWORD_FILE": str(ingest_secret),
                    "CLICKHOUSE_QUERY_PASSWORD_FILE": str(query_secret),
                    "CLICKHOUSE_INGEST_PASSWORD": "must-not-be-read",
                    "CLICKHOUSE_QUERY_PASSWORD": "must-not-be-read",
                },
                client_factory=factory,
            )

        self.assertIsNot(ingest, query)
        self.assertEqual(
            calls,
            [
                {
                    "host": "clickhouse.local",
                    "port": 8123,
                    "username": "benchmark-ingest",
                    "password": "ingest-file-secret",
                },
                {
                    "host": "clickhouse.local",
                    "port": 8123,
                    "username": "benchmark-query",
                    "password": "query-file-secret",
                    "autogenerate_session_id": False,
                },
            ],
        )

    def test_target_requires_both_secret_files_before_creating_clients(self) -> None:
        from tools.structured_aggregation_benchmark import build_clickhouse_clients

        calls = []
        with self.assertRaisesRegex(ValueError, "PASSWORD_FILE"):
            build_clickhouse_clients(
                {"CLICKHOUSE_HOST": "clickhouse.local"},
                client_factory=lambda **kwargs: calls.append(kwargs),
            )
        self.assertEqual(calls, [])

    def test_percentile_uses_nearest_rank_and_rejects_invalid_samples(self) -> None:
        from tools.structured_aggregation_benchmark import percentile

        samples = [0.4, 0.1, 0.3, 0.2]
        self.assertEqual(percentile(samples, 0.50), 0.2)
        self.assertEqual(percentile(samples, 0.95), 0.4)
        for invalid_samples in ([], [0.1, float("nan")], [0.1, -0.2]):
            with self.subTest(samples=invalid_samples), self.assertRaises(ValueError):
                percentile(invalid_samples, 0.95)

    def test_report_schema_contains_only_stable_acceptance_metrics(self) -> None:
        from tools.structured_aggregation_benchmark import build_report

        report = build_report(
            row_count=1_000_000,
            peak_rss_growth_mb=125.25,
            latencies=[0.2, 0.1, 0.4, 0.3],
            success_count=4,
            error_count=0,
        )

        self.assertEqual(
            set(report),
            {
                "rowCount",
                "peakRssGrowthMb",
                "successCount",
                "errorCount",
                "p50Seconds",
                "p95Seconds",
            },
        )
        self.assertEqual(report["rowCount"], 1_000_000)
        self.assertEqual(report["p50Seconds"], 0.2)
        self.assertEqual(report["p95Seconds"], 0.4)

    def test_workload_measures_every_bounded_batch_and_reports_actual_rows(
        self,
    ) -> None:
        from tools.structured_aggregation_benchmark import (
            BenchmarkConfig,
            execute_workload,
        )

        rss_values = iter([100, 110, 120, 125, 150, 160])
        published = []
        report = execute_workload(
            BenchmarkConfig(rows=4, concurrency=1, requests=2),
            ingestion_batches=([1, 2], [3]),
            publish_batch=lambda batch: published.append(list(batch)),
            finish_publication=lambda: None,
            execute_query=lambda _index: None,
            rss_reader=lambda: next(rss_values),
            clock=iter([0.0, 0.1, 1.0, 1.2]).__next__,
        )

        self.assertEqual(published, [[1, 2], [3]])
        self.assertEqual(report["rowCount"], 3)
        self.assertGreater(report["peakRssGrowthMb"], 0)
        with self.assertRaises(StopIteration):
            next(rss_values)

    def test_rss_baseline_precedes_generator_and_samples_publication_completion(
        self,
    ) -> None:
        from tools.structured_aggregation_benchmark import (
            BenchmarkConfig,
            MEBIBYTE,
            execute_workload,
        )

        events = []
        readings = iter([100, 110, 130, 140, 170, 190])

        class Batches:
            def __iter__(self):
                events.append("iterate")
                yield [1]
                yield [2]

        def rss() -> int:
            events.append("rss")
            return next(readings)

        report = execute_workload(
            BenchmarkConfig(rows=2, concurrency=1, requests=1),
            ingestion_batches=Batches(),
            publish_batch=lambda batch: events.append(f"publish-{batch[0]}"),
            finish_publication=lambda: events.append("finish"),
            execute_query=lambda _index: None,
            rss_reader=rss,
            clock=iter([0.0, 0.1]).__next__,
        )

        self.assertEqual(
            events[:10],
            [
                "rss",
                "iterate",
                "rss",
                "publish-1",
                "rss",
                "rss",
                "publish-2",
                "rss",
                "finish",
                "rss",
            ],
        )
        self.assertEqual(report["peakRssGrowthMb"], round(90 / MEBIBYTE, 6))

    def test_empty_ingestion_fails_before_publication_or_queries(self) -> None:
        from tools.structured_aggregation_benchmark import (
            BenchmarkConfig,
            execute_workload,
        )

        events = []
        with self.assertRaisesRegex(ValueError, "no batches"):
            execute_workload(
                BenchmarkConfig(rows=1, concurrency=1, requests=1),
                ingestion_batches=(),
                publish_batch=lambda _batch: events.append("publish"),
                finish_publication=lambda: events.append("finish"),
                execute_query=lambda _index: events.append("query"),
                rss_reader=lambda: 100,
            )
        self.assertEqual(events, [])

    def test_fixed_query_references_cover_every_aggregate_and_reject_wrong_results(
        self,
    ) -> None:
        from tools.structured_aggregation_benchmark import (
            AGGREGATE_QUERIES,
            _ClickHouseTarget,
            expected_aggregate_values,
        )

        expected = expected_aggregate_values(1_000)
        self.assertEqual(len(expected), len(AGGREGATE_QUERIES))
        self.assertTrue(all(value is not None for value in expected))
        self.assertIsInstance(expected[0], Decimal)
        self.assertIsInstance(expected[1], Decimal)
        self.assertIsInstance(expected[2], int)

        class Client:
            def __init__(self, values):
                self.values = list(values)

            def query(self, _statement):
                return type("Result", (), {"result_rows": [(self.values.pop(0),)]})()

        ingest = type("Ingest", (), {})()
        correct = _ClickHouseTarget(ingest, Client(expected), "benchmark", 1_000)
        for index in range(len(AGGREGATE_QUERIES)):
            correct.query(index)

        wrong = _ClickHouseTarget(ingest, Client([Decimal("999")]), "benchmark", 1_000)
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            wrong.query(0)

    def test_count_result_requires_exact_integer_semantics(self) -> None:
        from tools.structured_aggregation_benchmark import _ClickHouseTarget

        class Client:
            def __init__(self, value):
                self.value = value

            def query(self, _statement):
                return type("Result", (), {"result_rows": [(self.value,)]})()

        ingest = object()
        for value in (Decimal("10.9"), 10.1, True, "10"):
            with self.subTest(value=value):
                target = _ClickHouseTarget(ingest, Client(value), "benchmark", 1_000)
                target.expected_values = (
                    *target.expected_values[:2],
                    10,
                    *target.expected_values[3:],
                )
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    target.query(2)

        for value in (Decimal("10.0"), 10):
            with self.subTest(value=value):
                target = _ClickHouseTarget(ingest, Client(value), "benchmark", 1_000)
                target.expected_values = (
                    *target.expected_values[:2],
                    10,
                    *target.expected_values[3:],
                )
                target.query(2)

    def test_workload_uses_requested_concurrency_and_counts_query_errors(self) -> None:
        from tools.structured_aggregation_benchmark import (
            BenchmarkConfig,
            execute_workload,
        )

        all_started = Event()
        release = Event()
        lock = Lock()
        active = 0
        peak_active = 0

        def query(index: int) -> None:
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
                if peak_active == 3:
                    all_started.set()
            self.assertTrue(release.wait(timeout=2))
            with lock:
                active -= 1
            if index == 4:
                raise RuntimeError("query failed")

        def release_when_ready() -> None:
            self.assertTrue(all_started.wait(timeout=2))
            release.set()

        import threading

        coordinator = threading.Thread(target=release_when_ready)
        coordinator.start()
        report = execute_workload(
            BenchmarkConfig(rows=1, concurrency=3, requests=6),
            ingestion_batches=([1],),
            publish_batch=lambda _batch: None,
            finish_publication=lambda: None,
            execute_query=query,
            rss_reader=lambda: 100,
        )
        coordinator.join(timeout=2)

        self.assertEqual(peak_active, 3)
        self.assertEqual(report["successCount"], 5)
        self.assertEqual(report["errorCount"], 1)

    def test_main_emits_json_and_fails_when_any_bound_is_exceeded(self) -> None:
        from tools.structured_aggregation_benchmark import main

        captured_configs = []

        def runner(config):
            captured_configs.append(config)
            return {
                "rowCount": config.rows,
                "peakRssGrowthMb": 513.0,
                "successCount": config.requests - 1,
                "errorCount": 1,
                "p50Seconds": 1.0,
                "p95Seconds": 5.1,
            }

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--rows",
                    "1000000",
                    "--concurrency",
                    "15",
                    "--requests",
                    "150",
                    "--p95-seconds",
                    "5",
                    "--max-rss-growth-mb",
                    "512",
                ],
                runner=runner,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(len(captured_configs), 1)
        self.assertEqual(captured_configs[0].concurrency, 15)
        self.assertEqual(json.loads(output.getvalue())["errorCount"], 1)

    def test_main_returns_zero_only_for_complete_bounded_run(self) -> None:
        from tools.structured_aggregation_benchmark import main

        def runner(config):
            return {
                "rowCount": config.rows,
                "peakRssGrowthMb": 128.0,
                "successCount": config.requests,
                "errorCount": 0,
                "p50Seconds": 0.5,
                "p95Seconds": 2.0,
            }

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([], runner=runner)

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["rowCount"], 1_000_000)
        self.assertEqual(payload["successCount"], 150)

    def test_main_emits_fail_closed_report_when_target_runner_crashes(self) -> None:
        from tools.structured_aggregation_benchmark import main

        def runner(_config):
            raise ConnectionError("ClickHouse target unavailable")

        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = main([], runner=runner)

        self.assertEqual(exit_code, 2)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["rowCount"], 0)
        self.assertEqual(payload["successCount"], 0)
        self.assertEqual(payload["errorCount"], 150)
        self.assertIn("target unavailable", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
