from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from threading import Event, Lock
import unittest


class StructuredAggregationBenchmarkTest(unittest.TestCase):
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

        rss_values = iter([100, 120, 125, 150])
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
