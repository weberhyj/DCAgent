import hashlib
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from dataclasses import FrozenInstanceError


ROOT = Path(__file__).parents[1]


class _ImmediateProcess:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.pid = 123

    def wait(self, timeout=None) -> int:
        return self.exit_code


class BenchmarkReportTest(unittest.TestCase):
    def test_metric_gate_validates_shape_and_is_frozen_with_slots(self) -> None:
        from tools.benchmarks.report import MetricGate

        gate = MetricGate("document_p95_ms", "lte", 5_000)
        self.assertFalse(hasattr(gate, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            gate.limit = 6_000  # type: ignore[misc]

        for args in (
            ("", "lte", 1),
            ("metric", "eq", 1),
            ("metric", "lte", True),
            ("metric", "lte", "1"),
            ("metric", "lte", math.inf),
            ("metric", "gte", math.nan),
            ("metric", "lte", 10**10_000),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                MetricGate(*args)

    def test_gate_evaluation_fails_closed_in_stable_gate_order(self) -> None:
        from tools.benchmarks.report import MetricGate, evaluate_capacity

        gates = (
            MetricGate("document_p95_ms", "lte", 5_000),
            MetricGate("error_rate", "lte", 0.01),
            MetricGate("warm_cache_hit_rate", "gte", 0.20),
            MetricGate("missing", "lte", 1),
            MetricGate("not_available", "lte", 1),
            MetricGate("nan", "lte", 1),
        )
        result = evaluate_capacity(
            gates,
            {
                "document_p95_ms": 6_200,
                "error_rate": 0.05,
                "warm_cache_hit_rate": 0.10,
                "not_available": "not_available",
                "nan": float("nan"),
            },
        )

        self.assertFalse(result.passed)
        self.assertFalse(hasattr(result, "__dict__"))
        self.assertEqual(
            result.failures,
            [
                "document_p95_ms",
                "error_rate",
                "warm_cache_hit_rate",
                "missing",
                "not_available",
                "nan",
            ],
        )
        with self.assertRaises(TypeError):
            result.failures.append("another")

    def test_write_report_is_deterministic_atomic_and_complete(self) -> None:
        from tools.benchmarks.report import MetricGate, evaluate_capacity, write_report

        gates = (MetricGate("latency", "lte", 10),)
        result = evaluate_capacity(gates, {"latency": 9})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            kwargs = {
                "path": path,
                "manifest": {"virtual_users": 15},
                "profile": {"name": "acceptance", "vector_dimensions": 768, "model_slots": 2},
                "mode": "phase4-online",
                "cache_label": "warm",
                "gates": gates,
                "hardware": {"logicalCores": 16},
                "metrics": {"latency": 9},
                "result": result,
                "command_exit_codes": {"locust": 0},
                "checksums": {"manifestSha256": "a" * 64, "profileSha256": "b" * 64},
                "software_versions": {"python": "3.13"},
            }
            write_report(**kwargs)
            first = path.read_bytes()
            write_report(**kwargs)
            self.assertEqual(first, path.read_bytes())
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

            payload = json.loads(first)
            self.assertEqual(
                set(payload),
                {
                    "manifest",
                    "profile",
                    "mode",
                    "cacheLabel",
                    "gates",
                    "hardware",
                    "metrics",
                    "gateResult",
                    "commandExitCodes",
                    "checksums",
                    "softwareVersions",
                },
            )
            self.assertTrue(payload["gateResult"]["passed"])


class CapacityRunnerTest(unittest.TestCase):
    def test_selects_exact_profile_gates_and_rejects_phase_mismatch(self) -> None:
        from tools.benchmarks.manifest import BenchmarkManifest
        from tools.benchmarks.run_capacity_benchmark import (
            derive_benchmark_timeout_seconds,
            select_profile,
        )

        smoke = BenchmarkManifest.load(ROOT / "benchmarks/manifests/smoke.json")
        profile = select_profile(smoke, "service-round-trip", "phase1-smoke")
        self.assertEqual(profile.name, "service-round-trip")
        self.assertEqual(profile.gates[0].name, "postgresql_round_trip_ms")

        acceptance = BenchmarkManifest.load(ROOT / "benchmarks/manifests/acceptance-30m-5m.json")
        warm = select_profile(acceptance, "online-warm", "phase4-online")
        self.assertEqual(warm.gates[-1].name, "error_rate")
        with self.assertRaises(ValueError):
            select_profile(acceptance, "batch-daily", "phase4-online")
        with self.assertRaises(ValueError):
            select_profile(smoke, "service-round-trip", "phase4-online")

        batch = select_profile(acceptance, "batch-initial", "phase4-batch")
        self.assertEqual(derive_benchmark_timeout_seconds(acceptance, batch), 86_700)

    def test_rejects_missing_and_non_numeric_metrics_before_reporting_pass(self) -> None:
        from tools.benchmarks.report import MetricGate
        from tools.benchmarks.run_capacity_benchmark import validate_metrics

        gates = (MetricGate("latency", "lte", 10),)
        for metrics in ({}, {"latency": "not_available"}, {"latency": None}, {"latency": math.inf}):
            with self.subTest(metrics=metrics):
                self.assertEqual(validate_metrics(gates, metrics), ["latency"])
        self.assertEqual(validate_metrics(gates, {"latency": 8}), [])

    def test_create_report_serializes_non_finite_metric_as_failed_not_available(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import create_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            report_path = root / "report.json"
            metrics_path.write_text('{"postgresql_round_trip_ms": NaN}', encoding="utf-8")

            passed = create_report(
                manifest_path=ROOT / "benchmarks/manifests/smoke.json",
                metrics_path=metrics_path,
                report_path=report_path,
                profile_name="service-round-trip",
                mode="phase1-smoke",
                cache_label="not-applicable",
                vector_dimension=32,
                model_slots=1,
                disk_path=root,
                hardware_collector=lambda _path: {"diskDevice": "test"},
                version_collector=lambda: ({"python": "test"}, {"version:python": 0}),
                benchmark_command=("benchmark", "--run"),
                benchmark_popen_factory=lambda _command, **_kwargs: _ImmediateProcess(),
            )

            self.assertFalse(passed)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["postgresql_round_trip_ms"], "not_available")
            self.assertEqual(
                set(payload["metrics"]),
                {
                    "postgresql_round_trip_ms",
                    "clickhouse_round_trip_ms",
                    "qdrant_round_trip_ms",
                    "redis_round_trip_ms",
                    "clamav_round_trip_ms",
                    "embedding_round_trip_ms",
                },
            )
            self.assertTrue(
                all(value == "not_available" for value in payload["metrics"].values())
            )
            self.assertFalse(payload["gateResult"]["passed"])

    def test_create_report_ignores_stale_metrics_and_requires_fresh_generation(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import create_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            report_path = root / "report.json"
            metrics_path.write_text(
                json.dumps({
                    "postgresql_round_trip_ms": 1,
                    "clickhouse_round_trip_ms": 1,
                    "qdrant_round_trip_ms": 1,
                    "redis_round_trip_ms": 1,
                    "clamav_round_trip_ms": 1,
                    "embedding_round_trip_ms": 1,
                }),
                encoding="utf-8",
            )
            passed = create_report(
                manifest_path=ROOT / "benchmarks/manifests/smoke.json",
                metrics_path=metrics_path,
                report_path=report_path,
                profile_name="service-round-trip",
                mode="phase1-smoke",
                cache_label="not-applicable",
                vector_dimension=32,
                model_slots=1,
                disk_path=root,
                benchmark_command=("benchmark", "--run"),
                benchmark_popen_factory=lambda _command, **_kwargs: _ImmediateProcess(),
                hardware_collector=lambda _path: {},
                version_collector=lambda: ({}, {}),
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(passed)
            self.assertTrue(all(value == "not_available" for value in payload["metrics"].values()))

    def test_create_report_accepts_only_metrics_generated_at_injected_path(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import create_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            report_path = root / "report.json"
            expected_metrics = {
                "postgresql_round_trip_ms": 1,
                "clickhouse_round_trip_ms": 1,
                "qdrant_round_trip_ms": 1,
                "redis_round_trip_ms": 1,
                "clamav_round_trip_ms": 1,
                "embedding_round_trip_ms": 1,
            }

            def popen(_command, **kwargs):
                generated_path = Path(kwargs["env"]["BENCHMARK_METRICS_PATH"])
                generated_path.write_text(json.dumps(expected_metrics), encoding="utf-8")
                self.assertEqual(
                    kwargs["env"]["BENCHMARK_MANIFEST"],
                    str((ROOT / "benchmarks/manifests/smoke.json").resolve()),
                )
                return _ImmediateProcess()

            passed = create_report(
                manifest_path=ROOT / "benchmarks/manifests/smoke.json",
                metrics_path=metrics_path,
                report_path=report_path,
                profile_name="service-round-trip",
                mode="phase1-smoke",
                cache_label="not-applicable",
                vector_dimension=32,
                model_slots=1,
                disk_path=root,
                benchmark_command=("benchmark", "--run"),
                benchmark_popen_factory=popen,
                hardware_collector=lambda _path: {},
                version_collector=lambda: ({}, {}),
            )
            self.assertTrue(passed)
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["metrics"],
                expected_metrics,
            )

    def test_hardware_inventory_and_checksums_are_reproducible(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import collect_hardware, sha256_file

        class Memory:
            total = 64 * 1024**3
            available = 12 * 1024**3

        class Partition:
            device = "Disk0"
            mountpoint = "C:\\"

        class Psutil:
            @staticmethod
            def cpu_count(logical=True):
                return 16 if logical else 8

            @staticmethod
            def virtual_memory():
                return Memory()

            @staticmethod
            def disk_partitions(all=False):
                return [Partition()]

        class Platform:
            @staticmethod
            def processor():
                return "Offline CPU"

        hardware = collect_hardware(Path("C:/data"), psutil_module=Psutil, platform_module=Platform)
        self.assertEqual(hardware["cpuModel"], "Offline CPU")
        self.assertEqual(hardware["physicalCores"], 8)
        self.assertEqual(hardware["logicalCores"], 16)
        self.assertEqual(hardware["totalRamBytes"], 64 * 1024**3)
        self.assertEqual(hardware["availableRamBytes"], 12 * 1024**3)
        self.assertEqual(hardware["diskDevice"], "Disk0")

        class RootPartition:
            device = "DiskRoot"
            mountpoint = "C:\\"

        class BothPsutil(Psutil):
            @staticmethod
            def disk_partitions(all=False):
                return [Partition(), RootPartition()]

        bounded = collect_hardware(Path("C:/database"), psutil_module=BothPsutil, platform_module=Platform)
        self.assertEqual(bounded["diskDevice"], "DiskRoot")

        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            handle.write(b"offline")
            path = Path(handle.name)
        try:
            self.assertEqual(sha256_file(path), hashlib.sha256(b"offline").hexdigest())
        finally:
            path.unlink(missing_ok=True)

    def test_hardware_inventory_degrades_without_optional_psutil(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import collect_hardware

        hardware = collect_hardware(Path.cwd(), psutil_module=None, platform_module=type(
            "Platform", (), {"processor": staticmethod(lambda: "Offline CPU")}
        ))
        self.assertEqual(hardware["cpuModel"], "Offline CPU")
        self.assertIn("physicalCores", hardware)
        self.assertEqual(hardware["totalRamBytes"], "not_available")
        self.assertEqual(hardware["diskDevice"], "not_available")

    def test_subprocess_wrapper_uses_argument_vector_and_shell_false(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import run_benchmark_command, run_fixed_command

        calls = []

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            if hasattr(kwargs["stdout"], "write"):
                kwargs["stdout"].write(b"tool 1.2\n")
            return _ImmediateProcess()

        result = run_fixed_command(("tool", "--version"), popen_factory=popen)
        self.assertEqual(result, {"exitCode": 0, "version": "tool 1.2"})
        self.assertEqual(calls[0][0], ["tool", "--version"])
        self.assertIs(calls[0][1]["shell"], False)
        with self.assertRaises(ValueError):
            run_fixed_command("tool --version", popen_factory=popen)  # type: ignore[arg-type]

        self.assertEqual(run_benchmark_command(("benchmark", "--run"), popen_factory=popen), 0)
        self.assertEqual(calls[1][0], ["benchmark", "--run"])
        self.assertIs(calls[1][1]["shell"], False)
        self.assertIs(calls[1][1]["stdout"], __import__("subprocess").DEVNULL)
        self.assertIs(calls[1][1]["stderr"], __import__("subprocess").DEVNULL)
        self.assertEqual(
            run_benchmark_command(
                ("benchmark", "--run"),
                popen_factory=lambda *_a, **_k: _ImmediateProcess(7),
            ),
            7,
        )

    def test_benchmark_timeout_kills_process_tree_without_unbounded_capture(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import run_benchmark_command

        calls = []
        killed = []

        class Process:
            pid = 321

            def wait(self, timeout=None):
                calls.append(("wait", timeout))
                if len([item for item in calls if item[0] == "wait"]) == 1:
                    raise __import__("subprocess").TimeoutExpired("benchmark", timeout)
                return -9

        process = Process()

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            return process

        exit_code = run_benchmark_command(
            ("benchmark", "--run"),
            timeout_seconds=1,
            popen_factory=popen,
            kill_strategy=lambda item: killed.append(item),
        )
        self.assertEqual(exit_code, 124)
        self.assertEqual(killed, [process])
        self.assertIs(calls[0][1]["shell"], False)
        self.assertIs(calls[0][1]["stdout"], __import__("subprocess").DEVNULL)
        self.assertIs(calls[0][1]["stderr"], __import__("subprocess").DEVNULL)

    def test_nonzero_benchmark_command_fails_even_with_numeric_metrics(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import create_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            report_path = root / "report.json"
            metrics_path.write_text(
                json.dumps({
                    "postgresql_round_trip_ms": 1,
                    "clickhouse_round_trip_ms": 1,
                    "qdrant_round_trip_ms": 1,
                    "redis_round_trip_ms": 1,
                    "clamav_round_trip_ms": 1,
                    "embedding_round_trip_ms": 1,
                }),
                encoding="utf-8",
            )
            passed = create_report(
                manifest_path=ROOT / "benchmarks/manifests/smoke.json",
                metrics_path=metrics_path,
                report_path=report_path,
                profile_name="service-round-trip",
                mode="phase1-smoke",
                cache_label="not-applicable",
                vector_dimension=32,
                model_slots=1,
                disk_path=root,
                hardware_collector=lambda _path: {},
                version_collector=lambda: ({}, {}),
                benchmark_command=("benchmark", "--run"),
                benchmark_popen_factory=lambda _command, **_kwargs: _ImmediateProcess(9),
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(passed)
            self.assertEqual(payload["commandExitCodes"]["benchmark"], 9)
            self.assertFalse(payload["gateResult"]["passed"])
            self.assertIn("benchmark_command", payload["gateResult"]["failures"])

    def test_missing_benchmark_command_is_reported_as_failed(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import create_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            report_path = root / "report.json"
            metrics_path.write_text("{}", encoding="utf-8")
            passed = create_report(
                manifest_path=ROOT / "benchmarks/manifests/smoke.json",
                metrics_path=metrics_path,
                report_path=report_path,
                profile_name="service-round-trip",
                mode="phase1-smoke",
                cache_label="not-applicable",
                vector_dimension=32,
                model_slots=1,
                disk_path=root,
                hardware_collector=lambda _path: {},
                version_collector=lambda: ({}, {}),
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(passed)
            self.assertIsNone(payload["commandExitCodes"]["benchmark"])
            self.assertIn("benchmark_command", payload["gateResult"]["failures"])

    def test_explicit_benchmark_timeout_rejects_non_positive_values(self) -> None:
        from tools.benchmarks.run_capacity_benchmark import create_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            metrics_path.write_text("{}", encoding="utf-8")
            for timeout in (0, -1, True):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    create_report(
                        manifest_path=ROOT / "benchmarks/manifests/smoke.json",
                        metrics_path=metrics_path,
                        report_path=root / "report.json",
                        profile_name="service-round-trip",
                        mode="phase1-smoke",
                        cache_label="not-applicable",
                        vector_dimension=32,
                        model_slots=1,
                        disk_path=root,
                        benchmark_command=("benchmark", "--run"),
                        benchmark_timeout_seconds=timeout,
                        hardware_collector=lambda _path: {},
                        version_collector=lambda: ({}, {}),
                    )

    def test_locust_contract_is_import_safe_and_scoped_by_trusted_identity(self) -> None:
        source = (ROOT / "benchmarks/locustfile.py").read_text(encoding="utf-8")
        self.assertIn("@task(4)", source)
        self.assertIn("@task(2)", source)
        self.assertIn("think_time_seconds", source)
        self.assertIn('"X-Identity"', source)
        self.assertNotIn("X-Tenant", source)
        self.assertNotIn("X-Classification", source)
        self.assertIn("queue_feedback_ms", source)
        self.assertIn("first_token_ms", source)
        self.assertIn("/api/conversations/{conversation_id}/messages/stream", source)
        self.assertIn("Idempotency-Key", source)
        self.assertIn('"content"', source)
        self.assertIn('"mode"', source)
        from tools.benchmarks.locustfile import parse_sse_events, record_sse_timings

        events = list(parse_sse_events([
            "event: accepted\n",
            'data: {"requestId":"r1"}\n',
            "\n",
            "event: queued\n",
            'data: {"position":1}\n',
            "\n",
            "event: delta\n",
            'data: {"text":"答"}\n',
            "\n",
        ]))
        self.assertEqual([item["event"] for item in events], ["accepted", "queued", "delta"])
        multiline = list(parse_sse_events([
            "event: delta\n",
            'data: {"text":\n',
            'data: "答案"}\n',
            "\n",
        ]))
        self.assertEqual(multiline, [{"event": "delta", "data": {"text": "答案"}}])
        context: dict[str, object] = {}
        record_sse_timings(events, context, elapsed_ms=[10, 25, 80])
        self.assertEqual(context["queue_feedback_ms"], 10)
        self.assertEqual(context["first_token_ms"], 80)

        from tools.benchmarks.locustfile import stream_failure_reason

        self.assertEqual(stream_failure_reason([], 200), "empty_stream")
        self.assertEqual(stream_failure_reason([{"event": "accepted", "data": {}}], 200), "missing_terminal_event")
        self.assertEqual(stream_failure_reason([{"event": "error", "data": {}}], 200), "sse_error")

    def test_locust_metrics_are_atomically_persisted_from_full_stream_contexts(self) -> None:
        from tools.benchmarks import locustfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            locustfile.RECORDED_REQUESTS.clear()
            locustfile.RECORDED_REQUESTS.extend(
                [
                    {"request_kind": "structured", "full_stream_ms": 100.0, "failed": False, "queue_feedback_ms": 10.0, "first_token_ms": 60.0},
                    {"request_kind": "structured", "full_stream_ms": 200.0, "failed": False, "queue_feedback_ms": 20.0, "first_token_ms": 80.0, "cache_hit": True},
                    {"request_kind": "document", "full_stream_ms": 300.0, "failed": True},
                    {"request_kind": "mixed", "full_stream_ms": 400.0, "failed": False, "queue_feedback_ms": 30.0, "first_token_ms": 100.0, "cache_hit": True},
                ]
            )
            locustfile.RECORDED_REQUESTS[0]["cache_hit"] = True
            environment = type("Environment", (), {"stats": object()})()
            locustfile.persist_benchmark_metrics(environment, path=path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["structured_p95_ms"], 200.0)
            self.assertNotIn("document_p95_ms", payload)
            self.assertEqual(payload["mixed_p95_ms"], 400.0)
            self.assertEqual(payload["queue_feedback_p95_ms"], 30.0)
            self.assertEqual(payload["first_token_p95_ms"], 100.0)
            self.assertEqual(payload["error_rate"], 0.25)
            self.assertEqual(payload["warm_cache_hit_rate"], 1.0)

    def test_locust_connection_failure_is_included_in_error_rate_once(self) -> None:
        from tools.benchmarks import locustfile

        locustfile.RECORDED_REQUESTS.clear()
        context: dict[str, object] = {}
        locustfile.record_stream_timings(
            name="query:structured",
            context=context,
            response_time=25.0,
            exception=ConnectionError("offline"),
        )
        locustfile.record_stream_timings(
            name="query:structured",
            context=context,
            response_time=25.0,
            exception=ConnectionError("offline"),
        )
        self.assertEqual(len(locustfile.RECORDED_REQUESTS), 1)
        metrics = locustfile.build_metrics(locustfile.RECORDED_REQUESTS)
        self.assertEqual(metrics["error_rate"], 1.0)
        self.assertNotIn("structured_p95_ms", metrics)

    def test_partial_success_timing_or_cache_coverage_omits_metric(self) -> None:
        from tools.benchmarks.locustfile import build_metrics

        metrics = build_metrics([
            {
                "request_kind": "structured",
                "full_stream_ms": 100.0,
                "failed": False,
                "queue_feedback_ms": 10.0,
                "first_token_ms": 50.0,
                "cache_hit": True,
            },
            {
                "request_kind": "document",
                "full_stream_ms": 200.0,
                "failed": False,
            },
        ])
        self.assertEqual(metrics["error_rate"], 0.0)
        self.assertNotIn("queue_feedback_p95_ms", metrics)
        self.assertNotIn("first_token_p95_ms", metrics)
        self.assertNotIn("warm_cache_hit_rate", metrics)
        module = importlib.import_module("tools.benchmarks.locustfile")
        self.assertEqual(module.REQUEST_WEIGHTS, {"structured": 4, "document": 4, "mixed": 2})


if __name__ == "__main__":
    unittest.main()
