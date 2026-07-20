from __future__ import annotations

import inspect
import unittest
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from threading import Barrier, BrokenBarrierError, Event, Lock, Thread
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.infra import health as health_module
from app.infra.health import DependencyCheck, DependencyHealthRegistry
from app.main import create_app
from app.repository import InMemoryChatRepository
from app.routes import router as app_router
from app.seed import build_seed_state


class InfraHealthTest(unittest.TestCase):
    def test_liveness_does_not_require_external_services(self) -> None:
        calls = 0

        def qdrant_check() -> tuple[bool, str]:
            nonlocal calls
            calls += 1
            return False, "unavailable"

        registry = DependencyHealthRegistry([DependencyCheck("qdrant", qdrant_check)])
        client = TestClient(
            create_app(
                InMemoryChatRepository(build_seed_state()),
                health_registry=registry,
            )
        )

        liveness = client.get("/api/healthz")

        self.assertEqual(liveness.status_code, 200)
        self.assertEqual(liveness.json(), {"status": "ok"})
        self.assertEqual(calls, 0)

        readiness = client.get("/api/readyz")

        self.assertEqual(readiness.status_code, 503)
        self.assertEqual(
            readiness.json(),
            {
                "status": "not_ready",
                "dependencies": {"qdrant": {"ok": False, "detail": "unavailable"}},
            },
        )
        self.assertEqual(calls, 1)

    def test_registry_reports_success_and_empty_registry_is_ready(self) -> None:
        registry = DependencyHealthRegistry([DependencyCheck("redis", lambda: (True, "ready"))])

        self.assertEqual(
            registry.report(),
            {"redis": {"ok": True, "detail": "ready"}},
        )
        self.assertTrue(registry.ready())
        self.assertEqual(DependencyHealthRegistry().report(), {})
        self.assertTrue(DependencyHealthRegistry().ready())

    def test_registry_sanitizes_check_exceptions(self) -> None:
        def failing_check() -> tuple[bool, str]:
            raise RuntimeError("postgresql://admin:secret@example.invalid/private-database")

        registry = DependencyHealthRegistry([DependencyCheck("postgresql", failing_check)])

        report = registry.report()

        self.assertEqual(report["postgresql"]["ok"], False)
        self.assertEqual(report["postgresql"]["detail"], "check failed")
        self.assertNotIn("secret", str(report))
        self.assertFalse(registry.ready())

    def test_registry_rejects_malformed_check_results(self) -> None:
        cases = (
            lambda: True,
            lambda: (True,),
            lambda: ("yes", "ready"),
            lambda: (True, object()),
        )

        for index, check in enumerate(cases):
            with self.subTest(index=index):
                registry = DependencyHealthRegistry(
                    [DependencyCheck("invalid", check)]  # type: ignore[arg-type]
                )
                self.assertEqual(
                    registry.report(),
                    {
                        "invalid": {
                            "ok": False,
                            "detail": "invalid check result",
                        }
                    },
                )
                self.assertFalse(registry.ready())

    def test_registry_rejects_duplicate_dependency_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate dependency check"):
            DependencyHealthRegistry(
                [
                    DependencyCheck("redis", lambda: (False, "unavailable")),
                    DependencyCheck("redis", lambda: (True, "ready")),
                ]
            )

    def test_registry_runs_dependency_checks_concurrently(self) -> None:
        barrier = Barrier(2, timeout=1.0)

        def concurrent_check() -> tuple[bool, str]:
            try:
                barrier.wait()
            except BrokenBarrierError:
                return False, "checks were sequential"
            return True, "ready"

        registry = DependencyHealthRegistry(
            [
                DependencyCheck("clickhouse", concurrent_check),
                DependencyCheck("qdrant", concurrent_check),
            ]
        )

        self.assertEqual(
            registry.report(),
            {
                "clickhouse": {"ok": True, "detail": "ready"},
                "qdrant": {"ok": True, "detail": "ready"},
            },
        )

    def test_registries_share_one_bounded_executor(self) -> None:
        registry = DependencyHealthRegistry([DependencyCheck("redis", lambda: (True, "ready"))])
        shared_executor = None
        with patch.object(
            health_module,
            "_SHARED_EXECUTOR",
            None,
            create=True,
        ):
            with patch.object(
                health_module,
                "ThreadPoolExecutor",
                side_effect=RealThreadPoolExecutor,
            ) as constructor:
                registry.report()
                registry.report()
                shared_executor = health_module._SHARED_EXECUTOR
            self.assertEqual(constructor.call_count, 1)

        if shared_executor is not None:
            shared_executor.shutdown(wait=True, cancel_futures=True)

    def test_registry_close_shuts_down_and_later_report_rebuilds_executor(
        self,
    ) -> None:
        registry = DependencyHealthRegistry([DependencyCheck("redis", lambda: (True, "ready"))])
        with patch.object(
            health_module,
            "_SHARED_EXECUTOR",
            None,
            create=True,
        ):
            with patch.object(
                health_module,
                "ThreadPoolExecutor",
                side_effect=RealThreadPoolExecutor,
            ) as constructor:
                registry.report()
                first_executor = health_module._SHARED_EXECUTOR
                registry.close()
                self.assertIsNone(health_module._SHARED_EXECUTOR)

                registry.report()
                second_executor = health_module._SHARED_EXECUTOR
                self.assertIsNot(first_executor, second_executor)
                self.assertEqual(constructor.call_count, 2)
                registry.close()

    def test_registry_single_flights_overlapping_reports_without_waiting(
        self,
    ) -> None:
        started = Event()
        release = Event()
        follower_done = Event()
        calls = 0
        calls_lock = Lock()

        def slow_check() -> tuple[bool, str]:
            nonlocal calls
            with calls_lock:
                calls += 1
            started.set()
            release.wait(timeout=1.0)
            return True, "ready"

        registry = DependencyHealthRegistry([DependencyCheck("redis", slow_check)])
        leader_results: list[dict[str, dict[str, bool | str]]] = []
        follower_results: list[dict[str, dict[str, bool | str]]] = []

        def follow_inflight_report() -> None:
            follower_results.append(registry.report())
            follower_done.set()

        first = Thread(target=lambda: leader_results.append(registry.report()))
        first.start()
        self.assertTrue(started.wait(timeout=1.0))
        second = Thread(target=follow_inflight_report)
        second.start()

        try:
            self.assertTrue(follower_done.wait(timeout=0.5))
            self.assertEqual(
                follower_results,
                [
                    {
                        "redis": {
                            "ok": False,
                            "detail": "check in progress",
                        }
                    }
                ],
            )
            self.assertEqual(calls, 1)
        finally:
            release.set()
            first.join(timeout=1.0)
            second.join(timeout=1.0)

        self.assertEqual(calls, 1)
        self.assertEqual(
            leader_results,
            [{"redis": {"ok": True, "detail": "ready"}}],
        )

    def test_registry_reuses_defensive_copy_within_cache_ttl(self) -> None:
        calls = 0
        clock = [10.0]

        def check() -> tuple[bool, str]:
            nonlocal calls
            calls += 1
            return True, f"ready {calls}"

        registry = DependencyHealthRegistry(
            [DependencyCheck("redis", check)],
            cache_ttl_seconds=0.5,
        )

        with patch.object(
            health_module,
            "monotonic",
            side_effect=lambda: clock[0],
        ):
            first = registry.report()
            first["redis"]["ok"] = False
            clock[0] = 10.25
            cached = registry.report()
            clock[0] = 10.51
            refreshed = registry.report()

        self.assertEqual(calls, 2)
        self.assertEqual(
            cached,
            {"redis": {"ok": True, "detail": "ready 1"}},
        )
        self.assertEqual(
            refreshed,
            {"redis": {"ok": True, "detail": "ready 2"}},
        )

    def test_registry_bounds_stale_cache_while_refresh_is_inflight(self) -> None:
        refresh_started = Event()
        release_refresh = Event()
        follower_done = Event()
        calls = 0
        clock = [20.0]

        def check() -> tuple[bool, str]:
            nonlocal calls
            calls += 1
            if calls == 2:
                refresh_started.set()
                release_refresh.wait(timeout=1.0)
            return True, f"ready {calls}"

        registry = DependencyHealthRegistry(
            [DependencyCheck("redis", check)],
            cache_ttl_seconds=0.5,
            max_stale_seconds=2.0,
        )
        refresh_results: list[dict[str, dict[str, bool | str]]] = []
        follower_results: list[dict[str, dict[str, bool | str]]] = []

        with patch.object(
            health_module,
            "monotonic",
            side_effect=lambda: clock[0],
        ):
            self.assertEqual(
                registry.report(),
                {"redis": {"ok": True, "detail": "ready 1"}},
            )
            clock[0] = 21.0
            refresh = Thread(target=lambda: refresh_results.append(registry.report()))
            refresh.start()
            self.assertTrue(refresh_started.wait(timeout=1.0))

            def follow_refresh() -> None:
                follower_results.append(registry.report())
                follower_done.set()

            follower = Thread(target=follow_refresh)
            follower.start()
            try:
                self.assertTrue(follower_done.wait(timeout=0.5))
                self.assertEqual(
                    follower_results,
                    [{"redis": {"ok": True, "detail": "ready 1"}}],
                )
                self.assertEqual(calls, 2)

                clock[0] = 22.1
                self.assertEqual(
                    registry.report(),
                    {
                        "redis": {
                            "ok": False,
                            "detail": "check in progress",
                        }
                    },
                )
            finally:
                release_refresh.set()
                refresh.join(timeout=1.0)
                follower.join(timeout=1.0)

        self.assertEqual(
            refresh_results,
            [{"redis": {"ok": True, "detail": "ready 2"}}],
        )

    def test_dependency_check_is_frozen_and_slotted(self) -> None:
        check = DependencyCheck("redis", lambda: (True, "ready"))

        with self.assertRaises((AttributeError, TypeError)):
            check.name = "changed"  # type: ignore[misc]
        self.assertFalse(hasattr(check, "__dict__"))

    def test_health_compatibility_endpoint_is_unchanged(self) -> None:
        client = TestClient(create_app(InMemoryChatRepository(build_seed_state())))

        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_liveness_handlers_do_not_use_the_worker_thread_pool(self) -> None:
        endpoints = {
            route.path: route.endpoint for route in app_router.routes if hasattr(route, "endpoint")
        }

        self.assertTrue(inspect.iscoroutinefunction(endpoints["/api/health"]))
        self.assertTrue(inspect.iscoroutinefunction(endpoints["/api/healthz"]))


if __name__ == "__main__":
    unittest.main()
