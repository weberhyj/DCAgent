from __future__ import annotations

import inspect
import unittest
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from threading import Barrier, BrokenBarrierError, Event, Lock, Thread
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.infra import health as health_module
from app.infra.health import DependencyCheck, DependencyHealthRegistry, build_dependency_checks
from app.main import create_app
from app.offline_settings import OfflineSettings
from app.repository import InMemoryChatRepository
from app.retrieval_settings import RetrievalSettings
from app.routes import router as app_router
from app.seed import build_seed_state


def retrieval_health_environment(mode: str = "qwen3") -> dict[str, str]:
    checksum = "a" * 64
    return {
        "OFFLINE_MODE": "true",
        "DATABASE_URL": "postgresql+psycopg://dc_agent@127.0.0.1/dc_agent",
        "CLICKHOUSE_URL": "http://127.0.0.1:8123",
        "QDRANT_URL": "http://127.0.0.1:6333",
        "REDIS_URL": "redis://127.0.0.1:6379/0",
        "CLAMAV_HOST": "127.0.0.1",
        "EMBEDDING_SERVICE_URL": "http://127.0.0.1:8081",
        "RERANKER_SERVICE_URL": "http://127.0.0.1:8082",
        "LLAMA_SERVER_URL": "http://127.0.0.1:8080",
        "LLM_PROVIDER": "physoc_deepseek",
        "RETRIEVAL_MODE": mode,
        "RETRIEVAL_PERMISSION_TAGS": "internal",
        "EMBEDDING_MODEL_NAME": "Qwen/Qwen3-Embedding-0.6B",
        "EMBEDDING_MODEL_VERSION": "1.0.0",
        "EMBEDDING_MODEL_SHA256": checksum,
        "EMBEDDING_MODEL_DIMENSIONS": "1024",
        "EMBEDDING_MODEL_NORMALIZED": "true",
        "EMBEDDING_ENCODING_PROFILE_SHA256": checksum,
        "EMBEDDING_PROTOCOL_VERSION": "v1",
        "RERANKER_MODEL_NAME": "Qwen/Qwen3-Reranker-0.6B",
        "RERANKER_MODEL_VERSION": "1.0.0",
        "RERANKER_MODEL_SHA256": checksum,
        "RERANKER_PROMPT_PROFILE_SHA256": checksum,
        "RERANKER_PROTOCOL_VERSION": "v1",
    }


class MetadataResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def json(self) -> dict[str, object]:
        return dict(self.payload)


class RetrievalHealthHttpClient:
    def __init__(self, responses: dict[str, MetadataResponse]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str) -> MetadataResponse:
        self.urls.append(url)
        return self.responses[url]


class RetrievalHealthGateway:
    def __init__(self, *, alias: str | None = "knowledge_chunks_qwen3_v1") -> None:
        self.alias = alias
        self.validations: list[tuple[str, int]] = []

    def resolve_alias(self) -> str | None:
        return self.alias

    def validate_collection(
        self,
        collection_name: str,
        *,
        dense_dimensions: int,
    ) -> int:
        self.validations.append((collection_name, dense_dimensions))
        return 10


def retrieval_health_responses(*, reranker_version: str = "1.0.0") -> dict[str, MetadataResponse]:
    checksum = "a" * 64
    return {
        "http://127.0.0.1:6333/readyz": MetadataResponse({"status": "ready"}),
        "http://127.0.0.1:8081/v1/metadata": MetadataResponse(
            {
                "modelName": "Qwen/Qwen3-Embedding-0.6B",
                "modelVersion": "1.0.0",
                "modelChecksum": checksum,
                "dimensions": 1024,
                "normalized": True,
                "encodingProfileSha256": checksum,
                "protocolVersion": "v1",
            }
        ),
        "http://127.0.0.1:8082/v1/metadata": MetadataResponse(
            {
                "modelName": "Qwen/Qwen3-Reranker-0.6B",
                "modelVersion": reranker_version,
                "modelChecksum": checksum,
                "promptProfileSha256": checksum,
                "protocolVersion": "v1",
            }
        ),
    }


class InfraHealthTest(unittest.TestCase):
    def test_qwen3_retrieval_health_checks_metadata_alias_and_dimension(self) -> None:
        environ = retrieval_health_environment()
        offline = OfflineSettings.from_environ(environ)
        retrieval = RetrievalSettings.from_environ(environ)
        http_client = RetrievalHealthHttpClient(retrieval_health_responses())
        gateway = RetrievalHealthGateway()

        checks = build_dependency_checks(
            offline,
            database=object(),
            environ=environ,
            http_client=http_client,
            retrieval_settings=retrieval,
            retrieval_gateway=gateway,
        )
        retrieval_checks = {
            check.name: check.check()
            for check in checks
            if check.name in {"qdrant", "embedding", "reranker"}
        }

        self.assertEqual(
            retrieval_checks,
            {
                "qdrant": (True, "ready"),
                "embedding": (True, "ready"),
                "reranker": (True, "ready"),
            },
        )
        self.assertEqual(gateway.validations, [("knowledge_chunks_qwen3_v1", 1024)])
        self.assertIn("http://127.0.0.1:8081/v1/metadata", http_client.urls)
        self.assertIn("http://127.0.0.1:8082/v1/metadata", http_client.urls)

    def test_qwen3_metadata_mismatch_is_not_ready(self) -> None:
        environ = retrieval_health_environment()
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(
                retrieval_health_responses(reranker_version="unexpected")
            ),
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=RetrievalHealthGateway(),
        )
        reranker = next(check for check in checks if check.name == "reranker")

        self.assertEqual(reranker.check(), (False, "metadata mismatch"))

    def test_shadow_retrieval_dependency_failure_is_degraded_but_ready(self) -> None:
        environ = retrieval_health_environment("shadow")
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(
                retrieval_health_responses(reranker_version="unexpected")
            ),
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=RetrievalHealthGateway(),
        )
        registry = DependencyHealthRegistry(
            [
                check
                for check in checks
                if check.name in {"qdrant", "embedding", "reranker", "retrieval_shadow"}
            ]
        )

        report = registry.report()

        self.assertTrue(registry.ready())
        self.assertEqual(report["reranker"], {"ok": True, "detail": "degraded"})
        self.assertEqual(report["retrieval_shadow"], {"ok": True, "detail": "ready"})

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
