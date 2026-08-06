from __future__ import annotations

import inspect
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, BrokenBarrierError, Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from app.infra import health as health_module
from app.infra.health import DependencyCheck, DependencyHealthRegistry, build_dependency_checks
from app.main import create_app
from app.offline_settings import OfflineSettings
from app.repository import InMemoryChatRepository
from app.retrieval_scope import DynamicRetrievalScopeProvider
from app.retrieval_settings import RetrievalSettings
from app.routes import router as app_router
from app.seed import build_seed_state
from fastapi.testclient import TestClient


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

    def __init__(
        self,
        payload: dict[str, object] | None = None,
        *,
        chunks: list[bytes] | None = None,
        read_delay_seconds: float = 0.0,
    ) -> None:
        self.payload = payload
        body = json.dumps(payload or {}).encode("utf-8")
        self.chunks = list(chunks or [body])
        self.read_delay_seconds = read_delay_seconds
        self.closed = False

    def __enter__(self) -> MetadataResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def iter_raw(self) -> object:
        for chunk in self.chunks:
            if self.read_delay_seconds:
                time.sleep(self.read_delay_seconds)
            yield chunk

    def close(self) -> None:
        self.closed = True


class RetrievalHealthHttpClient:
    def __init__(self, responses: dict[str, MetadataResponse]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def stream(self, method: str, url: str, **kwargs: object) -> MetadataResponse:
        del method, kwargs
        self.urls.append(url)
        return self.responses[url]

    def get(self, url: str) -> MetadataResponse:
        raise AssertionError(f"metadata health must stream instead of buffering {url}")


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


class RetrievalHealthAudit:
    def __init__(
        self,
        collection_name: str | None = "knowledge_chunks_qwen3_v1",
        *,
        embedding_fingerprint: object | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_fingerprint = embedding_fingerprint

    def active_publication(self, alias_name: str) -> object | None:
        del alias_name
        if self.collection_name is None:
            return None
        return SimpleNamespace(
            id="publication-id-not-used-as-scope",
            collection_name=self.collection_name,
            embedding_fingerprint=self.embedding_fingerprint,
        )


def retrieval_scope_provider(
    gateway: RetrievalHealthGateway,
    *,
    active_collection: str | None = "knowledge_chunks_qwen3_v1",
    publication_fingerprint: object | None = None,
) -> DynamicRetrievalScopeProvider:
    configured = RetrievalSettings.from_environ(retrieval_health_environment())
    if publication_fingerprint is None:
        publication_fingerprint = configured.embedding_fingerprint
    return DynamicRetrievalScopeProvider(
        audit=RetrievalHealthAudit(
            active_collection,
            embedding_fingerprint=publication_fingerprint,
        ),
        gateway=gateway,
        alias_name="knowledge_chunks_current",
        knowledge_base_id="default",
        permission_tags=("internal",),
        embedding_fingerprint=configured.embedding_fingerprint,
    )


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
    def test_qwen3_rrf_only_health_omits_reranker_dependency(self) -> None:
        environ = retrieval_health_environment()
        environ["RERANKER_ENABLED"] = "false"
        for key in tuple(environ):
            if key == "RERANKER_SERVICE_URL" or key.startswith("RERANKER_MODEL_"):
                environ.pop(key)
        gateway = RetrievalHealthGateway()
        http_client = RetrievalHealthHttpClient(retrieval_health_responses())

        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=http_client,
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=gateway,
            retrieval_scope_provider=retrieval_scope_provider(gateway),
        )
        retrieval_checks = [
            check for check in checks if check.name in {"qdrant", "embedding", "reranker"}
        ]
        for check in retrieval_checks:
            check.check()

        self.assertNotIn("reranker", {check.name for check in checks})
        self.assertNotIn("http://127.0.0.1:8082/v1/metadata", http_client.urls)

    def test_qwen3_retrieval_health_checks_metadata_alias_and_dimension(self) -> None:
        environ = retrieval_health_environment()
        offline = OfflineSettings.from_environ(environ)
        retrieval = RetrievalSettings.from_environ(environ)
        http_client = RetrievalHealthHttpClient(retrieval_health_responses())
        gateway = RetrievalHealthGateway()
        scope_provider = retrieval_scope_provider(gateway)

        checks = build_dependency_checks(
            offline,
            database=object(),
            environ=environ,
            http_client=http_client,
            retrieval_settings=retrieval,
            retrieval_gateway=gateway,
            retrieval_scope_provider=scope_provider,
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
        self.assertTrue(all(response.closed for response in http_client.responses.values()))

    def test_qwen3_metadata_mismatch_is_not_ready(self) -> None:
        environ = retrieval_health_environment()
        gateway = RetrievalHealthGateway()
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(
                retrieval_health_responses(reranker_version="unexpected")
            ),
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=gateway,
            retrieval_scope_provider=retrieval_scope_provider(gateway),
        )
        reranker = next(check for check in checks if check.name == "reranker")

        self.assertEqual(reranker.check(), (False, "metadata mismatch"))

    def test_shadow_retrieval_dependency_failure_is_degraded_but_ready(self) -> None:
        environ = retrieval_health_environment("shadow")
        gateway = RetrievalHealthGateway()
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(
                retrieval_health_responses(reranker_version="unexpected")
            ),
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=gateway,
            retrieval_scope_provider=retrieval_scope_provider(gateway),
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

    def test_shadow_degraded_wrapper_preserves_probe_cancellation(self) -> None:
        class CancellableCheck:
            def __init__(self) -> None:
                self.cancelled = False

            def __call__(self) -> tuple[bool, str]:
                return False, "unavailable"

            def cancel(self) -> None:
                self.cancelled = True

        check = CancellableCheck()
        degraded = health_module._degraded_dependency(check)

        degraded.cancel()

        self.assertTrue(check.cancelled)

    def test_retrieval_dependency_timeouts_degrade_only_in_shadow_mode(self) -> None:
        class BlockingCancellableCheck:
            def __init__(self) -> None:
                self.cancelled = Event()

            def __call__(self) -> tuple[bool, str]:
                self.cancelled.wait(1.0)
                return False, "unavailable"

            def cancel(self) -> None:
                self.cancelled.set()

        for mode, expected in (
            ("shadow", (True, "degraded")),
            ("qwen3", (False, "unavailable")),
        ):
            with self.subTest(mode=mode):
                environ = retrieval_health_environment(mode)
                environ["DEPENDENCY_TIMEOUT_SECONDS"] = "0.01"
                blockers = {
                    name: BlockingCancellableCheck() for name in ("qdrant", "embedding", "reranker")
                }
                with (
                    patch.object(
                        health_module,
                        "_qdrant_retrieval_check",
                        return_value=blockers["qdrant"],
                    ),
                    patch.object(
                        health_module,
                        "_embedding_metadata_check",
                        return_value=blockers["embedding"],
                    ),
                    patch.object(
                        health_module,
                        "_reranker_metadata_check",
                        return_value=blockers["reranker"],
                    ),
                ):
                    checks = build_dependency_checks(
                        OfflineSettings.from_environ(environ),
                        database=object(),
                        environ=environ,
                        retrieval_settings=RetrievalSettings.from_environ(environ),
                    )

                retrieval_checks = {
                    check.name: check.check() for check in checks if check.name in blockers
                }

                self.assertEqual(
                    retrieval_checks,
                    dict.fromkeys(blockers, expected),
                )
                self.assertTrue(all(blocker.cancelled.is_set() for blocker in blockers.values()))

    def test_retrieval_health_fails_closed_when_alias_and_audit_diverge(self) -> None:
        environ = retrieval_health_environment()
        gateway = RetrievalHealthGateway(alias="knowledge_chunks_qwen3_v2")
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(retrieval_health_responses()),
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=gateway,
            retrieval_scope_provider=retrieval_scope_provider(
                gateway,
                active_collection="knowledge_chunks_qwen3_v1",
            ),
        )
        qdrant = next(check for check in checks if check.name == "qdrant")

        self.assertEqual(qdrant.check(), (False, "scope unavailable"))

    def test_retrieval_health_reports_embedding_fingerprint_mismatch(self) -> None:
        environ = retrieval_health_environment()
        retrieval = RetrievalSettings.from_environ(environ)
        gateway = RetrievalHealthGateway()
        mismatched = replace(
            retrieval.embedding_fingerprint,
            model_version="old-qwen3-v1",
        )
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(retrieval_health_responses()),
            retrieval_settings=retrieval,
            retrieval_gateway=gateway,
            retrieval_scope_provider=retrieval_scope_provider(
                gateway,
                publication_fingerprint=mismatched,
            ),
        )
        qdrant = next(check for check in checks if check.name == "qdrant")

        self.assertEqual(qdrant.check(), (False, "embedding_fingerprint_mismatch"))
        self.assertEqual(gateway.validations, [])

    def test_metadata_stream_rejects_chunked_body_over_cap_and_closes_response(self) -> None:
        environ = retrieval_health_environment()
        gateway = RetrievalHealthGateway()
        responses = retrieval_health_responses()
        oversized = MetadataResponse(chunks=[b"{" + b"x" * 700, b"y" * 700 + b"}"])
        responses["http://127.0.0.1:8081/v1/metadata"] = oversized
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(responses),
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=gateway,
            retrieval_scope_provider=retrieval_scope_provider(gateway),
        )
        embedding = next(check for check in checks if check.name == "embedding")

        self.assertEqual(embedding.check(), (False, "invalid metadata response"))
        self.assertTrue(oversized.closed)

    def test_metadata_stream_enforces_absolute_deadline_and_leaves_no_probe_thread(
        self,
    ) -> None:
        environ = retrieval_health_environment()
        environ["DEPENDENCY_TIMEOUT_SECONDS"] = "0.03"
        gateway = RetrievalHealthGateway()
        responses = retrieval_health_responses()
        trickle = MetadataResponse(
            chunks=[b"{" if index == 0 else b" " for index in range(8)],
            read_delay_seconds=0.01,
        )
        responses["http://127.0.0.1:8081/v1/metadata"] = trickle
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(responses),
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=gateway,
            retrieval_scope_provider=retrieval_scope_provider(gateway),
        )
        embedding = next(check for check in checks if check.name == "embedding")

        self.assertEqual(embedding.check(), (False, "unavailable"))
        time.sleep(0.05)
        self.assertFalse(
            any(
                thread.name == "dependency-probe-embedding" and thread.is_alive()
                for thread in threading.enumerate()
            )
        )
        self.assertTrue(trickle.closed)

    def test_metadata_stream_deadline_also_applies_when_the_final_read_ends(self) -> None:
        class SlowEndResponse(MetadataResponse):
            def iter_raw(self) -> object:
                time.sleep(0.04)
                return
                yield b""  # pragma: no cover

        slow_end = SlowEndResponse()
        endpoint = "http://127.0.0.1:8081/v1/metadata"
        payload, detail = health_module._metadata_payload(
            RetrievalHealthHttpClient({endpoint: slow_end}),
            endpoint,
            0.03,
        )

        self.assertIsNone(payload)
        self.assertEqual(detail, "unavailable")
        self.assertTrue(slow_end.closed)

    def test_real_httpx_metadata_probe_cancels_stalled_response_and_recovers(self) -> None:
        first_chunk_sent = Event()
        client_disconnected = Event()
        request_lock = Lock()
        request_count = 0
        metadata_body = json.dumps(
            retrieval_health_responses()["http://127.0.0.1:8081/v1/metadata"].payload
        ).encode("utf-8")

        class MetadataHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                nonlocal request_count
                with request_lock:
                    request_count += 1
                    current_request = request_count
                if current_request == 1:
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("connection", "keep-alive")
                    self.end_headers()
                    self.wfile.write(b"{")
                    self.wfile.flush()
                    first_chunk_sent.set()
                    self.connection.settimeout(1.0)
                    try:
                        while self.connection.recv(1):
                            pass
                    except OSError:
                        pass
                    finally:
                        client_disconnected.set()
                        self.close_connection = True
                    return

                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(metadata_body)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(metadata_body)
                self.wfile.flush()

            def log_message(self, _format: str, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), MetadataHandler)
        server.daemon_threads = True
        server_thread = Thread(target=server.serve_forever, name="metadata-test-server")
        server_thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server_thread.join, 1.0)
        self.addCleanup(server.shutdown)
        service_url = f"http://127.0.0.1:{server.server_port}"
        environ = retrieval_health_environment()
        # Keep the production timeout realistic while excluding cold client
        # construction from the behavior under test. This test measures
        # cancellation of a stalled response and recovery on later probes.
        environ["DEPENDENCY_TIMEOUT_SECONDS"] = "0.5"
        environ["EMBEDDING_SERVICE_URL"] = service_url
        clients = [
            httpx.Client(
                timeout=httpx.Timeout(5.0),
                follow_redirects=False,
                trust_env=False,
            )
            for _ in range(3)
        ]
        available_clients = iter(clients)

        def metadata_client_factory() -> httpx.Client:
            return next(available_clients)

        gateway = RetrievalHealthGateway()
        checks = build_dependency_checks(
            OfflineSettings.from_environ(environ),
            database=object(),
            environ=environ,
            http_client=RetrievalHealthHttpClient(retrieval_health_responses()),
            metadata_http_client_factory=metadata_client_factory,
            retrieval_settings=RetrievalSettings.from_environ(environ),
            retrieval_gateway=gateway,
            retrieval_scope_provider=retrieval_scope_provider(gateway),
        )
        embedding = next(check for check in checks if check.name == "embedding")

        started = time.monotonic()
        first = embedding.check()
        elapsed = time.monotonic() - started

        self.assertEqual(first, (False, "unavailable"))
        self.assertTrue(first_chunk_sent.is_set())
        self.assertLess(elapsed, 0.9)
        self.assertTrue(client_disconnected.wait(0.5))
        self.assertFalse(
            any(
                thread.name == "dependency-probe-embedding" and thread.is_alive()
                for thread in threading.enumerate()
            )
        )
        self.assertEqual(embedding.check(), (True, "ready"))
        self.assertEqual(embedding.check(), (True, "ready"))
        self.assertEqual(request_count, 3)
        self.assertEqual(len(clients), 3)
        self.assertTrue(all(client.is_closed for client in clients))

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
