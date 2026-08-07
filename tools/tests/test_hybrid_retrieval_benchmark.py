from __future__ import annotations

import json
import math
import multiprocessing
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@dataclass(frozen=True, slots=True)
class _Candidate:
    chunk_id: str


@dataclass(frozen=True, slots=True)
class _Outcome:
    mode: str
    candidates: tuple[_Candidate, ...]
    stage_ms: dict[str, float]
    fallback_reason: str | None = None


class _FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, request):
        self.queries.append(request.query)
        return _Outcome(
            mode="qwen3",
            candidates=(_Candidate(f"chunk-{request.routing_key}"),),
            stage_ms={"embedding": 10.0, "qdrant": 20.0, "reranker": 30.0},
        )


class _ClosingResource:
    def __init__(self, name: str, events: list[str], method: str = "close") -> None:
        self.name = name
        self.events = events
        setattr(self, method, self._close)

    def _close(self) -> None:
        self.events.append(self.name)


class _DatabaseResource:
    def __init__(self, events: list[str]) -> None:
        self.engine = SimpleNamespace(dispose=lambda: events.append("database"))


class _GatewayResource(_ClosingResource):
    def __init__(
        self,
        events: list[str],
        *,
        fail_alias: bool = False,
    ) -> None:
        super().__init__("gateway", events)
        self.fail_alias = fail_alias

    def resolve_alias(self) -> str:
        if self.fail_alias:
            raise RuntimeError("alias failed")
        return "knowledge_chunks_qwen3_v1"


def _production_runtime_probe(results) -> None:
    from app.database import Database
    from app.llm import PhysocDeepSeekLLMProvider, TemplateLLMProvider
    from app.models import KnowledgeChunkModel
    from app.retrieval_models import RetrievalMode
    from app.retrieval_router import RetrievalRouter
    from app.sql_repository import SqlChatRepository

    from tools.hybrid_retrieval_benchmark import (
        BenchmarkQuestion,
        build_production_runtime,
        run_benchmark,
    )

    try:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        setup_repository = SqlChatRepository(database)
        setup_repository.add_uploaded_knowledge_source(
            source_id="benchmark-source",
            name="benchmark.txt",
            source_type="TXT",
            classification="internal",
            records=0,
            file_path="benchmark.txt",
            file_size=128,
            mime_type="text/plain",
        )
        setup_repository.complete_knowledge_source_indexing(
            "benchmark-source",
            [
                KnowledgeChunkModel(
                    id="benchmark-legacy-chunk",
                    source_id="benchmark-source",
                    chunk_index=0,
                    text="benchmark policy evidence",
                    token_count=3,
                )
            ],
        )
        setup_repository.add_uploaded_knowledge_source(
            source_id="benchmark-restricted-source",
            name="restricted-benchmark.txt",
            source_type="TXT",
            classification="restricted",
            records=0,
            file_path="restricted-benchmark.txt",
            file_size=128,
            mime_type="text/plain",
        )
        setup_repository.complete_knowledge_source_indexing(
            "benchmark-restricted-source",
            [
                KnowledgeChunkModel(
                    id="benchmark-restricted-chunk",
                    source_id="benchmark-restricted-source",
                    chunk_index=0,
                    text="benchmark policy evidence",
                    token_count=3,
                )
            ],
        )
        events: list[str] = []

        class FailingHybrid(_ClosingResource):
            def __init__(self) -> None:
                super().__init__("hybrid", events)

            def retrieve(self, _request):
                raise RuntimeError("hybrid failed")

        class Factory:
            def __init__(self) -> None:
                self.hybrid = FailingHybrid()
                self.router = None
                self.legacy_repository = None

            def create_qdrant_client(self, _settings):
                return _ClosingResource("qdrant", events)

            def create_gateway(self, _qdrant, _settings):
                return _GatewayResource(events)

            def create_embedding_client(self, _settings):
                return _ClosingResource("embedding", events)

            def create_reranker_client(self, _settings):
                return _ClosingResource("reranker", events)

            def create_sparse_encoder(self, _environment):
                return _ClosingResource("sparse", events)

            def create_hybrid_retriever(self, **_dependencies):
                return self.hybrid

            def create_audit(self, _database):
                return _ClosingResource("audit", events)

            def create_router(self, **dependencies):
                self.legacy_repository = dependencies["legacy_search"].__self__
                self.router = RetrievalRouter(
                    mode=RetrievalMode.QWEN3,
                    legacy_search=dependencies["legacy_search"],
                    hybrid=dependencies["hybrid"],
                    audit=dependencies["audit"],
                    canary_percent=100,
                    request_id_factory=lambda: "benchmark-request",
                )
                return self.router

        factory = Factory()
        settings = SimpleNamespace(
            mode=RetrievalMode.QWEN3,
            canary_percent=100.0,
            knowledge_base_id="default",
            permission_tags=("internal",),
        )
        with (
            patch("app.database.Database", return_value=database),
            patch(
                "app.retrieval_settings.RetrievalSettings.from_environ",
                return_value=settings,
            ),
            patch("app.main._DefaultRetrievalResourceFactory", return_value=factory),
            patch.object(
                PhysocDeepSeekLLMProvider,
                "generate_reply",
                side_effect=AssertionError("Physoc must not be called"),
            ) as physoc,
        ):
            runtime = build_production_runtime({"RETRIEVAL_MODE": "qwen3"})
            try:
                report = run_benchmark(
                    retriever=runtime.retriever,
                    scope=runtime.scope,
                    questions=[BenchmarkQuestion("case-router", "benchmark policy")],
                    concurrency=1,
                    requests=1,
                    p95_limit=1.0,
                    error_rate_limit=0.0,
                    fallback_rate_limit=1.0,
                )
            finally:
                runtime.close()
        results.put(
            {
                "adapter": runtime.retriever is not factory.hybrid,
                "router": isinstance(factory.router, RetrievalRouter),
                "template": isinstance(
                    factory.legacy_repository._llm_provider, TemplateLLMProvider
                ),
                "physoc_calls": physoc.call_count,
                "scope_permission_tags": runtime.scope.permission_tags,
                "report": report,
            }
        )
    except BaseException as error:
        results.put({"error": repr(error)})
        raise


def _fake_runtime_dependencies(
    *,
    settings: object,
    database_factory,
    repository_factory,
    resource_factory,
) -> SimpleNamespace:
    return SimpleNamespace(
        settings_from_environ=lambda _environment: settings,
        database_url_resolver=lambda _environment: "sqlite://",
        database_factory=database_factory,
        repository_factory=repository_factory,
        resource_factory=resource_factory,
        publication_version=lambda _collection: "v1",
        production_scope_factory=lambda knowledge_base_id, permission_tags, publication_version: (
            SimpleNamespace(
                knowledge_base_id=knowledge_base_id,
                permission_tags=permission_tags,
                publication_version=publication_version,
            )
        ),
        production_request_factory=lambda **values: SimpleNamespace(**values),
    )


class HybridRetrievalBenchmarkTest(unittest.TestCase):
    def test_report_fails_p95_error_and_fallback_thresholds(self) -> None:
        from tools.hybrid_retrieval_benchmark import summarize_results

        report = summarize_results(
            latencies=[1.0] * 14 + [5.2],
            errors=1,
            fallbacks=2,
            requests=100,
            p95_limit=5.0,
            error_rate_limit=0.01,
            fallback_rate_limit=0.01,
        )

        self.assertFalse(report.passed)
        self.assertIn("p95_seconds", report.failed_gates)
        self.assertIn("fallback_rate", report.failed_gates)
        self.assertNotIn("error_rate", report.failed_gates)

    def test_threshold_boundaries_pass(self) -> None:
        from tools.hybrid_retrieval_benchmark import summarize_results

        report = summarize_results(
            latencies=[5.0],
            errors=1,
            fallbacks=1,
            requests=100,
            p95_limit=5.0,
            error_rate_limit=0.01,
            fallback_rate_limit=0.01,
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.failed_gates, ())
        self.assertEqual(report.passed_gates, ("p95_seconds", "error_rate", "fallback_rate"))

    def test_zero_requests_and_non_finite_latencies_fail_closed(self) -> None:
        from tools.hybrid_retrieval_benchmark import summarize_results

        empty = summarize_results(
            latencies=[],
            errors=0,
            fallbacks=0,
            requests=0,
            p95_limit=5.0,
            error_rate_limit=0.01,
            fallback_rate_limit=0.01,
        )
        self.assertFalse(empty.passed)
        self.assertIn("requests", empty.failed_gates)
        for latency in (math.nan, math.inf, -math.inf):
            with self.subTest(latency=latency):
                report = summarize_results(
                    latencies=[latency],
                    errors=0,
                    fallbacks=0,
                    requests=1,
                    p95_limit=5.0,
                    error_rate_limit=0.01,
                    fallback_rate_limit=0.01,
                )
                self.assertFalse(report.passed)
                self.assertIn("p95_seconds", report.failed_gates)

    def test_fake_retriever_runs_closed_loop_without_live_services(self) -> None:
        from tools.hybrid_retrieval_benchmark import (
            BenchmarkQuestion,
            RetrievalScope,
            run_benchmark,
        )

        retriever = _FakeRetriever()
        report = run_benchmark(
            retriever=retriever,
            scope=RetrievalScope("kb", ("internal",), "v1"),
            questions=[
                BenchmarkQuestion("case-a", "sensitive question a"),
                BenchmarkQuestion("case-b", "sensitive question b"),
            ],
            concurrency=3,
            requests=7,
            p95_limit=0.1,
            error_rate_limit=0.0,
            fallback_rate_limit=0.0,
        )

        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["summary"]["requests"], 7)
        self.assertEqual(len(retriever.queries), 7)
        self.assertEqual({item["caseId"] for item in report["records"]}, {"case-a", "case-b"})
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("sensitive question", serialized)

    def test_production_runtime_uses_router_sql_fallback_and_routed_hit_ids(
        self,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        process = context.Process(target=_production_runtime_probe, args=(results,))
        process.start()
        process.join(30.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            self.fail("production runtime probe timed out")
        self.assertEqual(process.exitcode, 0)
        result = results.get(timeout=5.0)
        self.assertNotIn("error", result)
        self.assertTrue(result["adapter"])
        self.assertTrue(result["router"])
        self.assertTrue(result["template"])
        self.assertEqual(result["physoc_calls"], 0)
        self.assertEqual(result["scope_permission_tags"], ("internal",))
        report = result["report"]
        self.assertEqual(report["summary"]["errorRate"], 0.0)
        self.assertEqual(report["summary"]["fallbackRate"], 1.0)
        self.assertEqual(
            report["records"][0]["chunkIds"],
            ["benchmark-legacy-chunk"],
        )

    def test_production_runtime_requires_exact_qwen3_mode_before_allocating(
        self,
    ) -> None:
        from tools.hybrid_retrieval_benchmark import build_production_runtime

        database_calls: list[str] = []
        dependencies = _fake_runtime_dependencies(
            settings=SimpleNamespace(mode="shadow"),
            database_factory=lambda url: database_calls.append(url),
            repository_factory=lambda _database, _permission_tags: object(),
            resource_factory=lambda: object(),
        )
        with self.assertRaisesRegex(ValueError, "RETRIEVAL_MODE=qwen3"):
            build_production_runtime(
                {"RETRIEVAL_MODE": "shadow"},
                _dependencies=dependencies,
            )

        self.assertEqual(database_calls, [])

    def test_production_runtime_requires_100_percent_qwen3_before_allocating(
        self,
    ) -> None:
        from tools.hybrid_retrieval_benchmark import build_production_runtime

        for canary_percent in (0.0, 99.99):
            with self.subTest(canary_percent=canary_percent):
                database_calls: list[str] = []

                def allocate_database(url: str):
                    database_calls.append(url)
                    raise RuntimeError("database allocated")

                dependencies = _fake_runtime_dependencies(
                    settings=SimpleNamespace(
                        mode="qwen3",
                        canary_percent=canary_percent,
                    ),
                    database_factory=allocate_database,
                    repository_factory=lambda _database, _permission_tags: object(),
                    resource_factory=lambda: object(),
                )
                with self.assertRaisesRegex(ValueError, "100% Qwen3 routing"):
                    build_production_runtime(
                        {
                            "RETRIEVAL_MODE": "qwen3",
                            "RETRIEVAL_CANARY_PERCENT": str(canary_percent),
                        },
                        _dependencies=dependencies,
                    )
                self.assertEqual(database_calls, [])

        database_calls = []

        def allocate_database(url: str):
            database_calls.append(url)
            raise RuntimeError("database allocated")

        dependencies = _fake_runtime_dependencies(
            settings=SimpleNamespace(mode="qwen3", canary_percent=100.0),
            database_factory=allocate_database,
            repository_factory=lambda _database, _permission_tags: object(),
            resource_factory=lambda: object(),
        )
        with self.assertRaisesRegex(RuntimeError, "database allocated"):
            build_production_runtime(
                {
                    "RETRIEVAL_MODE": "qwen3",
                    "RETRIEVAL_CANARY_PERCENT": "100",
                },
                _dependencies=dependencies,
            )
        self.assertEqual(database_calls, ["sqlite://"])

    def test_construction_failures_close_prior_resources_once_in_reverse_order(
        self,
    ) -> None:
        from tools.hybrid_retrieval_benchmark import build_production_runtime

        stages = (
            "repository",
            "qdrant",
            "gateway",
            "embedding",
            "reranker",
            "sparse",
            "hybrid",
            "audit",
            "router",
            "alias",
        )
        resource_order = [
            "database",
            "repository",
            "qdrant",
            "gateway",
            "embedding",
            "reranker",
            "sparse",
            "hybrid",
            "audit",
            "router",
        ]
        settings = SimpleNamespace(
            mode="qwen3",
            canary_percent=100.0,
            knowledge_base_id="default",
            permission_tags=("internal",),
        )

        for stage in stages:
            with self.subTest(stage=stage):
                events: list[str] = []
                database = _DatabaseResource(events)
                repository = _ClosingResource("repository", events)
                repository.search_knowledge_chunks = lambda _query, _limit: []

                class Factory:
                    def create(self, name: str):
                        if stage == name:
                            raise RuntimeError(f"{name} failed")
                        if name == "gateway":
                            return _GatewayResource(events, fail_alias=stage == "alias")
                        return _ClosingResource(name, events)

                    def create_qdrant_client(self, _settings):
                        return self.create("qdrant")

                    def create_gateway(self, _qdrant, _settings):
                        return self.create("gateway")

                    def create_embedding_client(self, _settings):
                        return self.create("embedding")

                    def create_reranker_client(self, _settings):
                        return self.create("reranker")

                    def create_sparse_encoder(self, _environment):
                        return self.create("sparse")

                    def create_hybrid_retriever(self, **_dependencies):
                        return self.create("hybrid")

                    def create_audit(self, _database):
                        return self.create("audit")

                    def create_router(self, **_dependencies):
                        return self.create("router")

                def repository_factory(_database, _permission_tags):
                    if stage == "repository":
                        raise RuntimeError("repository failed")
                    return repository

                dependencies = _fake_runtime_dependencies(
                    settings=settings,
                    database_factory=lambda _url: database,
                    repository_factory=repository_factory,
                    resource_factory=Factory,
                )
                with (
                    self.assertRaisesRegex(RuntimeError, f"{stage} failed"),
                ):
                    build_production_runtime(
                        {"RETRIEVAL_MODE": "qwen3"},
                        _dependencies=dependencies,
                    )

                failure_index = (
                    resource_order.index(stage) if stage != "alias" else len(resource_order)
                )
                expected = list(reversed(resource_order[:failure_index]))
                self.assertEqual(events, expected)

    def test_runtime_close_supports_close_shutdown_dispose_and_engine_once(
        self,
    ) -> None:
        from tools.hybrid_retrieval_benchmark import ProductionRuntime, RetrievalScope

        events: list[str] = []
        close_resource = _ClosingResource("close", events)
        runtime = ProductionRuntime(
            retriever=_FakeRetriever(),
            scope=RetrievalScope("kb", ("internal",), "v1"),
            resources=(
                _DatabaseResource(events),
                _ClosingResource("shutdown", events, "shutdown"),
                _ClosingResource("dispose", events, "dispose"),
                close_resource,
                close_resource,
            ),
        )

        runtime.close()

        self.assertEqual(events, ["close", "dispose", "shutdown", "database"])

    def test_written_json_excludes_question_credentials_and_upstream_exception(
        self,
    ) -> None:
        from tools.hybrid_retrieval_benchmark import write_report

        report = {
            "summary": {
                "passed": False,
                "requests": 1,
                "p95Seconds": 0.1,
                "errorRate": 1.0,
                "fallbackRate": 0.0,
                "passedGates": [],
                "failedGates": ["error_rate"],
            },
            "records": [
                {
                    "caseId": "question text must not escape",
                    "chunkIds": ["http://internal.example/raw"],
                    "mode": "Authorization: Bearer credential",
                    "latencySeconds": 0.1,
                    "fallbackReason": "upstream stack trace",
                    "error": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, report)
            serialized = path.read_text(encoding="utf-8")

        for secret in (
            "question",
            "Authorization",
            "Bearer",
            "http://internal.example",
            "upstream stack trace",
        ):
            self.assertNotIn(secret, serialized)

    def test_write_failure_removes_only_current_process_temporary_file(self) -> None:
        from tools.hybrid_retrieval_benchmark import write_report

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            unrelated = destination.with_name(f".{destination.name}.999999.tmp")
            temporary.touch()
            unrelated.touch()

            with (
                patch.object(Path, "write_text", side_effect=OSError("write failed")),
                self.assertRaisesRegex(OSError, "write failed"),
            ):
                write_report(destination, {"summary": {}, "records": []})

            self.assertFalse(temporary.exists())
            self.assertTrue(unrelated.exists())

    def test_replace_failure_removes_only_current_process_temporary_file(self) -> None:
        from tools.hybrid_retrieval_benchmark import write_report

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            unrelated = destination.with_name(f".{destination.name}.999999.tmp")
            unrelated.touch()

            with (
                patch(
                    "tools.hybrid_retrieval_benchmark.os.replace",
                    side_effect=OSError("replace failed"),
                ),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                write_report(destination, {"summary": {}, "records": []})

            self.assertFalse(temporary.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
