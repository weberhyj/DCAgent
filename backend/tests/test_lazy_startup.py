from __future__ import annotations

import importlib
import tempfile
import time
import unittest
from collections.abc import Mapping
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from app.database import Database
from app.infra.health import (
    DependencyCheck,
    DependencyHealthRegistry,
    build_dependency_checks,
    create_postgres_health_engine,
    create_redis_health_client,
    postgres_schema_revision_check,
)
from app.main import _database_url_with_connect_timeout, create_app
from app.offline_settings import OfflineSettings
from app.qdrant_retrieval import QdrantRetrievalGateway
from app.retrieval_settings import RetrievalSettings


class ClosableFake:
    def __init__(self, name: str) -> None:
        self.name = name
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class RecordingClosable:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self.closed = closed
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed.append(self.name)


class ConfigurableRepository(ClosableFake):
    def __init__(self) -> None:
        super().__init__("repository")
        self.retrieval_router = None
        self.retrieval_scope_provider = None

    def configure_retrieval(self, router: object, scope_provider: object) -> None:
        self.retrieval_router = router
        self.retrieval_scope_provider = scope_provider

    @property
    def retrieval_scope(self) -> object | None:
        if self.retrieval_scope_provider is None:
            return None
        return self.retrieval_scope_provider.resolve().scope

    def search_knowledge_chunks(self, query: str, limit: int = 5) -> list[object]:
        del query, limit
        return []


class RecordingRetrievalResourceFactory:
    def __init__(self, *, fail_at: str | None = None, shared_clients: bool = False) -> None:
        self.fail_at = fail_at
        self.calls: list[str] = []
        self.closed: list[str] = []
        self.qdrant = RecordingClosable("qdrant", self.closed)
        self.embedding = (
            self.qdrant if shared_clients else RecordingClosable("embedding", self.closed)
        )
        self.reranker = RecordingClosable("reranker", self.closed)
        self.hybrid = RecordingClosable("hybrid", self.closed)
        self.router = RecordingClosable("shadow_queue", self.closed)
        self.alias_collection: str | None = "knowledge_chunks_qwen3_v17"
        self.gateway = SimpleNamespace(
            name="gateway",
            resolve_alias=lambda: self.alias_collection,
        )
        self.sparse = SimpleNamespace(name="sparse")
        self.active_publication: object | None = SimpleNamespace(
            id="publication-uuid-must-not-be-scope",
            collection_name="knowledge_chunks_qwen3_v17",
            embedding_fingerprint=RetrievalSettings.from_environ(
                private_qwen_environment()
            ).embedding_fingerprint,
        )
        self.audit = SimpleNamespace(active_publication=lambda alias: self.active_publication)
        self.index_lifecycle = SimpleNamespace(name="index-lifecycle")

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"failed at {name}")

    def create_qdrant_client(self, settings: object) -> object:
        del settings
        self._record("qdrant")
        return self.qdrant

    def create_gateway(self, client: object, settings: object) -> object:
        del client, settings
        self._record("gateway")
        return self.gateway

    def create_embedding_client(self, settings: object) -> object:
        del settings
        self._record("embedding")
        return self.embedding

    def create_reranker_client(self, settings: object) -> object:
        del settings
        self._record("reranker")
        return self.reranker

    def create_sparse_encoder(self, environ: Mapping[str, str]) -> object:
        del environ
        self._record("sparse")
        return self.sparse

    def create_hybrid_retriever(self, **dependencies: object) -> object:
        self._record("hybrid")
        self.hybrid.dependencies = dependencies  # type: ignore[attr-defined]
        return self.hybrid

    def create_audit(self, database: object) -> object:
        del database
        self._record("audit")
        return self.audit

    def create_router(self, **dependencies: object) -> object:
        self._record("router")
        self.router.dependencies = dependencies  # type: ignore[attr-defined]
        return self.router

    def create_index_lifecycle(self, **dependencies: object) -> object:
        self._record("index_lifecycle")
        self.index_lifecycle.dependencies = dependencies
        return self.index_lifecycle


class ScopeRecordingQdrant(RecordingClosable):
    def __init__(self, closed: list[str]) -> None:
        super().__init__("qdrant", closed)
        self.query_calls: list[SimpleNamespace] = []
        self.indexed_payload = {
            "knowledge_base_id": "default",
            "publication_version": "v17",
            "permission_tags": ["internal"],
            "source_id": "source-1",
            "source_name": "Policy.txt",
            "source_type": "TXT",
            "classification": "internal",
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "text": "Safe evidence",
            "parent_chunk_id": None,
            "previous_chunk_id": None,
            "next_chunk_id": None,
        }

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(SimpleNamespace(**kwargs))
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.9,
                    payload=dict(self.indexed_payload),
                )
            ]
        )

    def get_aliases(self) -> object:
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(
                    alias_name="knowledge_chunks_current",
                    collection_name="knowledge_chunks_qwen3_v17",
                )
            ]
        )


class ScopeQueryingRetrievalResourceFactory(RecordingRetrievalResourceFactory):
    def __init__(self) -> None:
        super().__init__()
        self.qdrant = ScopeRecordingQdrant(self.closed)

    def create_gateway(self, client: object, settings: object) -> object:
        self._record("gateway")
        self.gateway = QdrantRetrievalGateway(
            client,  # type: ignore[arg-type]
            alias_name=settings.qdrant_collection_alias,
        )
        return self.gateway


def private_environment(**changes: str) -> dict[str, str]:
    values = {
        "OFFLINE_MODE": "true",
        "DATABASE_URL": "postgresql+psycopg://dc_agent@127.0.0.1/dc_agent",
        "CLICKHOUSE_URL": "http://127.0.0.1:8123",
        "QDRANT_URL": "http://127.0.0.1:6333",
        "REDIS_URL": "redis://127.0.0.1:6379/0",
        "CLAMAV_HOST": "127.0.0.1",
        "EMBEDDING_SERVICE_URL": "http://127.0.0.1:8081",
        "LLAMA_SERVER_URL": "http://127.0.0.1:8080",
        "LLM_PROVIDER": "physoc_deepseek",
        "LLM_API_BASE": "http://127.0.0.1:8090",
        "LLM_STREAM_PATH": "/api/physoc/deepseeks/stream",
        "LLM_MODEL": "my_deepseek_r1_7b",
    }
    values.update(changes)
    return values


def private_qwen_environment(mode: str = "qwen3", **changes: str) -> dict[str, str]:
    checksum = "a" * 64
    values = private_environment(
        RETRIEVAL_MODE=mode,
        RETRIEVAL_PERMISSION_TAGS="internal,restricted",
        RERANKER_SERVICE_URL="http://127.0.0.1:8082",
        EMBEDDING_MODEL_NAME="Qwen/Qwen3-Embedding-0.6B",
        EMBEDDING_MODEL_VERSION="1.0.0",
        EMBEDDING_MODEL_SHA256=checksum,
        EMBEDDING_MODEL_DIMENSIONS="1024",
        EMBEDDING_MODEL_NORMALIZED="true",
        EMBEDDING_ENCODING_PROFILE_SHA256=checksum,
        EMBEDDING_PROTOCOL_VERSION="v1",
        RERANKER_MODEL_NAME="Qwen/Qwen3-Reranker-0.6B",
        RERANKER_MODEL_VERSION="1.0.0",
        RERANKER_MODEL_SHA256=checksum,
        RERANKER_PROMPT_PROFILE_SHA256=checksum,
        RERANKER_PROTOCOL_VERSION="v1",
        SPARSE_MODEL_ROOT="C:/models/bm25",
        SPARSE_PROFILE_SHA256=checksum,
    )
    values.update(changes)
    return values


def build_settings(environ: Mapping[str, str] | None = None) -> OfflineSettings:
    return OfflineSettings.from_environ(private_environment(**dict(environ or {})))


class LazyStartupTest(unittest.TestCase):
    def test_importing_main_does_not_connect_or_create_schema(self) -> None:
        database_module = importlib.import_module("app.database")
        with (
            patch.object(
                database_module.Database,
                "__init__",
                side_effect=AssertionError("connected"),
            ) as database_init,
            patch.object(
                database_module.Database,
                "create_schema",
                side_effect=AssertionError("schema mutated"),
            ) as create_schema,
        ):
            module = importlib.import_module("app.main")
            module = importlib.reload(module)

        self.assertTrue(callable(module.create_production_app))
        self.assertIsNotNone(module.app)
        database_init.assert_not_called()
        create_schema.assert_not_called()

    def test_production_factory_construction_is_lazy(self) -> None:
        module = importlib.import_module("app.main")
        with patch.object(
            module.Database,
            "__init__",
            side_effect=AssertionError("database constructed before startup"),
        ) as database_init:
            app = module.create_production_app(environ=private_environment())

        self.assertIsNotNone(app)
        database_init.assert_not_called()
        client = TestClient(app)
        self.assertEqual(client.get("/api/healthz").status_code, 200)
        self.assertEqual(client.get("/api/readyz").status_code, 503)

    def test_production_startup_rejects_template_before_resource_factories(
        self,
    ) -> None:
        module = importlib.import_module("app.main")
        calls: list[str] = []
        app = module.create_production_app(
            environ=private_environment(LLM_PROVIDER="template"),
            database_factory=lambda _url: calls.append("database"),
            llm_provider_factory=lambda _environment: calls.append("llm"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Production runtime requires a real LLM provider",
        ):
            with TestClient(app):
                pass

        self.assertEqual(calls, [])

    def test_development_app_still_allows_injected_template_repository(self) -> None:
        repository = ClosableFake("development-repository")

        with TestClient(create_app(repository=repository)) as client:
            response = client.get("/api/healthz")

        self.assertEqual(response.status_code, 200)

    def test_production_lifespan_does_not_precreate_migration_owned_schema(self) -> None:
        module = importlib.import_module("app.main")
        resources = {
            "repository": ClosableFake("repository"),
            "queue": ClosableFake("queue"),
            "storage": ClosableFake("storage"),
            "evaluation": ClosableFake("evaluation"),
        }
        factory_calls = {name: 0 for name in resources}
        ready = False
        health_calls = 0

        def build_repository() -> ClosableFake:
            factory_calls["repository"] += 1
            return resources["repository"]

        def build_queue(repository: object) -> ClosableFake:
            self.assertIs(repository, resources["repository"])
            factory_calls["queue"] += 1
            return resources["queue"]

        def build_storage(root: Path) -> ClosableFake:
            self.assertIsInstance(root, Path)
            factory_calls["storage"] += 1
            return resources["storage"]

        def build_evaluation_service() -> ClosableFake:
            factory_calls["evaluation"] += 1
            return resources["evaluation"]

        def dependency_check() -> tuple[bool, str]:
            nonlocal health_calls
            health_calls += 1
            return ready, "ready" if ready else "starting"

        class TrackingHealthRegistry(DependencyHealthRegistry):
            def __init__(self) -> None:
                super().__init__([DependencyCheck("fake", dependency_check)])
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1
                super().close()

        registry = TrackingHealthRegistry()
        app = module.create_production_app(
            environ=private_environment(),
            repository_factory=build_repository,
            health_registry_factory=lambda: registry,
            ingestion_queue_factory=build_queue,
            storage_factory=build_storage,
            evaluation_import_service_factory=build_evaluation_service,
        )

        self.assertEqual(factory_calls, {name: 0 for name in resources})
        with patch.object(
            module.Database,
            "create_schema",
            side_effect=AssertionError("schema mutation is migration-owned"),
        ) as create_schema:
            client = TestClient(app)
            with client:
                self.assertEqual(factory_calls, {name: 1 for name in resources})
                self.assertIs(
                    app.state.repository,
                    resources["repository"],
                )
                self.assertIs(
                    app.state.knowledge_ingestion_queue,
                    resources["queue"],
                )
                self.assertIs(
                    app.state.knowledge_file_storage,
                    resources["storage"],
                )
                self.assertIs(
                    app.state.evaluation_import_service,
                    resources["evaluation"],
                )
                self.assertIs(app.state.health_registry, registry)

                self.assertEqual(client.get("/api/healthz").status_code, 200)
                self.assertEqual(health_calls, 0)
                self.assertEqual(client.get("/api/readyz").status_code, 503)
                self.assertEqual(health_calls, 1)
                ready = True
                self.assertEqual(client.get("/api/readyz").status_code, 200)
                self.assertEqual(health_calls, 2)

            self.assertEqual(client.get("/api/healthz").status_code, 200)
            self.assertEqual(client.get("/api/readyz").status_code, 503)
            self.assertEqual(health_calls, 2)

        create_schema.assert_not_called()
        self.assertEqual(factory_calls, {name: 1 for name in resources})
        self.assertEqual(
            {name: resource.close_calls for name, resource in resources.items()},
            {name: 1 for name in resources},
        )
        self.assertEqual(registry.close_calls, 1)

    def test_qwen3_startup_injects_and_closes_owned_retrieval_resources(self) -> None:
        module = importlib.import_module("app.main")
        resources = RecordingRetrievalResourceFactory()
        repository = ConfigurableRepository()
        observed_queue: dict[str, object] = {}

        def build_queue(*, repository: object, index_lifecycle: object) -> ClosableFake:
            observed_queue["repository"] = repository
            observed_queue["index_lifecycle"] = index_lifecycle
            return ClosableFake("queue")

        app = module.create_production_app(
            environ=private_qwen_environment(),
            repository_factory=lambda: repository,
            retrieval_resource_factory=resources,
            health_registry_factory=DependencyHealthRegistry,
            ingestion_queue_factory=build_queue,
            database_factory=lambda _url: ClosableFake("database"),
            llm_provider_factory=lambda _environment: ClosableFake("llm"),
            storage_factory=lambda _root: ClosableFake("storage"),
            evaluation_import_service_factory=lambda: ClosableFake("evaluation"),
        )

        with TestClient(app):
            self.assertIs(repository.retrieval_router, resources.router)
            self.assertEqual(
                repository.retrieval_scope.publication_version,
                "v17",
            )
            self.assertIs(observed_queue["repository"], repository)
            self.assertIs(observed_queue["index_lifecycle"], resources.index_lifecycle)
            self.assertIs(app.state.retrieval_gateway, resources.gateway)
            self.assertIs(
                repository.retrieval_scope_provider,
                app.state.retrieval_scope_provider,
            )

        self.assertEqual(
            resources.calls,
            [
                "qdrant",
                "gateway",
                "embedding",
                "reranker",
                "sparse",
                "hybrid",
                "audit",
                "router",
                "index_lifecycle",
            ],
        )
        self.assertEqual(
            resources.closed,
            ["shadow_queue", "hybrid", "reranker", "embedding", "qdrant"],
        )

    def test_startup_scope_matches_the_version_in_qdrant_payload_filters(self) -> None:
        module = importlib.import_module("app.main")
        resources = ScopeQueryingRetrievalResourceFactory()
        repository = ConfigurableRepository()
        app = module.create_production_app(
            environ=private_qwen_environment(),
            repository_factory=lambda: repository,
            retrieval_resource_factory=resources,
            health_registry_factory=DependencyHealthRegistry,
            ingestion_queue_factory=lambda **_kwargs: ClosableFake("queue"),
            database_factory=lambda _url: ClosableFake("database"),
            llm_provider_factory=lambda _environment: ClosableFake("llm"),
            storage_factory=lambda _root: ClosableFake("storage"),
            evaluation_import_service_factory=lambda: ClosableFake("evaluation"),
        )

        with TestClient(app):
            scope = repository.retrieval_scope_provider.resolve().scope
            hits = resources.gateway.search_dense(
                [1.0],
                scope=scope,
                limit=1,
            )

        self.assertEqual([item.chunk_id for item in hits], ["chunk-1"])
        query_filter = resources.qdrant.query_calls[0].query_filter
        version_condition = next(
            condition for condition in query_filter.must if condition.key == "publication_version"
        )
        self.assertEqual(version_condition.match.value, "v17")
        self.assertEqual(
            version_condition.match.value,
            resources.qdrant.indexed_payload["publication_version"],
        )

    def test_shadow_and_qwen3_start_without_active_publication_and_transition_live(
        self,
    ) -> None:
        module = importlib.import_module("app.main")

        for mode in ("shadow", "qwen3"):
            with self.subTest(mode=mode):
                resources = RecordingRetrievalResourceFactory()
                resources.active_publication = None
                resources.alias_collection = None
                repository = ConfigurableRepository()
                app = module.create_production_app(
                    environ=private_qwen_environment(mode),
                    repository_factory=lambda: repository,
                    retrieval_resource_factory=resources,
                    health_registry_factory=DependencyHealthRegistry,
                    ingestion_queue_factory=lambda **_kwargs: ClosableFake("queue"),
                    database_factory=lambda _url: ClosableFake("database"),
                    llm_provider_factory=lambda _environment: ClosableFake("llm"),
                    storage_factory=lambda _root: ClosableFake("storage"),
                    evaluation_import_service_factory=lambda: ClosableFake("evaluation"),
                )

                with TestClient(app):
                    unavailable = repository.retrieval_scope_provider.resolve()
                    self.assertIsNone(unavailable.scope)
                    self.assertEqual(unavailable.detail, "unavailable")

                    resources.active_publication = SimpleNamespace(
                        id="later-publication-id",
                        collection_name="knowledge_chunks_qwen3_v18",
                        embedding_fingerprint=RetrievalSettings.from_environ(
                            private_qwen_environment(mode)
                        ).embedding_fingerprint,
                    )
                    resources.alias_collection = "knowledge_chunks_qwen3_v18"
                    available = repository.retrieval_scope_provider.resolve()
                    self.assertEqual(available.scope.publication_version, "v18")

    def test_legacy_startup_does_not_construct_retrieval_resources(self) -> None:
        module = importlib.import_module("app.main")
        resources = RecordingRetrievalResourceFactory(fail_at="qdrant")

        app = module.create_production_app(
            environ=private_environment(RETRIEVAL_MODE="legacy"),
            repository_factory=ConfigurableRepository,
            retrieval_resource_factory=resources,
            health_registry_factory=DependencyHealthRegistry,
            ingestion_queue_factory=lambda _repository: ClosableFake("queue"),
            database_factory=lambda _url: ClosableFake("database"),
            llm_provider_factory=lambda _environment: ClosableFake("llm"),
            storage_factory=lambda _root: ClosableFake("storage"),
            evaluation_import_service_factory=lambda: ClosableFake("evaluation"),
        )

        with TestClient(app):
            self.assertIsNone(app.state.repository.retrieval_router)

        self.assertEqual(resources.calls, [])
        self.assertEqual(resources.closed, [])

    def test_retrieval_partial_factory_failure_closes_constructed_resources(self) -> None:
        module = importlib.import_module("app.main")
        resources = RecordingRetrievalResourceFactory(fail_at="reranker")
        app = module.create_production_app(
            environ=private_qwen_environment(),
            repository_factory=ConfigurableRepository,
            retrieval_resource_factory=resources,
            health_registry_factory=DependencyHealthRegistry,
            database_factory=lambda _url: ClosableFake("database"),
            llm_provider_factory=lambda _environment: ClosableFake("llm"),
        )

        with self.assertRaisesRegex(RuntimeError, "failed at reranker"):
            with TestClient(app):
                pass

        self.assertEqual(resources.closed, ["embedding", "qdrant"])

    def test_retrieval_owned_resources_are_registered_only_once(self) -> None:
        module = importlib.import_module("app.main")
        resources = RecordingRetrievalResourceFactory(shared_clients=True)
        app = module.create_production_app(
            environ=private_qwen_environment(),
            repository_factory=ConfigurableRepository,
            retrieval_resource_factory=resources,
            health_registry_factory=DependencyHealthRegistry,
            ingestion_queue_factory=lambda **_kwargs: ClosableFake("queue"),
            database_factory=lambda _url: ClosableFake("database"),
            llm_provider_factory=lambda _environment: ClosableFake("llm"),
            storage_factory=lambda _root: ClosableFake("storage"),
            evaluation_import_service_factory=lambda: ClosableFake("evaluation"),
        )

        with TestClient(app):
            pass

        self.assertEqual(resources.qdrant.close_calls, 1)
        self.assertEqual(resources.closed.count("qdrant"), 1)

    def test_production_readiness_uses_mode_aware_retrieval_health(self) -> None:
        module = importlib.import_module("app.main")

        for mode, expected_status, expected_detail in (
            ("qwen3", 503, "unavailable"),
            ("shadow", 200, "degraded"),
        ):
            with self.subTest(mode=mode):
                resources = RecordingRetrievalResourceFactory()
                captured: dict[str, object] = {}

                def build_checks(
                    _settings: object,
                    **dependencies: object,
                ) -> list[DependencyCheck]:
                    captured.update(dependencies)
                    retrieval = dependencies["retrieval_settings"]
                    detail = (
                        "degraded"
                        if retrieval.mode.value == "shadow"
                        else "unavailable"
                    )
                    return [
                        DependencyCheck(
                            "reranker",
                            lambda: (mode == "shadow", detail),
                        )
                    ]

                app = module.create_production_app(
                    environ=private_qwen_environment(mode),
                    repository_factory=ConfigurableRepository,
                    retrieval_resource_factory=resources,
                    ingestion_queue_factory=lambda **_kwargs: ClosableFake("queue"),
                    database_factory=lambda _url: ClosableFake("database"),
                    llm_provider_factory=lambda _environment: ClosableFake("llm"),
                    storage_factory=lambda _root: ClosableFake("storage"),
                    evaluation_import_service_factory=lambda: ClosableFake("evaluation"),
                    health_http_client_factory=lambda **_kwargs: ClosableFake("http"),
                    health_redis_client_factory=lambda *_args, **_kwargs: ClosableFake("redis"),
                    postgres_health_engine_factory=lambda *_args, **_kwargs: ClosableFake(
                        "health-database"
                    ),
                )

                with patch.object(module, "build_dependency_checks", side_effect=build_checks):
                    with TestClient(app) as client:
                        response = client.get("/api/readyz")

                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(
                    response.json()["dependencies"]["reranker"]["detail"],
                    expected_detail,
                )
                self.assertIs(captured["retrieval_gateway"], resources.gateway)
                self.assertEqual(
                    captured["retrieval_settings"].mode.value,
                    mode,
                )

    def test_production_lifespan_reuses_and_closes_health_clients(self) -> None:
        module = importlib.import_module("app.main")
        database = Database("sqlite+pysqlite:///:memory:")
        with database.engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"
            )
            connection.exec_driver_sql(
                "INSERT INTO alembic_version (version_num) VALUES ('20260730_06')"
            )

        class FakeResponse:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self, url: str) -> None:
                self.url = url

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def iter_bytes(self, **_kwargs: object) -> object:
                if self.url.endswith("/ping"):
                    yield b"Ok.\n"
                else:
                    yield b'{"status":"ready"}'

        class FakeHttpClient:
            def __init__(self) -> None:
                self.get_calls: list[str] = []
                self.close_calls = 0

            def get(self, url: str) -> FakeResponse:
                self.get_calls.append(url)
                return FakeResponse(url)

            def stream(
                self,
                method: str,
                url: str,
                **kwargs: object,
            ) -> FakeResponse:
                self.get_calls.append(url)
                return FakeResponse(url)

            def close(self) -> None:
                self.close_calls += 1

        class FakeRedisClient:
            def __init__(self) -> None:
                self.ping_calls = 0
                self.close_calls = 0

            def ping(self) -> bool:
                self.ping_calls += 1
                return True

            def close(self) -> None:
                self.close_calls += 1

        class TrackingHealthEngine:
            def __init__(self) -> None:
                self.dispose_calls = 0

            def connect(self) -> object:
                return database.engine.connect()

            def dispose(self) -> None:
                self.dispose_calls += 1

        class FakeClamAVSocket:
            def __enter__(self) -> FakeClamAVSocket:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def settimeout(self, timeout: float) -> None:
                return None

            def sendall(self, payload: bytes) -> None:
                return None

            def recv(self, size: int) -> bytes:
                return b"PONG\0"

        http_client = FakeHttpClient()
        redis_client = FakeRedisClient()
        health_engine = TrackingHealthEngine()
        http_factory_calls = 0
        redis_factory_calls = 0
        health_engine_factory_calls = 0

        def build_http_client(**_kwargs: object) -> FakeHttpClient:
            nonlocal http_factory_calls
            http_factory_calls += 1
            return http_client

        def build_redis_client(
            _url: str,
            **_kwargs: object,
        ) -> FakeRedisClient:
            nonlocal redis_factory_calls
            redis_factory_calls += 1
            return redis_client

        def build_health_engine(
            _url: str,
            **_kwargs: object,
        ) -> TrackingHealthEngine:
            nonlocal health_engine_factory_calls
            health_engine_factory_calls += 1
            return health_engine

        try:
            app = module.create_production_app(
                environ=private_environment(),
                database_factory=lambda _url: database,
                repository_factory=lambda: ClosableFake("repository"),
                ingestion_queue_factory=lambda _repository: ClosableFake("queue"),
                storage_factory=lambda _root: ClosableFake("storage"),
                evaluation_import_service_factory=lambda: ClosableFake("evaluation"),
                health_http_client_factory=build_http_client,
                health_redis_client_factory=build_redis_client,
                postgres_health_engine_factory=build_health_engine,
            )

            with patch(
                "app.infra.health.socket.create_connection",
                return_value=FakeClamAVSocket(),
            ):
                with TestClient(app) as client:
                    self.assertEqual(http_factory_calls, 1)
                    self.assertEqual(redis_factory_calls, 1)
                    self.assertEqual(health_engine_factory_calls, 1)
                    self.assertEqual(client.get("/api/readyz").status_code, 200)
                    self.assertEqual(client.get("/api/readyz").status_code, 200)
                    self.assertEqual(len(http_client.get_calls), 3)
                    self.assertEqual(redis_client.ping_calls, 1)

            self.assertEqual(http_client.close_calls, 1)
            self.assertEqual(redis_client.close_calls, 1)
            self.assertEqual(health_engine.dispose_calls, 1)
        finally:
            database.engine.dispose()

    def test_template_provider_skips_llama_dependency(self) -> None:
        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(LLM_PROVIDER="template"),
        )

        self.assertNotIn("llama", {check.name for check in checks})

    def test_generation_enabled_includes_llama_dependency(self) -> None:
        cases = (
            private_environment(
                LLM_PROVIDER="openai_compatible",
                LLM_API_KEY="local-test",
                LLM_MODEL="local-model",
                LLM_API_BASE="http://127.0.0.1:8080/v1",
            ),
            private_environment(GENERATION_ENABLED="true"),
        )

        for environ in cases:
            with self.subTest(environ=environ):
                checks = build_dependency_checks(
                    build_settings(environ),
                    database=object(),
                    environ=environ,
                )
                self.assertIn("llama", {check.name for check in checks})

    def test_postgres_revision_check_uses_current_alembic_heads(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        try:
            with database.engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"
                )
                connection.exec_driver_sql(
                    "INSERT INTO alembic_version (version_num) VALUES ('future_head')"
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                versions = root / "alembic" / "versions"
                versions.mkdir(parents=True)
                config_path = root / "alembic.ini"
                config_path.write_text(
                    "[alembic]\nscript_location = %(here)s/alembic\n",
                    encoding="utf-8",
                )
                (versions / "future_head.py").write_text(
                    "revision = 'future_head'\n"
                    "down_revision = None\n"
                    "branch_labels = None\n"
                    "depends_on = None\n",
                    encoding="utf-8",
                )

                check = postgres_schema_revision_check(
                    database,
                    config_path=config_path,
                )

                self.assertEqual(check(), (True, "schema current"))
                with database.engine.begin() as connection:
                    connection.exec_driver_sql(
                        "UPDATE alembic_version SET version_num = 'stale_head'"
                    )
                self.assertEqual(
                    check(),
                    (False, "schema revision mismatch"),
                )

                (versions / "second_head.py").write_text(
                    "revision = 'second_head'\n"
                    "down_revision = None\n"
                    "branch_labels = None\n"
                    "depends_on = None\n",
                    encoding="utf-8",
                )
                with database.engine.begin() as connection:
                    connection.exec_driver_sql(
                        "UPDATE alembic_version SET version_num = 'future_head'"
                    )
                    connection.exec_driver_sql(
                        "INSERT INTO alembic_version (version_num) VALUES ('second_head')"
                    )
                self.assertEqual(
                    check(),
                    (False, "schema revision mismatch"),
                )
        finally:
            database.engine.dispose()

    def test_postgres_revision_check_sets_scoped_statement_timeout(self) -> None:
        events: list[str] = []

        class FakeConnection:
            class Dialect:
                name = "postgresql"

            dialect = Dialect()

            def __enter__(self) -> FakeConnection:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def exec_driver_sql(self, statement: str) -> None:
                events.append(statement)

        class FakeEngine:
            def connect(self) -> FakeConnection:
                return FakeConnection()

        class FakeDatabase:
            engine = FakeEngine()

        class FakeScriptDirectory:
            @staticmethod
            def get_heads() -> tuple[str]:
                return ("future_head",)

        class FakeMigrationContext:
            @staticmethod
            def get_current_heads() -> tuple[str]:
                return ("future_head",)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "alembic.ini"
            config_path.write_text("[alembic]\n", encoding="utf-8")
            with (
                patch(
                    "alembic.script.ScriptDirectory.from_config",
                    return_value=FakeScriptDirectory(),
                ),
                patch(
                    "alembic.runtime.migration.MigrationContext.configure",
                    side_effect=lambda _connection: (
                        events.append("migration-context") or FakeMigrationContext()
                    ),
                ),
            ):
                check = postgres_schema_revision_check(
                    FakeDatabase(),
                    config_path=config_path,
                    timeout_seconds=1.25,
                )

                self.assertEqual(check(), (True, "schema current"))

        self.assertEqual(events[0], "SET LOCAL statement_timeout = 1250")
        self.assertEqual(events[1], "migration-context")

    def test_database_connect_timeout_is_forcibly_bounded(self) -> None:
        for configured_timeout in ("999999", "nan", "inf", "0"):
            with self.subTest(configured_timeout=configured_timeout):
                url = _database_url_with_connect_timeout(
                    "postgresql+psycopg://dc_agent:secret@127.0.0.1/dc_agent"
                    f"?connect_timeout={configured_timeout}",
                    2.0,
                )
                parsed = make_url(url)
                self.assertEqual(parsed.query["connect_timeout"], "2")

    def test_database_connect_timeout_handles_invalid_runtime_timeout(self) -> None:
        for runtime_timeout in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(runtime_timeout=runtime_timeout):
                url = _database_url_with_connect_timeout(
                    "postgresql+psycopg://dc_agent:secret@127.0.0.1/dc_agent"
                    "?connect_timeout=999999",
                    runtime_timeout,
                )
                parsed = make_url(url)
                self.assertEqual(parsed.query["connect_timeout"], "2")

    def test_redis_health_client_strips_timeout_overrides_from_url(self) -> None:
        captured: dict[str, object] = {}
        sentinel = object()

        def factory(url: str, **kwargs: object) -> object:
            captured["url"] = url
            captured.update(kwargs)
            return sentinel

        client = create_redis_health_client(
            "rediss://health:secret@127.0.0.1/4"
            "?ssl_cert_reqs=required"
            "&client_name=readiness"
            "&db=4"
            "&socket_connect_timeout=999999"
            "&Socket_Timeout=999999"
            "&health_check_interval=999999"
            "&Retry_On_Timeout=true"
            "&retry_on_error=TimeoutError"
            "&TIMEOUT=999999"
            "&max_connections=999999"
            "&PoRt=80"
            "#fragment",
            2.0,
            client_factory=factory,
        )

        self.assertIs(client, sentinel)
        self.assertEqual(
            captured["url"],
            "rediss://health:secret@127.0.0.1/4?ssl_cert_reqs=required&client_name=readiness&db=4",
        )
        self.assertEqual(captured["socket_connect_timeout"], 2.0)
        self.assertEqual(captured["socket_timeout"], 2.0)
        self.assertEqual(captured["health_check_interval"], 0)
        self.assertEqual(captured["max_connections"], 8)

    def test_redis_ping_hard_timeout_single_flights_stuck_client(self) -> None:
        started = Event()
        release = Event()
        finished = Event()

        class FakeRedisClient:
            def __init__(self) -> None:
                self.ping_calls = 0

            def ping(self) -> bool:
                self.ping_calls += 1
                started.set()
                if self.ping_calls == 1:
                    release.wait()
                    finished.set()
                return True

        client = FakeRedisClient()
        environ = private_environment(DEPENDENCY_TIMEOUT_SECONDS="0.05")
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
            redis_client=client,
        )
        redis_check = next(check for check in checks if check.name == "redis")

        started_at = time.monotonic()
        try:
            self.assertEqual(redis_check.check(), (False, "unavailable"))
            self.assertTrue(started.wait(timeout=0.2))
            self.assertLess(time.monotonic() - started_at, 0.2)

            follower_results: list[tuple[bool, str]] = []
            follower_done = Event()

            def follow_stuck_ping() -> None:
                follower_results.append(redis_check.check())
                follower_done.set()

            follower = Thread(target=follow_stuck_ping)
            follower.start()
            self.assertTrue(follower_done.wait(timeout=0.2))
            follower.join(timeout=0.2)
            self.assertEqual(follower_results, [(False, "unavailable")])
            self.assertEqual(client.ping_calls, 1)
        finally:
            release.set()

        self.assertTrue(finished.wait(timeout=0.5))
        retry_deadline = time.monotonic() + 0.5
        retry_result = (False, "unavailable")
        while time.monotonic() < retry_deadline:
            retry_result = redis_check.check()
            if retry_result == (True, "ready"):
                break
        self.assertEqual(retry_result, (True, "ready"))
        self.assertEqual(client.ping_calls, 2)

    def test_health_urls_allow_omitted_ports_but_reject_zero(self) -> None:
        environ = private_environment(
            DATABASE_URL="postgresql+psycopg://dc_agent:secret@postgres/dc_agent",
            CLICKHOUSE_URL="https://localhost",
            QDRANT_URL="http://qdrant",
            REDIS_URL="redis://redis/0",
            EMBEDDING_SERVICE_URL="http://embedding-service",
        )

        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
        )
        self.assertEqual(
            {check.name for check in checks},
            {"postgresql", "clickhouse", "qdrant", "redis", "clamav", "embedding"},
        )

        invalid = private_environment(QDRANT_URL="http://127.0.0.1:0")
        with self.assertRaisesRegex(ValueError, "private or loopback"):
            build_dependency_checks(
                build_settings(invalid),
                database=object(),
                environ=invalid,
            )

    def test_postgres_health_engine_uses_nullpool(self) -> None:
        captured: dict[str, object] = {}
        sentinel = object()

        def factory(url: str, **kwargs: object) -> object:
            captured["url"] = url
            captured.update(kwargs)
            return sentinel

        engine = create_postgres_health_engine(
            "postgresql+psycopg://dc_agent:secret@127.0.0.1/dc_agent?connect_timeout=2",
            engine_factory=factory,
        )

        self.assertIs(engine, sentinel)
        self.assertEqual(
            getattr(captured["poolclass"], "__name__", ""),
            "NullPool",
        )
        self.assertIs(captured["pool_pre_ping"], False)

    def test_health_urls_reject_link_local_and_unspecified_before_client_creation(
        self,
    ) -> None:
        module = importlib.import_module("app.main")
        cases = (
            {"CLICKHOUSE_URL": "http://169.254.169.254:8123"},
            {"QDRANT_URL": "http://0.0.0.0:6333"},
            {"REDIS_URL": "redis://169.254.169.254:6379/0"},
            {"EMBEDDING_SERVICE_URL": "http://0.0.0.0:8081"},
            {"QDRANT_URL": "http://attacker:6333"},
            {
                "DATABASE_URL": (
                    "postgresql+psycopg://dc_agent:secret@169.254.169.254:5432/dc_agent"
                )
            },
            {
                "LLAMA_SERVER_URL": "http://169.254.169.254:8080",
                "LLM_PROVIDER": "openai_compatible",
            },
        )

        for changes in cases:
            with self.subTest(changes=changes):
                calls = {"http": 0, "redis": 0}

                def build_http(**_kwargs: object) -> object:
                    calls["http"] += 1
                    return ClosableFake("http")

                def build_redis(_url: str, **_kwargs: object) -> object:
                    calls["redis"] += 1
                    return ClosableFake("redis")

                app = module.create_production_app(
                    environ=private_environment(**changes),
                    repository_factory=lambda: ClosableFake("repository"),
                    ingestion_queue_factory=lambda _repository: ClosableFake("queue"),
                    storage_factory=lambda _root: ClosableFake("storage"),
                    evaluation_import_service_factory=lambda: ClosableFake("evaluation"),
                    health_http_client_factory=build_http,
                    health_redis_client_factory=build_redis,
                )

                with self.assertRaisesRegex(ValueError, "private or loopback"):
                    with TestClient(app):
                        pass
                self.assertEqual(calls, {"http": 0, "redis": 0})

    def test_embedding_readiness_uses_service_root_readyz(self) -> None:
        class FakeResponse:
            status_code = 200
            headers: dict[str, str] = {}

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def iter_bytes(self, **_kwargs: object) -> object:
                yield b'{"status":"ready"}'

        class FakeHttpClient:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def __enter__(self) -> FakeHttpClient:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def get(self, url: str) -> FakeResponse:
                self.urls.append(url)
                return FakeResponse()

            def stream(
                self,
                method: str,
                url: str,
                **kwargs: object,
            ) -> FakeResponse:
                self.urls.append(url)
                return FakeResponse()

        fake_client = FakeHttpClient()
        environ = private_environment(EMBEDDING_SERVICE_URL="http://127.0.0.1:8081/v1")
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
            http_client_factory=lambda **_kwargs: fake_client,
        )
        embedding_check = next(check for check in checks if check.name == "embedding")

        self.assertEqual(embedding_check.check(), (True, "ready"))
        self.assertEqual(
            fake_client.urls,
            ["http://127.0.0.1:8081/readyz"],
        )

    def test_http_health_clients_disable_environment_proxy_inheritance(self) -> None:
        class FakeResponse:
            status_code = 200
            text = "Ok.\n"

        class FakeHttpClient:
            def __enter__(self) -> FakeHttpClient:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def get(self, url: str) -> FakeResponse:
                return FakeResponse()

        captured: dict[str, object] = {}

        def factory(**kwargs: object) -> FakeHttpClient:
            captured.update(kwargs)
            return FakeHttpClient()

        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(),
            http_client_factory=factory,
        )
        clickhouse_check = next(check for check in checks if check.name == "clickhouse")

        self.assertEqual(clickhouse_check.check(), (True, "ready"))
        self.assertIs(captured["trust_env"], False)

    def test_clickhouse_ping_requires_the_expected_response_body(self) -> None:
        class FakeResponse:
            status_code = 200
            text = "not clickhouse"

        class FakeHttpClient:
            def __enter__(self) -> FakeHttpClient:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def get(self, url: str) -> FakeResponse:
                return FakeResponse()

        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(),
            http_client_factory=lambda **_kwargs: FakeHttpClient(),
        )
        clickhouse_check = next(check for check in checks if check.name == "clickhouse")

        self.assertEqual(
            clickhouse_check.check(),
            (False, "invalid readiness response"),
        )

    def test_http_health_rejects_oversized_content_length_without_reading(
        self,
    ) -> None:
        class FakeResponse:
            status_code = 200
            headers = {"content-length": "4097"}

            def __init__(self) -> None:
                self.iterated = False

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            @staticmethod
            def json() -> dict[str, str]:
                return {"status": "ready"}

            def iter_bytes(self) -> object:
                self.iterated = True
                yield b'{"status":"ready"}'

        class FakeHttpClient:
            def __init__(self, response: FakeResponse) -> None:
                self.response = response
                self.stream_calls = 0

            def get(self, url: str) -> FakeResponse:
                return self.response

            def stream(
                self,
                method: str,
                url: str,
                **kwargs: object,
            ) -> FakeResponse:
                self.stream_calls += 1
                return self.response

        response = FakeResponse()
        client = FakeHttpClient(response)
        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(),
            http_client=client,
        )
        embedding_check = next(check for check in checks if check.name == "embedding")

        self.assertEqual(
            embedding_check.check(),
            (False, "invalid readiness response"),
        )
        self.assertEqual(client.stream_calls, 1)
        self.assertFalse(response.iterated)

    def test_http_health_stream_stops_when_body_limit_is_exceeded(self) -> None:
        class FakeResponse:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self) -> None:
                self.chunks_read = 0

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            @staticmethod
            def json() -> dict[str, str]:
                return {"status": "ready"}

            def iter_bytes(self) -> object:
                for chunk in (b"x" * 700, b"y" * 700, b"z" * 700):
                    self.chunks_read += 1
                    yield chunk

        class FakeHttpClient:
            def __init__(self, response: FakeResponse) -> None:
                self.response = response

            def get(self, url: str) -> FakeResponse:
                return self.response

            def stream(
                self,
                method: str,
                url: str,
                **kwargs: object,
            ) -> FakeResponse:
                return self.response

        response = FakeResponse()
        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(),
            http_client=FakeHttpClient(response),
        )
        embedding_check = next(check for check in checks if check.name == "embedding")

        self.assertEqual(
            embedding_check.check(),
            (False, "invalid readiness response"),
        )
        self.assertEqual(response.chunks_read, 2)

    def test_http_health_rejects_encoded_body_without_decompressing(self) -> None:
        class FakeResponse:
            status_code = 200
            headers = {"content-encoding": "gzip"}

            def __init__(self) -> None:
                self.iterated = False

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            @staticmethod
            def json() -> dict[str, str]:
                return {"status": "ready"}

            def iter_bytes(self, **_kwargs: object) -> object:
                self.iterated = True
                yield b'{"status":"ready"}'

        class FakeHttpClient:
            def __init__(self, response: FakeResponse) -> None:
                self.response = response

            def get(self, url: str) -> FakeResponse:
                return self.response

            def stream(
                self,
                method: str,
                url: str,
                **kwargs: object,
            ) -> FakeResponse:
                return self.response

        response = FakeResponse()
        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(),
            http_client=FakeHttpClient(response),
        )
        embedding_check = next(check for check in checks if check.name == "embedding")

        self.assertEqual(
            embedding_check.check(),
            (False, "invalid readiness response"),
        )
        self.assertFalse(response.iterated)

    def test_http_health_reads_unbuffered_raw_chunks(self) -> None:
        class FakeResponse:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self) -> None:
                self.raw_calls = 0
                self.decoded_calls = 0

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def iter_raw(self) -> object:
                self.raw_calls += 1
                yield b'{"status":"ready"}'

            def iter_bytes(self, **_kwargs: object) -> object:
                self.decoded_calls += 1
                yield b'{"status":"ready"}'

        class FakeHttpClient:
            def __init__(self, response: FakeResponse) -> None:
                self.response = response

            def stream(
                self,
                method: str,
                url: str,
                **kwargs: object,
            ) -> FakeResponse:
                return self.response

        response = FakeResponse()
        checks = build_dependency_checks(
            build_settings(),
            database=object(),
            environ=private_environment(),
            http_client=FakeHttpClient(response),
        )
        embedding_check = next(check for check in checks if check.name == "embedding")

        self.assertEqual(embedding_check.check(), (True, "ready"))
        self.assertEqual(response.raw_calls, 1)
        self.assertEqual(response.decoded_calls, 0)

    def test_http_health_stream_uses_one_total_deadline(self) -> None:
        clock = [0.0]

        class FakeResponse:
            status_code = 200
            headers: dict[str, str] = {}

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            @staticmethod
            def json() -> dict[str, str]:
                return {"status": "ready"}

            def iter_bytes(self) -> object:
                yield b'{"status":'
                clock[0] = 1.1
                yield b'"ready"}'

        class FakeHttpClient:
            def __init__(self) -> None:
                self.timeouts: list[float] = []

            def get(self, url: str) -> FakeResponse:
                return FakeResponse()

            def stream(
                self,
                method: str,
                url: str,
                **kwargs: object,
            ) -> FakeResponse:
                self.timeouts.append(float(kwargs["timeout"]))
                return FakeResponse()

        client = FakeHttpClient()
        environ = private_environment(DEPENDENCY_TIMEOUT_SECONDS="1.0")
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
            http_client=client,
        )
        embedding_check = next(check for check in checks if check.name == "embedding")

        with patch(
            "app.infra.health.monotonic",
            side_effect=lambda: clock[0],
        ):
            self.assertEqual(
                embedding_check.check(),
                (False, "unavailable"),
            )
        self.assertEqual(client.timeouts, [1.0])

    def test_public_clamav_host_is_rejected_without_a_socket_call(self) -> None:
        environ = private_environment(CLAMAV_HOST="8.8.8.8")
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
        )
        clamav_check = next(check for check in checks if check.name == "clamav")

        with patch(
            "app.infra.health.socket.create_connection",
            side_effect=AssertionError("public ClamAV host was contacted"),
        ):
            self.assertEqual(
                clamav_check.check(),
                (False, "invalid configuration"),
            )

    def test_arbitrary_clamav_service_name_is_rejected_without_dns(self) -> None:
        environ = private_environment(CLAMAV_HOST="attacker")
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
        )
        clamav_check = next(check for check in checks if check.name == "clamav")

        with patch(
            "app.infra.health.socket.create_connection",
            side_effect=AssertionError("unapproved ClamAV host was contacted"),
        ):
            self.assertEqual(
                clamav_check.check(),
                (False, "invalid configuration"),
            )

    def test_clamav_ping_reads_partial_pong_until_terminator(self) -> None:
        class FakeClamAVSocket:
            def __init__(self) -> None:
                self.chunks = [b"PO", b"NG", b"\0"]
                self.recv_calls = 0

            def __enter__(self) -> FakeClamAVSocket:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def settimeout(self, timeout: float) -> None:
                return None

            def sendall(self, payload: bytes) -> None:
                return None

            def recv(self, size: int) -> bytes:
                self.recv_calls += 1
                return self.chunks.pop(0)

        fake_socket = FakeClamAVSocket()
        environ = private_environment(CLAMAV_HOST="clamav")
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
        )
        clamav_check = next(check for check in checks if check.name == "clamav")

        with patch(
            "app.infra.health.socket.create_connection",
            return_value=fake_socket,
        ):
            self.assertEqual(clamav_check.check(), (True, "ready"))
        self.assertEqual(fake_socket.recv_calls, 3)

    def test_clamav_ping_enforces_one_total_socket_deadline(self) -> None:
        class SlowDripSocket:
            def __init__(self) -> None:
                self.recv_calls = 0

            def __enter__(self) -> SlowDripSocket:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def settimeout(self, timeout: float) -> None:
                return None

            def sendall(self, payload: bytes) -> None:
                return None

            def recv(self, size: int) -> bytes:
                self.recv_calls += 1
                return b"P"

        fake_socket = SlowDripSocket()
        environ = private_environment(
            CLAMAV_HOST="clamav",
            DEPENDENCY_TIMEOUT_SECONDS="1.0",
        )
        checks = build_dependency_checks(
            build_settings(environ),
            database=object(),
            environ=environ,
        )
        clamav_check = next(check for check in checks if check.name == "clamav")

        with (
            patch(
                "app.infra.health.socket.create_connection",
                return_value=fake_socket,
            ) as create_connection,
            patch(
                "app.infra.health.monotonic",
                side_effect=(0.0, 0.2, 0.4, 0.6, 1.1),
                create=True,
            ),
        ):
            self.assertEqual(clamav_check.check(), (False, "unavailable"))
        create_connection.assert_called_once_with(
            ("clamav", 3310),
            timeout=0.8,
        )
        self.assertEqual(fake_socket.recv_calls, 1)


if __name__ == "__main__":
    unittest.main()
