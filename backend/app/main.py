from __future__ import annotations

import inspect
import math
import os
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from threading import Condition
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.engine import make_url

from .clickhouse_gateway import ClickHouseGateway, StructuredStorageError
from .database import Database, resolve_database_url
from .evaluation_import import EvaluationImportService
from .infra.health import (
    DependencyHealthRegistry,
    build_dependency_checks,
    create_http_health_client,
    create_postgres_health_engine,
    create_redis_health_client,
    validate_health_service_urls,
)
from .ingestion import KnowledgeIngestionQueue
from .llm import LLMProvider, create_llm_provider
from .offline_settings import OfflineSettings, parse_bool
from .repository import ChatRepository
from .routes import router
from .runtime_env import load_runtime_environment
from .sql_repository import SqlChatRepository
from .storage import KnowledgeFileStorage
from .structured_answer import StructuredAnswerService
from .structured_repository import StructuredRepository


def create_default_repository(
    llm_provider: LLMProvider | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    structured_query_enabled: bool | None = None,
    database_factory: Callable[[str], Database] = Database,
    clickhouse_client_factory: Callable[..., object] | None = None,
) -> ChatRepository:
    if environ is None:
        load_runtime_environment()
        source: Mapping[str, str] = os.environ
    else:
        source = environ
    if structured_query_enabled is not None:
        source = {
            **source,
            "STRUCTURED_QUERY_ENABLED": "true" if structured_query_enabled else "false",
        }
    enabled = parse_bool(source.get("STRUCTURED_QUERY_ENABLED"), default=False)
    settings = OfflineSettings.from_environ(source) if enabled else None
    database_url = settings.database_url if settings is not None else resolve_database_url(source)
    database = database_factory(database_url)
    database.create_schema()
    structured_service = None
    if settings is not None:
        structured_repository = create_structured_repository(database)
        structured_service, _clients = _create_structured_answer_service(
            settings,
            structured_repository,
            clickhouse_client_factory=clickhouse_client_factory,
        )
    repository = SqlChatRepository(
        database,
        llm_provider=llm_provider or create_llm_provider(source),
        structured_service=structured_service,
    )
    return repository


def create_structured_repository(database: Database) -> StructuredRepository:
    return StructuredRepository(database)


def create_app(
    repository: ChatRepository | None = None,
    upload_dir: Path | None = None,
    ingestion_queue: KnowledgeIngestionQueue | None = None,
    structured_repository: StructuredRepository | None = None,
    structured_query_enabled: bool = False,
    llm_provider: LLMProvider | None = None,
    health_registry: DependencyHealthRegistry | None = None,
) -> FastAPI:
    owns_repository = repository is None
    if repository is None:
        # The create_app flag is authoritative so ambient env values cannot change legacy defaults.
        repository = create_default_repository(
            llm_provider,
            structured_query_enabled=structured_query_enabled,
        )

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        try:
            yield
        finally:
            if owns_repository:
                await _close_owned_resource(repository)

    app = _build_app(lifespan=lifespan if owns_repository else None)
    app.state.repository = repository
    app.state.structured_repository = structured_repository
    app.state.structured_query_enabled = structured_query_enabled
    app.state.knowledge_ingestion_queue = ingestion_queue or KnowledgeIngestionQueue(
        app.state.repository,
        structured_repository=structured_repository,
        structured_query_enabled=structured_query_enabled,
    )
    app.state.knowledge_file_storage = KnowledgeFileStorage(
        upload_dir or Path(__file__).resolve().parents[1] / "uploads" / "knowledge"
    )
    app.state.evaluation_import_service = EvaluationImportService(ttl_seconds=1800)
    app.state.health_registry = (
        health_registry if health_registry is not None else DependencyHealthRegistry()
    )
    app.state.health_checks_active = True

    return app


def _build_app(*, lifespan: Any | None = None) -> FastAPI:
    app = FastAPI(
        title="DC-Agent API",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.health_checks_active = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5177",
            "http://127.0.0.1:5177",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


def _database_url_with_connect_timeout(
    database_url: str,
    timeout_seconds: float,
) -> str:
    try:
        url = make_url(database_url)
    except Exception:
        return database_url
    if url.get_backend_name() != "postgresql":
        return database_url

    try:
        normalized_timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        normalized_timeout = 2.0
    if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
        normalized_timeout = 2.0
    bounded_timeout = max(1, min(10, math.ceil(normalized_timeout)))
    query = dict(url.query)
    query["connect_timeout"] = str(bounded_timeout)
    return url.set(query=query).render_as_string(hide_password=False)


async def _close_owned_resource(resource: object) -> None:
    for method_name in ("aclose", "close", "shutdown", "dispose"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass
        return

    engine = getattr(resource, "engine", None)
    dispose = getattr(engine, "dispose", None)
    if callable(dispose):
        with suppress(Exception):
            result = dispose()
            if inspect.isawaitable(result):
                await result


def create_production_app(
    *,
    environ: Mapping[str, str] | None = None,
    repository_factory: Callable[[], ChatRepository] | None = None,
    health_registry_factory: Callable[[], DependencyHealthRegistry] | None = None,
    database_factory: Callable[[str], object] | None = None,
    llm_provider_factory: Callable[[Mapping[str, str]], LLMProvider] | None = None,
    ingestion_queue_factory: Callable[..., object] | None = None,
    storage_factory: Callable[[Path], object] | None = None,
    evaluation_import_service_factory: Callable[[], object] | None = None,
    health_http_client_factory: Callable[..., object] | None = None,
    postgres_health_engine_factory: Callable[..., object] | None = None,
    health_redis_client_factory: Callable[..., object] | None = None,
    clickhouse_client_factory: Callable[..., object] | None = None,
    upload_dir: Path | None = None,
) -> FastAPI:
    environment_override = dict(environ) if environ is not None else None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        owned_resources: list[object] = []
        owned_resource_ids: set[int] = set()

        def own(resource: object) -> object:
            if id(resource) not in owned_resource_ids:
                owned_resource_ids.add(id(resource))
                owned_resources.append(resource)
            return resource

        try:
            if environment_override is None:
                load_runtime_environment()
                source: Mapping[str, str] = os.environ
            else:
                source = environment_override

            settings = OfflineSettings.from_environ(source)
            if health_registry_factory is None:
                validate_health_service_urls(settings, source)
            provider_builder = llm_provider_factory or create_llm_provider
            llm_provider = own(provider_builder(source))

            database_builder = database_factory or Database
            database_url = _database_url_with_connect_timeout(
                settings.database_url,
                settings.dependency_timeout_seconds,
            )
            database = own(database_builder(database_url))

            structured_repository = create_structured_repository(database)  # type: ignore[arg-type]
            structured_service = None
            if settings.structured_query_enabled:
                structured_service, clickhouse_clients = _create_structured_answer_service(
                    settings,
                    structured_repository,
                    clickhouse_client_factory=clickhouse_client_factory,
                )
                for client in clickhouse_clients:
                    own(client)

            if repository_factory is None:
                repository: ChatRepository = SqlChatRepository(
                    database,  # type: ignore[arg-type]
                    llm_provider=llm_provider,  # type: ignore[arg-type]
                    structured_service=structured_service,
                )
            else:
                repository = repository_factory()
                if structured_service is not None:
                    if not hasattr(repository, "_structured_service"):
                        raise TypeError(
                            "When STRUCTURED_QUERY_ENABLED=true, repository_factory must "
                            "return a structured-service-aware repository"
                        )
                    repository._structured_service = structured_service  # type: ignore[attr-defined]
            own(repository)
            if ingestion_queue_factory is None:
                ingestion_queue = own(
                    KnowledgeIngestionQueue(
                        repository,
                        structured_repository=structured_repository,
                        structured_query_enabled=settings.structured_query_enabled,
                    )
                )
            else:
                ingestion_queue = own(
                    _create_custom_ingestion_queue(
                        ingestion_queue_factory,
                        repository,
                        structured_repository,
                        settings.structured_query_enabled,
                    )
                )

            storage_builder = storage_factory or KnowledgeFileStorage
            storage_root = (
                upload_dir or Path(__file__).resolve().parents[1] / "uploads" / "knowledge"
            )
            storage = own(storage_builder(storage_root))

            evaluation_builder = evaluation_import_service_factory or (
                lambda: EvaluationImportService(ttl_seconds=1800)
            )
            evaluation_service = own(evaluation_builder())

            if health_registry_factory is None:
                postgres_health_engine = own(
                    create_postgres_health_engine(
                        database_url,
                        engine_factory=postgres_health_engine_factory,
                    )
                )
                health_http_client = own(
                    create_http_health_client(
                        settings.dependency_timeout_seconds,
                        client_factory=health_http_client_factory,
                    )
                )
                health_redis_client = own(
                    create_redis_health_client(
                        settings.redis_url,
                        settings.dependency_timeout_seconds,
                        client_factory=health_redis_client_factory,
                    )
                )
                bounded_health_timeout = settings.dependency_timeout_seconds
                if not math.isfinite(bounded_health_timeout) or bounded_health_timeout <= 0:
                    bounded_health_timeout = 2.0
                bounded_health_timeout = min(bounded_health_timeout, 10.0)
                health_registry = DependencyHealthRegistry(
                    build_dependency_checks(
                        settings,
                        database=postgres_health_engine,
                        environ=source,
                        http_client=health_http_client,
                        redis_client=health_redis_client,
                    ),
                    cache_ttl_seconds=0.5,
                    max_stale_seconds=bounded_health_timeout + 0.5,
                )
                own(health_registry)
            else:
                health_registry = health_registry_factory()
                if not isinstance(
                    health_registry,
                    DependencyHealthRegistry,
                ):
                    raise TypeError("health_registry_factory must return DependencyHealthRegistry")
                own(health_registry)

            application.state.llm_provider = llm_provider
            application.state.database = database
            application.state.repository = repository
            application.state.structured_repository = structured_repository
            application.state.structured_query_enabled = settings.structured_query_enabled
            application.state.knowledge_ingestion_queue = ingestion_queue
            application.state.knowledge_file_storage = storage
            application.state.evaluation_import_service = evaluation_service
            application.state.health_registry = health_registry
            application.state.health_checks_active = True
            yield
        finally:
            application.state.health_checks_active = False
            for resource in reversed(owned_resources):
                await _close_owned_resource(resource)

    return _build_app(lifespan=lifespan)


def _create_structured_answer_service(
    settings: OfflineSettings,
    structured_repository: StructuredRepository,
    *,
    clickhouse_client_factory: Callable[..., object] | None = None,
) -> tuple[StructuredAnswerService, tuple[object, ...]]:
    if not settings.structured_query_enabled:
        raise ValueError("structured answer service requires STRUCTURED_QUERY_ENABLED=true")
    if clickhouse_client_factory is None:
        import clickhouse_connect

        clickhouse_client_factory = clickhouse_connect.get_client
    gateway = _LazyStructuredQueryGateway(clickhouse_client_factory, settings.clickhouse_url)
    service = StructuredAnswerService(structured_repository.get_catalog, gateway)
    return service, (gateway,)


class _LazyStructuredQueryGateway:
    def __init__(self, client_factory: Callable[..., object], dsn: str) -> None:
        self._client_factory = client_factory
        self._dsn = dsn
        self._condition = Condition()
        self._closed = False
        self._initializing = False
        self._active_queries = 0
        self._gateway: ClickHouseGateway | None = None
        self._clients: tuple[object, object] | None = None

    def query(self, statement: str, parameters: Mapping[str, object] | None = None) -> object:
        gateway = self._acquire_gateway_for_query()
        try:
            return gateway.query(statement, parameters)
        finally:
            with self._condition:
                self._active_queries -= 1
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._initializing or self._active_queries:
                self._condition.wait()
            clients = self._clients or ()
            self._clients = None
            self._gateway = None
        _close_clickhouse_clients(clients)

    def _acquire_gateway_for_query(self) -> ClickHouseGateway:
        with self._condition:
            while True:
                if self._closed:
                    raise StructuredStorageError("ClickHouse structured query gateway is closed")
                if self._gateway is not None:
                    self._active_queries += 1
                    return self._gateway
                if not self._initializing:
                    self._initializing = True
                    break
                self._condition.wait()

        try:
            gateway, clients = self._build_gateway()
        except Exception:
            with self._condition:
                self._initializing = False
                self._condition.notify_all()
            raise

        with self._condition:
            if not self._closed:
                self._gateway = gateway
                self._clients = clients
                self._active_queries += 1
                self._initializing = False
                self._condition.notify_all()
                return gateway

        _close_clickhouse_clients(clients)
        with self._condition:
            self._initializing = False
            self._condition.notify_all()
        raise StructuredStorageError("ClickHouse structured query gateway is closed")

    def _build_gateway(self) -> tuple[ClickHouseGateway, tuple[object, object]]:
        ingest_client = self._client_factory(dsn=self._dsn)
        try:
            query_client = self._client_factory(dsn=self._dsn)
        except Exception:
            _close_clickhouse_clients((ingest_client,))
            raise
        clients = (ingest_client, query_client)
        try:
            gateway = ClickHouseGateway(ingest_client, query_client=query_client)
        except Exception:
            _close_clickhouse_clients(clients)
            raise
        return gateway, clients


def _close_clickhouse_clients(clients: tuple[object, ...]) -> None:
    closed_ids: set[int] = set()
    for client in reversed(clients):
        if id(client) in closed_ids:
            continue
        closed_ids.add(id(client))
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _create_custom_ingestion_queue(
    factory: Callable[..., object],
    repository: ChatRepository,
    structured_repository: StructuredRepository,
    structured_query_enabled: bool,
) -> object:
    if not structured_query_enabled:
        return factory(repository)

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "When STRUCTURED_QUERY_ENABLED=true, ingestion_queue_factory must be "
            "structured-aware and expose an inspectable three-argument signature"
        ) from error

    keyword_arguments = {
        "repository": repository,
        "structured_repository": structured_repository,
        "structured_query_enabled": True,
    }
    try:
        signature.bind(**keyword_arguments)
    except TypeError:
        positional_arguments = (repository, structured_repository, True)
        try:
            signature.bind(*positional_arguments)
        except TypeError as error:
            raise TypeError(
                "When STRUCTURED_QUERY_ENABLED=true, ingestion_queue_factory must be "
                "structured-aware and accept repository, structured_repository, and "
                "structured_query_enabled"
            ) from error
        return factory(*positional_arguments)
    return factory(**keyword_arguments)


# Legacy development commands still import ``app.main:app``.  This construction
# only registers routes and the lifespan; all stateful work remains in startup.
app = create_production_app()
