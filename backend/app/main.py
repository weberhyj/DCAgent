from __future__ import annotations

import inspect
import math
import os
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.engine import make_url

from .database import Database, resolve_database_url
from .evaluation_import import EvaluationImportService
from .infra.health import (
    DependencyHealthRegistry,
    build_dependency_checks,
)
from .ingestion import KnowledgeIngestionQueue
from .llm import LLMProvider, create_llm_provider
from .offline_settings import OfflineSettings
from .repository import ChatRepository
from .routes import router
from .runtime_env import load_runtime_environment
from .sql_repository import SqlChatRepository
from .storage import KnowledgeFileStorage


def create_default_repository(llm_provider: LLMProvider | None = None) -> ChatRepository:
    load_runtime_environment()
    database = Database(resolve_database_url())
    database.create_schema()
    repository = SqlChatRepository(database, llm_provider=llm_provider or create_llm_provider())
    return repository


def create_app(
    repository: ChatRepository | None = None,
    upload_dir: Path | None = None,
    ingestion_queue: KnowledgeIngestionQueue | None = None,
    llm_provider: LLMProvider | None = None,
    health_registry: DependencyHealthRegistry | None = None,
) -> FastAPI:
    app = _build_app()
    app.state.repository = repository or create_default_repository(llm_provider)
    app.state.knowledge_ingestion_queue = ingestion_queue or KnowledgeIngestionQueue(app.state.repository)
    app.state.knowledge_file_storage = KnowledgeFileStorage(
        upload_dir or Path(__file__).resolve().parents[1] / "uploads" / "knowledge"
    )
    app.state.evaluation_import_service = EvaluationImportService(ttl_seconds=1800)
    app.state.health_registry = (
        health_registry
        if health_registry is not None
        else DependencyHealthRegistry()
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

    bounded_timeout = max(1, min(10, math.ceil(timeout_seconds)))
    query = dict(url.query)
    query.setdefault("connect_timeout", str(bounded_timeout))
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
    health_registry_factory: Callable[
        [], DependencyHealthRegistry
    ] | None = None,
    database_factory: Callable[[str], object] | None = None,
    llm_provider_factory: Callable[
        [Mapping[str, str]], LLMProvider
    ] | None = None,
    ingestion_queue_factory: Callable[[ChatRepository], object] | None = None,
    storage_factory: Callable[[Path], object] | None = None,
    evaluation_import_service_factory: Callable[[], object] | None = None,
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
            provider_builder = llm_provider_factory or create_llm_provider
            llm_provider = own(provider_builder(source))

            database_builder = database_factory or Database
            database_url = _database_url_with_connect_timeout(
                settings.database_url,
                settings.dependency_timeout_seconds,
            )
            database = own(database_builder(database_url))

            if repository_factory is None:
                repository: ChatRepository = SqlChatRepository(
                    database,  # type: ignore[arg-type]
                    llm_provider=llm_provider,  # type: ignore[arg-type]
                )
            else:
                repository = repository_factory()
            own(repository)

            queue_builder = ingestion_queue_factory or KnowledgeIngestionQueue
            ingestion_queue = own(queue_builder(repository))

            storage_builder = storage_factory or KnowledgeFileStorage
            storage_root = (
                upload_dir
                or Path(__file__).resolve().parents[1]
                / "uploads"
                / "knowledge"
            )
            storage = own(storage_builder(storage_root))

            evaluation_builder = (
                evaluation_import_service_factory
                or (lambda: EvaluationImportService(ttl_seconds=1800))
            )
            evaluation_service = own(evaluation_builder())

            if health_registry_factory is None:
                health_registry = DependencyHealthRegistry(
                    build_dependency_checks(
                        settings,
                        database=database,
                        environ=source,
                    )
                )
            else:
                health_registry = health_registry_factory()
                if not isinstance(
                    health_registry,
                    DependencyHealthRegistry,
                ):
                    raise TypeError(
                        "health_registry_factory must return "
                        "DependencyHealthRegistry"
                    )

            application.state.llm_provider = llm_provider
            application.state.database = database
            application.state.repository = repository
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


# Legacy development commands still import ``app.main:app``.  This construction
# only registers routes and the lifespan; all stateful work remains in startup.
app = create_production_app()
