from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Database, resolve_database_url
from .evaluation_import import EvaluationImportService
from .ingestion import KnowledgeIngestionQueue
from .llm import LLMProvider, create_llm_provider
from .repository import ChatRepository, InMemoryChatRepository
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
) -> FastAPI:
    app = FastAPI(title="DC-Agent API", version="0.2.0")
    app.state.repository = repository or create_default_repository(llm_provider)
    app.state.knowledge_ingestion_queue = ingestion_queue or KnowledgeIngestionQueue(app.state.repository)
    app.state.knowledge_file_storage = KnowledgeFileStorage(
        upload_dir or Path(__file__).resolve().parents[1] / "uploads" / "knowledge"
    )
    app.state.evaluation_import_service = EvaluationImportService(ttl_seconds=1800)

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


app = create_app()
