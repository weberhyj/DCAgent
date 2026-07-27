from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from fastapi import HTTPException

from .repository import ChatRepository
from .spreadsheet_schema import infer_spreadsheet_schema
from .structured_repository import StructuredRepository
from .text_parser import parse_knowledge_file


@dataclass(slots=True)
class KnowledgeIngestionJob:
    source_id: str
    file_path: Path
    source_type: str


class KnowledgeIndexLifecycle(Protocol):
    def upsert_source(self, source_id: str): ...

    def delete_source(self, source_id: str) -> None: ...


class KnowledgeIndexUnavailableError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=503,
            detail="Hybrid retrieval index is temporarily unavailable; source was not deleted",
        )


class KnowledgeIngestionQueue:
    def __init__(
        self,
        repository: ChatRepository,
        structured_repository: StructuredRepository | None = None,
        structured_query_enabled: bool = False,
        index_lifecycle: KnowledgeIndexLifecycle | None = None,
    ) -> None:
        self._repository = repository
        self._structured_repository = structured_repository
        self._structured_query_enabled = structured_query_enabled
        self._index_lifecycle = index_lifecycle
        self._pending: list[KnowledgeIngestionJob] = []
        self._lock = Lock()

    def enqueue(self, source_id: str, file_path: str | Path, source_type: str) -> None:
        job = KnowledgeIngestionJob(
            source_id=source_id,
            file_path=Path(file_path),
            source_type=source_type,
        )
        with self._lock:
            self._pending.append(job)

    def discard_source(self, source_id: str) -> None:
        if self._index_lifecycle is not None:
            try:
                self._index_lifecycle.delete_source(source_id)
            except Exception as error:
                raise KnowledgeIndexUnavailableError() from error
        with self._lock:
            self._pending = [job for job in self._pending if job.source_id != source_id]

    def drain(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    return
                job = self._pending.pop(0)
            self._process(job)

    def _process(self, job: KnowledgeIngestionJob) -> None:
        try:
            if self._should_infer_structured_preview(job):
                if self._structured_repository is None:
                    raise RuntimeError("Structured repository is unavailable")
                preview = infer_spreadsheet_schema(job.file_path, job.source_id)
                self._structured_repository.save_preview(preview)
                return
            chunks = parse_knowledge_file(job.file_path, job.source_id, job.source_type)
            self._repository.complete_knowledge_source_indexing(job.source_id, chunks)
        except Exception as exc:
            try:
                message = str(exc) or exc.__class__.__name__
                self._repository.fail_knowledge_source_indexing(job.source_id, message)
            except Exception:
                return
            return
        if self._index_lifecycle is None:
            return
        try:
            publication_id, indexed_chunk_count = self._index_lifecycle.upsert_source(job.source_id)
            self._repository.complete_retrieval_source_indexing(
                job.source_id,
                publication_id,
                indexed_chunk_count,
            )
        except Exception as error:
            try:
                self._repository.fail_retrieval_source_indexing(
                    job.source_id,
                    error.__class__.__name__,
                )
            except Exception:
                return

    def _should_infer_structured_preview(self, job: KnowledgeIngestionJob) -> bool:
        return self._structured_query_enabled and job.file_path.suffix.lower() in {".xlsx", ".csv"}
