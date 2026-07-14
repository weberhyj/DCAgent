from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .repository import ChatRepository
from .text_parser import parse_knowledge_file


@dataclass(slots=True)
class KnowledgeIngestionJob:
    source_id: str
    file_path: Path
    source_type: str


class KnowledgeIngestionQueue:
    def __init__(self, repository: ChatRepository) -> None:
        self._repository = repository
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
            chunks = parse_knowledge_file(job.file_path, job.source_id, job.source_type)
            self._repository.complete_knowledge_source_indexing(job.source_id, chunks)
        except Exception as exc:
            try:
                message = str(exc) or exc.__class__.__name__
                self._repository.fail_knowledge_source_indexing(job.source_id, message)
            except Exception:
                return
