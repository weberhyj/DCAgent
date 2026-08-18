from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from threading import Event, Lock

from .ingestion import KnowledgeIngestionQueue
from .models import KnowledgeSourceModel
from .repository import STATUS_INDEXING, ChatRepository


class KnowledgeIngestionWorker:
    """Process uploaded knowledge sources outside the API process.

    The source row is the durable queue: an uploaded or reindexed source stays
    in ``解析中`` until this worker finishes it.  That makes unfinished work
    discoverable again after an API or worker restart without relying on an
    in-memory queue.
    """

    def __init__(
        self,
        repository: ChatRepository,
        ingestion_queue: KnowledgeIngestionQueue,
        *,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._repository = repository
        self._ingestion_queue = ingestion_queue
        self._poll_interval_seconds = poll_interval_seconds
        self._stop = Event()
        self._run_lock = Lock()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> bool:
        if not self._run_lock.acquire(blocking=False):
            return False
        try:
            source = self._next_source()
            if source is None or not source.file_path:
                return False
            self._ingestion_queue.process(source.id, source.file_path, source.source_type)
            return True
        finally:
            self._run_lock.release()

    def _next_source(self) -> KnowledgeSourceModel | None:
        sources = self._repository.list_knowledge_sources()
        candidates = [
            source
            for source in sources
            if source.status == STATUS_INDEXING and source.file_path
        ]
        if not candidates:
            return None
        # list_knowledge_sources returns newest first in both repository
        # implementations.  Use the stable source ID as a tie-breaker so a
        # restart cannot continuously starve older uploads.
        return min(candidates, key=lambda source: (source.updated_at, source.id))

    def run_forever(self) -> None:
        while not self._stop.is_set():
            if not self.run_once():
                self._stop.wait(self._poll_interval_seconds)


async def _run_production_worker() -> None:
    # Importing the application gives the worker the same database, storage,
    # retrieval lifecycle, and compatibility settings as the API deployment.
    from .main import app

    async with app.router.lifespan_context(app):
        worker = KnowledgeIngestionWorker(
            app.state.repository,
            app.state.knowledge_ingestion_queue,
            poll_interval_seconds=float(os.environ.get("KNOWLEDGE_INGESTION_POLL_SECONDS", "1")),
        )
        for signum in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                asyncio.get_running_loop().add_signal_handler(signum, worker.stop)
        await asyncio.to_thread(worker.run_forever)


def main() -> None:
    asyncio.run(_run_production_worker())


if __name__ == "__main__":
    main()
