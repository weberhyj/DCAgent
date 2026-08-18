from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ingestion import KnowledgeIngestionQueue
from app.ingestion_worker import KnowledgeIngestionWorker
from app.repository import STATUS_INDEXED, STATUS_INDEXING, InMemoryChatRepository
from app.seed import build_seed_state


class KnowledgeIngestionWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repository = InMemoryChatRepository(build_seed_state())
        self.queue = KnowledgeIngestionQueue(self.repository)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_pending_source(self, source_id: str = "kb-async") -> Path:
        path = self.root / f"{source_id}.txt"
        path.write_text("异步解析后的知识内容。" * 40, encoding="utf-8")
        self.repository.add_uploaded_knowledge_source(
            source_id=source_id,
            name=path.name,
            source_type="文档",
            classification="内部",
            records=0,
            file_path=str(path),
            file_size=path.stat().st_size,
            mime_type="text/plain",
        )
        return path

    def test_worker_processes_persisted_indexing_source(self) -> None:
        self.add_pending_source()
        worker = KnowledgeIngestionWorker(self.repository, self.queue)

        self.assertTrue(worker.run_once())

        source = next(item for item in self.repository.list_knowledge_sources() if item.id == "kb-async")
        self.assertEqual(source.status, STATUS_INDEXED)
        self.assertGreater(source.records, 0)
        self.assertGreater(len(self.repository.list_knowledge_chunks("kb-async")), 0)
        self.assertFalse(worker.run_once())

    def test_new_worker_recovers_source_left_indexing_after_restart(self) -> None:
        self.add_pending_source("kb-recovered")
        before_restart = next(
            item for item in self.repository.list_knowledge_sources() if item.id == "kb-recovered"
        )
        self.assertEqual(before_restart.status, STATUS_INDEXING)

        restarted_worker = KnowledgeIngestionWorker(self.repository, self.queue)

        self.assertTrue(restarted_worker.run_once())
        recovered = next(
            item for item in self.repository.list_knowledge_sources() if item.id == "kb-recovered"
        )
        self.assertEqual(recovered.status, STATUS_INDEXED)

    def test_worker_ignores_manual_source_without_uploaded_file(self) -> None:
        self.repository.add_knowledge_source("manual", "文档", "内部")
        worker = KnowledgeIngestionWorker(self.repository, self.queue)

        self.assertFalse(worker.run_once())

        source = self.repository.list_knowledge_sources()[0]
        self.assertEqual(source.status, STATUS_INDEXING)
        self.assertIsNone(source.file_path)


if __name__ == "__main__":
    unittest.main()
