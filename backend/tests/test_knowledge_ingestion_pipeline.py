from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.ingestion import KnowledgeIndexUnavailableError, KnowledgeIngestionQueue
from app.main import create_app
from app.repository import InMemoryChatRepository
from app.seed import build_seed_state
from app.text_parser import parse_knowledge_file


class KnowledgeIngestionPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = InMemoryChatRepository(build_seed_state())
        self.queue = KnowledgeIngestionQueue(self.repository)
        self.client = TestClient(
            create_app(
                repository=self.repository,
                upload_dir=Path(self.temp_dir.name),
                ingestion_queue=self.queue,
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_upload_registers_pending_source_then_background_task_indexes_chunks(self) -> None:
        body = ("第一段现金流分析。" * 80).encode("utf-8")

        response = self.client.post(
            "/api/knowledge/uploads",
            data={"classification": "内部·机密"},
            files={"file": ("cashflow-note.txt", body, "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        uploaded = response.json()[0]
        self.assertEqual(uploaded["name"], "cashflow-note.txt")
        self.assertEqual(uploaded["status"], "解析中")
        self.assertEqual(uploaded["records"], 0)

        sources = self.client.get("/api/knowledge/sources").json()
        indexed = sources[0]
        self.assertEqual(indexed["id"], uploaded["id"])
        self.assertEqual(indexed["status"], "已索引")
        self.assertGreater(indexed["records"], 1)

        chunks_response = self.client.get(f"/api/knowledge/sources/{uploaded['id']}/chunks")
        self.assertEqual(chunks_response.status_code, 200)
        chunks = chunks_response.json()
        self.assertEqual(len(chunks), indexed["records"])
        self.assertEqual(chunks[0]["sourceId"], uploaded["id"])
        self.assertEqual(chunks[0]["chunkIndex"], 0)
        self.assertIn("现金流", chunks[0]["text"])

    def test_parser_splits_long_text_file_into_ordered_chunks(self) -> None:
        path = Path(self.temp_dir.name) / "risk-note.md"
        path.write_text(
            "\n".join([f"风险提示 {index}: 回款周期变化" for index in range(120)]), encoding="utf-8"
        )

        chunks = parse_knowledge_file(path, source_id="kb-parser", source_type="文档")

        self.assertGreater(len(chunks), 1)
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(chunk.source_id == "kb-parser" for chunk in chunks))
        self.assertIn("风险提示", chunks[0].text)

    def test_parser_removes_nul_bytes_before_chunks_are_indexed(self) -> None:
        path = Path(self.temp_dir.name) / "nul-note.txt"
        path.write_bytes("差旅制度".encode() + b"\x00" + "审批流程".encode())

        chunks = parse_knowledge_file(path, source_id="kb-nul", source_type="文档")

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all("\x00" not in chunk.text for chunk in chunks))
        self.assertIn("差旅制度", chunks[0].text)
        self.assertIn("审批流程", chunks[0].text)

    def test_new_document_updates_postgres_then_active_qdrant_collection(self) -> None:
        events = []

        class Lifecycle:
            def upsert_source(inner_self, source_id):
                events.append(("qdrant", len(self.repository.list_knowledge_chunks(source_id))))
                return "publication-1", 1

        source_id = "source-lifecycle"
        path = Path(self.temp_dir.name) / "lifecycle.txt"
        path.write_text("one searchable passage", encoding="utf-8")
        self.repository.add_uploaded_knowledge_source(
            source_id, "lifecycle.txt", "TXT", "internal", 0, str(path), 22, "text/plain"
        )
        queue = KnowledgeIngestionQueue(self.repository, index_lifecycle=Lifecycle())

        queue.enqueue(source_id, path, "TXT")
        queue.drain()

        self.assertEqual(events, [("qdrant", 1)])
        self.assertEqual(self.repository.get_source_index_status(source_id), "indexed")

    def test_qdrant_failure_does_not_delete_postgres_chunks(self) -> None:
        class FailingLifecycle:
            def upsert_source(self, source_id):
                raise RuntimeError("qdrant unavailable")

        source_id = "source-failed-index"
        path = Path(self.temp_dir.name) / "failed-index.txt"
        path.write_text("legacy passage remains available", encoding="utf-8")
        self.repository.add_uploaded_knowledge_source(
            source_id,
            "failed-index.txt",
            "TXT",
            "internal",
            0,
            str(path),
            32,
            "text/plain",
        )
        queue = KnowledgeIngestionQueue(self.repository, index_lifecycle=FailingLifecycle())

        queue.enqueue(source_id, path, "TXT")
        queue.drain()

        self.assertEqual(len(self.repository.list_knowledge_chunks(source_id)), 1)
        self.assertEqual(self.repository.get_source_index_status(source_id), "failed")

    def test_source_deletion_stops_before_postgres_when_qdrant_delete_fails(self) -> None:
        class FailingDeleteLifecycle:
            def delete_source(self, source_id):
                raise RuntimeError("qdrant unavailable")

        source_id = "source-delete"
        self.repository.add_knowledge_source("delete.txt", "TXT", "internal")
        self.repository._state.knowledge_sources[0].id = source_id
        queue = KnowledgeIngestionQueue(
            self.repository,
            index_lifecycle=FailingDeleteLifecycle(),
        )

        with self.assertRaises(KnowledgeIndexUnavailableError) as captured:
            queue.discard_source(source_id)

        self.assertEqual(captured.exception.status_code, 503)
        self.assertTrue(
            any(source.id == source_id for source in self.repository.list_knowledge_sources())
        )

    def test_delete_endpoint_returns_503_and_preserves_source_when_qdrant_fails(self) -> None:
        class FailingDeleteLifecycle:
            def delete_source(self, source_id, *, finalize=None):
                raise RuntimeError("retrieval index reconciliation required")

        source_id = "source-api-delete"
        self.repository.add_knowledge_source("delete.txt", "TXT", "internal")
        self.repository._state.knowledge_sources[0].id = source_id
        queue = KnowledgeIngestionQueue(
            self.repository,
            index_lifecycle=FailingDeleteLifecycle(),
        )
        client = TestClient(
            create_app(
                repository=self.repository,
                upload_dir=Path(self.temp_dir.name),
                ingestion_queue=queue,
            )
        )

        response = client.delete(f"/api/knowledge/sources/{source_id}")

        self.assertEqual(response.status_code, 503)
        self.assertIn("reconciliation", response.json()["detail"])
        self.assertTrue(
            any(source.id == source_id for source in self.repository.list_knowledge_sources())
        )

    def test_delete_endpoint_commits_postgres_inside_source_maintenance_fence(self) -> None:
        events: list[str] = []

        class FencedDeleteLifecycle:
            def delete_source(inner_self, source_id, *, finalize=None):
                events.append("qdrant-delete")
                self.assertIsNotNone(finalize)
                result = finalize()
                events.append("fence-release")
                return result

        source_id = "source-api-fenced-delete"
        self.repository.add_knowledge_source("delete.txt", "TXT", "internal")
        self.repository._state.knowledge_sources[0].id = source_id
        original_delete = self.repository.delete_knowledge_source

        def delete_from_postgres(target_source_id):
            events.append("postgres-delete")
            return original_delete(target_source_id)

        self.repository.delete_knowledge_source = delete_from_postgres
        queue = KnowledgeIngestionQueue(
            self.repository,
            index_lifecycle=FencedDeleteLifecycle(),
        )
        client = TestClient(
            create_app(
                repository=self.repository,
                upload_dir=Path(self.temp_dir.name),
                ingestion_queue=queue,
            )
        )

        response = client.delete(f"/api/knowledge/sources/{source_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(events, ["qdrant-delete", "postgres-delete", "fence-release"])

    def test_reindex_qdrant_failure_preserves_old_postgres_chunks(self) -> None:
        class FailingReindexLifecycle:
            def delete_source(self, source_id, *, finalize=None):
                raise RuntimeError("qdrant unavailable")

        source_id = "source-api-reindex"
        path = Path(self.temp_dir.name) / "reindex.txt"
        path.write_text("replacement content", encoding="utf-8")
        self.repository.add_uploaded_knowledge_source(
            source_id,
            "reindex.txt",
            "TXT",
            "internal",
            0,
            str(path),
            19,
            "text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            source_id,
            parse_knowledge_file(path, source_id, "TXT"),
        )
        old_chunks = self.repository.list_knowledge_chunks(source_id)
        old_status = next(
            item.status for item in self.repository.list_knowledge_sources() if item.id == source_id
        )
        queue = KnowledgeIngestionQueue(
            self.repository,
            index_lifecycle=FailingReindexLifecycle(),
        )
        client = TestClient(
            create_app(
                repository=self.repository,
                upload_dir=Path(self.temp_dir.name),
                ingestion_queue=queue,
            )
        )

        response = client.post(f"/api/knowledge/sources/{source_id}/reindex")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.repository.list_knowledge_chunks(source_id), old_chunks)
        source = next(
            item for item in self.repository.list_knowledge_sources() if item.id == source_id
        )
        self.assertEqual(source.status, old_status)


if __name__ == "__main__":
    unittest.main()
