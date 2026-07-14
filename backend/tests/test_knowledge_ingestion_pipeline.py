from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.ingestion import KnowledgeIngestionQueue
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
        path.write_text("\n".join([f"风险提示 {index}: 回款周期变化" for index in range(120)]), encoding="utf-8")

        chunks = parse_knowledge_file(path, source_id="kb-parser", source_type="文档")

        self.assertGreater(len(chunks), 1)
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(chunk.source_id == "kb-parser" for chunk in chunks))
        self.assertIn("风险提示", chunks[0].text)

    def test_parser_removes_nul_bytes_before_chunks_are_indexed(self) -> None:
        path = Path(self.temp_dir.name) / "nul-note.txt"
        path.write_bytes("差旅制度".encode("utf-8") + b"\x00" + "审批流程".encode("utf-8"))

        chunks = parse_knowledge_file(path, source_id="kb-nul", source_type="文档")

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all("\x00" not in chunk.text for chunk in chunks))
        self.assertIn("差旅制度", chunks[0].text)
        self.assertIn("审批流程", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
