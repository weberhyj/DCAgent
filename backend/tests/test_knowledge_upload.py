from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.repository import InMemoryChatRepository
from app.seed import build_seed_state


class KnowledgeUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        repository = InMemoryChatRepository(build_seed_state())
        self.client = TestClient(create_app(repository=repository, upload_dir=Path(self.temp_dir.name)))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_uploads_file_and_registers_pending_source_then_indexes_on_refresh(self) -> None:
        response = self.client.post(
            "/api/knowledge/uploads",
            data={"classification": "内部·机密"},
            files={"file": ("董事会纪要.pdf", b"quarterly board memo" * 20, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        source = response.json()[0]
        self.assertEqual(source["name"], "董事会纪要.pdf")
        self.assertEqual(source["sourceType"], "PDF")
        self.assertEqual(source["classification"], "内部·机密")
        self.assertEqual(source["status"], "解析中")
        self.assertEqual(source["records"], 0)
        self.assertGreater(source["fileSize"], 0)
        self.assertEqual(source["mimeType"], "application/pdf")

        stored_files = list(Path(self.temp_dir.name).glob("*.pdf"))
        self.assertEqual(len(stored_files), 1)
        self.assertEqual(stored_files[0].read_bytes(), b"quarterly board memo" * 20)

        indexed = self.client.get("/api/knowledge/sources").json()[0]
        self.assertEqual(indexed["id"], source["id"])
        self.assertEqual(indexed["status"], "已索引")
        self.assertGreater(indexed["records"], 0)

    def test_uploads_multiple_files_in_one_request(self) -> None:
        response = self.client.post(
            "/api/knowledge/uploads",
            data={"classification": "内部"},
            files=[
                ("files", ("policy-a.txt", b"policy a content" * 20, "text/plain")),
                ("files", ("policy-b.md", b"policy b content" * 20, "text/markdown")),
            ],
        )

        self.assertEqual(response.status_code, 200)
        sources_by_name = {source["name"]: source for source in response.json()}
        self.assertEqual(sources_by_name["policy-a.txt"]["status"], "解析中")
        self.assertEqual(sources_by_name["policy-b.md"]["status"], "解析中")
        self.assertEqual(sources_by_name["policy-a.txt"]["classification"], "内部")
        self.assertEqual(sources_by_name["policy-b.md"]["classification"], "内部")

        stored_files = sorted(path.suffix for path in Path(self.temp_dir.name).iterdir())
        self.assertEqual(stored_files, [".md", ".txt"])

    def test_deletes_uploaded_source_chunks_and_file(self) -> None:
        response = self.client.post(
            "/api/knowledge/uploads",
            data={"classification": "内部"},
            files={"file": ("policy.txt", b"travel policy approval flow" * 20, "text/plain")},
        )
        source = response.json()[0]
        stored_file = next(Path(self.temp_dir.name).glob("*.txt"))

        indexed = self.client.get("/api/knowledge/sources").json()[0]
        self.assertEqual(indexed["id"], source["id"])
        chunks_response = self.client.get(f"/api/knowledge/sources/{source['id']}/chunks")
        self.assertGreater(len(chunks_response.json()), 0)

        delete_response = self.client.delete(f"/api/knowledge/sources/{source['id']}")

        self.assertEqual(delete_response.status_code, 200)
        self.assertNotIn(source["id"], [item["id"] for item in delete_response.json()])
        self.assertFalse(stored_file.exists())
        missing_chunks = self.client.get(f"/api/knowledge/sources/{source['id']}/chunks")
        self.assertEqual(missing_chunks.status_code, 404)

    def test_deletes_pending_uploaded_source_without_indexing_it_later(self) -> None:
        response = self.client.post(
            "/api/knowledge/uploads",
            data={"classification": "内部"},
            files={"file": ("pending-policy.txt", b"pending policy" * 20, "text/plain")},
        )
        source = response.json()[0]
        stored_file = next(Path(self.temp_dir.name).glob("*.txt"))

        delete_response = self.client.delete(f"/api/knowledge/sources/{source['id']}")
        refreshed = self.client.get("/api/knowledge/sources")

        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(refreshed.status_code, 200)
        self.assertNotIn(source["id"], [item["id"] for item in refreshed.json()])
        self.assertFalse(stored_file.exists())

    def test_rejects_unsupported_upload_type(self) -> None:
        response = self.client.post(
            "/api/knowledge/uploads",
            data={"classification": "内部"},
            files={"file": ("script.exe", b"binary", "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
