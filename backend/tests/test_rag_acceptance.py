from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.llm import LLMProviderError
from app.main import create_app
from app.models import ChatState
from app.repository import InMemoryChatRepository


class RagAcceptanceTest(unittest.TestCase):
    def test_model_failure_returns_explicit_unavailable_without_raw_retrieval_chunks(self) -> None:
        raw_chunk = "CONFIDENTIAL raw retrieved payroll passage"
        internal_url = "http://physoc.internal.example/private"

        class FailingPhysocProvider:
            def generate_reply(self, request):
                raise LLMProviderError("大模型服务暂时不可用，请稍后重试。")

        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[]),
            llm_provider=FailingPhysocProvider(),
        )
        client = TestClient(create_app(repository=repository, upload_dir=Path(self.temp_dir.name)))
        upload = client.post(
            "/api/knowledge/uploads",
            data={"classification": "internal"},
            files={"file": ("policy.txt", raw_chunk.encode(), "text/plain")},
        )
        self.assertEqual(upload.status_code, 200)
        conversation_id = client.post("/api/conversations").json()["activeConversationId"]

        with patch("app.routes.logger"):
            response = client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"content": "payroll policy", "mode": "source"},
            )

        self.assertEqual(response.status_code, 502)
        serialized = response.text
        self.assertIn("大模型服务暂时不可用", serialized)
        self.assertNotIn(raw_chunk, serialized)
        self.assertNotIn(internal_url, serialized)

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        self.client = TestClient(
            create_app(repository=repository, upload_dir=Path(self.temp_dir.name))
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_upload_index_and_user_question_returns_grounded_answer_without_source_internals(
        self,
    ) -> None:
        policy_text = (
            "差旅报销制度规定：员工出差前必须先提交审批单。"
            "返程后需要在五个工作日内上传发票、行程单和审批记录。"
        )
        upload_response = self.client.post(
            "/api/knowledge/uploads",
            data={"classification": "内部·机密"},
            files={"file": ("travel-policy.txt", policy_text.encode("utf-8"), "text/plain")},
        )
        self.assertEqual(upload_response.status_code, 200)
        source = upload_response.json()[0]
        self.assertEqual(source["status"], "解析中")

        indexed_sources = self.client.get("/api/knowledge/sources").json()
        indexed = next(item for item in indexed_sources if item["id"] == source["id"])
        self.assertEqual(indexed["status"], "已索引")
        self.assertGreater(indexed["records"], 0)

        conversation_id = self.client.post("/api/conversations").json()["activeConversationId"]
        answer_response = self.client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "差旅票据材料", "mode": "source"},
        )

        self.assertEqual(answer_response.status_code, 200)
        assistant = answer_response.json()["messages"][-1]
        answer_payload = str(assistant)
        self.assertIn("差旅报销", assistant["paragraphs"][0]["text"])
        self.assertIn("发票", assistant["paragraphs"][0]["text"])
        self.assertEqual(assistant["paragraphs"][0]["citations"], [])
        self.assertNotIn("sourceId", answer_payload)
        self.assertNotIn("chunkId", answer_payload)
        self.assertNotIn("travel-policy.txt", answer_payload)

        audit_response = self.client.get("/api/admin/agent/runs")
        self.assertEqual(audit_response.status_code, 200)
        audit_run = audit_response.json()[0]
        self.assertEqual(audit_run["conversationId"], conversation_id)
        self.assertEqual(audit_run["query"], "差旅票据材料")
        self.assertEqual(audit_run["status"], "completed")
        self.assertGreater(audit_run["evidenceCount"], 0)
        tool_names = [step["toolName"] for step in audit_run["steps"]]
        self.assertIn("search_knowledge", tool_names)
        self.assertIn("inspect_document", tool_names)
        self.assertIn("compare_evidence", tool_names)
        self.assertIn("compose_answer", tool_names)
        self.assertTrue(all(step["readOnly"] for step in audit_run["steps"]))


if __name__ == "__main__":
    unittest.main()
