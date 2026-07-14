from __future__ import annotations

import os
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.llm import LLMProviderError, LLMRequest
from app.main import create_app
from app.models import (
    ChatMessageModel,
    ChatState,
    CitationModel,
    ConversationModel,
    KnowledgeChunkModel,
    ResponseParagraphModel,
)
from app.repository import InMemoryChatRepository
from app.seed import build_seed_state


DISPLAY_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def assert_display_timestamp(test_case: unittest.TestCase, value: str) -> None:
    test_case.assertRegex(value, DISPLAY_TIMESTAMP_PATTERN)


class ApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryChatRepository(build_seed_state())
        self.client = TestClient(create_app(self.repository))

    def test_lists_conversations_with_frontend_contract_fields(self) -> None:
        response = self.client.get("/api/conversations")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("activeConversationId", payload)
        self.assertIn("conversations", payload)
        self.assertIn("messages", payload)
        self.assertGreaterEqual(len(payload["conversations"]), 1)

        conversation = payload["conversations"][0]
        self.assertEqual(
            set(conversation),
            {"id", "title", "topic", "group", "updatedAt", "pinned"},
        )
        for item in payload["conversations"]:
            assert_display_timestamp(self, item["updatedAt"])

        assistant_message = next(
            message for message in payload["messages"] if message["role"] == "assistant"
        )
        for message in payload["messages"]:
            assert_display_timestamp(self, message["time"])
        self.assertEqual(assistant_message["paragraphs"][0]["citations"], [])

    def test_empty_repository_lists_blank_search_without_seed_data(self) -> None:
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        client = TestClient(create_app(repository))

        response = client.get("/api/conversations")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["messages"], [])
        self.assertEqual(len(payload["conversations"]), 1)
        self.assertEqual(payload["activeConversationId"], payload["conversations"][0]["id"])
        self.assertNotEqual(payload["activeConversationId"], "conv-q4")
        self.assertNotIn("Q4", str(payload))

    def test_user_api_removes_inline_citation_numbers_from_saved_answers(self) -> None:
        conversation = ConversationModel(
            id="conv-citations",
            title="引用编号清理",
            topic="知识检索",
            group="今天",
            updated_at="2026-07-14 10:00:00",
        )
        assistant = ChatMessageModel(
            id="msg-citations",
            role="assistant",
            time="2026-07-14 10:00:01",
            paragraphs=[
                ResponseParagraphModel(
                    text="第一条结论[1]，第二条结论 [2]。",
                    citations=[
                        CitationModel(
                            label="[1] 内部资料",
                            classification="内部",
                            source_id="source-1",
                        )
                    ],
                )
            ],
        )
        repository = InMemoryChatRepository(
            ChatState(
                conversations=[conversation],
                messages_by_conversation={conversation.id: [assistant]},
                knowledge_sources=[],
            )
        )
        client = TestClient(create_app(repository))

        response = client.get("/api/conversations")

        self.assertEqual(response.status_code, 200)
        paragraph = response.json()["messages"][0]["paragraphs"][0]
        self.assertEqual(paragraph["text"], "第一条结论，第二条结论。")
        self.assertEqual(paragraph["citations"], [])

    def test_default_app_does_not_seed_demo_conversations(self) -> None:
        with patch.dict(
            os.environ,
            {"DATABASE_URL": "sqlite+pysqlite:///:memory:", "LLM_PROVIDER": "template"},
        ):
            client = TestClient(create_app())

            response = client.get("/api/conversations")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["messages"], [])
        self.assertEqual(len(payload["conversations"]), 1)
        self.assertNotEqual(payload["activeConversationId"], "conv-q4")
        self.assertNotIn("Q4", str(payload))

    def test_admin_frontend_origin_is_allowed_for_local_development(self) -> None:
        response = self.client.options(
            "/api/knowledge/sources",
            headers={
                "Origin": "http://127.0.0.1:5174",
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:5174")

    def test_creates_empty_conversation_and_saves_first_exchange(self) -> None:
        create_response = self.client.post("/api/conversations")

        self.assertEqual(create_response.status_code, 200)
        created = create_response.json()
        conversation_id = created["activeConversationId"]
        self.assertEqual(created["messages"], [])
        self.assertEqual(created["conversations"][0]["id"], conversation_id)
        assert_display_timestamp(self, created["conversations"][0]["updatedAt"])
        self.assertEqual(created["conversations"][0]["title"], "未命名搜查档案")

        send_response = self.client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "分析一下本季度现金流风险", "mode": "deep"},
        )

        self.assertEqual(send_response.status_code, 200)
        sent = send_response.json()
        self.assertEqual(sent["activeConversationId"], conversation_id)
        self.assertEqual(sent["conversations"][0]["id"], conversation_id)
        assert_display_timestamp(self, sent["conversations"][0]["updatedAt"])
        self.assertEqual(sent["conversations"][0]["title"], "分析一下本季度现金流风险")
        self.assertEqual([message["role"] for message in sent["messages"]], ["user", "assistant"])
        for message in sent["messages"]:
            assert_display_timestamp(self, message["time"])
        self.assertEqual(sent["messages"][0]["content"], "分析一下本季度现金流风险")
        self.assertGreaterEqual(len(sent["messages"][1]["paragraphs"]), 1)
        self.assertIn("未检索到足够依据", sent["messages"][1]["paragraphs"][0]["text"])
        self.assertEqual(sent["messages"][1]["artifacts"], [])
        for artifact in sent["messages"][1]["artifacts"]:
            self.assertEqual(artifact.get("source"), "")

    def test_rejects_blank_message_content(self) -> None:
        conversation_id = self.client.post("/api/conversations").json()["activeConversationId"]

        response = self.client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "   ", "mode": "deep"},
        )

        self.assertEqual(response.status_code, 422)

    def test_model_failure_returns_user_safe_gateway_error(self) -> None:
        class FailingLLMProvider:
            def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
                raise LLMProviderError("大模型服务暂时不可用，请稍后重试。")

        repository = InMemoryChatRepository(build_seed_state(), llm_provider=FailingLLMProvider())
        client = TestClient(create_app(repository=repository), raise_server_exceptions=False)
        conversation_id = client.post("/api/conversations").json()["activeConversationId"]

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "请查询差旅制度", "mode": "deep"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "大模型服务暂时不可用，请稍后重试。",
        )

    def test_deletes_conversation_and_returns_next_active_bundle(self) -> None:
        created_id = self.client.post("/api/conversations").json()["activeConversationId"]

        response = self.client.delete(f"/api/conversations/{created_id}")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotEqual(payload["activeConversationId"], created_id)
        self.assertNotIn(created_id, [item["id"] for item in payload["conversations"]])

    def test_adds_knowledge_source_with_ingestion_status(self) -> None:
        response = self.client.post(
            "/api/knowledge/sources",
            json={
                "name": "董事会纪要.pdf",
                "sourceType": "PDF",
                "classification": "内部·机密",
            },
        )

        self.assertEqual(response.status_code, 200)
        source = response.json()[0]
        self.assertEqual(source["name"], "董事会纪要.pdf")
        self.assertEqual(source["sourceType"], "PDF")
        self.assertEqual(source["classification"], "内部·机密")
        self.assertEqual(source["status"], "解析中")
        self.assertEqual(source["records"], 0)
        assert_display_timestamp(self, source["updatedAt"])

    def test_assistant_reply_uses_indexed_knowledge_without_exposing_sources(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-cashflow",
            name="cashflow-note.txt",
            source_type="文档",
            classification="内部·机密",
            records=0,
            file_path="cashflow-note.txt",
            file_size=128,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "kb-cashflow",
            [
                KnowledgeChunkModel(
                    id="chunk-cashflow-0",
                    source_id="kb-cashflow",
                    chunk_index=0,
                    text="现金流风险来自回款周期拉长和应收账款增加。",
                    token_count=31,
                )
            ],
        )
        conversation_id = self.client.post("/api/conversations").json()["activeConversationId"]

        response = self.client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "请分析现金流风险", "mode": "source"},
        )

        self.assertEqual(response.status_code, 200)
        assistant = response.json()["messages"][-1]
        self.assertEqual(len(assistant["paragraphs"]), 1)
        self.assertEqual(assistant["artifacts"], [])
        self.assertNotIn("Q4", str(assistant))
        self.assertNotIn("已按", str(assistant))
        for paragraph in assistant["paragraphs"]:
            self.assertEqual(paragraph["citations"], [])
            paragraph_payload = str(paragraph)
            self.assertNotIn("sourceId", paragraph_payload)
            self.assertNotIn("sourceName", paragraph_payload)
            self.assertNotIn("chunkId", paragraph_payload)
            self.assertNotIn("chunkIndex", paragraph_payload)
            self.assertNotIn("matchedTerms", paragraph_payload)
            self.assertNotIn("excerpt", paragraph_payload)
            self.assertNotIn("cashflow-note.txt", paragraph_payload)
            self.assertNotIn("chunk-cashflow-0", paragraph_payload)
        self.assertIn("现金流", assistant["paragraphs"][0]["text"])

    def test_admin_knowledge_chunk_endpoint_keeps_source_text_available(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-admin",
            name="policy.txt",
            source_type="文档",
            classification="内部",
            records=0,
            file_path="policy.txt",
            file_size=128,
            mime_type="text/plain",
        )
        self.repository.complete_knowledge_source_indexing(
            "kb-admin",
            [
                KnowledgeChunkModel(
                    id="chunk-admin-0",
                    source_id="kb-admin",
                    chunk_index=0,
                    text="管理员可以查看这段原始资料。",
                    token_count=18,
                )
            ],
        )

        response = self.client.get("/api/knowledge/sources/kb-admin/chunks")

        self.assertEqual(response.status_code, 200)
        chunk = response.json()[0]
        self.assertEqual(chunk["sourceId"], "kb-admin")
        self.assertEqual(chunk["id"], "chunk-admin-0")
        self.assertIn("原始资料", chunk["text"])

    def test_admin_sources_include_parse_failure_reason(self) -> None:
        self.repository.add_uploaded_knowledge_source(
            source_id="kb-failed",
            name="broken-report.txt",
            source_type="文档",
            classification="内部",
            records=0,
            file_path="broken-report.txt",
            file_size=64,
            mime_type="text/plain",
        )
        self.repository.fail_knowledge_source_indexing(
            "kb-failed",
            "文件内容无法解析",
        )

        response = self.client.get("/api/knowledge/sources")

        self.assertEqual(response.status_code, 200)
        failed = next(source for source in response.json() if source["id"] == "kb-failed")
        self.assertEqual(failed["status"], "解析失败")
        self.assertEqual(failed["errorMessage"], "文件内容无法解析")

    def test_reindex_failed_source_resets_status_and_queues_ingestion(self) -> None:
        class RecordingIngestionQueue:
            def __init__(self) -> None:
                self.jobs: list[tuple[str, str, str]] = []

            def enqueue(self, source_id: str, file_path: str, source_type: str) -> None:
                self.jobs.append((source_id, file_path, source_type))

            def discard_source(self, source_id: str) -> None:
                self.jobs = [job for job in self.jobs if job[0] != source_id]

            def drain(self) -> None:
                return None

        queue = RecordingIngestionQueue()
        repository = InMemoryChatRepository(build_seed_state())
        client = TestClient(create_app(repository=repository, ingestion_queue=queue))
        repository.add_uploaded_knowledge_source(
            source_id="kb-retry",
            name="retry-policy.txt",
            source_type="文档",
            classification="内部",
            records=0,
            file_path="retry-policy.txt",
            file_size=128,
            mime_type="text/plain",
        )
        repository.fail_knowledge_source_indexing("kb-retry", "首次解析失败")

        response = client.post("/api/knowledge/sources/kb-retry/reindex")

        self.assertEqual(response.status_code, 200)
        retried = next(source for source in response.json() if source["id"] == "kb-retry")
        self.assertEqual(retried["status"], "解析中")
        self.assertIsNone(retried["errorMessage"])
        self.assertEqual(queue.jobs, [("kb-retry", "retry-policy.txt", "文档")])


if __name__ == "__main__":
    unittest.main()
