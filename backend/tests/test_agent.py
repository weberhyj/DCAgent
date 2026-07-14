from __future__ import annotations

import unittest

from app.agent import KnowledgeAgentTools, ReadOnlyKnowledgeAgent
from app.llm import LLMRequest
from app.models import (
    ChatMessageModel,
    KnowledgeChunkModel,
    KnowledgeSearchHitModel,
    KnowledgeSourceModel,
    ResponseParagraphModel,
)


def source(source_id: str, name: str) -> KnowledgeSourceModel:
    return KnowledgeSourceModel(
        id=source_id,
        name=name,
        source_type="文档",
        records=2,
        status="已索引",
        updated_at="2026-07-10 10:00:00",
        classification="内部·机密",
    )


def chunk(source_id: str, index: int, text: str) -> KnowledgeChunkModel:
    return KnowledgeChunkModel(
        id=f"chunk-{source_id}-{index}",
        source_id=source_id,
        chunk_index=index,
        text=text,
        token_count=len(text),
    )


def hit(
    item_source: KnowledgeSourceModel,
    item_chunk: KnowledgeChunkModel,
    score: float,
) -> KnowledgeSearchHitModel:
    return KnowledgeSearchHitModel(
        source=item_source,
        chunk=item_chunk,
        score=score,
        rank=1,
        matched_terms=["票据"],
    )


class RecordingProvider:
    def __init__(self) -> None:
        self.request: LLMRequest | None = None

    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        self.request = request
        return ChatMessageModel(
            id="msg-agent-answer",
            role="assistant",
            time="2026-07-10 10:00:01",
            paragraphs=[ResponseParagraphModel(text="已完成多步调查。[1]")],
        )


class AgentTest(unittest.TestCase):
    def test_deep_mode_retries_with_expanded_query_when_first_search_is_weak(self) -> None:
        policy = source("kb-policy", "差旅制度.txt")
        finance = source("kb-finance", "财务规则.txt")
        policy_chunk = chunk("kb-policy", 0, "差旅申请需要审批。")
        finance_chunk = chunk("kb-finance", 0, "票据材料包括发票、行程单和审批记录。")
        search_calls: list[str] = []

        def search(query: str, limit: int) -> list[KnowledgeSearchHitModel]:
            search_calls.append(query)
            if len(search_calls) == 1:
                return [hit(policy, policy_chunk, 0.8)]
            return [hit(finance, finance_chunk, 8.2)]

        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=search,
                inspect_document=lambda source_id: [],
            ),
            llm_provider=provider,
        )

        result = agent.run(
            conversation_id="conv-agent",
            content="差旅票据材料需要什么",
            mode="deep",
            previous_messages=[],
        )

        self.assertEqual(len(search_calls), 2)
        self.assertNotEqual(search_calls[0], search_calls[1])
        self.assertEqual(result.reply.id, "msg-agent-answer")
        self.assertIsNotNone(provider.request)
        self.assertEqual({item.source.id for item in provider.request.knowledge_hits}, {"kb-policy", "kb-finance"})
        search_steps = [step for step in result.steps if step.tool_name == "search_knowledge"]
        self.assertEqual(len(search_steps), 2)
        self.assertTrue(all(step.read_only for step in result.steps))

    def test_agent_inspects_documents_and_compares_multiple_sources(self) -> None:
        policy = source("kb-policy", "差旅制度.txt")
        finance = source("kb-finance", "财务规则.txt")
        policy_hit = hit(policy, chunk("kb-policy", 0, "差旅材料需在五日内提交。"), 9.2)
        finance_hit = hit(finance, chunk("kb-finance", 0, "财务要求提交发票。"), 8.8)
        inspected: list[str] = []

        def inspect(source_id: str) -> list[KnowledgeChunkModel]:
            inspected.append(source_id)
            if source_id == "kb-policy":
                return [chunk(source_id, 1, "差旅材料还需要行程单和审批记录。")]
            return [chunk(source_id, 1, "缺少发票时财务会退回补充。")]

        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=lambda query, limit: [policy_hit, finance_hit],
                inspect_document=inspect,
            ),
            llm_provider=provider,
        )

        result = agent.run(
            conversation_id="conv-agent",
            content="对比差旅制度和财务票据要求",
            mode="source",
            previous_messages=[],
        )

        self.assertEqual(set(inspected), {"kb-policy", "kb-finance"})
        self.assertIn("inspect_document", [step.tool_name for step in result.steps])
        self.assertIn("compare_evidence", [step.tool_name for step in result.steps])
        self.assertIsNotNone(provider.request)
        self.assertIn("多来源", provider.request.agent_context)
        self.assertIn("差旅制度.txt", provider.request.agent_context)
        self.assertIn("财务规则.txt", provider.request.agent_context)
        self.assertGreaterEqual(len(provider.request.knowledge_hits), 2)

    def test_agent_stops_without_mutating_tools_when_no_evidence_exists(self) -> None:
        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=lambda query, limit: [],
                inspect_document=lambda source_id: self.fail("empty search must not inspect a document"),
            ),
            llm_provider=provider,
        )

        result = agent.run(
            conversation_id="conv-agent",
            content="查询不存在的内部制度",
            mode="quick",
            previous_messages=[],
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual([step.tool_name for step in result.steps], ["plan_retrieval", "search_knowledge", "compose_answer"])
        self.assertIsNotNone(provider.request)
        self.assertEqual(provider.request.knowledge_hits, [])


if __name__ == "__main__":
    unittest.main()
