from __future__ import annotations

import unittest
from collections.abc import Sequence

from app.agent import (
    GREETING_REPLY,
    KnowledgeAgentTools,
    ReadOnlyKnowledgeAgent,
    is_greeting_message,
)
from app.llm import LLMRequest
from app.models import (
    ChatMessageModel,
    ChatState,
    KnowledgeChunkModel,
    KnowledgeSearchHitModel,
    KnowledgeSourceModel,
    ResponseParagraphModel,
)
from app.repository import InMemoryChatRepository
from app.word_fact_answer import WordFactAnswerService
from app.word_facts import KnowledgeFactModel, WordFactMatch, WordFactualIntent


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


def word_fact_match(
    entity: str,
    field: str,
    value: str,
) -> WordFactMatch:
    fact = KnowledgeFactModel.create(
        id=f"fact-{entity}-{field}-{value}",
        source_id="kb-people",
        chunk_id="chunk-people-1",
        entity=entity,
        field=field,
        value=value,
        confidence=0.98,
        locator={"paragraph": 1},
    )
    return WordFactMatch(
        fact=fact,
        source_name="people.docx",
        classification="内部",
    )


class FakeFacts:
    def __init__(self, matches: Sequence[WordFactMatch]) -> None:
        self._matches = list(matches)

    def find_knowledge_facts(
        self,
        _intent: WordFactualIntent,
        *,
        permission_tags: Sequence[str] = (),
    ) -> list[WordFactMatch]:
        del permission_tags
        return list(self._matches)


class RecordingRouteProvider:
    def __init__(self) -> None:
        self.generation_calls = 0

    def generate_reply(self, _request: LLMRequest) -> ChatMessageModel:
        self.generation_calls += 1
        return ChatMessageModel(
            id=f"msg-route-{self.generation_calls}",
            role="assistant",
            time="2026-08-11 12:00:00",
            paragraphs=[ResponseParagraphModel(text="hybrid RAG answer")],
        )


class AgentTest(unittest.TestCase):
    def test_pure_greeting_builds_welcome_run_without_external_dependencies(self) -> None:
        def unexpected_call(*args: object, **kwargs: object) -> None:
            raise AssertionError("greeting must not call external dependencies")

        class UnexpectedProvider:
            def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
                raise AssertionError("greeting must not call external dependencies")

        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=unexpected_call,
                inspect_document=unexpected_call,
            ),
            llm_provider=UnexpectedProvider(),
        )

        result = agent.try_answer_greeting(
            conversation_id="conv-greeting",
            content="  您好！ ",
            mode="quick",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.reply.paragraphs[0].text, GREETING_REPLY)
        self.assertEqual(result.evidence_count, 0)
        self.assertEqual(result.source_count, 0)
        self.assertEqual([step.tool_name for step in result.steps], ["respond_greeting"])
        self.assertTrue(result.steps[0].read_only)

    def test_supported_greeting_phrases_are_recognized(self) -> None:
        for greeting in (
            "你好",
            "您好",
            "嗨",
            "哈喽",
            "在吗",
            "你在吗",
            "你是谁",
            "介绍一下你自己",
        ):
            with self.subTest(greeting=greeting):
                self.assertTrue(is_greeting_message(greeting))

    def test_common_unicode_greeting_punctuation_is_ignored(self) -> None:
        for greeting in (
            "您好…",
            "您好～～",
            "您好~",
            "您好；",
            "您好：",
            "“您好”",
            "您好......",
        ):
            with self.subTest(greeting=greeting):
                self.assertTrue(is_greeting_message(greeting))

    def test_non_punctuation_symbols_remain_significant(self) -> None:
        for greeting in ("您好😀", "您好＋", "您好$", "您好©"):
            with self.subTest(greeting=greeting):
                self.assertFalse(is_greeting_message(greeting))

    def test_greeting_with_substantive_question_falls_through(self) -> None:
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=lambda query, limit, routing_key: [],
                inspect_document=lambda source_id: [],
            ),
            llm_provider=RecordingProvider(),
        )

        result = agent.try_answer_greeting(
            conversation_id="conv-greeting",
            content="“你好”，请问报销制度是什么",
            mode="quick",
        )

        self.assertIsNone(result)

    def test_deep_mode_retries_with_expanded_query_when_first_search_is_weak(self) -> None:
        policy = source("kb-policy", "差旅制度.txt")
        finance = source("kb-finance", "财务规则.txt")
        policy_chunk = chunk("kb-policy", 0, "差旅申请需要审批。")
        finance_chunk = chunk("kb-finance", 0, "票据材料包括发票、行程单和审批记录。")
        search_calls: list[str] = []

        routing_keys: list[str] = []

        def search(
            query: str,
            limit: int,
            routing_key: str,
        ) -> list[KnowledgeSearchHitModel]:
            search_calls.append(query)
            routing_keys.append(routing_key)
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
        self.assertEqual(routing_keys, ["conv-agent", "conv-agent"])
        self.assertNotEqual(search_calls[0], search_calls[1])
        self.assertEqual(result.reply.id, "msg-agent-answer")
        self.assertIsNotNone(provider.request)
        self.assertEqual(
            {item.source.id for item in provider.request.knowledge_hits},
            {"kb-policy", "kb-finance"},
        )
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
                search_knowledge=lambda query, limit, routing_key: [policy_hit, finance_hit],
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
                search_knowledge=lambda query, limit, routing_key: [],
                inspect_document=lambda source_id: self.fail(
                    "empty search must not inspect a document"
                ),
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
        self.assertEqual(
            [step.tool_name for step in result.steps],
            ["plan_retrieval", "search_knowledge", "compose_answer"],
        )
        self.assertIsNotNone(provider.request)
        self.assertEqual(provider.request.knowledge_hits, [])


class WordFactRouteRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recording_llm = RecordingRouteProvider()
        self.fact_service = WordFactAnswerService(
            FakeFacts([word_fact_match("张三", "年龄", "28岁")])
        )
        self.search_calls = 0
        self.inspect_calls = 0

    def build_repository(
        self,
        *,
        word_fact_service: WordFactAnswerService | None = None,
    ) -> tuple[InMemoryChatRepository, str]:
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[]),
            llm_provider=self.recording_llm,
            word_fact_service=word_fact_service,
        )

        def search(
            _query: str,
            _limit: int,
            _routing_key: str,
        ) -> list[KnowledgeSearchHitModel]:
            self.search_calls += 1
            return []

        def inspect(_source_id: str) -> list[KnowledgeChunkModel]:
            self.inspect_calls += 1
            return []

        repository._agent = ReadOnlyKnowledgeAgent(
            KnowledgeAgentTools(search_knowledge=search, inspect_document=inspect),
            self.recording_llm,
        )
        _, conversation_id, _ = repository.create_conversation()
        return repository, conversation_id

    def test_fact_route_precedes_agent_and_llm(self) -> None:
        repository, conversation_id = self.build_repository(
            word_fact_service=self.fact_service
        )

        _, _, messages = repository.send_message(conversation_id, "张三几岁", "deep")

        self.assertEqual(messages[-1].paragraphs[0].text, "张三的年龄是28岁。")
        self.assertEqual(self.recording_llm.generation_calls, 0)
        self.assertEqual(self.search_calls, 0)
        self.assertEqual(self.inspect_calls, 0)

    def test_open_word_question_continues_to_hybrid_rag(self) -> None:
        repository, conversation_id = self.build_repository(
            word_fact_service=self.fact_service
        )

        repository.send_message(conversation_id, "介绍张三", "deep")

        self.assertGreater(self.search_calls, 0)
        self.assertEqual(self.recording_llm.generation_calls, 1)

    def test_missing_fact_is_terminal_and_never_inspects_document(self) -> None:
        repository, conversation_id = self.build_repository(
            word_fact_service=WordFactAnswerService(FakeFacts([]))
        )

        repository.send_message(conversation_id, "张三几岁", "deep")

        self.assertEqual(self.search_calls, 0)
        self.assertEqual(self.inspect_calls, 0)


if __name__ == "__main__":
    unittest.main()
