from __future__ import annotations

import math
import unittest
from collections.abc import Sequence

from app.agent import (
    GREETING_REPLY,
    AgentSearchResult,
    KnowledgeAgentTools,
    ReadOnlyKnowledgeAgent,
    _has_query_overlap,
    filter_relevant_hits,
    is_follow_up_message,
    is_greeting_message,
    merge_ranked_hits,
)
from app.knowledge_route_models import KnowledgeRouteType
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
from app.retrieval_models import (
    EvidenceExpansionPolicy,
    RetrievalMode,
    RetrievalRequest,
    RetrievalScope,
)
from app.retrieval_router import RoutedRetrievalOutcome
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
    def test_history_is_only_enabled_for_context_dependent_follow_ups(self) -> None:
        previous_messages = [
            ChatMessageModel(
                id="msg-user-previous",
                role="user",
                time="2026-08-18 10:00:00",
                paragraphs=[ResponseParagraphModel(text="张三的职务是什么")],
            )
        ]
        policy = source("kb-policy", "people.docx")
        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=lambda query, limit, routing_key: AgentSearchResult(
                    hits=(hit(policy, chunk("kb-policy", 0, "张三负责财务审批。"), 0.95),)
                )
            ),
            llm_provider=provider,
        )

        agent.run(
            conversation_id="conv-history",
            content="张三负责哪些工作",
            mode="quick",
            previous_messages=previous_messages,
        )
        assert provider.request is not None
        self.assertFalse(provider.request.include_history)

        agent.run(
            conversation_id="conv-history",
            content="那他的审批范围呢？",
            mode="quick",
            previous_messages=previous_messages,
        )
        assert provider.request is not None
        self.assertTrue(provider.request.include_history)

    def test_follow_up_detection_requires_an_explicit_context_reference(self) -> None:
        independent = ("张三负责哪些工作", "报销流程是什么", "介绍一下蜘蛛侠")
        follow_ups = ("那他的职务呢？", "继续介绍", "刚才提到的制度适用谁？")

        self.assertTrue(all(not is_follow_up_message(item) for item in independent))
        self.assertTrue(all(is_follow_up_message(item) for item in follow_ups))

    def test_search_audit_records_stable_chunk_ids_and_scores(self) -> None:
        policy = source("kb-policy", "policy.docx")
        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=lambda query, limit, routing_key: AgentSearchResult(
                    hits=(hit(policy, chunk("kb-policy", 3, "差旅申请需要审批。"), 0.937654),)
                )
            ),
            llm_provider=provider,
        )

        result = agent.run(
            conversation_id="conv-audit",
            content="差旅申请如何审批",
            mode="quick",
            previous_messages=[],
        )

        search_step = next(step for step in result.steps if step.tool_name == "search_knowledge")
        self.assertIn("evidence=chunk-kb-policy-3:0.937654", search_step.output_summary)

    def test_relevance_gate_drops_low_scoring_unrelated_sources(self) -> None:
        town = source("kb-town", "test.docx")
        spider = source("kb-spider", "蜘蛛侠角色介绍.docx")
        candidates = [
            hit(town, chunk("kb-town", 0, "欧洲小镇名称源于犀牛陂的谐音。"), 0.938),
            hit(town, chunk("kb-town", 1, "欧洲小镇园区规划。"), 0.006),
            hit(town, chunk("kb-town", 2, "欧洲小镇参观须知。"), 0.0001),
            hit(spider, chunk("kb-spider", 0, "蜘蛛侠由彼得·帕克担任。"), 0.0014),
            hit(spider, chunk("kb-spider", 1, "蜘蛛侠的主要能力。"), 0.0),
        ]

        filtered = filter_relevant_hits(candidates)

        self.assertEqual([item.source.id for item in filtered], ["kb-town", "kb-town"])
        self.assertEqual([item.chunk.chunk_index for item in filtered], [0, 1])
        self.assertEqual([item.rank for item in filtered], [1, 2])

    def test_relevance_gate_keeps_independently_relevant_sources(self) -> None:
        policy = source("kb-policy", "差旅制度.txt")
        finance = source("kb-finance", "财务规则.txt")

        filtered = filter_relevant_hits(
            [
                hit(policy, chunk("kb-policy", 0, "差旅申请需要审批。"), 0.91),
                hit(finance, chunk("kb-finance", 0, "票据需要提交发票。"), 0.82),
            ]
        )

        self.assertEqual(
            {item.source.id for item in filtered},
            {"kb-policy", "kb-finance"},
        )

    def test_relevance_gate_reports_filter_diagnostics(self) -> None:
        town = source("kb-town", "test.docx")
        spider = source("kb-spider", "蜘蛛侠角色介绍.docx")
        diagnostics: dict[str, object] = {}

        filtered = filter_relevant_hits(
            [
                hit(town, chunk("kb-town", 0, "欧洲小镇名称源于犀牛陂的谐音。"), 0.938),
                hit(town, chunk("kb-town", 1, "欧洲小镇园区规划。"), 0.006),
                hit(spider, chunk("kb-spider", 0, "蜘蛛侠角色起源。"), 0.0014),
            ],
            diagnostics=diagnostics,
        )

        self.assertEqual(len(filtered), 2)
        self.assertEqual(diagnostics["candidate_count"], 3)
        self.assertEqual(diagnostics["filtered_count"], 2)
        self.assertEqual(diagnostics["dropped_by_score_or_adjacency"], 1)
        self.assertGreater(float(diagnostics["cutoff"]), 0.0)
        self.assertNotIn("reason", diagnostics)

    def test_relevance_gate_keeps_query_matching_top_hit_when_all_scores_are_low(self) -> None:
        spider = source("kb-spider", "蜘蛛侠角色介绍.docx")
        diagnostics: dict[str, object] = {}

        filtered = filter_relevant_hits(
            [
                hit(
                    spider,
                    chunk("kb-spider", 0, "蜘蛛侠的主要活动区域是纽约市。"),
                    0.021,
                ),
                hit(
                    spider,
                    chunk("kb-spider", 1, "蜘蛛侠的战斗方式和装备。"),
                    0.004,
                ),
            ],
            query="蜘蛛侠的主要活动区域是什么？",
            diagnostics=diagnostics,
        )

        self.assertEqual([item.chunk.chunk_index for item in filtered], [0, 1])
        self.assertEqual(diagnostics["reason"], "low_score_top_candidate_retained")
        self.assertEqual(diagnostics["configured_cutoff"], 0.03)
        self.assertEqual(diagnostics["cutoff"], 0.021)

    def test_relevance_gate_does_not_keep_low_score_without_query_overlap(self) -> None:
        source_model = source("kb-unrelated", "unrelated.docx")
        diagnostics: dict[str, object] = {}

        filtered = filter_relevant_hits(
            [
                hit(
                    source_model,
                    chunk("kb-unrelated", 0, "财务报销需要提交发票。"),
                    0.021,
                ),
            ],
            query="蜘蛛侠的主要活动区域是什么？",
            diagnostics=diagnostics,
        )

        self.assertEqual(filtered, [])
        self.assertEqual(diagnostics["reason"], "all_candidates_below_cutoff")

    def test_query_overlap_accepts_document_field_synonym(self) -> None:
        self.assertTrue(
            _has_query_overlap(
                "蜘蛛侠的位置是什么？",
                "蜘蛛侠的主要活动区域是纽约市。",
            )
        )

    def test_query_overlap_accepts_natural_age_question_forms(self) -> None:
        people = source("kb-people", "people.docx")
        age_chunk = chunk("kb-people", 0, "张三年龄：30岁。")
        for question in (
            "张三有多少岁",
            "张三多少岁",
            "张三今年多少岁",
            "张三现在多少岁",
            "张三现年多少岁",
        ):
            with self.subTest(question=question):
                diagnostics: dict[str, object] = {}
                filtered = filter_relevant_hits(
                    [hit(people, age_chunk, 0.02)],
                    query=question,
                    diagnostics=diagnostics,
                )
                self.assertEqual([item.chunk.id for item in filtered], [age_chunk.id])

    def test_query_overlap_rejects_same_subject_with_wrong_field(self) -> None:
        self.assertFalse(
            _has_query_overlap(
                "蜘蛛侠的年龄是多少？",
                "蜘蛛侠的性别：男。",
            )
        )

    def test_relevance_gate_filters_non_finite_scores(self) -> None:
        people = source("kb-people", "people.docx")
        diagnostics: dict[str, object] = {}
        filtered = filter_relevant_hits(
            [
                hit(people, chunk("kb-people", 0, "张三的年龄是30岁。"), math.nan),
                hit(people, chunk("kb-people", 1, "张三的年龄是30岁。"), 0.8),
            ],
            query="张三的年龄是多少？",
            diagnostics=diagnostics,
        )

        self.assertEqual([item.chunk.chunk_index for item in filtered], [1])
        self.assertEqual(diagnostics["invalid_score_count"], 1)
        self.assertEqual(diagnostics["candidate_count_after_score_validation"], 1)

    def test_relevance_gate_drops_adjacent_chunk_with_conflicting_field(self) -> None:
        people = source("kb-people", "people.docx")
        filtered = filter_relevant_hits(
            [
                hit(people, chunk("kb-people", 0, "蜘蛛侠年龄：30岁。"), 0.9),
                hit(people, chunk("kb-people", 1, "蜘蛛侠性别：男。"), 0.8),
            ],
            query="蜘蛛侠年龄是多少？",
            adjacency_distance=1,
        )
        self.assertEqual([item.chunk.chunk_index for item in filtered], [0])

    def test_relevance_gate_fails_closed_when_all_scores_are_non_finite(self) -> None:
        people = source("kb-people", "people.docx")
        diagnostics: dict[str, object] = {}
        filtered = filter_relevant_hits(
            [
                hit(people, chunk("kb-people", 0, "张三的年龄是30岁。"), math.nan),
                hit(people, chunk("kb-people", 1, "张三的年龄是30岁。"), math.inf),
            ],
            query="张三的年龄是多少？",
            diagnostics=diagnostics,
        )

        self.assertEqual(filtered, [])
        self.assertEqual(diagnostics["invalid_score_count"], 2)
        self.assertEqual(diagnostics["reason"], "invalid_scores_filtered")
        self.assertEqual(diagnostics["filtered_count"], 0)

    def test_merge_ranked_hits_drops_non_finite_scores(self) -> None:
        people = source("kb-people", "people.docx")
        merged = merge_ranked_hits(
            [hit(people, chunk("kb-people", 0, "有效"), math.nan)],
            [
                hit(people, chunk("kb-people", 1, "有效年龄"), 0.8),
                hit(people, chunk("kb-people", 2, "无效"), math.inf),
            ],
            limit=5,
        )
        self.assertEqual([item.chunk.chunk_index for item in merged], [1])

    def test_query_overlap_rejects_field_only_match_for_wrong_subject(self) -> None:
        self.assertFalse(
            _has_query_overlap(
                "蜘蛛侠的位置是什么？",
                "蝙蝠侠的主要活动区域是哥谭市。",
            )
        )

    def test_query_overlap_allows_field_only_question(self) -> None:
        self.assertTrue(
            _has_query_overlap(
                "位置是什么？",
                "主要活动区域是纽约市。",
            )
        )

    def test_query_overlap_does_not_use_filename_as_field_evidence(self) -> None:
        self.assertFalse(
            _has_query_overlap(
                "位置是什么？",
                "销售额 | 100",
                source_text="地址簿.xlsx",
            )
        )

    def test_query_overlap_keeps_aggregate_metric_families_separate(self) -> None:
        self.assertFalse(_has_query_overlap("平均温度", "最低温度 | 10"))
        self.assertTrue(_has_query_overlap("平均温度", "温度 | 10"))

    def test_query_overlap_can_use_filename_for_entity_only(self) -> None:
        self.assertTrue(
            _has_query_overlap(
                "多伦多的平均温度",
                "[每分钟温度]\ntoronto_edt | 19.7",
                source_text="多伦多_天气.xlsx",
                source_type="XLSX",
            )
        )

    def test_query_overlap_allows_headerless_tabular_row_without_conflicting_label(self) -> None:
        self.assertTrue(
            _has_query_overlap(
                "多伦多的平均温度",
                "toronto_edt | 2026-08-16T00:00:00 | 19.7",
                source_text="多伦多_天气.xlsx",
                source_type="XLSX",
            )
        )

    def test_query_overlap_rejects_tabular_row_with_conflicting_field_label(self) -> None:
        self.assertFalse(
            _has_query_overlap(
                "多伦多的平均温度",
                "多伦多 | 最低温度 | 10",
                source_text="多伦多_天气.xlsx",
                source_type="XLSX",
            )
        )

    def test_query_overlap_accepts_narrative_subject_without_literal_field_label(self) -> None:
        self.assertTrue(
            _has_query_overlap(
                "蜘蛛侠的位置是什么？",
                "蜘蛛侠常在纽约活动。",
            )
        )

    def test_query_overlap_accepts_location_semantic_phrases(self) -> None:
        for text in (
            "蜘蛛侠位于纽约市。",
            "蜘蛛侠的主要活动区域是纽约市。",
        ):
            with self.subTest(text=text):
                self.assertTrue(_has_query_overlap("蜘蛛侠的位置是什么？", text))

    def test_query_overlap_rejects_bare_activity_or_department_as_location(self) -> None:
        for text in (
            "蜘蛛侠参加了公益活动。",
            "蜘蛛侠所在部门是英雄联盟。",
            "蜘蛛侠负责活动策划。",
        ):
            with self.subTest(text=text):
                self.assertFalse(_has_query_overlap("蜘蛛侠的位置是什么？", text))

    def test_query_overlap_does_not_use_narrative_filename_as_entity(self) -> None:
        self.assertFalse(
            _has_query_overlap(
                "蜘蛛侠的位置是什么？",
                "蝙蝠侠常在哥谭活动。",
                source_text="蜘蛛侠资料.docx",
                source_type="DOCX",
            )
        )

    def test_low_score_guard_can_use_source_filename_for_tabular_entity(self) -> None:
        weather = source("kb-weather", "多伦多_2026-08-16_每分钟天气温度.xlsx")
        hit = KnowledgeSearchHitModel(
            source=weather,
            chunk=KnowledgeChunkModel(
                id="weather-row",
                source_id=weather.id,
                chunk_index=0,
                text="[每分钟温度]\ntoronto_edt | 2026-08-16T00:00:00 | 19.7",
                token_count=20,
            ),
            score=0.006,
            rank=1,
        )

        filtered = filter_relevant_hits(
            [hit],
            query="多伦多在2026年8月16日00:00到1:00这段时间的平均温度",
        )

        self.assertEqual([item.chunk.id for item in filtered], ["weather-row"])

    def test_explicit_file_reference_scopes_candidates_to_named_source(self) -> None:
        named = source("kb-named", "蜘蛛侠资料.docx")
        unrelated = source("kb-other", "蝙蝠侠资料.docx")
        filtered = filter_relevant_hits(
            [
                hit(
                    unrelated,
                    chunk("kb-other", 0, "蝙蝠侠的主要活动区域是哥谭市。"),
                    0.9,
                ),
                hit(
                    named,
                    chunk("kb-named", 0, "蜘蛛侠的主要活动区域是纽约市。"),
                    0.2,
                ),
            ],
            query="请查询蜘蛛侠资料.docx中的位置",
        )

        self.assertEqual([item.source.id for item in filtered], ["kb-named"])

    def test_explicit_file_reference_fails_closed_when_not_in_candidates(self) -> None:
        unrelated = source("kb-other", "蝙蝠侠资料.docx")
        diagnostics: dict[str, object] = {}
        filtered = filter_relevant_hits(
            [
                hit(
                    unrelated,
                    chunk("kb-other", 0, "蝙蝠侠的主要活动区域是哥谭市。"),
                    0.9,
                )
            ],
            query="请查询蜘蛛侠资料.docx中的位置",
            diagnostics=diagnostics,
        )

        self.assertEqual(filtered, [])
        self.assertEqual(diagnostics["reason"], "explicit_file_reference_not_in_candidates")

    def test_explicit_file_reference_matches_basename_not_substring(self) -> None:
        named = source("kb-named", "upload-report.xlsx")
        similar = source("kb-similar", "my_report.xlsx")
        filtered = filter_relevant_hits(
            [
                hit(named, chunk("kb-named", 0, "销售额 | 100"), 0.2),
                hit(similar, chunk("kb-similar", 0, "销售额 | 999"), 0.9),
            ],
            query=r"请查询 E:/data/report.xlsx 中的销售额",
        )

        self.assertEqual(filtered, [])

    def test_explicit_file_reference_matches_exact_basename(self) -> None:
        named = source("kb-named", "abc.xlsx")
        filtered = filter_relevant_hits(
            [hit(named, chunk("kb-named", 0, "销售额 | 100"), 0.2)],
            query="请查询abc.xlsx中的销售额",
        )

        self.assertEqual([item.source.id for item in filtered], ["kb-named"])

    def test_file_scope_prefers_longer_prefixed_basename_over_cleaned_candidate(self) -> None:
        exact = source("kb-exact", "关于项目.docx")
        cleaned = source("kb-cleaned", "项目.docx")
        filtered = filter_relevant_hits(
            [
                hit(exact, chunk("kb-exact", 0, "项目内容：A"), 0.2),
                hit(cleaned, chunk("kb-cleaned", 0, "项目内容：B"), 0.9),
            ],
            query="关于项目.docx中的内容",
        )

        self.assertEqual([item.source.id for item in filtered], ["kb-exact"])

    def test_file_scope_falls_back_to_cleaned_candidate_when_raw_basename_is_absent(self) -> None:
        cleaned = source("kb-cleaned", "项目.docx")
        filtered = filter_relevant_hits(
            [hit(cleaned, chunk("kb-cleaned", 0, "项目内容：B"), 0.2)],
            query="关于项目.docx中的内容",
        )

        self.assertEqual([item.source.id for item in filtered], ["kb-cleaned"])

    def test_agent_never_sends_low_scoring_unrelated_document_to_llm(self) -> None:
        town = source("kb-town", "test.docx")
        spider = source("kb-spider", "蜘蛛侠角色介绍.docx")
        candidates = (
            hit(town, chunk("kb-town", 0, "欧洲小镇名称源于犀牛陂的谐音。"), 0.938),
            hit(town, chunk("kb-town", 1, "欧洲小镇园区规划。"), 0.006),
            hit(spider, chunk("kb-spider", 0, "蜘蛛侠角色起源。"), 0.0014),
            hit(spider, chunk("kb-spider", 1, "蜘蛛侠人物关系。"), 0.0),
        )
        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=lambda query, limit, routing_key: AgentSearchResult(
                    hits=candidates
                )
            ),
            llm_provider=provider,
        )

        result = agent.run(
            conversation_id="conv-town",
            content="欧洲小镇名字的由来",
            mode="quick",
            previous_messages=[],
        )

        self.assertIsNotNone(provider.request)
        assert provider.request is not None
        self.assertEqual(
            {item.source.id for item in provider.request.knowledge_hits},
            {"kb-town"},
        )
        self.assertEqual(result.evidence_count, 2)
        self.assertEqual(result.source_count, 1)
        self.assertEqual(result.route_metadata.candidate_source_ids, ("kb-town",))

    def test_pure_greeting_builds_welcome_run_without_external_dependencies(self) -> None:
        def unexpected_call(*args: object, **kwargs: object) -> None:
            raise AssertionError("greeting must not call external dependencies")

        class UnexpectedProvider:
            def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
                raise AssertionError("greeting must not call external dependencies")

        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(search_knowledge=unexpected_call),
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
                search_knowledge=lambda query, limit, routing_key: AgentSearchResult(hits=()),
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
        ) -> AgentSearchResult:
            search_calls.append(query)
            routing_keys.append(routing_key)
            if len(search_calls) == 1:
                return AgentSearchResult(hits=(hit(policy, policy_chunk, 0.8),))
            return AgentSearchResult(hits=(hit(finance, finance_chunk, 8.2),))

        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(search_knowledge=search),
            llm_provider=provider,
        )

        result = agent.run(
            conversation_id="conv-agent",
            content="对比差旅制度和财务票据材料需要什么",
            mode="deep",
            previous_messages=[],
            route_type=KnowledgeRouteType.SUMMARY_COMPARE,
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

    def test_agent_never_inspects_documents_after_reranked_search(self) -> None:
        policy = source("kb-policy", "差旅制度.txt")
        finance = source("kb-finance", "财务规则.txt")
        policy_hit = hit(policy, chunk("kb-policy", 0, "差旅材料需在五日内提交。"), 9.2)
        finance_hit = hit(finance, chunk("kb-finance", 0, "财务要求提交发票。"), 8.8)
        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=lambda query, limit, routing_key: AgentSearchResult(
                    hits=(policy_hit, finance_hit)
                ),
            ),
            llm_provider=provider,
        )

        result = agent.run(
            conversation_id="conv-agent",
            content="对比差旅制度和财务票据要求",
            mode="source",
            previous_messages=[],
            route_type=KnowledgeRouteType.SUMMARY_COMPARE,
        )

        self.assertNotIn("inspect_document", [step.tool_name for step in result.steps])
        self.assertIn("compare_evidence", [step.tool_name for step in result.steps])
        self.assertIsNotNone(provider.request)
        self.assertIn("多来源", provider.request.agent_context)
        self.assertIn("差旅制度.txt", provider.request.agent_context)
        self.assertIn("财务规则.txt", provider.request.agent_context)
        self.assertGreaterEqual(len(provider.request.knowledge_hits), 2)

    def test_document_route_does_not_expand_after_a_single_precise_search(self) -> None:
        policy = source("kb-policy", "policy.txt")
        finance = source("kb-finance", "finance.txt")
        search_calls = 0

        def search(query: str, limit: int, routing_key: str) -> AgentSearchResult:
            nonlocal search_calls
            search_calls += 1
            if search_calls == 1:
                return AgentSearchResult(
                    hits=(hit(policy, chunk("kb-policy", 0, "policy evidence"), 0.8),),
                    fallback_reason="reranker_service_error",
                )
            return AgentSearchResult(
                hits=(hit(finance, chunk("kb-finance", 0, "finance evidence"), 8.8),),
                fallback_reason="reranker_response_error",
            )

        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(search_knowledge=search),
            llm_provider=RecordingProvider(),
        )

        result = agent.run(
            conversation_id="conv-agent",
            content="reimbursement workflow",
            mode="deep",
            previous_messages=[],
            route_type=KnowledgeRouteType.DOCUMENT_QA,
        )

        self.assertEqual(search_calls, 1)
        self.assertEqual(result.route_metadata.degradation_reason, "reranker_service_error")
        self.assertEqual(
            result.route_metadata.candidate_source_ids,
            ("kb-policy",),
        )
        self.assertTrue(result.route_metadata.adjacency_allowed)

    def test_agent_stops_without_mutating_tools_when_no_evidence_exists(self) -> None:
        provider = RecordingProvider()
        agent = ReadOnlyKnowledgeAgent(
            tools=KnowledgeAgentTools(
                search_knowledge=lambda query, limit, routing_key: AgentSearchResult(hits=()),
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
        self.recorded_retrieval_requests: list[RetrievalRequest] = []

    def build_repository(
        self,
        *,
        word_fact_service: WordFactAnswerService | None = None,
    ) -> tuple[InMemoryChatRepository, str]:
        recorded_requests = self.recorded_retrieval_requests

        class RecordingRouter:
            def search(self, request: RetrievalRequest) -> RoutedRetrievalOutcome:
                recorded_requests.append(request)
                return RoutedRetrievalOutcome(
                    mode=RetrievalMode.LEGACY,
                    hits=(),
                    stage_ms={"legacy": 0.0},
                )

        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[]),
            llm_provider=self.recording_llm,
            word_fact_service=word_fact_service,
            retrieval_router=RecordingRouter(),
            retrieval_scope=RetrievalScope("default", ("internal",), "v1"),
        )
        _, conversation_id, _ = repository.create_conversation()
        return repository, conversation_id

    def test_fact_route_precedes_agent_and_llm(self) -> None:
        repository, conversation_id = self.build_repository(word_fact_service=self.fact_service)

        _, _, messages = repository.send_message(conversation_id, "张三几岁", "deep")

        self.assertEqual(messages[-1].paragraphs[0].text, "张三的年龄是28岁。")
        self.assertEqual(self.recording_llm.generation_calls, 0)
        self.assertEqual(self.recorded_retrieval_requests, [])
        run = repository.list_agent_runs(1)[0]
        self.assertFalse(run.route_metadata.adjacency_allowed)

    def test_open_word_question_continues_to_hybrid_rag(self) -> None:
        repository, conversation_id = self.build_repository(word_fact_service=self.fact_service)

        repository.send_message(conversation_id, "介绍张三", "deep")

        self.assertGreater(len(self.recorded_retrieval_requests), 0)
        self.assertTrue(
            all(
                request.expansion_policy is EvidenceExpansionPolicy.BOUNDED_ADJACENCY
                for request in self.recorded_retrieval_requests
            )
        )
        self.assertEqual(self.recording_llm.generation_calls, 1)

    def test_document_route_preserves_router_degradation_metadata(self) -> None:
        document_source = source("kb-policy", "policy.txt")
        document_hit = hit(
            document_source,
            chunk("kb-policy", 0, "reimbursement evidence"),
            9.2,
        )
        recorded_requests = self.recorded_retrieval_requests

        class DegradedRouter:
            def search(self, request: RetrievalRequest) -> RoutedRetrievalOutcome:
                recorded_requests.append(request)
                return RoutedRetrievalOutcome(
                    mode=RetrievalMode.QWEN3,
                    hits=(document_hit,),
                    stage_ms={"reranker": 1.0, "adjacency": 1.0},
                    fallback_reason="reranker_service_error",
                )

        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[]),
            llm_provider=self.recording_llm,
            retrieval_router=DegradedRouter(),
            retrieval_scope=RetrievalScope("default", ("internal",), "v1"),
        )
        _, conversation_id, _ = repository.create_conversation()

        repository.send_message(conversation_id, "reimbursement workflow", "quick")

        self.assertEqual(len(recorded_requests), 1)
        self.assertIs(
            recorded_requests[0].expansion_policy,
            EvidenceExpansionPolicy.BOUNDED_ADJACENCY,
        )
        run = repository.list_agent_runs(1)[0]
        self.assertEqual(run.route_metadata.degradation_reason, "reranker_service_error")
        self.assertEqual(run.route_metadata.candidate_source_ids, ("kb-policy",))
        self.assertTrue(run.route_metadata.adjacency_allowed)

    def test_open_questions_with_field_vocabulary_continue_to_hybrid_rag(self) -> None:
        for question in (
            "介绍财务部门",
            "公司的岗位职责是什么",
            "张三的年龄变化趋势是什么",
        ):
            with self.subTest(question=question):
                self.recording_llm = RecordingRouteProvider()
                self.recorded_retrieval_requests = []
                repository, conversation_id = self.build_repository(
                    word_fact_service=self.fact_service
                )

                repository.send_message(conversation_id, question, "deep")

                self.assertGreater(len(self.recorded_retrieval_requests), 0)
                self.assertEqual(self.recording_llm.generation_calls, 1)

    def test_malformed_fact_is_safe_terminal_and_never_calls_rag(self) -> None:
        repository, conversation_id = self.build_repository(
            word_fact_service=WordFactAnswerService(
                FakeFacts([word_fact_match("张三", "年龄", "28岁 性别：女")])
            )
        )

        _, _, messages = repository.send_message(conversation_id, "张三几岁", "deep")

        self.assertEqual(
            messages[-1].paragraphs[0].text,
            "无法安全返回张三的年龄，请核对来源数据。",
        )
        self.assertNotIn("女", messages[-1].paragraphs[0].text)
        self.assertEqual(self.recording_llm.generation_calls, 0)
        self.assertEqual(self.recorded_retrieval_requests, [])

    def test_missing_fact_is_terminal_and_never_inspects_document(self) -> None:
        repository, conversation_id = self.build_repository(
            word_fact_service=WordFactAnswerService(FakeFacts([]))
        )

        repository.send_message(conversation_id, "张三几岁", "deep")

        self.assertEqual(self.recorded_retrieval_requests, [])


if __name__ == "__main__":
    unittest.main()
