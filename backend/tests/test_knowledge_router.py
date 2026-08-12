from __future__ import annotations

import unittest

from app.agent import AgentRunResult, AgentStep
from app.knowledge_route_models import KnowledgeRouteMetadata, KnowledgeRouteType
from app.knowledge_router import (
    KnowledgeAnswerRouter,
    LegacyKnowledgeAnswerRouter,
    classify_document_route,
)
from app.models import ChatMessageModel, ResponseParagraphModel


def run_result(
    *,
    route_type: KnowledgeRouteType = KnowledgeRouteType.DOCUMENT_QA,
    metadata: KnowledgeRouteMetadata | None = None,
) -> AgentRunResult:
    reply = ChatMessageModel(
        id="msg-1",
        role="assistant",
        time="2026-08-12 09:00:00",
        paragraphs=[ResponseParagraphModel(text="answer")],
    )
    step = AgentStep(
        id="step-1",
        step_index=0,
        tool_name="answer",
        status="completed",
        input_summary="question",
        output_summary="answer",
        started_at="2026-08-12 09:00:00",
        completed_at="2026-08-12 09:00:00",
    )
    return AgentRunResult(
        id="agent-1",
        conversation_id="conv-1",
        query="question",
        mode="deep",
        status="completed",
        started_at="2026-08-12 09:00:00",
        completed_at="2026-08-12 09:00:00",
        reply=reply,
        steps=[step],
        evidence_count=0,
        source_count=0,
        route_type=route_type,
        route_metadata=metadata or KnowledgeRouteMetadata(),
    )


class RecordingAgent:
    def __init__(self, calls: list[str], greeting: AgentRunResult | None = None) -> None:
        self._calls = calls
        self._greeting = greeting

    def try_answer_greeting(self, **_kwargs: object) -> AgentRunResult | None:
        self._calls.append("greeting")
        return self._greeting

    def run(self, **kwargs: object) -> AgentRunResult:
        self._calls.append("document")
        route_type = kwargs.get("route_type", KnowledgeRouteType.DOCUMENT_QA)
        return run_result(route_type=route_type)  # type: ignore[arg-type]


class RecordingStructured:
    def __init__(self, calls: list[str], result: AgentRunResult | None = None) -> None:
        self._calls = calls
        self._result = result

    def try_answer(self, *_args: object) -> AgentRunResult | None:
        self._calls.append("excel")
        return self._result


class RecordingWordFacts:
    def __init__(self, calls: list[str], result: AgentRunResult | None = None) -> None:
        self._calls = calls
        self._result = result

    def try_answer(self, *_args: object) -> AgentRunResult | None:
        self._calls.append("word")
        return self._result


def excel_unavailable_run() -> AgentRunResult:
    return run_result(
        route_type=KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE,
        metadata=KnowledgeRouteMetadata(degradation_reason="clickhouse_unavailable"),
    )


def word_not_found_run() -> AgentRunResult:
    return run_result(route_type=KnowledgeRouteType.WORD_FACTUAL)


class KnowledgeAnswerRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[str] = []

    def router(
        self,
        *,
        structured_result: AgentRunResult | None = None,
        word_result: AgentRunResult | None = None,
    ) -> KnowledgeAnswerRouter:
        return KnowledgeAnswerRouter(
            agent=RecordingAgent(self.calls),
            structured_service=RecordingStructured(self.calls, structured_result),
            word_fact_service=RecordingWordFacts(self.calls, word_result),
        )

    def test_route_order_is_greeting_then_excel_then_word_then_document(self) -> None:
        router = self.router(
            structured_result=run_result(
                route_type=KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE
            )
        )

        result = router.answer("conv-1", "华东地区的销售额汇总", "deep", [])

        self.assertEqual(self.calls, ["greeting", "excel"])
        self.assertEqual(result.route_type, KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE)

    def test_excel_unavailable_is_terminal_and_never_calls_word_or_agent(self) -> None:
        result = self.router(structured_result=excel_unavailable_run()).answer(
            "conv-1", "华东地区的销售额汇总", "deep", []
        )

        self.assertEqual(result.route_metadata.degradation_reason, "clickhouse_unavailable")
        self.assertEqual(self.calls, ["greeting", "excel"])

    def test_word_not_found_is_terminal_and_never_calls_document_agent(self) -> None:
        result = self.router(word_result=word_not_found_run()).answer(
            "conv-1", "张三几岁", "deep", []
        )

        self.assertEqual(result.route_type, KnowledgeRouteType.WORD_FACTUAL)
        self.assertEqual(self.calls, ["greeting", "excel", "word"])

    def test_open_introduction_routes_to_summary_compare(self) -> None:
        result = self.router().answer("conv-1", "介绍张三", "deep", [])

        self.assertEqual(result.route_type, KnowledgeRouteType.SUMMARY_COMPARE)
        self.assertEqual(self.calls, ["greeting", "excel", "word", "document"])

    def test_policy_question_routes_to_document_qa(self) -> None:
        result = self.router().answer("conv-1", "报销流程是什么", "quick", [])

        self.assertEqual(result.route_type, KnowledgeRouteType.DOCUMENT_QA)

    def test_source_mode_routes_to_summary_compare(self) -> None:
        self.assertEqual(
            classify_document_route("报销流程是什么", "source"),
            KnowledgeRouteType.SUMMARY_COMPARE,
        )

    def test_greeting_is_tagged_with_a_validated_greeting_route(self) -> None:
        router = KnowledgeAnswerRouter(
            agent=RecordingAgent(self.calls, greeting=run_result()),
            structured_service=RecordingStructured(self.calls),
            word_fact_service=RecordingWordFacts(self.calls),
        )

        result = router.answer("conv-1", "你好", "quick", [])

        self.assertEqual(result.route_type, KnowledgeRouteType.GREETING)
        self.assertTrue(result.route_metadata.validation_passed)
        self.assertEqual(self.calls, ["greeting"])

    def test_legacy_router_omits_word_factual_routing(self) -> None:
        result = LegacyKnowledgeAnswerRouter(
            agent=RecordingAgent(self.calls),
            structured_service=RecordingStructured(self.calls),
        ).answer("conv-1", "张三几岁", "deep", [])

        self.assertEqual(self.calls, ["greeting", "excel", "document"])
        self.assertEqual(result.route_type, KnowledgeRouteType.DOCUMENT_QA)


class KnowledgeRouteMetadataTests(unittest.TestCase):
    def test_metadata_round_trip_preserves_serializable_route_contract(self) -> None:
        metadata = KnowledgeRouteMetadata(
            dataset_id="sales",
            entity="华东",
            target_fields=("销售额",),
            candidate_source_ids=("source-1",),
            origin_route=KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE,
            degradation_reason="clickhouse_unavailable",
            validation_passed=False,
            adjacency_allowed=True,
        )

        restored = KnowledgeRouteMetadata.from_dict(metadata.to_dict())

        self.assertEqual(restored, metadata)

    def test_to_dict_rejects_oversized_scalar_metadata(self) -> None:
        metadata = KnowledgeRouteMetadata(dataset_id="x" * 257)

        with self.assertRaises(ValueError):
            metadata.to_dict()

    def test_to_dict_rejects_oversized_or_unbounded_metadata_lists(self) -> None:
        metadata = KnowledgeRouteMetadata(target_fields=tuple("field" for _ in range(33)))

        with self.assertRaises(ValueError):
            metadata.to_dict()

    def test_to_dict_rejects_oversized_list_items(self) -> None:
        metadata = KnowledgeRouteMetadata(candidate_source_ids=("s" * 129,))

        with self.assertRaises(ValueError):
            metadata.to_dict()

    def test_from_dict_requires_actual_booleans_and_string_origin_route(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeRouteMetadata.from_dict({"adjacency_allowed": "false"})
        with self.assertRaises(ValueError):
            KnowledgeRouteMetadata.from_dict({"validation_passed": 1})
        with self.assertRaises(ValueError):
            KnowledgeRouteMetadata.from_dict({"origin_route": 1})

    def test_from_dict_rejects_oversized_scalar_and_list_values(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeRouteMetadata.from_dict({"entity": "x" * 257})
        with self.assertRaises(ValueError):
            KnowledgeRouteMetadata.from_dict({"target_fields": ["field"] * 33})
        with self.assertRaises(ValueError):
            KnowledgeRouteMetadata.from_dict({"candidate_source_ids": ["s" * 129]})


if __name__ == "__main__":
    unittest.main()
