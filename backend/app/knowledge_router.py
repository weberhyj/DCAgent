from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from .agent import AgentRunResult, ReadOnlyKnowledgeAgent
from .knowledge_route_models import KnowledgeRouteMetadata, KnowledgeRouteType
from .models import ChatMessageModel, ComposerMode
from .structured_answer import is_structured_candidate


class StructuredAnswerServiceProtocol(Protocol):
    def try_answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult | None: ...

    def catalog_snapshot(self) -> object | None: ...


class WordFactAnswerServiceProtocol(Protocol):
    def try_answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult | None: ...


def classify_document_route(content: str, mode: ComposerMode) -> KnowledgeRouteType:
    if mode == "source":
        return KnowledgeRouteType.SUMMARY_COMPARE
    normalized = content.casefold()
    if any(term in normalized for term in ("介绍", "总结", "概括", "比较", "对比", "异同", "分别")):
        return KnowledgeRouteType.SUMMARY_COMPARE
    return KnowledgeRouteType.DOCUMENT_QA


class KnowledgeAnswerRouter:
    def __init__(
        self,
        agent: ReadOnlyKnowledgeAgent,
        structured_service: StructuredAnswerServiceProtocol | None = None,
        word_fact_service: WordFactAnswerServiceProtocol | None = None,
    ) -> None:
        self._agent = agent
        self._structured_service = structured_service
        self._word_fact_service = word_fact_service

    def answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult:
        greeting = self._agent.try_answer_greeting(
            conversation_id=conversation_id,
            content=content,
            mode=mode,
        )
        if greeting is not None:
            return replace(
                greeting,
                route_type=KnowledgeRouteType.GREETING,
                route_metadata=KnowledgeRouteMetadata(validation_passed=True),
            )

        if self._structured_service is not None:
            structured = self._structured_service.try_answer(
                conversation_id=conversation_id,
                content=content,
                mode=mode,
                previous_messages=previous_messages,
            )
            if structured is not None:
                return structured

        # A question that is clearly about a structured workbook is terminal
        # even when parsing could not produce a usable plan. Falling through to
        # Word facts or vector RAG is what caused unrelated document answers.
        if self._structured_service is not None:
            catalog = getattr(self._structured_service, "catalog_snapshot", None)
            if callable(catalog):
                try:
                    snapshot = catalog()
                except Exception:
                    snapshot = None
                if snapshot is not None and _looks_like_structured_question(content, snapshot):
                    return _structured_clarification_run(conversation_id, content, mode)

        if self._word_fact_service is not None:
            factual = self._word_fact_service.try_answer(
                conversation_id, content, mode, previous_messages
            )
            if factual is not None:
                return factual

        document_route = classify_document_route(content, mode)
        return self._agent.run(
            conversation_id=conversation_id,
            content=content,
            mode=mode,
            previous_messages=list(previous_messages),
            route_type=document_route,
        )


def _looks_like_structured_question(content: str, catalog: object) -> bool:
    try:
        return is_structured_candidate(content, catalog)  # type: ignore[arg-type]
    except Exception:
        return False


def _structured_clarification_run(
    conversation_id: str,
    content: str,
    mode: ComposerMode,
) -> AgentRunResult:
    from .structured_answer import _structured_run

    return _structured_run(
        conversation_id,
        content.strip(),
        mode,
        "未能解析这条 Excel 查询。请明确写出筛选列、筛选值和要返回的列。",
        "structured query clarification required",
        route_type=KnowledgeRouteType.CLARIFICATION,
        route_metadata=KnowledgeRouteMetadata(
            origin_route=KnowledgeRouteType.EXCEL_ROW_LOOKUP,
            degradation_reason="intent_unavailable",
            validation_passed=False,
        ),
    )


class LegacyKnowledgeAnswerRouter:
    def __init__(
        self,
        agent: ReadOnlyKnowledgeAgent,
        structured_service: StructuredAnswerServiceProtocol | None = None,
    ) -> None:
        self._agent = agent
        self._structured_service = structured_service

    def answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult:
        greeting = self._agent.try_answer_greeting(
            conversation_id=conversation_id,
            content=content,
            mode=mode,
        )
        if greeting is not None:
            return replace(
                greeting,
                route_type=KnowledgeRouteType.GREETING,
                route_metadata=KnowledgeRouteMetadata(validation_passed=True),
            )
        if self._structured_service is not None:
            structured = self._structured_service.try_answer(
                conversation_id, content, mode, previous_messages
            )
            if structured is not None:
                return structured
        return self._agent.run(
            conversation_id=conversation_id,
            content=content,
            mode=mode,
            previous_messages=list(previous_messages),
        )
