from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from .agent import AgentRunResult, ReadOnlyKnowledgeAgent
from .knowledge_route_models import KnowledgeRouteMetadata, KnowledgeRouteType
from .models import ChatMessageModel, ComposerMode


class StructuredAnswerServiceProtocol(Protocol):
    def try_answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult | None: ...


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
                conversation_id, content, mode, previous_messages
            )
            if structured is not None:
                return structured

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
