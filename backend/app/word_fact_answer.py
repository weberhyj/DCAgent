from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from .agent import AgentRunResult, AgentStep
from .knowledge_route_models import KnowledgeRouteMetadata, KnowledgeRouteType
from .models import ChatMessageModel, CitationModel, ComposerMode, ResponseParagraphModel
from .time_utils import display_datetime_label
from .word_facts import (
    WordFactClarification,
    WordFactMatch,
    WordFactRepository,
    WordFactualIntent,
    conflicting_word_fact_answer,
    exact_word_fact_answer,
    missing_word_fact_answer,
    resolve_word_factual_intent,
    source_ambiguity_word_fact_answer,
    unsafe_word_fact_answer,
    validate_word_fact_answer,
)


def _deduplicate_selected_matches(
    intent: WordFactualIntent,
    matches: Sequence[WordFactMatch],
) -> tuple[WordFactMatch, ...]:
    """Keep one exact value per source, preserving repository order."""

    unique: dict[tuple[str, str], WordFactMatch] = {}
    for match in matches:
        if (
            match.fact.entity_normalized != intent.entity_normalized
            or match.fact.field_normalized != intent.field_normalized
        ):
            continue
        unique.setdefault((match.fact.source_id, match.fact.value), match)
    return tuple(unique.values())


def _fact_answer_text(intent: WordFactualIntent, matches: Sequence[WordFactMatch]) -> str:
    if not matches:
        return missing_word_fact_answer(intent)
    source_ids = {match.fact.source_id for match in matches}
    values = {match.fact.value for match in matches}
    if len(source_ids) != 1:
        return source_ambiguity_word_fact_answer(intent)
    if len(values) != 1:
        return conflicting_word_fact_answer(intent)
    return exact_word_fact_answer(intent, next(iter(values)))


def _fact_citations(
    intent: WordFactualIntent,
    matches: Sequence[WordFactMatch],
) -> list[CitationModel]:
    return [
        CitationModel(
            label=f"[{rank}] {match.classification} · {match.source_name}",
            classification=match.classification,
            source_id=match.fact.source_id,
            source_name=match.source_name,
            chunk_id=match.fact.chunk_id,
            excerpt=f"{intent.field}：{match.fact.value}",
            rank=rank,
            matched_terms=[intent.field],
        )
        for rank, match in enumerate(matches, start=1)
    ]


def build_word_fact_run(
    conversation_id: str,
    question: str,
    mode: ComposerMode,
    answer: str,
    *,
    intent: WordFactualIntent | None = None,
    matches: Sequence[WordFactMatch] = (),
    route_type: KnowledgeRouteType = KnowledgeRouteType.WORD_FACTUAL,
    route_metadata: KnowledgeRouteMetadata | None = None,
) -> AgentRunResult:
    """Build the standard completed agent result for a deterministic fact outcome."""

    safe_matches = tuple(matches)
    candidate_source_ids = tuple(sorted({match.fact.source_id for match in safe_matches}))
    validation_passed = intent is not None
    if intent is not None and not validate_word_fact_answer(intent, safe_matches, answer):
        answer = unsafe_word_fact_answer(intent)
        safe_matches = ()
        validation_passed = False
    timestamp = display_datetime_label()
    unique_source_ids = list(dict.fromkeys(match.fact.source_id for match in safe_matches))
    reply = ChatMessageModel(
        id=f"msg-{uuid4().hex[:8]}",
        role="assistant",
        time=timestamp,
        paragraphs=[
            ResponseParagraphModel(
                text=answer,
                citations=(
                    _fact_citations(intent, safe_matches) if intent is not None else []
                ),
            )
        ],
    )
    step = AgentStep(
        id=f"step-{uuid4().hex[:12]}",
        step_index=0,
        tool_name="query_word_fact",
        status="completed",
        input_summary=question.strip(),
        output_summary="已完成精确事实查询",
        source_ids=unique_source_ids,
        read_only=True,
        started_at=timestamp,
        completed_at=timestamp,
    )
    return AgentRunResult(
        id=f"agent-{uuid4().hex[:12]}",
        conversation_id=conversation_id,
        query=question.strip(),
        mode=mode,
        status="completed",
        started_at=timestamp,
        completed_at=timestamp,
        reply=reply,
        steps=[step],
        evidence_count=len(safe_matches),
        source_count=len(unique_source_ids),
        route_type=route_type,
        route_metadata=route_metadata or KnowledgeRouteMetadata(
            entity=intent.entity if intent is not None else None,
            target_fields=(intent.field,) if intent is not None else (),
            candidate_source_ids=candidate_source_ids,
            validation_passed=validation_passed,
        ),
    )


def answer_word_fact(
    conversation_id: str,
    question: str,
    mode: ComposerMode,
    intent: WordFactualIntent,
    matches: Sequence[WordFactMatch],
) -> AgentRunResult:
    selected = _deduplicate_selected_matches(intent, matches)
    answer = _fact_answer_text(intent, selected)
    return build_word_fact_run(
        conversation_id,
        question,
        mode,
        answer,
        intent=intent,
        matches=selected,
    )


class WordFactAnswerService:
    def __init__(
        self,
        repository: WordFactRepository,
        permission_tags: Sequence[str] = (),
    ) -> None:
        self._repository = repository
        self._permission_tags = tuple(permission_tags)

    def try_answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult | None:
        del previous_messages
        resolution = resolve_word_factual_intent(content)
        if resolution is None:
            return None
        if isinstance(resolution, WordFactClarification):
            return build_word_fact_run(
                conversation_id,
                content,
                mode,
                resolution.message,
                route_type=KnowledgeRouteType.CLARIFICATION,
                route_metadata=KnowledgeRouteMetadata(
                    origin_route=KnowledgeRouteType.WORD_FACTUAL,
                    validation_passed=True,
                ),
            )
        matches = self._repository.find_knowledge_facts(
            resolution,
            permission_tags=self._permission_tags,
        )
        return answer_word_fact(conversation_id, content, mode, resolution, matches)
