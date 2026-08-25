from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .embeddings import expand_terms, extract_embedding_terms
from .knowledge_route_models import KnowledgeRouteMetadata, KnowledgeRouteType
from .llm import LLMProvider, LLMRequest
from .models import (
    ChatMessageModel,
    ComposerMode,
    KnowledgeSearchHitModel,
    ResponseParagraphModel,
)
from .retrieval_models import EvidenceExpansionPolicy
from .time_utils import display_datetime_label

AgentRunStatus = Literal["completed", "failed"]
AgentStepStatus = Literal["completed", "failed"]

GREETING_REPLY = (
    "您好，我是 DCAgent 企业知识库智能助手。您可以向我询问 Word、Excel、PDF 等知识库资料中的内容，"
    "我会检索相关依据并为您汇总回答。"
)
GREETING_PHRASES = frozenset(
    {"你好", "您好", "嗨", "哈喽", "在吗", "你在吗", "你是谁", "介绍一下你自己"}
)
FOLLOW_UP_PREFIXES = (
    "继续",
    "再说",
    "再补充",
    "补充一下",
    "还有",
    "那",
    "那么",
    "这个",
    "那个",
    "这两个",
    "那两个",
    "两者",
    "两份",
    "各自",
    "哪个",
    "上述",
    "前面",
    "刚才",
    "他",
    "她",
    "它",
    "他们",
    "她们",
    "它们",
    "该人",
    "该项目",
    "该制度",
    "对此",
    "what about",
    "how about",
    "continue",
)
FOLLOW_UP_REFERENCES = (
    "上一个",
    "上一条",
    "前一个",
    "前面提到",
    "刚才提到",
    "上面说的",
    "上面的内容",
    "上述内容",
    "前述",
    "前文",
    "之前",
    "刚刚的",
    "这个答案",
    "这个结果",
    "它们之间",
)
MULTI_SOURCE_SYNTHESIS_TERMS = (
    "比较",
    "对比",
    "异同",
    "分别",
    "各自",
    "综合",
    "多份",
    "多个",
    "两份",
    "两者",
    "这些资料",
    "这些文档",
    "跨文档",
    "全库",
)


@dataclass(slots=True)
class AgentStep:
    id: str
    step_index: int
    tool_name: str
    status: AgentStepStatus
    input_summary: str
    output_summary: str
    started_at: str
    completed_at: str
    source_ids: list[str] = field(default_factory=list)
    read_only: bool = True


@dataclass(slots=True)
class AgentRunResult:
    id: str
    conversation_id: str
    query: str
    mode: ComposerMode
    status: AgentRunStatus
    started_at: str
    completed_at: str
    reply: ChatMessageModel
    steps: list[AgentStep]
    evidence_count: int
    source_count: int
    route_type: KnowledgeRouteType = KnowledgeRouteType.DOCUMENT_QA
    route_metadata: KnowledgeRouteMetadata = field(default_factory=KnowledgeRouteMetadata)

    def to_audit(self) -> AgentRunAudit:
        return AgentRunAudit(
            id=self.id,
            conversation_id=self.conversation_id,
            query=self.query,
            mode=self.mode,
            status=self.status,
            started_at=self.started_at,
            completed_at=self.completed_at,
            answer_message_id=self.reply.id,
            evidence_count=self.evidence_count,
            source_count=self.source_count,
            steps=self.steps,
            route_type=self.route_type,
            route_metadata=self.route_metadata,
        )


@dataclass(slots=True)
class AgentRunAudit:
    id: str
    conversation_id: str
    query: str
    mode: ComposerMode
    status: AgentRunStatus
    started_at: str
    completed_at: str
    answer_message_id: str
    evidence_count: int
    source_count: int
    steps: list[AgentStep] = field(default_factory=list)
    route_type: KnowledgeRouteType = KnowledgeRouteType.DOCUMENT_QA
    route_metadata: KnowledgeRouteMetadata = field(default_factory=KnowledgeRouteMetadata)


@dataclass(frozen=True, slots=True)
class AgentSearchResult:
    hits: tuple[KnowledgeSearchHitModel, ...]
    fallback_reason: str | None = None


@dataclass(slots=True)
class KnowledgeAgentTools:
    search_knowledge: Callable[..., AgentSearchResult]


class AgentState(TypedDict):
    run_id: str
    conversation_id: str
    content: str
    mode: ComposerMode
    route_type: KnowledgeRouteType
    previous_messages: list[ChatMessageModel]
    started_at: str
    search_queries: list[str]
    query_index: int
    knowledge_hits: list[KnowledgeSearchHitModel]
    fallback_reasons: list[str]
    steps: list[AgentStep]
    agent_context: str
    reply: ChatMessageModel | None


def now_label() -> str:
    return display_datetime_label()


def is_greeting_message(content: str) -> bool:
    normalized = "".join(
        character
        for character in content
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
        and character not in {"~", "～"}
    ).casefold()
    return normalized in GREETING_PHRASES


def is_follow_up_message(content: str) -> bool:
    """Return whether a question depends on preceding conversation context."""

    normalized = re.sub(r"\s+", " ", content).strip().casefold()
    if not normalized:
        return False
    if normalized.startswith(FOLLOW_UP_PREFIXES):
        return True
    if any(reference in normalized for reference in FOLLOW_UP_REFERENCES):
        return True
    return normalized.rstrip("？? ").endswith(("呢", "又如何", "还有吗"))


def build_agent_search_queries(
    content: str,
    mode: ComposerMode,
    route_type: KnowledgeRouteType | None = None,
) -> list[str]:
    """Build bounded retrieval queries without broadening ordinary fact lookup.

    ``deep`` is useful for an explicit multi-document summary/ comparison, but
    broadening every question with generic policy terms causes unrelated files
    to enter the evidence set.  Production callers pass the classified route;
    the ``None`` default keeps the small public helper backwards compatible.
    """
    query = content.strip()
    if mode == "quick" or route_type == KnowledgeRouteType.DOCUMENT_QA:
        return [query]

    # Deep retrieval is reserved for routes that can legitimately use more
    # than one document.  Do not invent generic terms when no useful expansion
    # exists; the original query is more precise in that case.
    if route_type is not None and route_type != KnowledgeRouteType.SUMMARY_COMPARE:
        return [query]
    if route_type == KnowledgeRouteType.SUMMARY_COMPARE and not any(
        term in query for term in MULTI_SOURCE_SYNTHESIS_TERMS
    ):
        return [query]

    terms = extract_embedding_terms(query)
    expanded = [term for term in expand_terms(terms) if term not in terms]
    if expanded:
        broader_query = " ".join([query, *expanded[:12]])
        return list(dict.fromkeys([query, broader_query]))
    return [query]


def merge_ranked_hits(
    existing: list[KnowledgeSearchHitModel],
    incoming: Sequence[KnowledgeSearchHitModel],
    limit: int,
) -> list[KnowledgeSearchHitModel]:
    by_chunk_id = {hit.chunk.id: hit for hit in existing}
    for hit in incoming:
        current = by_chunk_id.get(hit.chunk.id)
        if current is None or hit.score > current.score:
            by_chunk_id[hit.chunk.id] = hit

    ranked = sorted(
        by_chunk_id.values(),
        key=lambda hit: (-hit.score, hit.source.name, hit.chunk.chunk_index),
    )[:limit]
    return [
        KnowledgeSearchHitModel(
            source=hit.source,
            chunk=hit.chunk,
            score=hit.score,
            keyword_score=hit.keyword_score,
            vector_score=hit.vector_score,
            rank=index,
            matched_terms=hit.matched_terms,
        )
        for index, hit in enumerate(ranked, start=1)
    ]


def filter_relevant_hits(
    hits: Sequence[KnowledgeSearchHitModel],
    *,
    relative_score_floor: float = 0.05,
    minimum_score_floor: float = 0.01,
    adjacency_distance: int = 1,
    single_source: bool = False,
) -> list[KnowledgeSearchHitModel]:
    """Drop unrelated sources before evidence is passed to the LLM.

    Hybrid retrieval intentionally returns a bounded candidate set, which can
    include low-scoring documents (especially when bounded adjacency expands a
    result).  Those candidates must not become LLM evidence just because they
    fit inside ``top_k``.  A hit is retained when its source has an anchor hit
    above a relative score floor, plus at most the immediately adjacent chunks
    around each anchor so a split passage remains readable.
    """
    hit_list = list(hits)
    if not hit_list:
        return []
    if any(value < 0 for value in (relative_score_floor, minimum_score_floor)):
        raise ValueError("score floors must be non-negative")
    if adjacency_distance < 0:
        raise ValueError("adjacency_distance must be non-negative")

    top_score = max(hit.score for hit in hit_list)
    if top_score <= 0:
        return []

    # llama.cpp BGE reranker scores are converted to probabilities in [0, 1].
    # The old 5% floor is too permissive at that scale: a score of .04 could
    # still become evidence beside a .8 match.  Keep the historical behavior
    # for legacy/high-scale test and fallback scores, while applying a stricter
    # normalized-score gate to the production reranker output.
    normalized_scores = top_score <= 1.0 and all(0.0 <= hit.score <= 1.0 for hit in hit_list)
    effective_relative_floor = max(relative_score_floor, 0.10) if normalized_scores else relative_score_floor
    effective_minimum_floor = max(minimum_score_floor, 0.03) if normalized_scores else minimum_score_floor
    cutoff = max(effective_minimum_floor, top_score * effective_relative_floor)
    anchors = [hit for hit in hit_list if hit.score >= cutoff]
    if not anchors:
        return []

    allowed_source_ids: set[str] | None = None
    if single_source:
        # Ordinary document QA should answer from the strongest source only.
        # Multi-source evidence remains available to the explicit comparison
        # route, which calls this function with single_source=False.
        source_scores: dict[str, float] = {}
        for anchor in anchors:
            source_scores[anchor.source.id] = max(
                source_scores.get(anchor.source.id, 0.0), anchor.score
            )
        if source_scores:
            ordered_sources = sorted(
                source_scores.items(), key=lambda item: (-item[1], item[0])
            )
            strongest_source, strongest_score = ordered_sources[0]
            second_score = ordered_sources[1][1] if len(ordered_sources) > 1 else 0.0
            # Only collapse to one source when the leading result is both
            # confident and clearly separated.  If the scores are close, the
            # expected document may be rank 2+ in the retrieval Top-K; keep
            # the independently relevant sources instead of returning no
            # evidence to the answer composer.
            if (
                not normalized_scores
                or (strongest_score >= 0.70 and strongest_score - second_score >= 0.15)
            ):
                allowed_source_ids = {strongest_source}

    anchor_indices: dict[str, list[int]] = {}
    for hit in anchors:
        anchor_indices.setdefault(hit.source.id, []).append(hit.chunk.chunk_index)

    filtered: list[KnowledgeSearchHitModel] = []
    for hit in hit_list:
        if allowed_source_ids is not None and hit.source.id not in allowed_source_ids:
            continue
        indices = anchor_indices.get(hit.source.id)
        if not indices:
            continue
        if hit.score >= cutoff or any(
            abs(hit.chunk.chunk_index - anchor_index) <= adjacency_distance
            for anchor_index in indices
        ):
            filtered.append(hit)

    filtered.sort(key=lambda hit: (-hit.score, hit.source.name, hit.chunk.chunk_index))
    return [
        KnowledgeSearchHitModel(
            source=hit.source,
            chunk=hit.chunk,
            score=hit.score,
            keyword_score=hit.keyword_score,
            vector_score=hit.vector_score,
            rank=index,
            matched_terms=hit.matched_terms,
        )
        for index, hit in enumerate(filtered, start=1)
    ]


def build_comparison_context(hits: list[KnowledgeSearchHitModel], search_rounds: int) -> str:
    source_names = list(dict.fromkeys(hit.source.name for hit in hits))
    if not source_names:
        return f"Agent 已完成 {search_rounds} 轮检索，但没有找到可用证据。"

    scope = "多来源" if len(source_names) > 1 else "单来源"
    conflict_terms = ("不得", "禁止", "无需", "不需要", "必须", "应当", "需要")
    observed = {term for hit in hits for term in conflict_terms if term in hit.chunk.text}
    conflict_summary = "检测到可能需要核对的约束措辞" if len(observed) >= 2 else "未检测到明显冲突"
    return (
        f"Agent 已完成 {search_rounds} 轮检索和{scope}证据检查。"
        f"来源：{'、'.join(source_names)}。{conflict_summary}。"
    )


class ReadOnlyKnowledgeAgent:
    def __init__(
        self,
        tools: KnowledgeAgentTools,
        llm_provider: LLMProvider,
        max_hits: int = 5,
    ) -> None:
        self.tools = tools
        self.llm_provider = llm_provider
        self.max_hits = max_hits
        self.graph = self._build_graph()

    def try_answer_greeting(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
    ) -> AgentRunResult | None:
        if not is_greeting_message(content):
            return None

        timestamp = now_label()
        query = content.strip()
        step = AgentStep(
            id=f"step-{uuid4().hex[:12]}",
            step_index=0,
            tool_name="respond_greeting",
            status="completed",
            input_summary=query,
            output_summary="已返回固定欢迎词",
            started_at=timestamp,
            completed_at=timestamp,
            read_only=True,
        )
        reply = ChatMessageModel(
            id=f"msg-agent-{uuid4().hex[:12]}",
            role="assistant",
            time=timestamp,
            paragraphs=[ResponseParagraphModel(text=GREETING_REPLY)],
        )
        return AgentRunResult(
            id=f"agent-{uuid4().hex[:12]}",
            conversation_id=conversation_id,
            query=query,
            mode=mode,
            status="completed",
            started_at=timestamp,
            completed_at=timestamp,
            reply=reply,
            steps=[step],
            evidence_count=0,
            source_count=0,
        )

    def run(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: list[ChatMessageModel],
        route_type: KnowledgeRouteType = KnowledgeRouteType.DOCUMENT_QA,
    ) -> AgentRunResult:
        run_id = f"agent-{uuid4().hex[:12]}"
        started_at = now_label()
        final_state = self.graph.invoke(
            AgentState(
                run_id=run_id,
                conversation_id=conversation_id,
                content=content.strip(),
                mode=mode,
                route_type=route_type,
                previous_messages=previous_messages,
                started_at=started_at,
                search_queries=[],
                query_index=0,
                knowledge_hits=[],
                fallback_reasons=[],
                steps=[],
                agent_context="",
                reply=None,
            )
        )
        reply = final_state["reply"]
        if reply is None:
            raise RuntimeError("Agent graph completed without a reply")
        hits = final_state["knowledge_hits"]
        fallback_reasons = final_state["fallback_reasons"]
        return AgentRunResult(
            id=run_id,
            conversation_id=conversation_id,
            query=content.strip(),
            mode=mode,
            status="completed",
            started_at=started_at,
            completed_at=now_label(),
            reply=reply,
            steps=final_state["steps"],
            evidence_count=len(hits),
            source_count=len({hit.source.id for hit in hits}),
            route_type=route_type,
            route_metadata=KnowledgeRouteMetadata(
                candidate_source_ids=tuple(sorted({hit.source.id for hit in hits})),
                degradation_reason=fallback_reasons[0] if fallback_reasons else None,
                adjacency_allowed=route_type
                in {
                    KnowledgeRouteType.DOCUMENT_QA,
                    KnowledgeRouteType.SUMMARY_COMPARE,
                },
            ),
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("search", self._search)
        graph.add_node("advance_query", self._advance_query)
        graph.add_node("compare", self._compare)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "search")
        graph.add_conditional_edges(
            "search",
            self._route_after_search,
            {
                "advance_query": "advance_query",
                "compare": "compare",
                "answer": "answer",
            },
        )
        graph.add_edge("advance_query", "search")
        graph.add_edge("compare", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def _step(
        self,
        state: AgentState,
        tool_name: str,
        input_summary: str,
        output_summary: str,
        source_ids: list[str] | None = None,
    ) -> AgentStep:
        timestamp = now_label()
        return AgentStep(
            id=f"step-{uuid4().hex[:12]}",
            step_index=len(state["steps"]),
            tool_name=tool_name,
            status="completed",
            input_summary=input_summary,
            output_summary=output_summary,
            started_at=timestamp,
            completed_at=timestamp,
            source_ids=source_ids or [],
            read_only=True,
        )

    def _plan(self, state: AgentState) -> dict:
        queries = build_agent_search_queries(
            state["content"], state["mode"], state["route_type"]
        )
        step = self._step(
            state,
            "plan_retrieval",
            state["content"],
            f"生成 {len(queries)} 个有界检索策略",
        )
        return {"search_queries": queries, "steps": [*state["steps"], step]}

    def _search(self, state: AgentState) -> dict:
        query = state["search_queries"][state["query_index"]]
        expansion_policy = (
            EvidenceExpansionPolicy.NONE
            if state["route_type"]
            in {
                KnowledgeRouteType.WORD_FACTUAL,
                KnowledgeRouteType.EXCEL_ROW_LOOKUP,
                KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE,
                KnowledgeRouteType.EXCEL_MULTI_AGGREGATE,
            }
            else EvidenceExpansionPolicy.BOUNDED_ADJACENCY
        )
        try:
            search_result = self.tools.search_knowledge(
                query,
                self.max_hits,
                state["conversation_id"],
                expansion_policy=expansion_policy,
            )
        except TypeError as error:
            if "expansion_policy" not in str(error):
                raise
            search_result = self.tools.search_knowledge(
                query,
                self.max_hits,
                state["conversation_id"],
            )
        hits = search_result.hits
        merged = merge_ranked_hits(state["knowledge_hits"], hits, self.max_hits)
        fallback_reasons = list(state["fallback_reasons"])
        if (
            search_result.fallback_reason is not None
            and search_result.fallback_reason not in fallback_reasons
            and len(fallback_reasons) < 8
        ):
            fallback_reasons.append(search_result.fallback_reason)
        source_ids = list(dict.fromkeys(hit.source.id for hit in hits))
        evidence_audit = ", ".join(
            f"{hit.chunk.id}:{hit.score:.6f}" for hit in hits[: self.max_hits]
        )
        output_summary = f"命中 {len(hits)} 个片段，累计保留 {len(merged)} 个片段"
        if evidence_audit:
            output_summary += f"；evidence={evidence_audit}"
        step = self._step(
            state,
            "search_knowledge",
            query,
            output_summary,
            source_ids,
        )
        return {
            "knowledge_hits": merged,
            "fallback_reasons": fallback_reasons,
            "steps": [*state["steps"], step],
        }

    def _route_after_search(self, state: AgentState) -> str:
        has_next_query = state["query_index"] + 1 < len(state["search_queries"])
        if has_next_query and self._needs_more_evidence(state):
            return "advance_query"
        if state["knowledge_hits"]:
            return "compare"
        return "answer"

    def _needs_more_evidence(self, state: AgentState) -> bool:
        if state["mode"] == "quick":
            return False
        hits = state["knowledge_hits"]
        if not hits:
            return True
        source_count = len({hit.source.id for hit in hits})
        top_score = max(hit.score for hit in hits)
        # Scores from the BGE reranker are probabilities; legacy fallback
        # scores historically used a larger scale.  Use the matching floor so
        # deep mode does not keep expanding a clearly good normalized hit.
        score_floor = 0.6 if 0.0 <= top_score <= 1.0 else 6.0
        return source_count < 2 or top_score < score_floor

    def _advance_query(self, state: AgentState) -> dict:
        return {"query_index": state["query_index"] + 1}

    def _compare(self, state: AgentState) -> dict:
        rounds = state["query_index"] + 1
        filtered_hits = filter_relevant_hits(
            state["knowledge_hits"],
            single_source=(
                state["route_type"] == KnowledgeRouteType.DOCUMENT_QA
                and state["mode"] != "source"
            ),
        )
        context = build_comparison_context(filtered_hits, rounds)
        source_ids = list(dict.fromkeys(hit.source.id for hit in filtered_hits))
        step = self._step(
            state,
            "compare_evidence",
            f"{len(source_ids)} 个来源",
            context,
            source_ids,
        )
        return {
            "knowledge_hits": filtered_hits,
            "agent_context": context,
            "steps": [*state["steps"], step],
        }

    def _answer(self, state: AgentState) -> dict:
        context = state["agent_context"] or build_comparison_context(
            state["knowledge_hits"],
            state["query_index"] + 1,
        )
        reply = self.llm_provider.generate_reply(
            LLMRequest(
                content=state["content"],
                mode=state["mode"],
                knowledge_hits=state["knowledge_hits"],
                previous_messages=state["previous_messages"],
                agent_context=context,
                include_history=bool(state["previous_messages"])
                and is_follow_up_message(state["content"])
                and state["route_type"]
                in {
                    KnowledgeRouteType.DOCUMENT_QA,
                    KnowledgeRouteType.SUMMARY_COMPARE,
                },
            )
        )
        step = self._step(
            state,
            "compose_answer",
            f"使用 {len(state['knowledge_hits'])} 个证据片段",
            "已生成最终回答",
            list(dict.fromkeys(hit.source.id for hit in state["knowledge_hits"])),
        )
        return {
            "agent_context": context,
            "reply": reply,
            "steps": [*state["steps"], step],
        }
