from __future__ import annotations

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
    search_knowledge: Callable[[str, int, str], AgentSearchResult]


class AgentState(TypedDict):
    run_id: str
    conversation_id: str
    content: str
    mode: ComposerMode
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


def build_agent_search_queries(content: str, mode: ComposerMode) -> list[str]:
    query = content.strip()
    if mode == "quick":
        return [query]

    terms = extract_embedding_terms(query)
    expanded = [term for term in expand_terms(terms) if term not in terms]
    if expanded:
        broader_query = " ".join([query, *expanded[:12]])
    else:
        broader_query = f"{query} 相关制度 规定 流程 依据"
    return list(dict.fromkeys([query, broader_query]))


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
                adjacency_allowed=True,
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
        queries = build_agent_search_queries(state["content"], state["mode"])
        step = self._step(
            state,
            "plan_retrieval",
            state["content"],
            f"生成 {len(queries)} 个有界检索策略",
        )
        return {"search_queries": queries, "steps": [*state["steps"], step]}

    def _search(self, state: AgentState) -> dict:
        query = state["search_queries"][state["query_index"]]
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
        step = self._step(
            state,
            "search_knowledge",
            query,
            f"命中 {len(hits)} 个片段，累计保留 {len(merged)} 个片段",
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
        return source_count < 2 or top_score < 6.0

    def _advance_query(self, state: AgentState) -> dict:
        return {"query_index": state["query_index"] + 1}

    def _compare(self, state: AgentState) -> dict:
        rounds = state["query_index"] + 1
        context = build_comparison_context(state["knowledge_hits"], rounds)
        source_ids = list(dict.fromkeys(hit.source.id for hit in state["knowledge_hits"]))
        step = self._step(
            state,
            "compare_evidence",
            f"{len(source_ids)} 个来源",
            context,
            source_ids,
        )
        return {"agent_context": context, "steps": [*state["steps"], step]}

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
