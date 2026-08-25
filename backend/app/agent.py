from __future__ import annotations

import math
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
from .word_facts import (
    find_query_field_aliases,
    normalize_fact_key,
    query_field_matches,
    query_file_reference_terms,
    query_overlap_terms,
    query_primary_field_terms,
    query_subject_terms,
)

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
LOW_CONFIDENCE_TOP_SCORE = 0.005
LOW_CONFIDENCE_QUERY_OVERLAP_MIN_LENGTH = 4
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
    retrieval_mode: str | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)


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
    # Retrieval output is an external/legacy boundary.  Do not let NaN/Inf
    # scores enter the merged set: Python's ordering with NaN is not a total
    # order and can make the selected source depend on insertion order.
    by_chunk_id = {
        hit.chunk.id: hit
        for hit in existing
        if _is_finite_score(hit.score)
    }
    for hit in incoming:
        if not _is_finite_score(hit.score):
            continue
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


def _is_finite_score(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def _format_audit_score(value: object) -> str:
    """Format an untrusted retrieval score without breaking the audit run."""

    if not _is_finite_score(value):
        return "n/a"
    return f"{float(value):.6f}"


def filter_relevant_hits(
    hits: Sequence[KnowledgeSearchHitModel],
    *,
    query: str = "",
    relative_score_floor: float = 0.05,
    minimum_score_floor: float = 0.01,
    adjacency_distance: int = 1,
    single_source: bool = False,
    diagnostics: dict[str, object] | None = None,
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
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics["candidate_count"] = len(hit_list)

    # Scores are an untrusted boundary: legacy retrieval rows, mocks, or a
    # partially-corrupt reranker response can contain NaN/Inf.  Letting one
    # of those values reach max()/sorting/threshold arithmetic makes the
    # result order and cutoff platform-dependent.  Drop them before any
    # source or score decisions and fail closed when nothing finite remains.
    finite_hits: list[KnowledgeSearchHitModel] = []
    invalid_score_count = 0
    for hit in hit_list:
        score = hit.score
        if not _is_finite_score(score):
            invalid_score_count += 1
            continue
        finite_hits.append(hit)
    if diagnostics is not None:
        diagnostics["invalid_score_count"] = invalid_score_count
        if invalid_score_count:
            diagnostics["candidate_count_after_score_validation"] = len(finite_hits)
    hit_list = finite_hits
    if not hit_list:
        if diagnostics is not None:
            diagnostics["reason"] = (
                "invalid_scores_filtered" if invalid_score_count else "no_candidates"
            )
            diagnostics["filtered_count"] = 0
        return []
    # An explicit filename is a hard user constraint. Apply it before score
    # thresholds so an unrelated document with a stronger semantic score
    # cannot become the answer source. The retrieval layer still receives the
    # original query (and therefore can rank the named file), while this final
    # guard fails closed if the bounded candidate set contains no such file.
    file_references = query_file_reference_terms(query) if query.strip() else ()
    if file_references and hit_list:
        normalized_references = tuple(
            normalize_fact_key(reference) for reference in file_references if reference
        )
        matching_by_reference = {
            reference: {
                hit.source.id
                for hit in hit_list
                if _file_reference_matches_source(reference, hit.source.name)
            }
            for reference in normalized_references
        }
        # ``query_file_reference_terms`` may retain both a cleaned wording
        # candidate and a longer raw basename candidate (for example
        # ``项目docx`` + ``关于项目docx``).  If the longer candidate matches
        # an uploaded source, prefer it; otherwise fall back to the cleaned
        # candidate.  This prevents a legitimate basename prefix from
        # broadening the hard file scope to a similarly named second source.
        matching_source_ids: set[str] = set()
        for reference, source_ids in matching_by_reference.items():
            if not source_ids:
                continue
            shadowed_by_longer_match = any(
                len(other) > len(reference)
                and other.endswith(reference)
                and matching_by_reference.get(other)
                for other in matching_by_reference
            )
            if not shadowed_by_longer_match:
                matching_source_ids.update(source_ids)
        if diagnostics is not None:
            diagnostics["file_reference_terms"] = list(file_references)
            diagnostics["candidate_count_before_file_scope"] = len(hit_list)
        if not matching_source_ids:
            if diagnostics is not None:
                diagnostics["reason"] = "explicit_file_reference_not_in_candidates"
                diagnostics["filtered_count"] = 0
            return []
        hit_list = [hit for hit in hit_list if hit.source.id in matching_source_ids]
        if diagnostics is not None:
            diagnostics["file_reference_source_ids"] = sorted(matching_source_ids)
            diagnostics["candidate_count"] = len(hit_list)
    if not hit_list:
        if diagnostics is not None:
            diagnostics["reason"] = "no_candidates"
            diagnostics["filtered_count"] = 0
        return []
    incompatible_field_count = 0
    if query.strip() and find_query_field_aliases(query):
        compatible_hits: list[KnowledgeSearchHitModel] = []
        for hit in hit_list:
            if _has_incompatible_field_label(query, hit.chunk.text):
                incompatible_field_count += 1
                continue
            compatible_hits.append(hit)
        hit_list = compatible_hits
        if diagnostics is not None:
            diagnostics["incompatible_field_count"] = incompatible_field_count
            diagnostics["candidate_count_after_field_validation"] = len(hit_list)
        if not hit_list:
            if diagnostics is not None:
                diagnostics["reason"] = "incompatible_fields_filtered"
                diagnostics["filtered_count"] = 0
            return []
    elif diagnostics is not None:
        diagnostics["incompatible_field_count"] = 0
    if any(value < 0 for value in (relative_score_floor, minimum_score_floor)):
        raise ValueError("score floors must be non-negative")
    if adjacency_distance < 0:
        raise ValueError("adjacency_distance must be non-negative")

    top_score = max(hit.score for hit in hit_list)
    if diagnostics is not None:
        diagnostics["score_min"] = min(hit.score for hit in hit_list)
        diagnostics["score_max"] = top_score
    if top_score <= 0:
        if diagnostics is not None:
            diagnostics["reason"] = "top_score_not_positive"
            diagnostics["filtered_count"] = 0
        return []

    # llama.cpp BGE reranker scores are converted to probabilities in [0, 1].
    # The old 5% floor is too permissive at that scale: a score of .04 could
    # still become evidence beside a .8 match.  Keep the historical behavior
    # for legacy/high-scale test and fallback scores, while applying a stricter
    # normalized-score gate to the production reranker output.
    normalized_scores = top_score <= 1.0 and all(0.0 <= hit.score <= 1.0 for hit in hit_list)
    effective_relative_floor = (
        max(relative_score_floor, 0.10) if normalized_scores else relative_score_floor
    )
    effective_minimum_floor = (
        max(minimum_score_floor, 0.03) if normalized_scores else minimum_score_floor
    )
    cutoff = max(effective_minimum_floor, top_score * effective_relative_floor)
    anchors = [hit for hit in hit_list if hit.score >= cutoff]
    lexical_fallback_anchor_only = False
    if (
        not anchors
        and normalized_scores
        and top_score >= LOW_CONFIDENCE_TOP_SCORE
        and query.strip()
    ):
        # A reranker can score an adjacent/context row above the row that
        # contains the requested field. Inspect the complete bounded candidate
        # set instead of trusting only the numerical top hit. For an explicit
        # subject (including multiple entities), retain every lexical anchor;
        # for a field-only question, retain only the strongest lexical anchor
        # so a generic header shared by many files does not fan out evidence.
        lexical_anchors = [
            hit
            for hit in hit_list
            if _has_query_overlap(
                query,
                hit.chunk.text,
                source_text=hit.source.name,
                source_type=hit.source.source_type,
            )
        ]
        subject_terms = query_subject_terms(query)
        if subject_terms:
            anchors = lexical_anchors
        elif lexical_anchors:
            anchors = [max(lexical_anchors, key=lambda hit: (hit.score, -hit.rank))]
        if anchors:
            # A low but positive BGE probability can still be the only useful
            # result for a short or synonym-heavy query. The normal adjacency
            # rule may add immediately adjacent chunks from the same source.
            diagnostics_cutoff = cutoff
            cutoff = max(hit.score for hit in anchors)
            lexical_fallback_anchor_only = True
            if diagnostics is not None:
                diagnostics["configured_cutoff"] = diagnostics_cutoff
                diagnostics["reason"] = "low_score_top_candidate_retained"
                diagnostics["lexical_anchor_chunk_ids"] = [
                    hit.chunk.id for hit in anchors[:8]
                ]
    if diagnostics is not None:
        diagnostics.update(
            {
                "normalized_scores": normalized_scores,
                "relative_floor": effective_relative_floor,
                "minimum_floor": effective_minimum_floor,
                "cutoff": cutoff,
                "anchor_only_fallback": lexical_fallback_anchor_only,
                "anchor_count": len(anchors),
                "anchor_chunk_ids": [hit.chunk.id for hit in anchors[:8]],
            }
        )
    if not anchors:
        if diagnostics is not None:
            diagnostics["reason"] = "all_candidates_below_cutoff"
            diagnostics["filtered_count"] = 0
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

    if diagnostics is not None:
        diagnostics["single_source"] = single_source
        diagnostics["anchor_source_ids"] = list(dict.fromkeys(hit.source.id for hit in anchors))
        diagnostics["allowed_source_ids"] = sorted(allowed_source_ids or set())

    anchor_indices: dict[str, list[int]] = {}
    for hit in anchors:
        anchor_indices.setdefault(hit.source.id, []).append(hit.chunk.chunk_index)

    filtered: list[KnowledgeSearchHitModel] = []
    dropped_by_source = 0
    dropped_by_score = 0
    anchor_chunk_ids = {hit.chunk.id for hit in anchors}
    for hit in hit_list:
        if allowed_source_ids is not None and hit.source.id not in allowed_source_ids:
            dropped_by_source += 1
            continue
        indices = anchor_indices.get(hit.source.id)
        if not indices:
            dropped_by_score += 1
            continue
        is_adjacent = any(
            abs(hit.chunk.chunk_index - anchor_index) <= adjacency_distance
            for anchor_index in indices
        )
        adjacent_conflict = bool(query.strip()) and _has_incompatible_field_label(
            query,
            hit.chunk.text,
        )
        if hit.chunk.id in anchor_chunk_ids or (
            is_adjacent and not adjacent_conflict
        ) or (
            not lexical_fallback_anchor_only and hit.score >= cutoff
        ):
            filtered.append(hit)
        else:
            dropped_by_score += 1

    if diagnostics is not None:
        diagnostics.update(
            {
                "filtered_count": len(filtered),
                "dropped_by_source": dropped_by_source,
                "dropped_by_score_or_adjacency": dropped_by_score,
                "filtered_source_ids": list(dict.fromkeys(hit.source.id for hit in filtered)),
            }
        )

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


def _has_query_overlap(
    query: str,
    text: str,
    *,
    source_text: str = "",
    source_type: str = "",
) -> bool:
    """Require a lexical signal before accepting a low-confidence top hit.

    In addition to literal question n-grams, include the configured query
    field vocabulary.  A user may ask for ``位置`` while a document labels the
    same value ``主要活动区域`` or ``地理位置``; treating those aliases as
    overlap prevents a genuinely relevant low-score hit from being discarded.
    """
    normalized_query = re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", query)
    ).casefold()
    normalized_text = re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", text)
    ).casefold()
    normalized_source = re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", source_text)
    ).casefold()
    source_is_tabular = _is_tabular_source_hint(source_text, source_type)
    if not normalized_query or not (normalized_text or normalized_source):
        return False

    # Field labels are evidence when they occur in the chunk body. A filename
    # such as ``地址簿.xlsx`` is metadata and cannot prove that the retrieved
    # row answers a bare ``地址`` question. For an explicit subject, a
    # narrative chunk may express the field semantically without repeating a
    # configured label (``蜘蛛侠常在纽约活动``), so a subject hit in the body
    # is also accepted as the field signal.
    field_query = bool(find_query_field_aliases(query))
    field_terms = {
        re.sub(r"\s+", "", term).casefold()
        for term in query_primary_field_terms(query)
        if term
    }
    overlap_text = (
        f"{normalized_text}{normalized_source}" if source_is_tabular else normalized_text
    )

    candidates = set(query_overlap_terms(query))
    # query_overlap_terms intentionally operates on normalized vocabulary,
    # but normalize punctuation once more for defensive compatibility with
    # callers that pass unusual Unicode forms.
    candidates = {
        re.sub(r"\s+", "", candidate).casefold()
        for candidate in candidates
        if candidate
    }
    stop_terms = {"什么", "如何", "怎么", "请问", "一下", "是否", "多少", "是什么"}
    # A field synonym alone is not enough to retain a low-confidence hit when
    # the question names a concrete subject.  For example, a generic policy
    # mentioning ``主要活动区域`` must not be accepted for ``蜘蛛侠的位置``
    # unless the subject/topic also appears in that candidate.  Field-only
    # questions such as ``位置是什么`` intentionally remain eligible.
    if field_query:
        subject_terms = query_subject_terms(query)
        field_in_chunk = query_field_matches(query, normalized_text)
        if not subject_terms and field_terms and not field_in_chunk:
            return False
        if subject_terms:
            # A named subject is not, by itself, evidence for every field
            # attached to that subject.  In particular, a chunk saying
            # ``蜘蛛侠性别：男`` must never become evidence for
            # ``蜘蛛侠年龄是多少`` merely because the entity matches.  The
            # field matcher includes configured aliases and guarded narrative
            # semantic terms (``岁``/``出生`` for age, ``男``/``女`` for
            # gender, etc.), so valid prose remains eligible while unrelated
            # attributes are rejected before lexical fallback can promote
            # them to LLM context.
            if not field_in_chunk and (
                not source_is_tabular
                or _has_incompatible_field_label(query, normalized_text)
            ):
                return False
            # Named entities/regions are hard anchors. A year or long numeric
            # identifier is only a soft anchor because a long document may
            # carry the date in its title/header while the matching split
            # chunk contains the entity and value row. Date-only questions
            # still require at least one numeric hit.
            lexical_terms = [
                term for term in subject_terms if not re.fullmatch(r"\d{4,}", term)
            ]
            numeric_terms = [
                term for term in subject_terms if re.fullmatch(r"\d{4,}", term)
            ]
            # A multi-entity question is commonly represented as one row per
            # entity (for example, ``张三和李四的联系方式``). The low-score
            # guard runs on one candidate at a time, so requiring every entity
            # in the same chunk would discard both valid rows before the
            # source/adjacency stage can combine them. A single-entity query
            # remains strict; for multiple entities one explicit subject is
            # enough to establish that this candidate belongs to the request.
            if lexical_terms and not (
                any(term in overlap_text for term in lexical_terms)
                if len(lexical_terms) > 1
                else lexical_terms[0] in overlap_text
            ):
                return False
            if not lexical_terms and numeric_terms and not any(
                term in overlap_text for term in numeric_terms
            ):
                return False
            subject_in_chunk = any(term in normalized_text for term in lexical_terms)
            if not field_in_chunk and not subject_in_chunk and not source_is_tabular:
                return False
    matched_lengths = sorted(
        (
            len(candidate)
            for candidate in candidates
            if candidate not in stop_terms and candidate in overlap_text
        ),
        reverse=True,
    )
    if not matched_lengths:
        # A recognized field label is sufficient for a field-only query after
        # the subject/metric guards above. Two-character Chinese headers such
        # as ``年龄`` and ``温度`` are otherwise below the generic n-gram
        # overlap threshold.
        return bool(
            field_query
            and (
                field_in_chunk
                or (
                    source_is_tabular
                    and subject_terms
                    and any(term in overlap_text for term in lexical_terms)
                    and not _has_incompatible_field_label(query, normalized_text)
                )
            )
        )
    return (
        matched_lengths[0] >= LOW_CONFIDENCE_QUERY_OVERLAP_MIN_LENGTH
        or len(matched_lengths) >= 2
        or (field_query and field_in_chunk)
        or (
            field_query
            and source_is_tabular
            and subject_terms
            and any(term in overlap_text for term in lexical_terms)
            and not _has_incompatible_field_label(query, normalized_text)
        )
    )


def _is_tabular_source_hint(source_name: str, source_type: str) -> bool:
    normalized_name = (source_name or "").casefold().split("?", 1)[0]
    normalized_type = (source_type or "").casefold()
    return normalized_name.endswith((".xlsx", ".xls", ".xlsb", ".csv")) or any(
        marker in normalized_type
        for marker in ("xlsx", "excel", "xls", "csv", "表格", "电子表")
    )


def _has_incompatible_field_label(query: str, text: str) -> bool:
    """Return whether a tabular row explicitly names a different field.

    Wide sheets can force a data-row chunk to omit its repeated header.  Such
    a row is still useful when its subject/date filters match, provided it
    does not explicitly advertise a conflicting metric (for example ``成本``
    for a ``销售额`` question).  ``query_field_matches`` handles compatible
    aliases before this helper is called; the remaining observed labels are
    therefore safe to treat as conflicts.
    """

    target_fields = {match.field for match in find_query_field_aliases(query)}
    if not target_fields:
        return False
    non_metric_fields = {"日期", "地区", "姓名"}
    target_metric_fields = target_fields - non_metric_fields
    # A filter-only question may legitimately retrieve a row that also
    # contains metric columns; those columns are context, not competing
    # answer fields.  Do not reject such rows at this lexical guard.
    if not target_metric_fields:
        return False
    # Reuse the canonical compatibility rules first.  In particular, a raw
    # ``温度`` header is a valid metric for an ``平均温度`` request, while an
    # explicitly qualified ``最低温度`` header is not.
    if query_field_matches(query, text):
        return False
    # Date/region/name labels are commonly filters that remain in a wide
    # table row even when the requested measure header was omitted by chunk
    # budgeting. They are not competing answer fields and must not make a
    # valid row look incompatible.
    observed_fields = {
        match.field
        for match in find_query_field_aliases(text)
        if match.field not in non_metric_fields
    }
    return bool(observed_fields and observed_fields.isdisjoint(target_fields))


def _file_reference_matches_source(reference: str, source_name: str) -> bool:
    """Match an explicit file reference against an uploaded source basename.

    File scope is a hard constraint. Compare normalized basenames exactly so
    ``report.xlsx`` cannot accidentally select ``my_report.xlsx``.
    """

    normalized_reference = normalize_fact_key(reference)
    raw_source = (source_name or "").split("?", 1)[0]
    source_basename = re.split(r"[/\\]", raw_source)[-1]
    normalized_source = normalize_fact_key(source_basename)
    if not normalized_reference or not normalized_source:
        return False
    return normalized_source == normalized_reference


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
            f"{hit.chunk.id}:{_format_audit_score(hit.score)}"
            for hit in hits[: self.max_hits]
        )
        output_summary = f"命中 {len(hits)} 个片段，累计保留 {len(merged)} 个片段"
        if evidence_audit:
            output_summary += f"；evidence={evidence_audit}"
        if search_result.fallback_reason:
            output_summary += f"；fallback={search_result.fallback_reason}"
        if search_result.retrieval_mode:
            output_summary += f"；retrieval_mode={search_result.retrieval_mode}"
        if search_result.stage_ms:
            timing_summary = ", ".join(
                f"{name}={duration:.1f}ms"
                for name, duration in search_result.stage_ms.items()
            )
            output_summary += f"；stage_ms={timing_summary}"
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
        finite_hits = [hit for hit in hits if _is_finite_score(hit.score)]
        if not finite_hits:
            return True
        source_count = len({hit.source.id for hit in finite_hits})
        top_score = max(hit.score for hit in finite_hits)
        # Scores from the BGE reranker are probabilities; legacy fallback
        # scores historically used a larger scale.  Use the matching floor so
        # deep mode does not keep expanding a clearly good normalized hit.
        score_floor = 0.6 if 0.0 <= top_score <= 1.0 else 6.0
        return source_count < 2 or top_score < score_floor

    def _advance_query(self, state: AgentState) -> dict:
        return {"query_index": state["query_index"] + 1}

    def _compare(self, state: AgentState) -> dict:
        rounds = state["query_index"] + 1
        diagnostics: dict[str, object] = {}
        filtered_hits = filter_relevant_hits(
            state["knowledge_hits"],
            query=state["content"],
            single_source=(
                state["route_type"] == KnowledgeRouteType.DOCUMENT_QA
                and state["mode"] != "source"
            ),
            diagnostics=diagnostics,
        )
        context = build_comparison_context(filtered_hits, rounds)
        source_ids = list(dict.fromkeys(hit.source.id for hit in filtered_hits))
        reason_labels = {
            "no_candidates": "检索结果为空",
            "top_score_not_positive": "最高相关性分数不大于 0",
            "all_candidates_below_cutoff": "所有候选片段都低于证据阈值",
            "low_score_top_candidate_retained": "候选分数整体偏低，但保留了与问题有文本重合的最高候选",
            "invalid_scores_filtered": "候选包含非有限分数，已安全过滤",
            "incompatible_fields_filtered": "候选字段与问题目标不兼容，已安全过滤",
        }
        diagnostic_reason = reason_labels.get(
            str(diagnostics.get("reason")),
            "已保留满足阈值的证据" if filtered_hits else "过滤后没有可用证据",
        )
        diagnostic_summary = (
            "证据诊断："
            f"候选={diagnostics.get('candidate_count', 0)}，"
            f"无效分数={diagnostics.get('invalid_score_count', 0)}，"
            f"冲突字段={diagnostics.get('incompatible_field_count', 0)}，"
            f"锚点={diagnostics.get('anchor_count', 0)}，"
            f"保留={diagnostics.get('filtered_count', 0)}，"
            f"分数范围={diagnostics.get('score_min', 'n/a')}..{diagnostics.get('score_max', 'n/a')}，"
            f"阈值={diagnostics.get('cutoff', 'n/a')}，"
            f"来源丢弃={diagnostics.get('dropped_by_source', 0)}，"
            f"分数/邻接丢弃={diagnostics.get('dropped_by_score_or_adjacency', 0)}，"
            f"结论={diagnostic_reason}"
        )
        step = self._step(
            state,
            "compare_evidence",
            (
                f"过滤前 {len(state['knowledge_hits'])} 个片段，"
                f"{len(set(hit.source.id for hit in state['knowledge_hits']))} 个来源"
            ),
            f"{context}；{diagnostic_summary}",
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
            "已生成最终回答；已调用大模型" if state["knowledge_hits"] else "未调用大模型，直接返回无证据提示",
            list(dict.fromkeys(hit.source.id for hit in state["knowledge_hits"])),
        )
        return {
            "agent_context": context,
            "reply": reply,
            "steps": [*state["steps"], step],
        }
