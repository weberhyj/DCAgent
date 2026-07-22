from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import PurePath
from typing import Literal
from uuid import uuid4

from .agent import AgentRunResult, AgentStep
from .models import ChatMessageModel, ComposerMode, ResponseParagraphModel
from .structured_models import (
    StructuredAggregateResult,
    StructuredCatalog,
    StructuredClarification,
    StructuredColumnSchema,
    StructuredDatasetCatalog,
    StructuredFilter,
    StructuredIntent,
    StructuredUnavailable,
)
from .structured_query import (
    StructuredQueryExecutor,
    StructuredQueryPlanner,
    UnsafeStructuredQueryError,
    _explicit_equality_value_spans,
    resolve_structured_intent,
)
from .time_utils import display_datetime_label

_CHINESE_AGGREGATE_TERMS = (
    "平均值",
    "平均",
    "均值",
    "总和",
    "合计",
    "求和",
    "多少条",
    "数量",
    "计数",
    "最大值",
    "最大",
    "最高",
    "最小值",
    "最小",
    "最低",
)
_IMPLICIT_ROW_COUNT_RE = re.compile(
    r"^(?:(?:总共|一共|共有)有?)?多少条(?:记录|数据|明细|行)?[？?。.]?$"
)
_STRONG_AGGREGATE_SUFFIXES = tuple(
    sorted(
        (
            "平均值",
            "平均",
            "均值",
            "总和",
            "合计",
            "求和",
            "计数",
            "最大值",
            "最大",
            "最高",
            "最小值",
            "最小",
            "最低",
        ),
        key=len,
        reverse=True,
    )
)
_HAS_EXPLICIT_FILTER_RE = re.compile(
    r"(?:(?:大于|不少于|小于|不超过|[<>]=?)\s*-?\d+(?:\.\d+)?)|="
    r"|(?:\d{4}-\d{2}-\d{2}\s*至\s*\d{4}-\d{2}-\d{2})"
)
_CONCEPT_ANYWHERE_PHRASES = ("什么是", "什么叫", "何为", "是什么意思")
_CONCEPT_TERM_INTRODUCERS = ("解释一下", "讲讲", "介绍一下", "说明一下")
_CONCEPT_TERM_SUFFIXES = (
    "是什么",
    "是什么意思",
    "怎么理解",
    "如何理解",
    "的含义",
    "含义",
    "的概念",
    "概念",
    "的定义",
    "定义",
)
_COPULA_FRAGMENTS = frozenset(("因为", "作为", "称为", "成为", "认为", "何为"))
_AGGREGATE_CONCEPT_TERMS = tuple(
    sorted(("算术平均值", *_CHINESE_AGGREGATE_TERMS), key=len, reverse=True)
)
_NATURAL_AGGREGATE_TAILS = ("是多少", "有多少", "多少", "呢", "吗")
_EQUALITY_FIELD_DELIMITERS = ("，", ",", "。", "；", ";", "且", "或")


class StructuredAnswerService:
    def __init__(
        self,
        catalog_provider: Callable[[], StructuredCatalog],
        clickhouse_gateway: object,
    ) -> None:
        self._catalog_provider = catalog_provider
        self._clickhouse_gateway = clickhouse_gateway

    def close(self) -> None:
        close = getattr(self._clickhouse_gateway, "close", None)
        if callable(close):
            close()

    def try_answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult | None:
        del previous_messages
        question = content.strip()
        if not _has_aggregate_language(question):
            return None

        try:
            catalog = self._catalog_provider()
        except Exception:
            if _classify_without_catalog(question) != "strong":
                return None
            return _structured_run(
                conversation_id,
                question,
                mode,
                "结构化查询服务不可用：无法读取已发布的数据目录。",
                "catalog unavailable",
            )
        if not is_structured_candidate(question, catalog):
            return None

        resolution = resolve_structured_intent(question, catalog)
        if isinstance(resolution, StructuredClarification):
            candidates = "、".join(resolution.candidates)
            suffix = f" 可选项：{candidates}。" if candidates else ""
            return _structured_run(
                conversation_id,
                question,
                mode,
                f"需要澄清后才能查询结构化数据：{resolution.message}。{suffix}".strip(),
                "structured clarification required",
            )
        if isinstance(resolution, StructuredUnavailable):
            return _structured_run(
                conversation_id,
                question,
                mode,
                f"结构化查询服务不可用：{resolution.message}。",
                "structured intent unavailable",
            )

        publication = _active_publication(catalog, resolution)
        if publication is None:
            return _structured_run(
                conversation_id,
                question,
                mode,
                "结构化查询服务不可用：数据集没有有效的活动发布版本。",
                "active publication unavailable",
            )
        try:
            plan = StructuredQueryPlanner(catalog).plan(resolution, publication)
        except UnsafeStructuredQueryError:
            return _structured_run(
                conversation_id,
                question,
                mode,
                "结构化查询服务不可用：查询计划未通过安全校验。",
                "structured query planning failed",
            )
        result = StructuredQueryExecutor(catalog, self._clickhouse_gateway).execute(plan)
        if isinstance(result, StructuredUnavailable):
            return _structured_run(
                conversation_id,
                question,
                mode,
                f"结构化查询服务不可用：{result.message}。",
                "structured query unavailable",
            )
        return _structured_run(
            conversation_id,
            question,
            mode,
            _format_result(result),
            f"structured aggregate completed; audit_id={result.audit_id}",
            source_ids=[result.dataset_id],
        )


def is_structured_candidate(question: str, catalog: StructuredCatalog) -> bool:
    if not _has_aggregate_language(question):
        return False
    if (
        _is_implicit_row_count(question)
        and len([dataset for dataset in catalog.datasets if dataset.active_publication is not None])
        == 1
    ):
        return True

    catalog_names = {
        name for dataset in catalog.datasets for name in _dataset_names(dataset) if name
    }
    filter_columns = tuple(
        column
        for dataset in catalog.datasets
        for column in dataset.schema.columns
        if column.allow_filter
    )
    normalized = _normalize(
        _mask_aggregate_equality_values(question, filter_columns, catalog_names)
    )
    return _has_catalog_span_with_independent_aggregate(normalized, catalog_names)


def _dataset_names(dataset: StructuredDatasetCatalog) -> tuple[str, ...]:
    names = {
        _normalize(dataset.schema.dataset_id),
        _normalize(dataset.source_name),
        _normalize(PurePath(dataset.source_name).stem),
        _normalize(dataset.schema.worksheet_name),
    }
    for column in dataset.schema.columns:
        names.update(
            _normalize(value)
            for value in (
                column.physical_name,
                column.original_name,
                column.display_name,
                *column.aliases,
            )
        )
    return tuple(name for name in names if name)


def _has_aggregate_language(question: str) -> bool:
    normalized = _normalize(question)
    return any(_normalize(term) in normalized for term in _CHINESE_AGGREGATE_TERMS)


def _has_catalog_span_with_independent_aggregate(
    normalized_question: str,
    catalog_names: set[str],
) -> bool:
    spans: set[tuple[int, int]] = set()
    for name in catalog_names:
        start = normalized_question.find(name)
        while start >= 0:
            spans.add((start, start + len(name)))
            start = normalized_question.find(name, start + 1)

    maximal_spans = (
        span
        for span in spans
        if not any(
            other_start <= span[0]
            and span[1] <= other_end
            and other_end - other_start > span[1] - span[0]
            for other_start, other_end in spans
        )
    )
    for start, end in maximal_spans:
        remaining = normalized_question[:start] + "_" * (end - start) + normalized_question[end:]
        if _has_aggregate_language(remaining):
            return True
    return False


def _mask_aggregate_equality_values(
    question: str,
    filter_columns: tuple[StructuredColumnSchema, ...],
    catalog_names: set[str],
) -> str:
    masked = list(question)
    spans = {
        span
        for column in filter_columns
        for span in _explicit_equality_value_spans(question, (column,))
    }
    for value_start, value_end in spans:
        value = question[value_start:value_end]
        if _has_aggregate_language(value) and not _has_catalog_span_with_independent_aggregate(
            _normalize(value), catalog_names
        ):
            masked[value_start:value_end] = "_" * (value_end - value_start)
    return "".join(masked)


def _is_implicit_row_count(question: str) -> bool:
    return _IMPLICIT_ROW_COUNT_RE.fullmatch(question.strip()) is not None


def _classify_without_catalog(question: str) -> Literal["weak", "strong", "concept"]:
    stripped = question.strip()
    normalized = _normalize(stripped)
    if not _has_aggregate_language(normalized):
        return "weak"
    if _HAS_EXPLICIT_FILTER_RE.search(stripped) or _has_chinese_equality_filter(stripped):
        return "strong"
    if _has_metric_qualified_concept_shape(normalized):
        return "strong"
    if _is_aggregate_concept_question(normalized):
        return "concept"
    if _is_implicit_row_count(stripped) or _has_field_aggregate_suffix(normalized):
        return "strong"
    return "weak"


def _has_metric_qualified_concept_shape(normalized: str) -> bool:
    for phrase in _CONCEPT_ANYWHERE_PHRASES:
        start = normalized.find(phrase)
        while start >= 0:
            if _has_field_aggregate_suffix(normalized[start + len(phrase) :]):
                return True
            start = normalized.find(phrase, start + 1)
    if normalized.endswith("是什么"):
        return _has_field_aggregate_suffix(normalized[: -len("是什么")])
    return False


def _has_chinese_equality_filter(question: str) -> bool:
    for index, character in enumerate(question):
        context = _normalize(question[max(0, index - 2) : index + 1])
        if character != "为" or any(context.endswith(item) for item in _COPULA_FRAGMENTS):
            continue
        field_start = max(question.rfind(item, 0, index) for item in _EQUALITY_FIELD_DELIMITERS) + 1
        field = question[field_start:index]
        value = question[index + 1 :]
        if not _normalize(field) or not _normalize(value):
            continue
        remaining = question[:field_start] + "_" * len(field) + question[index:]
        if _has_aggregate_language(remaining):
            return True
    return False


def _has_field_aggregate_suffix(normalized: str) -> bool:
    base = _strip_natural_aggregate_tail(normalized)
    suffix = _matching_aggregate_suffix(base)
    if suffix is None:
        return False
    prefix = base[: -len(suffix)]
    if prefix.endswith("的"):
        prefix = prefix[:-1]
    return bool(prefix)


def _matching_aggregate_suffix(normalized: str) -> str | None:
    return next((term for term in _STRONG_AGGREGATE_SUFFIXES if normalized.endswith(term)), None)


def _strip_natural_aggregate_tail(normalized: str) -> str:
    remaining = normalized
    while remaining:
        tail = next((item for item in _NATURAL_AGGREGATE_TAILS if remaining.endswith(item)), None)
        if tail is None or len(tail) >= len(remaining):
            return remaining
        remaining = remaining[: -len(tail)]
    return remaining


def _is_aggregate_concept_question(normalized: str) -> bool:
    if not _has_aggregate_language(normalized):
        return False
    if any(phrase in normalized for phrase in _CONCEPT_ANYWHERE_PHRASES):
        return True
    for introducer in _CONCEPT_TERM_INTRODUCERS:
        start = normalized.find(introducer)
        while start >= 0:
            remainder = normalized[start + len(introducer) :]
            if _matches_concept_term_phrase(remainder, allow_bare=True):
                return True
            start = normalized.find(introducer, start + 1)
    if (
        "说明因为" in normalized
        and _contains_aggregate_concept_term(normalized)
        and normalized.endswith(("影响", "原因", "后果", "结果"))
    ):
        return True
    if (
        "介绍被称为" in normalized
        and _contains_aggregate_concept_term(normalized)
        and normalized.endswith(("概念", "含义", "定义"))
    ):
        return True
    return any(
        _matches_concept_term_phrase(normalized[start:], allow_bare=False)
        for term in _AGGREGATE_CONCEPT_TERMS
        for start in _find_occurrence_starts(normalized, term)
    )


def _find_occurrence_starts(value: str, term: str) -> tuple[int, ...]:
    starts: list[int] = []
    start = value.find(term)
    while start >= 0:
        starts.append(start)
        start = value.find(term, start + 1)
    return tuple(starts)


def _matches_concept_term_phrase(value: str, *, allow_bare: bool) -> bool:
    for term in _AGGREGATE_CONCEPT_TERMS:
        if not value.startswith(term):
            continue
        remainder = value[len(term) :]
        return (allow_bare and not remainder) or remainder in _CONCEPT_TERM_SUFFIXES
    return False


def _contains_aggregate_concept_term(value: str) -> bool:
    return any(term in value for term in _AGGREGATE_CONCEPT_TERMS)


def _normalize(value: str) -> str:
    return re.sub(r"[\s\W]+", "", value.casefold(), flags=re.UNICODE)


def _active_publication(catalog: StructuredCatalog, intent: StructuredIntent):
    matches = [
        dataset.active_publication
        for dataset in catalog.datasets
        if dataset.schema.dataset_id == intent.dataset_id and dataset.active_publication is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _format_result(result: StructuredAggregateResult) -> str:
    metric = result.metric_display_name or result.metric_physical_name or "all_rows"
    value = _format_numeric_value(result.value)
    return (
        "结构化查询结果："
        f"source_file={result.source_name}; "
        f"worksheet={result.worksheet_name}; "
        f"aggregate={result.aggregate}; "
        f"metric={metric}; "
        f"value={value}; "
        f"total={result.total_count}; "
        f"valid={result.valid_count}; "
        f"null={result.null_count}; "
        f"filters={_format_filters(result.filters)}; "
        f"schema_version={result.schema_version}; "
        f"publication_version={result.publication_id}; "
        f"publication_id={result.publication_id}; "
        f"elapsed_ms={result.elapsed_ms:.3f}; "
        f"audit_id={result.audit_id}"
    )


def _format_numeric_value(value: Decimal | int | None) -> str:
    if value is None:
        return "null"
    return format(value, ",")


def _format_filters(filters: tuple[StructuredFilter, ...]) -> str:
    if not filters:
        return "none"
    return ",".join(
        (
            f"{item.physical_name}:{item.operator}:{item.value}"
            if item.upper_value is None
            else f"{item.physical_name}:{item.operator}:{item.value}..{item.upper_value}"
        )
        for item in filters
    )


def _structured_run(
    conversation_id: str,
    question: str,
    mode: ComposerMode,
    answer: str,
    output_summary: str,
    *,
    source_ids: list[str] | None = None,
) -> AgentRunResult:
    timestamp = display_datetime_label()
    run_id = f"agent-{uuid4().hex[:12]}"
    reply = ChatMessageModel(
        id=f"msg-{uuid4().hex[:8]}",
        role="assistant",
        time=timestamp,
        paragraphs=[ResponseParagraphModel(text=answer)],
    )
    step = AgentStep(
        id=f"step-{uuid4().hex[:12]}",
        step_index=0,
        tool_name="query_structured_data",
        status="completed",
        input_summary=question,
        output_summary=output_summary,
        source_ids=source_ids or [],
        read_only=True,
        started_at=timestamp,
        completed_at=timestamp,
    )
    return AgentRunResult(
        id=run_id,
        conversation_id=conversation_id,
        query=question,
        mode=mode,
        status="completed",
        started_at=timestamp,
        completed_at=timestamp,
        reply=reply,
        steps=[step],
        evidence_count=0,
        source_count=len(set(source_ids or [])),
    )
