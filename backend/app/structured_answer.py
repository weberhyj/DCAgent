from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import PurePath
from uuid import uuid4

from .agent import AgentRunResult, AgentStep
from .models import ChatMessageModel, ComposerMode, ResponseParagraphModel
from .structured_models import (
    StructuredAggregateResult,
    StructuredCatalog,
    StructuredClarification,
    StructuredDatasetCatalog,
    StructuredFilter,
    StructuredIntent,
    StructuredUnavailable,
)
from .structured_query import (
    StructuredQueryExecutor,
    StructuredQueryPlanner,
    UnsafeStructuredQueryError,
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
_STRONG_AGGREGATE_SUFFIX_RE = re.compile(
    r"(?:的)?(?:平均值|平均|均值|总和|合计|求和|计数|最大值|最大|最高|最小值|最小|最低)"
    r"[？?。.]?$"
)
_STRUCTURED_FILTER_RE = re.compile(
    r"(?:大于|不少于|小于|不超过)|(?:\d{4}-\d{2}-\d{2}\s*至\s*\d{4}-\d{2}-\d{2})"
)


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
            if not _is_strong_structured_shape(question):
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
    normalized = _normalize(question)
    if not _has_aggregate_language(question):
        return False
    if (
        _is_implicit_row_count(question)
        and len([dataset for dataset in catalog.datasets if dataset.active_publication is not None])
        == 1
    ):
        return True

    remaining = normalized
    matched_catalog_name = False
    catalog_names = {
        name for dataset in catalog.datasets for name in _dataset_names(dataset) if name
    }
    for name in sorted(catalog_names, key=len, reverse=True):
        if name in remaining:
            matched_catalog_name = True
            remaining = remaining.replace(name, "")
    return matched_catalog_name and _has_aggregate_language(remaining)


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


def _is_implicit_row_count(question: str) -> bool:
    return _IMPLICIT_ROW_COUNT_RE.fullmatch(question.strip()) is not None


def _is_strong_structured_shape(question: str) -> bool:
    stripped = question.strip()
    if _is_aggregate_concept_question(stripped):
        return False
    return (
        _is_implicit_row_count(stripped)
        or _STRONG_AGGREGATE_SUFFIX_RE.search(stripped) is not None
        or (
            _has_aggregate_language(stripped) and _STRUCTURED_FILTER_RE.search(stripped) is not None
        )
    )


def _is_aggregate_concept_question(question: str) -> bool:
    normalized = _normalize(question)
    return normalized.startswith(("什么是", "何为")) or normalized.endswith(
        ("是什么", "是什么意思")
    )


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
