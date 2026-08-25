from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import PurePath
from threading import Lock
from typing import Literal
from uuid import uuid4

from .agent import AgentRunResult, AgentStep
from .clickhouse_compatibility import (
    ClickHouseCompatibilityMode,
    ClickHouseCompatibilityProfile,
)
from .knowledge_route_models import KnowledgeRouteMetadata, KnowledgeRouteType
from .models import (
    ArtifactModel,
    ChatMessageModel,
    ComposerMode,
    ResponseParagraphModel,
    TableArtifactModel,
)
from .structured_models import (
    StructuredAggregateResult,
    StructuredCatalog,
    StructuredClarification,
    StructuredColumnSchema,
    StructuredDatasetCatalog,
    StructuredIntent,
    StructuredMultiAggregateIntent,
    StructuredMultiAggregateResult,
    StructuredRowLookupIntent,
    StructuredRowLookupResult,
    StructuredUnavailable,
)
from .structured_query import (
    StructuredQueryExecutor,
    StructuredQueryPlanner,
    UnsafeStructuredQueryError,
    _candidate_equality_value_spans,
    _resolution_names,
    resolve_structured_intent,
)
from .time_utils import display_datetime_label

_CHINESE_AGGREGATE_TERMS = (
    "平均值",
    "平均",
    "均值",
    "均温",
    "总和",
    "合计",
    "求和",
    "多少条",
    "数量",
    "计数",
    "去重数量",
    "去重数",
    "不同数量",
    "唯一数量",
    "不重复数量",
    "最大值",
    "最大",
    "最高",
    "最小值",
    "最小",
    "最低",
    "中位数",
    "中位值",
    "标准差",
    "标准偏差",
    "方差",
    "变异数",
    "分位数",
    "百分位",
    "百分位数",
    "汇总",
    "统计",
)
_AGGREGATE_WORDS = (
    ("avg", ("平均值", "平均", "均值", "均温")),
    ("sum", ("总和", "合计", "求和")),
    ("count", ("多少条", "数量", "计数")),
    ("count_distinct", ("去重数量", "去重数", "不同数量", "唯一数量", "不重复数量")),
    ("max", ("最大", "最高")),
    ("min", ("最小", "最低")),
    ("median", ("中位数", "中位值")),
    ("stddev", ("标准差", "标准偏差")),
    ("variance", ("方差", "变异数")),
    ("percentile", ("分位数", "百分位", "百分位数")),
)
_IMPLICIT_ROW_COUNT_RE = re.compile(
    r"^(?:(?:总共|一共|共有)有?)?多少条(?:记录|数据|明细|行)?[？?。.]?$"
)
_IMPLICIT_SUMMARY_RE = re.compile(r"^(?:请)?(?:汇总|统计)[？?。.]?$")
_STRONG_AGGREGATE_SUFFIXES = tuple(
    sorted(
        (
            "平均值",
            "平均",
            "均值",
            "均温",
            "总和",
            "合计",
            "求和",
            "计数",
            "去重数量",
            "去重数",
            "不同数量",
            "唯一数量",
            "不重复数量",
            "最大值",
            "最大",
            "最高",
            "最小值",
            "最小",
            "最低",
            "中位数",
            "中位值",
            "标准差",
            "标准偏差",
            "方差",
            "分位数",
            "百分位数",
            "百分位",
            "汇总",
            "统计",
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
_CONCEPT_COPULA_REQUEST_PREFIXES = ("请解释一下", "请介绍一下", "请说明")
_CONCEPT_COPULA_PREDICATE_SUFFIXES = (
    "指标",
    "概念",
    "术语",
    "方法",
    "统计量",
    "度量",
    "定义",
    "含义",
)
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
_NAMED_AVERAGE_CONCEPT_TERMS = (
    "算术平均值",
    "加权平均值",
    "移动平均值",
    "几何平均值",
    "调和平均值",
)
_NATURAL_QUESTION_PARTICLES = ("呢", "吗", "吧", "呀", "啊")
_AGGREGATE_CONCEPT_TERMS = tuple(
    sorted((*_NAMED_AVERAGE_CONCEPT_TERMS, *_CHINESE_AGGREGATE_TERMS), key=len, reverse=True)
)
_NATURAL_AGGREGATE_TAILS = ("是多少", "有多少", "多少", "呢", "吗")
_EQUALITY_FIELD_DELIMITERS = ("，", ",", "。", "；", ";", "且", "或")


class StructuredAnswerService:
    def __init__(
        self,
        catalog_provider: Callable[[], StructuredCatalog],
        clickhouse_gateway: object,
        compatibility: ClickHouseCompatibilityProfile | None = None,
        implicit_summary_max_metrics: int = 12,
    ) -> None:
        self._catalog_provider = catalog_provider
        self._clickhouse_gateway = clickhouse_gateway
        self._compatibility = compatibility or ClickHouseCompatibilityProfile.for_mode(
            ClickHouseCompatibilityMode.MODERN
        )
        self._implicit_summary_max_metrics = implicit_summary_max_metrics
        self._catalog_snapshot: StructuredCatalog | None = None
        self._catalog_snapshot_lock = Lock()
        self._catalog_request_generation = 0
        self._latest_successful_catalog_generation = 0

    def close(self) -> None:
        close = getattr(self._clickhouse_gateway, "close", None)
        if callable(close):
            close()

    def catalog_snapshot(self) -> StructuredCatalog | None:
        """Return the last successfully loaded catalog for route decisions."""

        return self._get_catalog_snapshot()

    def try_answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult | None:
        del previous_messages
        question = content.strip()
        if not _has_aggregate_language(question) and not _has_row_lookup_markers(question):
            return None

        catalog_request_generation = self._next_catalog_request_generation()
        try:
            catalog = self._catalog_provider()
        except Exception:
            catalog_snapshot = self._get_catalog_snapshot()
            if catalog_snapshot is not None:
                if not is_structured_candidate(question, catalog_snapshot):
                    return None
                outage_resolution = resolve_structured_intent(
                    question,
                    catalog_snapshot,
                    implicit_summary_max_metrics=self._implicit_summary_max_metrics,
                )
                outage_metadata = _catalog_outage_route_metadata(
                    catalog_snapshot, outage_resolution
                )
                outage_route = _route_for_catalog_outage_resolution(outage_resolution)
            else:
                if _classify_without_catalog(question) != "strong":
                    return None
                outage_metadata = {}
                outage_route = KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE
            return _structured_run(
                conversation_id,
                question,
                mode,
                "结构化查询服务不可用：无法读取已发布的数据目录。",
                "catalog unavailable",
                route_type=outage_route,
                route_metadata=KnowledgeRouteMetadata(
                    **outage_metadata,
                    degradation_reason="catalog_unavailable",
                    validation_passed=False,
                ),
            )
        self._replace_catalog_snapshot(catalog, catalog_request_generation)
        if not is_structured_candidate(question, catalog):
            return None

        resolution = resolve_structured_intent(
            question,
            catalog,
            implicit_summary_max_metrics=self._implicit_summary_max_metrics,
        )
        if isinstance(resolution, StructuredClarification):
            candidates = "、".join(resolution.candidates)
            suffix = f" 可选项：{candidates}。" if candidates else ""
            return _structured_run(
                conversation_id,
                question,
                mode,
                f"需要澄清后才能查询结构化数据：{resolution.message}。{suffix}".strip(),
                "structured clarification required",
                route_type=KnowledgeRouteType.CLARIFICATION,
                route_metadata=KnowledgeRouteMetadata(
                    **_clarification_route_metadata(catalog, resolution),
                    validation_passed=True,
                ),
            )
        if isinstance(resolution, StructuredUnavailable):
            return _structured_run(
                conversation_id,
                question,
                mode,
                f"结构化查询服务不可用：{resolution.message}。",
                "structured intent unavailable",
                route_type=_route_for_parser_outcome(resolution),
                route_metadata=KnowledgeRouteMetadata(
                    **_unavailable_route_metadata(resolution),
                    degradation_reason="intent_unavailable",
                    validation_passed=False,
                ),
            )

        publication = _active_publication(catalog, resolution)
        if publication is None:
            return _structured_run(
                conversation_id,
                question,
                mode,
                "结构化查询服务不可用：数据集没有有效的活动发布版本。",
                "active publication unavailable",
                route_type=_route_for_intent(resolution),
                route_metadata=KnowledgeRouteMetadata(
                    **_intent_route_metadata(catalog, resolution),
                    degradation_reason="publication_unavailable",
                    validation_passed=False,
                ),
            )
        try:
            if isinstance(resolution, StructuredRowLookupIntent):
                plan = StructuredQueryPlanner(catalog, self._compatibility).plan_row_lookup(
                    resolution, publication
                )
                result = StructuredQueryExecutor(
                    catalog,
                    self._clickhouse_gateway,
                    compatibility=self._compatibility,
                ).execute_row_lookup(plan)
            elif isinstance(resolution, StructuredMultiAggregateIntent):
                plan = StructuredQueryPlanner(
                    catalog,
                    self._compatibility,
                    implicit_summary_max_metrics=self._implicit_summary_max_metrics,
                ).plan_multi(resolution, publication)
                result = StructuredQueryExecutor(
                    catalog,
                    self._clickhouse_gateway,
                    compatibility=self._compatibility,
                    implicit_summary_max_metrics=self._implicit_summary_max_metrics,
                ).execute_multi(plan)
            else:
                plan = StructuredQueryPlanner(catalog, self._compatibility).plan(
                    resolution, publication
                )
                result = StructuredQueryExecutor(
                    catalog,
                    self._clickhouse_gateway,
                    compatibility=self._compatibility,
                ).execute(plan)
        except UnsafeStructuredQueryError:
            return _structured_run(
                conversation_id,
                question,
                mode,
                "结构化查询服务不可用：查询计划未通过安全校验。",
                "structured query planning failed",
                route_type=_route_for_intent(resolution),
                route_metadata=KnowledgeRouteMetadata(
                    **_intent_route_metadata(catalog, resolution),
                    degradation_reason="plan_rejected",
                    validation_passed=False,
                ),
            )
        if isinstance(result, StructuredUnavailable):
            return _structured_run(
                conversation_id,
                question,
                mode,
                f"结构化查询服务不可用：{result.message}。",
                "structured query unavailable",
                route_type=_route_for_intent(resolution),
                route_metadata=KnowledgeRouteMetadata(
                    **_intent_route_metadata(catalog, resolution),
                    degradation_reason="clickhouse_unavailable",
                    validation_passed=False,
                ),
            )
        if isinstance(result, StructuredRowLookupResult):
            artifact = TableArtifactModel(
                type="table",
                title="Excel 行查询结果",
                source=result.source_name,
                columns=list(result.selected_display_names),
                rows=[list(row) for row in result.rows],
            )
            suffix = "（结果已截断）" if result.truncated else ""
            return _structured_run(
                conversation_id,
                question,
                mode,
                f"已返回 {len(result.rows)} 行{suffix}。",
                f"structured row lookup completed; audit_id={result.audit_id}",
                source_ids=[result.source_id],
                artifacts=[artifact],
                route_type=KnowledgeRouteType.EXCEL_ROW_LOOKUP,
                route_metadata=KnowledgeRouteMetadata(
                    dataset_id=result.dataset_id,
                    target_fields=result.selected_display_names,
                    candidate_source_ids=(result.source_id,),
                    validation_passed=True,
                ),
            )
        if isinstance(result, StructuredMultiAggregateResult):
            if result.group_by:
                dataset = next(
                    item for item in catalog.datasets if item.schema.dataset_id == result.dataset_id
                )
                display_by_physical = {
                    column.physical_name: column.display_name for column in dataset.schema.columns
                }
                group_labels = [display_by_physical[name] for name in result.group_by]
                metric_specs = result.groups[0].metrics if result.groups else ()
                metric_labels = [
                    _metric_result_label(item.metric_display_name, item.aggregate, item.percentile)
                    for item in metric_specs
                ]
                artifact = TableArtifactModel(
                    type="table",
                    title="Excel 分组汇总结果",
                    source=result.source_name,
                    columns=[*group_labels, *metric_labels, "匹配行数"],
                    rows=[
                        [
                            *group.group_values,
                            *[_format_numeric_value(item.value) for item in group.metrics],
                            str(group.total_count),
                        ]
                        for group in result.groups
                    ],
                )
                answer = (
                    f"已按{'、'.join(group_labels)}汇总，共得到 {len(result.groups)} 组。"
                    if result.groups
                    else "没有符合条件的分组数据。"
                )
                return _structured_run(
                    conversation_id,
                    question,
                    mode,
                    answer,
                    f"structured grouped aggregate completed; audit_id={result.audit_id}",
                    source_ids=[result.source_id],
                    artifacts=[artifact],
                    route_type=KnowledgeRouteType.EXCEL_MULTI_AGGREGATE,
                    route_metadata=KnowledgeRouteMetadata(
                        dataset_id=result.dataset_id,
                        target_fields=tuple([*group_labels, *metric_labels]),
                        candidate_source_ids=(result.source_id,),
                        validation_passed=True,
                    ),
                )
            paragraph = "；".join(
                _format_metric_result(
                    item.metric_display_name,
                    item.aggregate,
                    item.value,
                    item.percentile,
                )
                for item in result.metrics
            ) + "。"
            artifact = TableArtifactModel(
                type="table",
                title="结构化汇总结果",
                source=result.source_name,
                columns=["指标", "聚合", "值", "匹配行数", "有效值", "空值"],
                rows=[
                    [
                        item.metric_display_name,
                        item.aggregate,
                        _format_numeric_value(item.value),
                        str(result.total_count),
                        str(item.valid_count),
                        str(item.null_count),
                    ]
                    for item in result.metrics
                ],
            )
            return _structured_run(
                conversation_id,
                question,
                mode,
                paragraph,
                f"structured multi-aggregate completed; audit_id={result.audit_id}",
                source_ids=[result.source_id],
                artifacts=[artifact],
                route_type=KnowledgeRouteType.EXCEL_MULTI_AGGREGATE,
                route_metadata=KnowledgeRouteMetadata(
                    dataset_id=result.dataset_id,
                    target_fields=tuple(item.metric_display_name for item in result.metrics),
                    candidate_source_ids=(result.source_id,),
                    validation_passed=True,
                ),
            )
        return _structured_run(
            conversation_id,
            question,
            mode,
            _format_result(result),
            f"structured aggregate completed; audit_id={result.audit_id}",
            source_ids=[result.source_id],
            route_metadata=KnowledgeRouteMetadata(
                dataset_id=result.dataset_id,
                target_fields=(
                    result.metric_display_name or result.metric_physical_name or "all_rows",
                ),
                candidate_source_ids=(result.source_id,),
                validation_passed=True,
            ),
        )

    def _next_catalog_request_generation(self) -> int:
        with self._catalog_snapshot_lock:
            self._catalog_request_generation += 1
            return self._catalog_request_generation

    def _replace_catalog_snapshot(
        self,
        catalog: StructuredCatalog,
        request_generation: int,
    ) -> None:
        with self._catalog_snapshot_lock:
            if request_generation < self._latest_successful_catalog_generation:
                return
            self._catalog_snapshot = catalog
            self._latest_successful_catalog_generation = request_generation

    def _get_catalog_snapshot(self) -> StructuredCatalog | None:
        with self._catalog_snapshot_lock:
            return self._catalog_snapshot


def is_structured_candidate(question: str, catalog: StructuredCatalog) -> bool:
    normalized_question = _normalize(question)
    if _is_temperature_concept_question(normalized_question):
        return False
    if _has_row_lookup_language(question, catalog):
        return True
    if not _has_aggregate_language(question):
        return False
    # Temperature headers are commonly queried with a qualified alias such as
    # ``平均温度`` while the workbook stores only a raw ``温度`` column.  The
    # generic span gate cannot see through that qualified phrase because the
    # longer alias masks the aggregate token.  Allow this one bounded semantic
    # bridge only when the governed parser resolves it to an actual numeric,
    # aggregatable temperature column.  Concept questions must remain on RAG.
    if _is_temperature_aggregate_candidate(question, catalog):
        return True
    if _is_implicit_summary(question):
        return True
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
    metric_names = {
        normalized
        for dataset in catalog.datasets
        for column in dataset.schema.columns
        if column.allow_aggregate
        for _, value in _candidate_resolution_names(column)
        for normalized in (_normalize(value),)
        if normalized
    }
    normalized = _normalize(_mask_aggregate_equality_values(question, filter_columns, metric_names))
    return _has_catalog_span_with_independent_aggregate(normalized, catalog_names)


def _is_temperature_aggregate_candidate(
    question: str,
    catalog: StructuredCatalog,
) -> bool:
    normalized = _normalize(question)
    if not any(term in normalized for term in ("温度", "气温", "均温")):
        return False
    if _is_aggregate_concept_question(normalized):
        return False

    temperature_columns = {
        column.physical_name: column
        for dataset in catalog.datasets
        for column in dataset.schema.columns
        if column.allow_aggregate
        and column.data_type.value in {"integer", "decimal"}
        and any(
            term in _normalize(value)
            for value in (
                column.physical_name,
                column.original_name,
                column.display_name,
                *column.aliases,
            )
            for term in ("温度", "气温")
        )
    }
    if not temperature_columns:
        return False

    resolved = resolve_structured_intent(question, catalog)
    if not isinstance(resolved, StructuredIntent):
        return False
    return resolved.metric_physical_name in temperature_columns


def _is_temperature_concept_question(normalized: str) -> bool:
    """Recognize explanatory temperature questions without routing to SQL."""
    temperature_terms = (
        "温度",
        "气温",
        "均温",
        "平均温度",
        "平均气温",
        "最高温度",
        "最高气温",
        "最低温度",
        "最低气温",
    )
    temperature_concept_suffixes = (
        *_CONCEPT_TERM_SUFFIXES,
        "是什么含义",
        "是什么概念",
        "是什么定义",
    )
    temperature_introducers = (
        *_CONCEPT_TERM_INTRODUCERS,
        "我想知道",
        "想知道",
        "请告诉我",
        "告诉我",
    )
    if any(
        phrase in normalized
        and _temperature_concept_tail(
            normalized[normalized.find(phrase) + len(phrase) :],
            temperature_terms,
        )
        for phrase in _CONCEPT_ANYWHERE_PHRASES
    ):
        return True
    # Common explanatory requests whose wording does not contain ``什么是``.
    # Keep this anchored to the end of the question and to a short verb phrase
    # so a real filtered query such as ``多伦多在某时段的平均温度`` is not
    # mistaken for a concept question.
    explanation_re = re.compile(
        r"(?:请问|请|能否|可以|我想|想|帮我|请帮我|通俗地|简单地|麻烦)?"
        r"(?:介绍一下|解释一下|说明一下|介绍|解释|说明|了解|讲讲|知道)"
        r"(?:一下)?(?:平均温度|平均气温|最高温度|最高气温|最低温度|最低气温|温度|气温|均温)$"
    )
    if explanation_re.fullmatch(normalized):
        return True
    explanation_with_suffix_re = re.compile(
        r"(?:请问|请|能否|可以|我想|想|帮我|请帮我|通俗地|简单地|麻烦)?"
        r"(?:介绍一下|解释一下|说明一下|介绍|解释|说明|了解|讲讲|知道)"
        r"(?:一下)?(?:平均温度|平均气温|最高温度|最高气温|最低温度|最低气温|温度|气温|均温)"
        r"(?:是什么意思|是什么含义|是什么概念|是什么定义|怎么理解|如何理解|"
        r"的含义|含义|的概念|概念|的定义|定义)$"
    )
    if explanation_with_suffix_re.fullmatch(normalized):
        return True
    # Also accept the common noun-phrase order ``温度的平均值是什么`` as a
    # concept question when it has no entity/date/filter prefix.  Keep the
    # pattern anchored so ``多伦多的温度的平均值`` remains a data request.
    bare_average_noun_re = re.compile(
        r"(?:请问|请|能否|可以|我想|想|帮我|请帮我|通俗地|简单地|麻烦)?"
        r"(?:温度|气温)的(?:平均值|平均|均值|均温)"
        r"(?:是什么|是什么意思|是什么含义|是什么概念|是什么定义|怎么理解|如何理解|"
        r"的含义|含义|的概念|概念|的定义|定义)$"
    )
    if bare_average_noun_re.fullmatch(normalized):
        return True
    # Explanatory wording can have polite or adverbial prefixes, e.g.
    # ``请介绍一下平均温度`` and ``通俗地解释一下平均温度``.  Treat it as a
    # concept only when the tail is the bare temperature term (optionally with
    # a definition/meaning suffix); a tail containing a location, date, or
    # filter remains eligible for a real data query.
    for introducer in temperature_introducers:
        start = normalized.find(introducer)
        while start >= 0:
            remainder = normalized[start + len(introducer) :]
            remainder = _strip_concept_question_tail(remainder)
            if remainder in temperature_terms:
                return True
            if any(
                remainder == f"{term}{suffix}"
                for term in temperature_terms
                for suffix in temperature_concept_suffixes
            ):
                return True
            start = normalized.find(introducer, start + 1)
    for term in temperature_terms:
        for suffix in temperature_concept_suffixes:
            phrase = f"{term}{suffix}"
            if not normalized.endswith(phrase):
                continue
            prefix = normalized[: -len(phrase)]
            # Meaning/definition tails are unambiguously explanatory even if
            # a subject or location appears before the temperature term.
            # ``多伦多的平均温度是什么`` is intentionally different: its
            # bare ``是什么`` tail can still be a request for the value.
            if suffix in {
                "是什么意思",
                "是什么含义",
                "是什么概念",
                "是什么定义",
                "怎么理解",
                "如何理解",
                "的含义",
                "含义",
                "的概念",
                "概念",
                "的定义",
                "定义",
            }:
                return True
            # A bare field followed by a definition suffix is a concept
            # question.  If a subject/date/location precedes it (for example
            # ``多伦多的平均温度是什么``), it is a real data query and must
            # remain eligible for the structured route.
            if not prefix or _is_temperature_explanation_prefix(prefix):
                return True
    return False


def _temperature_concept_tail(
    remainder: str,
    temperature_terms: tuple[str, ...],
) -> bool:
    """Check that a ``什么是`` tail is a bare temperature concept."""
    remainder = _strip_concept_question_tail(remainder)
    if remainder in temperature_terms:
        return True
    concept_suffixes = (
        *_CONCEPT_TERM_SUFFIXES,
        "是什么含义",
        "是什么概念",
        "是什么定义",
    )
    return any(
        remainder == f"{term}{suffix}"
        for term in temperature_terms
        for suffix in concept_suffixes
    )


def _is_temperature_explanation_prefix(prefix: str) -> bool:
    """Return whether text before a temperature concept is only polite wording."""
    allowed = (
        "请问",
        "请",
        "能否",
        "可以",
        "我想",
        "我想知道",
        "想知道",
        "想",
        "帮我",
        "请帮我",
        "请告诉我",
        "告诉我",
        "通俗地",
        "简单地",
        "简单说",
        "关于",
    )
    return prefix in allowed


def _has_row_lookup_language(question: str, catalog: StructuredCatalog) -> bool:
    if not _has_row_lookup_markers(question):
        return False
    if any(term in question for _, words in _AGGREGATE_WORDS for term in words):
        return False
    return any(
        _normalize(value) in _normalize(question)
        for dataset in catalog.datasets
        for column in dataset.schema.columns
        for _, value in _resolution_names(column)
    )


def _has_row_lookup_markers(question: str) -> bool:
    return any(
        marker in question
        for marker in (
            "返回",
            "查找",
            "显示",
            "列出",
            "给出",
            "其他列",
            "所有",
            "全部",
            "每条",
            "每行",
            "明细",
        )
    )


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
            for _, value in _candidate_resolution_names(column)
        )
    return tuple(name for name in names if name)


def _candidate_resolution_names(
    column: StructuredColumnSchema,
) -> tuple[tuple[int, str], ...]:
    """Return schema names plus aggregate-specific query aliases for routing.

    The parser passes the actual aggregate context to the resolver. Candidate
    detection runs before that context is known, so it needs to see bounded
    aliases such as ``均温`` for a governed raw ``温度`` column as well. These
    names are used only for route gating; they never alter persisted schema or
    SQL planning.
    """
    names: list[tuple[int, str]] = list(_resolution_names(column))
    for aggregate in ("avg", "max", "min"):
        names.extend(_resolution_names(column, aggregate=aggregate))
    return tuple(dict.fromkeys(names))


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
    metric_names: set[str],
) -> str:
    masked = list(question)
    spans = {
        span
        for column in filter_columns
        for span in _candidate_equality_value_spans(question, (column,))
    }
    for value_start, value_end in spans:
        value = question[value_start:value_end]
        if not _has_aggregate_language(value):
            continue
        normalized_value = _normalize(value)
        preserve_offset = _metric_aggregate_start(normalized_value, metric_names)
        if preserve_offset is None:
            masked[value_start:value_end] = "_" * (value_end - value_start)
            continue
        raw_offset = _raw_offset_for_normalized_offset(value, preserve_offset)
        masked[value_start : value_start + raw_offset] = "_" * raw_offset
    return "".join(masked)


def _metric_aggregate_start(normalized_value: str, metric_names: set[str]) -> int | None:
    spans: set[tuple[int, int]] = set()
    for name in metric_names:
        start = normalized_value.find(name)
        while start >= 0:
            spans.add((start, start + len(name)))
            start = normalized_value.find(name, start + 1)
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
    candidates: list[tuple[int, int]] = []
    for start, end in maximal_spans:
        suffix = normalized_value[end:]
        suffix = suffix.removeprefix("的")
        suffix = _strip_natural_aggregate_tail(suffix)
        if _matching_aggregate_suffix(suffix) == suffix:
            candidates.append((start, end))
    if not candidates:
        return None
    return max(candidates, key=lambda span: (span[0], span[1] - span[0]))[0]


def _raw_offset_for_normalized_offset(value: str, normalized_offset: int) -> int:
    for index in range(len(value) + 1):
        if len(_normalize(value[:index])) >= normalized_offset:
            return index
    return len(value)


def _is_implicit_row_count(question: str) -> bool:
    return _IMPLICIT_ROW_COUNT_RE.fullmatch(question.strip()) is not None


def _is_implicit_summary(question: str) -> bool:
    return _IMPLICIT_SUMMARY_RE.fullmatch(question.strip()) is not None


def _classify_without_catalog(question: str) -> Literal["weak", "strong", "concept"]:
    stripped = question.strip()
    normalized = _normalize(stripped)
    # Keep explanatory temperature questions on the ordinary RAG path even
    # when the structured catalog is cold/unavailable.  Without this early
    # guard a phrase such as ``我想知道平均温度是什么`` is classified as a
    # weak/strong aggregate shape and can incorrectly return a structured
    # outage response instead of allowing the document retriever to answer.
    if _is_temperature_concept_question(normalized):
        return "concept"
    if _has_row_lookup_markers(stripped):
        return "strong"
    if not _has_aggregate_language(normalized):
        return "weak"
    concept_body = _normalize(_strip_concept_question_tail(stripped))
    if _has_prefixed_copula_concept_shape(concept_body):
        return "concept"
    if _HAS_EXPLICIT_FILTER_RE.search(stripped) or _has_chinese_equality_filter(stripped):
        return "strong"
    if _is_priority_aggregate_concept_shape(concept_body):
        return "concept"
    if _has_metric_qualified_concept_shape(normalized):
        return "strong"
    if _is_aggregate_concept_question(normalized):
        return "concept"
    if (
        _is_implicit_row_count(stripped)
        or _is_implicit_summary(stripped)
        or _has_field_aggregate_suffix(normalized)
    ):
        return "strong"
    return "weak"


def _is_priority_aggregate_concept_shape(normalized: str) -> bool:
    for opener in ("什么是", "什么叫", "何为"):
        start = normalized.find(opener)
        while start >= 0:
            remainder = normalized[start + len(opener) :]
            if remainder in _NAMED_AVERAGE_CONCEPT_TERMS:
                return True
            if start == 0 and remainder in _AGGREGATE_CONCEPT_TERMS:
                return True
            start = normalized.find(opener, start + 1)
    for term in _AGGREGATE_CONCEPT_TERMS:
        for suffix in _CONCEPT_TERM_SUFFIXES:
            phrase = f"{term}{suffix}"
            if normalized == phrase:
                return True
    return False


def _strip_concept_question_tail(value: str) -> str:
    remaining = value.rstrip()
    while remaining:
        previous = remaining
        while remaining and unicodedata.category(remaining[-1]).startswith("P"):
            remaining = remaining[:-1].rstrip()
        particle = next(
            (item for item in _NATURAL_QUESTION_PARTICLES if remaining.endswith(item)),
            None,
        )
        if particle is not None:
            remaining = remaining[: -len(particle)].rstrip()
        if remaining == previous:
            return remaining
    return remaining


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


def _has_prefixed_copula_concept_shape(normalized: str) -> bool:
    for prefix in _CONCEPT_COPULA_REQUEST_PREFIXES:
        if not normalized.startswith(prefix):
            continue
        body = normalized[len(prefix) :]
        for term in _AGGREGATE_CONCEPT_TERMS:
            marker = f"{term}为"
            if not body.startswith(marker):
                continue
            predicate = body[len(marker) :]
            return any(predicate.endswith(suffix) for suffix in _CONCEPT_COPULA_PREDICATE_SUFFIXES)
    return False


def _has_chinese_equality_filter(question: str) -> bool:
    for index, character in enumerate(question):
        context = _normalize(question[max(0, index - 2) : index + 1])
        if character != "为" or any(context.endswith(item) for item in _COPULA_FRAGMENTS):
            continue
        field_start = max(question.rfind(item, 0, index) for item in _EQUALITY_FIELD_DELIMITERS) + 1
        field = question[field_start:index]
        value = question[index + 1 :]
        normalized_field = _normalize(field)
        if not normalized_field or not _normalize(value):
            continue
        if normalized_field in _AGGREGATE_CONCEPT_TERMS:
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
    prefix = prefix.removesuffix("的")
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


def _active_publication(
    catalog: StructuredCatalog,
    intent: StructuredIntent | StructuredMultiAggregateIntent | StructuredRowLookupIntent,
):
    matches = [
        dataset.active_publication
        for dataset in catalog.datasets
        if dataset.schema.dataset_id == intent.dataset_id and dataset.active_publication is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _intent_route_metadata(
    catalog: StructuredCatalog,
    intent: StructuredIntent | StructuredMultiAggregateIntent | StructuredRowLookupIntent,
) -> dict[str, object]:
    dataset = next(item for item in catalog.datasets if item.schema.dataset_id == intent.dataset_id)
    if isinstance(intent, StructuredRowLookupIntent):
        physical_names = intent.selected_physical_names
    elif isinstance(intent, StructuredMultiAggregateIntent):
        physical_names = tuple(metric.metric_physical_name for metric in intent.metrics)
    else:
        physical_names = (intent.metric_physical_name,)
    fields = tuple(
        next(
            column.display_name for column in dataset.schema.columns if column.physical_name == name
        )
        if name is not None
        else "all_rows"
        for name in physical_names
    )
    return {
        "dataset_id": dataset.schema.dataset_id,
        "target_fields": fields,
        "candidate_source_ids": (dataset.schema.source_id,),
        "origin_route": _route_for_intent(intent),
    }


def _catalog_outage_route_metadata(
    catalog: StructuredCatalog,
    resolution: object,
) -> dict[str, object]:
    if isinstance(
        resolution, (StructuredIntent, StructuredMultiAggregateIntent, StructuredRowLookupIntent)
    ):
        return _intent_route_metadata(catalog, resolution)
    if isinstance(resolution, StructuredClarification):
        return _clarification_route_metadata(catalog, resolution)
    if isinstance(resolution, StructuredUnavailable):
        return _unavailable_route_metadata(resolution)
    return {}


def _route_for_catalog_outage_resolution(resolution: object) -> KnowledgeRouteType:
    if isinstance(
        resolution, (StructuredIntent, StructuredMultiAggregateIntent, StructuredRowLookupIntent)
    ):
        return _route_for_intent(resolution)
    if isinstance(resolution, StructuredClarification) and resolution.origin_route is not None:
        return KnowledgeRouteType(resolution.origin_route)
    if isinstance(resolution, StructuredUnavailable):
        return _route_for_parser_outcome(resolution)
    return KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE


def _route_for_intent(
    intent: StructuredIntent | StructuredMultiAggregateIntent | StructuredRowLookupIntent,
) -> KnowledgeRouteType:
    if isinstance(intent, StructuredRowLookupIntent):
        return KnowledgeRouteType.EXCEL_ROW_LOOKUP
    return (
        KnowledgeRouteType.EXCEL_MULTI_AGGREGATE
        if isinstance(intent, StructuredMultiAggregateIntent)
        else KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE
    )


def _clarification_route_metadata(
    catalog: StructuredCatalog,
    clarification: StructuredClarification,
) -> dict[str, object]:
    if clarification.origin_route is not None:
        return {
            "dataset_id": clarification.dataset_id,
            "target_fields": clarification.target_fields,
            "candidate_source_ids": clarification.candidate_source_ids,
            "origin_route": KnowledgeRouteType(clarification.origin_route),
        }
    candidates = set(clarification.candidates)
    matches = [
        dataset
        for dataset in catalog.datasets
        if candidates
        and candidates.issubset(
            {column.display_name for column in dataset.schema.columns if column.allow_aggregate}
        )
    ]
    if len(matches) == 1:
        dataset = matches[0]
        return {
            "dataset_id": dataset.schema.dataset_id,
            "target_fields": tuple(clarification.candidates),
            "candidate_source_ids": (dataset.schema.source_id,),
            "origin_route": KnowledgeRouteType.EXCEL_MULTI_AGGREGATE,
        }
    return {"origin_route": KnowledgeRouteType.EXCEL_MULTI_AGGREGATE}


def _unavailable_route_metadata(outcome: StructuredUnavailable) -> dict[str, object]:
    return {
        "dataset_id": outcome.dataset_id,
        "target_fields": outcome.target_fields,
        "candidate_source_ids": outcome.candidate_source_ids,
        "origin_route": None
        if outcome.origin_route is None
        else KnowledgeRouteType(outcome.origin_route),
    }


def _route_for_parser_outcome(outcome: StructuredUnavailable) -> KnowledgeRouteType:
    return (
        KnowledgeRouteType(outcome.origin_route)
        if outcome.origin_route is not None
        else KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE
    )


def _format_result(result: StructuredAggregateResult) -> str:
    metric = result.metric_display_name or result.metric_physical_name or "all_rows"
    if metric == "all_rows":
        return f"共有 {result.total_count} 条记录。"
    return _format_metric_result(metric, result.aggregate, result.value, result.percentile) + "。"


def _format_metric_result(
    metric: str,
    aggregate: str,
    value: Decimal | int | None,
    percentile: float | None = None,
) -> str:
    aggregate_label = {
        "avg": "平均值",
        "sum": "总和",
        "count": "记录数",
        "count_distinct": "去重数量",
        "min": "最小值",
        "max": "最大值",
        "median": "中位数",
        "stddev": "标准差",
        "variance": "方差",
        "percentile": f"P{percentile:g} 分位数" if percentile is not None else "分位数",
    }.get(aggregate, aggregate)
    if value is None:
        return f"未计算出{metric}的{aggregate_label}（没有有效数值）"
    return f"{metric}的{aggregate_label}为 {_format_numeric_value(value)}"


def _metric_result_label(metric: str, aggregate: str, percentile: float | None) -> str:
    label = {
        "avg": "平均值",
        "sum": "总和",
        "count": "记录数",
        "count_distinct": "去重数量",
        "min": "最小值",
        "max": "最大值",
        "median": "中位数",
        "stddev": "标准差",
        "variance": "方差",
        "percentile": f"P{percentile:g}" if percentile is not None else "分位数",
    }.get(aggregate, aggregate)
    return f"{metric}（{label}）"


def _format_numeric_value(value: Decimal | int | None) -> str:
    if value is None:
        return "null"
    return format(value, ",")


def _structured_run(
    conversation_id: str,
    question: str,
    mode: ComposerMode,
    answer: str,
    output_summary: str,
    *,
    source_ids: list[str] | None = None,
    artifacts: list[ArtifactModel] | None = None,
    route_type: KnowledgeRouteType = KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE,
    route_metadata: KnowledgeRouteMetadata | None = None,
) -> AgentRunResult:
    timestamp = display_datetime_label()
    run_id = f"agent-{uuid4().hex[:12]}"
    reply = ChatMessageModel(
        id=f"msg-{uuid4().hex[:8]}",
        role="assistant",
        time=timestamp,
        paragraphs=[ResponseParagraphModel(text=answer)],
        artifacts=artifacts or [],
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
        route_type=route_type,
        route_metadata=route_metadata
        or KnowledgeRouteMetadata(
            validation_passed=True,
            adjacency_allowed=False,
        ),
    )
