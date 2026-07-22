from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import PurePath
from typing import Literal

import sqlglot
from sqlglot import exp

from .structured_models import (
    StructuredAggregateResult,
    StructuredCatalog,
    StructuredClarification,
    StructuredColumnSchema,
    StructuredColumnType,
    StructuredDatasetCatalog,
    StructuredFilter,
    StructuredIntent,
    StructuredPublication,
    StructuredQueryPlan,
    StructuredUnavailable,
)

StructuredIntentResolution = StructuredIntent | StructuredClarification | StructuredUnavailable


@dataclass(frozen=True, order=True)
class _TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class _ClauseParseResult[T]:
    value: T | None = None
    consumed_spans: tuple[_TextSpan, ...] = ()
    issue: StructuredClarification | StructuredUnavailable | None = None


@dataclass(frozen=True)
class _FilterMatch:
    item: StructuredFilter
    span: _TextSpan
    consumed_spans: tuple[_TextSpan, ...]


_AGGREGATE_WORDS = (
    ("avg", ("平均值", "平均", "均值")),
    ("sum", ("总和", "合计", "求和")),
    ("count", ("多少条", "数量", "计数")),
    ("max", ("最大", "最高")),
    ("min", ("最小", "最低")),
)
_COMPARISON_OPERATORS = {"大于": "gt", "不少于": "gte", "小于": "lt", "不超过": "lte"}
_DATE_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*至\s*(\d{4}-\d{2}-\d{2})")
_NUMBER_RE = r"-?\d+(?:\.\d+)?"
_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]+$")
_ALLOWED_AGGREGATES = frozenset({"avg", "sum", "count", "min", "max"})
_ALLOWED_SQL_FUNCTIONS = frozenset({"AVG", "SUM", "COUNT", "MIN", "MAX"})


class UnsafeStructuredQueryError(ValueError):
    pass


class StructuredQueryPlanner:
    def __init__(self, catalog: StructuredCatalog) -> None:
        self._catalog = catalog

    def plan(
        self,
        intent: StructuredIntent,
        publication: StructuredPublication,
    ) -> StructuredQueryPlan:
        dataset = self._require_dataset(intent.dataset_id)
        active = dataset.active_publication
        if active is None:
            raise UnsafeStructuredQueryError("structured dataset is not published")
        if publication != active:
            raise UnsafeStructuredQueryError("publication is not the active catalog publication")
        if intent.aggregate not in _ALLOWED_AGGREGATES:
            raise UnsafeStructuredQueryError("unsupported aggregate")

        table_name = _require_identifier(publication.physical_table_name)
        columns = {column.physical_name: column for column in dataset.schema.columns}
        metric = None
        if intent.metric_physical_name is not None:
            metric = columns.get(intent.metric_physical_name)
            if metric is None or (intent.aggregate != "count" and not metric.allow_aggregate):
                raise UnsafeStructuredQueryError("unknown or disallowed aggregate column")
            _require_identifier(metric.physical_name)
        elif intent.aggregate != "count":
            raise UnsafeStructuredQueryError("aggregate requires a confirmed metric")

        aggregate_expression = (
            "count()" if metric is None else f"{intent.aggregate}({metric.physical_name})"
        )
        valid_expression = "count()" if metric is None else f"count({metric.physical_name})"
        null_expression = "0" if metric is None else f"count() - count({metric.physical_name})"
        projections = (
            f"{aggregate_expression} AS aggregate_value",
            "count() AS total_count",
            f"{valid_expression} AS valid_count",
            f"{null_expression} AS null_count",
        )

        parameters: dict[str, object] = {}
        predicates = []
        for index, item in enumerate(intent.filters):
            column = columns.get(item.physical_name)
            if column is None or not column.allow_filter:
                raise UnsafeStructuredQueryError("unknown or disallowed filter column")
            name = _require_identifier(column.physical_name)
            parameter_name = f"filter_{index}"
            parameter_type = _clickhouse_parameter_type(column.data_type)
            parameters[parameter_name] = _convert_parameter(item.value, column.data_type)
            placeholder = f"{{{parameter_name}:{parameter_type}}}"
            if item.operator == "between":
                if item.upper_value is None:
                    raise UnsafeStructuredQueryError("between filter requires an upper value")
                upper_name = f"filter_{index}_upper"
                upper_value = _convert_parameter(item.upper_value, column.data_type)
                upper_operator = "<="
                if column.data_type is StructuredColumnType.DATETIME and re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}", item.upper_value
                ):
                    upper_value = upper_value + timedelta(days=1)
                    upper_operator = "<"
                parameters[upper_name] = upper_value
                upper_placeholder = f"{{{upper_name}:{parameter_type}}}"
                predicates.append(
                    f"({name} >= {placeholder} AND {name} {upper_operator} {upper_placeholder})"
                )
            else:
                operator = {
                    "eq": "=",
                    "gt": ">",
                    "gte": ">=",
                    "lt": "<",
                    "lte": "<=",
                }.get(item.operator)
                if operator is None:
                    raise UnsafeStructuredQueryError("unsupported filter operator")
                if item.upper_value is not None:
                    raise UnsafeStructuredQueryError("non-range filter cannot have an upper value")
                predicates.append(f"{name} {operator} {placeholder}")

        sql = f"SELECT {', '.join(projections)} FROM {table_name}"
        if predicates:
            sql += " WHERE " + " AND ".join(predicates)
        _validate_generated_select(
            sql,
            table_name=table_name,
            allowed_columns=frozenset(columns),
        )
        return StructuredQueryPlan(
            publication_id=publication.publication_id,
            dataset_id=intent.dataset_id,
            metric_physical_name=intent.metric_physical_name,
            sql=sql,
            parameters=parameters,
            aggregate=intent.aggregate,
            filters=intent.filters,
        )

    def _require_dataset(self, dataset_id: str) -> StructuredDatasetCatalog:
        matches = [
            dataset for dataset in self._catalog.datasets if dataset.schema.dataset_id == dataset_id
        ]
        if len(matches) != 1:
            raise UnsafeStructuredQueryError("dataset must resolve to exactly one catalog entry")
        return matches[0]


class StructuredQueryExecutor:
    def __init__(
        self,
        catalog: StructuredCatalog,
        clickhouse_gateway: object,
        *,
        clock: Callable[[], float] = time.perf_counter,
        audit_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._catalog = catalog
        self._clickhouse = clickhouse_gateway
        self._clock = clock
        self._audit_id_factory = audit_id_factory

    def execute(
        self, plan: StructuredQueryPlan
    ) -> StructuredAggregateResult | StructuredUnavailable:
        dataset = self._require_active_dataset(plan)
        if isinstance(dataset, StructuredUnavailable):
            return dataset
        publication = dataset.active_publication
        assert publication is not None

        try:
            expected = StructuredQueryPlanner(self._catalog).plan(
                StructuredIntent(
                    dataset_id=plan.dataset_id,
                    aggregate=plan.aggregate,
                    metric_physical_name=plan.metric_physical_name,
                    filters=plan.filters,
                ),
                publication,
            )
        except UnsafeStructuredQueryError:
            return StructuredUnavailable("结构化查询计划已失效")
        if plan.sql != expected.sql or dict(plan.parameters) != dict(expected.parameters):
            return StructuredUnavailable("结构化查询计划未通过安全校验")

        query = getattr(self._clickhouse, "query", None)
        if query is None:
            return StructuredUnavailable("结构化查询服务暂时不可用")
        started = self._clock()
        try:
            raw_result = query(plan.sql, plan.parameters)
        except Exception as error:
            if (
                isinstance(error, TimeoutError)
                or "timeout" in str(error).casefold()
                or "timed out" in str(error).casefold()
            ):
                return StructuredUnavailable("结构化查询超时，请稍后重试")
            return StructuredUnavailable("结构化查询服务暂时不可用")
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)

        try:
            row = _aggregate_row(raw_result)
            value = _aggregate_value(row["aggregate_value"], plan.aggregate)
            total_count = int(row["total_count"])
            valid_count = int(row["valid_count"])
            null_count = int(row["null_count"])
        except (KeyError, TypeError, ValueError, IndexError, ArithmeticError):
            return StructuredUnavailable("结构化查询返回了无效结果")
        if min(total_count, valid_count, null_count) < 0 or valid_count + null_count != total_count:
            return StructuredUnavailable("结构化查询返回了不一致的计数")

        metric = next(
            (
                column
                for column in dataset.schema.columns
                if column.physical_name == plan.metric_physical_name
            ),
            None,
        )
        return StructuredAggregateResult(
            dataset_id=dataset.schema.dataset_id,
            schema_version=dataset.schema.schema_version,
            aggregate=plan.aggregate,
            metric_physical_name=plan.metric_physical_name,
            metric_display_name=None if metric is None else metric.display_name,
            value=value,
            total_count=total_count,
            valid_count=valid_count,
            null_count=null_count,
            source_name=dataset.source_name,
            worksheet_name=dataset.schema.worksheet_name,
            publication_id=publication.publication_id,
            filters=plan.filters,
            elapsed_ms=elapsed_ms,
            audit_id=self._audit_id_factory(),
        )

    def _require_active_dataset(
        self, plan: StructuredQueryPlan
    ) -> StructuredDatasetCatalog | StructuredUnavailable:
        matches = [
            dataset
            for dataset in self._catalog.datasets
            if dataset.schema.dataset_id == plan.dataset_id
        ]
        if len(matches) != 1:
            return StructuredUnavailable("结构化查询数据集不再唯一")
        dataset = matches[0]
        publication = dataset.active_publication
        if publication is None or publication.publication_id != plan.publication_id:
            return StructuredUnavailable("结构化查询发布版本已失效")
        return dataset


def parse_structured_intent(
    question: str, catalog: StructuredCatalog
) -> StructuredIntentResolution:
    return resolve_structured_intent(question, catalog)


def resolve_structured_intent(
    question: str, catalog: StructuredCatalog
) -> StructuredIntentResolution:
    dataset_result = _parse_dataset_clause(question, catalog)
    if dataset_result.issue is not None:
        return dataset_result.issue
    assert dataset_result.value is not None
    dataset = dataset_result.value

    filter_result = _parse_filter_clause(
        question,
        dataset.schema.columns,
        dataset_result.consumed_spans,
    )
    if filter_result.issue is not None:
        return filter_result.issue
    assert filter_result.value is not None

    consumed = (*dataset_result.consumed_spans, *filter_result.consumed_spans)
    aggregate_result = _parse_aggregate_clause(
        question,
        dataset.schema.columns,
        consumed,
    )
    if aggregate_result.issue is not None:
        return aggregate_result.issue
    assert aggregate_result.value is not None
    aggregate = aggregate_result.value

    consumed = (*consumed, *aggregate_result.consumed_spans)
    metric_result = _parse_metric_clause(
        question,
        dataset.schema.columns,
        aggregate,
        filter_result.value,
        consumed,
    )
    if metric_result.issue is not None:
        return metric_result.issue
    metric = metric_result.value

    consumed = (*consumed, *metric_result.consumed_spans)
    remaining = _mask_spans(question, consumed)
    if _DATE_RANGE_RE.search(remaining) or re.search(r"大于|不少于|小于|不超过|为|=", remaining):
        return StructuredUnavailable("结构化查询包含未识别的筛选条件")

    return StructuredIntent(
        dataset_id=dataset.schema.dataset_id,
        aggregate=aggregate,
        metric_physical_name=None if metric is None else metric.physical_name,
        filters=filter_result.value,
    )


def _parse_aggregate_clause(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
    excluded_spans: tuple[_TextSpan, ...],
) -> _ClauseParseResult[str]:
    available = _mask_spans(question, excluded_spans)
    field_spans = _column_name_spans(available, columns)
    matches: dict[str, list[_TextSpan]] = {}
    for aggregate, words in _AGGREGATE_WORDS:
        for word in words:
            for span in _find_normalized_spans(available, word):
                if any(_contains(field_span, span) for field_span in field_spans):
                    continue
                matches.setdefault(aggregate, []).append(span)
    if len(matches) > 1:
        return _ClauseParseResult(
            issue=StructuredClarification(
                "问题包含多个不同的聚合意图，请选择一个",
                tuple(sorted(matches)),
            )
        )
    if not matches:
        return _ClauseParseResult(issue=StructuredUnavailable("未识别到受支持的聚合意图"))
    aggregate = next(iter(matches))
    return _ClauseParseResult(
        value=aggregate,
        consumed_spans=_merge_spans(matches[aggregate]),
    )


def _parse_dataset_clause(
    question: str, catalog: StructuredCatalog
) -> _ClauseParseResult[StructuredDatasetCatalog]:
    column_spans = {
        span
        for dataset in catalog.datasets
        for span in _column_name_spans(question, dataset.schema.columns)
    }
    matches: list[tuple[int, int, _TextSpan, StructuredDatasetCatalog]] = []
    for dataset in catalog.datasets:
        names = (
            (0, dataset.schema.dataset_id),
            (1, dataset.source_name),
            (1, PurePath(dataset.source_name).stem),
            (2, dataset.schema.worksheet_name),
        )
        for priority, name in names:
            normalized_name = _normalize(name)
            if not normalized_name:
                continue
            for span in _find_normalized_spans(question, name):
                if any(_contains(column_span, span) for column_span in column_spans):
                    continue
                matches.append((priority, len(normalized_name), span, dataset))

    if matches:
        best_priority_by_span = {
            span: min(
                priority for priority, _, candidate_span, _ in matches if candidate_span == span
            )
            for _, _, span, _ in matches
        }
        prioritized = [match for match in matches if match[0] == best_priority_by_span[match[2]]]
        independent = [
            match
            for match in prioritized
            if not any(
                other_length > match[1] and _contains(other_span, match[2])
                for _, other_length, other_span, _ in prioritized
            )
        ]
        finalists = {dataset.schema.dataset_id: dataset for _, _, _, dataset in independent}
        if len(finalists) > 1:
            return _ClauseParseResult(
                issue=StructuredClarification(
                    "问题同时匹配多个数据集，请选择一个数据集",
                    tuple(sorted(finalists)),
                )
            )
        selected = next(iter(finalists.values()))
        if selected.active_publication is None:
            return _ClauseParseResult(issue=StructuredUnavailable("指定数据集尚未确认并发布"))
        return _ClauseParseResult(
            value=selected,
            consumed_spans=_merge_spans(
                span
                for _, _, span, dataset in independent
                if dataset.schema.dataset_id == selected.schema.dataset_id
            ),
        )

    published = [dataset for dataset in catalog.datasets if dataset.active_publication is not None]
    if len(published) == 1:
        return _ClauseParseResult(value=published[0])
    if not published:
        return _ClauseParseResult(issue=StructuredUnavailable("没有已确认并发布的结构化数据集"))
    return _ClauseParseResult(
        issue=StructuredClarification(
            "请指定要查询的数据集",
            tuple(sorted(dataset.schema.dataset_id for dataset in published)),
        )
    )


def _parse_metric_clause(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
    aggregate: str,
    filters: tuple[StructuredFilter, ...],
    excluded_spans: tuple[_TextSpan, ...],
) -> _ClauseParseResult[StructuredColumnSchema | None]:
    available = _mask_spans(question, excluded_spans)
    aggregate_columns = (
        columns
        if aggregate == "count"
        else tuple(column for column in columns if column.allow_aggregate)
    )
    matches = _resolve_columns(available, aggregate_columns)
    if isinstance(matches, StructuredClarification):
        return _ClauseParseResult(issue=matches)
    if matches:
        metric = matches[0]
        return _ClauseParseResult(
            value=metric,
            consumed_spans=_column_name_spans(available, (metric,)),
        )
    if aggregate == "count":
        return _ClauseParseResult(value=None)

    columns_by_name = {column.physical_name: column for column in columns}
    aggregate_filter_columns = {
        column
        for item in filters
        if (column := columns_by_name.get(item.physical_name)) is not None
        and column.allow_aggregate
    }
    if len(aggregate_filter_columns) == 1:
        return _ClauseParseResult(value=next(iter(aggregate_filter_columns)))
    if len(aggregate_filter_columns) > 1:
        return _ClauseParseResult(
            issue=StructuredClarification(
                "多个筛选字段都可作为聚合指标，请明确指标字段",
                tuple(sorted(column.physical_name for column in aggregate_filter_columns)),
            )
        )
    return _ClauseParseResult(issue=StructuredUnavailable("未识别到可聚合的指标字段"))


def _parse_filter_clause(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
    excluded_spans: tuple[_TextSpan, ...] = (),
) -> _ClauseParseResult[tuple[StructuredFilter, ...]]:
    available = _mask_spans(question, excluded_spans)
    filter_columns = tuple(column for column in columns if column.allow_filter)
    explicit = _parse_explicit_filter_clauses(available, filter_columns)
    if explicit.issue is not None:
        return _ClauseParseResult(issue=explicit.issue)
    explicit_matches = explicit.value or ()
    all_matches = list(explicit_matches)
    consumed = list(explicit.consumed_spans)

    date_ranges = tuple(_DATE_RANGE_RE.finditer(available))
    if len(date_ranges) > 1:
        return _ClauseParseResult(issue=StructuredUnavailable("结构化筛选暂不支持多个日期范围"))
    if date_ranges:
        date_columns = tuple(
            column
            for column in filter_columns
            if column.data_type in {StructuredColumnType.DATE, StructuredColumnType.DATETIME}
        )
        date_question = _mask_spans(available, consumed)
        explicit_date = _resolve_columns(date_question, date_columns)
        if isinstance(explicit_date, StructuredClarification):
            return _ClauseParseResult(issue=explicit_date)
        if explicit_date:
            date_column = explicit_date[0]
        elif len(date_columns) == 1:
            date_column = date_columns[0]
        elif len(date_columns) > 1:
            return _ClauseParseResult(
                issue=StructuredClarification(
                    "日期范围匹配多个日期字段，请选择一个字段",
                    tuple(sorted(column.physical_name for column in date_columns)),
                )
            )
        else:
            return _ClauseParseResult(
                issue=StructuredUnavailable("问题包含日期范围，但数据集没有可筛选的日期字段")
            )
        range_match = date_ranges[0]
        range_span = _TextSpan(range_match.start(), range_match.end())
        date_spans = list(_column_name_spans(date_question, (date_column,)))
        date_consumed = [range_span]
        bound_date_span = _bound_date_field_span(available, range_span, date_spans)
        if bound_date_span is not None:
            date_consumed.append(
                _TextSpan(
                    min(bound_date_span.start, range_span.start),
                    max(bound_date_span.end, range_span.end),
                )
            )
        date_item = StructuredFilter(
            date_column.physical_name,
            "between",
            range_match.group(1),
            range_match.group(2),
        )
        all_matches.append(_FilterMatch(date_item, range_span, tuple(date_consumed)))
        consumed.extend(date_consumed)

    implicit_columns = tuple(
        column
        for column in filter_columns
        if column.physical_name not in {match.item.physical_name for match in all_matches}
    )
    implicit_question = _mask_spans(available, consumed)
    implicit = _parse_implicit_filter_clauses(implicit_question, implicit_columns)
    if implicit.issue is not None:
        return _ClauseParseResult(issue=implicit.issue)
    implicit_matches = implicit.value or ()
    all_matches.extend(implicit_matches)
    consumed.extend(implicit.consumed_spans)

    if "或" in available and all_matches:
        return _ClauseParseResult(issue=StructuredUnavailable("结构化筛选暂不支持 OR 条件"))

    ordered: list[StructuredFilter] = []
    seen: set[StructuredFilter] = set()
    for match in sorted(all_matches, key=lambda item: item.span.start):
        if match.item not in seen:
            ordered.append(match.item)
            seen.add(match.item)
    return _ClauseParseResult(
        value=tuple(ordered),
        consumed_spans=_merge_spans(consumed),
    )


def _bound_date_field_span(
    question: str,
    range_span: _TextSpan,
    date_spans: Iterable[_TextSpan],
) -> _TextSpan | None:
    preceding = [
        span
        for span in date_spans
        if span.end <= range_span.start
        and re.fullmatch(r"[\s的]*", question[span.end : range_span.start])
    ]
    if preceding:
        return max(preceding, key=lambda span: span.end)
    following = [
        span
        for span in date_spans
        if range_span.end <= span.start
        and re.fullmatch(r"\s*", question[range_span.end : span.start])
    ]
    if following:
        return min(following, key=lambda span: span.start)
    return None


def _parse_explicit_filter_clauses(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
) -> _ClauseParseResult[tuple[_FilterMatch, ...]]:
    matches: list[_FilterMatch] = []
    for operator in re.finditer(r"大于|不少于|小于|不超过|为|=", question):
        prefix = question[: operator.start()]
        segment_start = max(
            (prefix.rfind(delimiter) + 1 for delimiter in ("且", "或", "，", ",", "。", "；", ";")),
            default=0,
        )
        resolved = _resolve_operator_field(
            question[segment_start : operator.start()],
            columns,
            segment_start,
        )
        if isinstance(resolved, StructuredClarification):
            return _ClauseParseResult(issue=resolved)
        if resolved is None:
            message = (
                "数值比较必须指定唯一已确认的数值字段"
                if operator.group() in _COMPARISON_OPERATORS
                else "等值筛选必须指定唯一已确认的字段"
            )
            return _ClauseParseResult(issue=StructuredUnavailable(message))
        column, field_span = resolved
        if operator.group() in _COMPARISON_OPERATORS:
            if column.data_type not in {
                StructuredColumnType.INTEGER,
                StructuredColumnType.DECIMAL,
            }:
                return _ClauseParseResult(
                    issue=StructuredUnavailable("数值比较仅支持整数或小数字段")
                )
            value_match = re.match(
                rf"\s*(?P<value>{_NUMBER_RE})(?=$|[\s，,。的且或；;])",
                question[operator.end() :],
            )
            if value_match is None:
                return _ClauseParseResult(issue=StructuredUnavailable("数值比较值格式无效"))
            item = StructuredFilter(
                column.physical_name,
                _COMPARISON_OPERATORS[operator.group()],
                value_match.group("value"),
            )
        else:
            value_match = re.match(
                r"\s*(?P<value>[^\s，,。的且或；;]+)",
                question[operator.end() :],
            )
            if value_match is None:
                return _ClauseParseResult(issue=StructuredUnavailable("等值筛选值格式无效"))
            item = StructuredFilter(
                column.physical_name,
                "eq",
                value_match.group("value").strip(),
            )
        clause_span = _TextSpan(
            field_span.start,
            operator.end() + value_match.end(),
        )
        matches.append(_FilterMatch(item, clause_span, (clause_span,)))
    return _ClauseParseResult(
        value=tuple(matches),
        consumed_spans=_merge_spans(span for match in matches for span in match.consumed_spans),
    )


def _resolve_operator_field(
    segment: str,
    columns: tuple[StructuredColumnSchema, ...],
    offset: int,
) -> tuple[StructuredColumnSchema, _TextSpan] | StructuredClarification | None:
    resolved = _resolve_columns(segment, columns)
    if isinstance(resolved, StructuredClarification):
        return resolved
    if not resolved:
        return None
    column = resolved[0]
    segment_end = len(segment.rstrip())
    candidates: list[tuple[int, int, _TextSpan]] = []
    for priority, name in _resolution_names(column):
        for span in _find_normalized_spans(segment, name):
            if span.end == segment_end:
                candidates.append((priority, len(_normalize(name)), span))
    if not candidates:
        return None
    best_priority = min(priority for priority, _, _ in candidates)
    at_priority = [candidate for candidate in candidates if candidate[0] == best_priority]
    longest = max(length for _, length, _ in at_priority)
    span = next(span for _, length, span in at_priority if length == longest)
    return column, _TextSpan(offset + span.start, offset + span.end)


def _parse_implicit_filter_clauses(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
) -> _ClauseParseResult[tuple[_FilterMatch, ...]]:
    matches: dict[
        int,
        list[tuple[int, int, StructuredColumnSchema, StructuredFilter, _TextSpan]],
    ] = {}
    for column in columns:
        if column.data_type is not StructuredColumnType.STRING:
            continue
        for priority, name in _resolution_names(column):
            field_pattern = _field_pattern(name, normalized=priority == 0)
            pattern = re.compile(
                rf"(?P<value>[A-Za-z0-9\u4e00-\u9fff]{{1,20}}){field_pattern}",
                re.IGNORECASE,
            )
            for match in pattern.finditer(question):
                if re.match(
                    r"\s*(?:为|=|大于|不少于|小于|不超过)",
                    question[match.end() :],
                ):
                    continue
                value = re.sub(r"^(?:请问|请|统计|计算|查询|求)", "", match.group("value"))
                if not value or any(
                    word in value for _, words in _AGGREGATE_WORDS for word in words
                ):
                    continue
                item = StructuredFilter(column.physical_name, "eq", value)
                matches.setdefault(match.end(), []).append(
                    (
                        priority,
                        len(_normalize(name)),
                        column,
                        item,
                        _TextSpan(match.start(), match.end()),
                    )
                )
    selected = _select_implicit_filter_matches(matches)
    if isinstance(selected, StructuredClarification):
        return _ClauseParseResult(issue=selected)
    return _ClauseParseResult(
        value=selected,
        consumed_spans=_merge_spans(span for match in selected for span in match.consumed_spans),
    )


def _select_implicit_filter_matches(
    matches: dict[
        int,
        list[tuple[int, int, StructuredColumnSchema, StructuredFilter, _TextSpan]],
    ],
) -> tuple[_FilterMatch, ...] | StructuredClarification:
    selected: list[_FilterMatch] = []
    for _, candidates in sorted(matches.items()):
        best_priority = min(priority for priority, _, _, _, _ in candidates)
        at_priority = [item for item in candidates if item[0] == best_priority]
        longest = max(length for _, length, _, _, _ in at_priority)
        finalists = [item for item in at_priority if item[1] == longest]
        by_column = {column.physical_name: (item, span) for _, _, column, item, span in finalists}
        if len(by_column) > 1:
            return StructuredClarification(
                "字段名称存在歧义，请选择一个字段",
                tuple(sorted(by_column)),
            )
        item, span = next(iter(by_column.values()))
        selected.append(_FilterMatch(item, span, (span,)))
    return tuple(selected)


def _resolve_columns(
    question: str, columns: Iterable[StructuredColumnSchema]
) -> tuple[StructuredColumnSchema, ...] | StructuredClarification:
    candidates = tuple(columns)
    normalized_question = _normalize(question)
    matches: list[tuple[int, int, int, int, StructuredColumnSchema]] = []
    for column in candidates:
        for priority, name in _resolution_names(column):
            normalized_name = _normalize(name)
            if not normalized_name:
                continue
            start = normalized_question.find(normalized_name)
            while start >= 0:
                matches.append(
                    (
                        priority,
                        len(normalized_name),
                        start,
                        start + len(normalized_name),
                        column,
                    )
                )
                start = normalized_question.find(normalized_name, start + 1)
    if not matches:
        return ()

    independent = [
        match
        for match in matches
        if not any(
            other_length > match[1] and other_start <= match[2] and match[3] <= other_end
            for _, other_length, other_start, other_end, _ in matches
        )
    ]
    by_span: dict[tuple[int, int], list[tuple[int, StructuredColumnSchema]]] = {}
    for priority, _, start, end, column in independent:
        by_span.setdefault((start, end), []).append((priority, column))
    selected: set[StructuredColumnSchema] = set()
    for span_matches in by_span.values():
        best_priority = min(priority for priority, _ in span_matches)
        selected.update(column for priority, column in span_matches if priority == best_priority)
    return _unique_or_clarification(selected)


def _column_name_spans(
    question: str,
    columns: Iterable[StructuredColumnSchema],
) -> tuple[_TextSpan, ...]:
    spans: set[_TextSpan] = set()
    for column in columns:
        for _, name in _resolution_names(column):
            spans.update(_find_normalized_spans(question, name))
    return tuple(sorted(spans))


def _unique_or_clarification(
    matches: set[StructuredColumnSchema],
) -> tuple[StructuredColumnSchema, ...] | StructuredClarification:
    ordered = tuple(sorted(matches, key=lambda column: column.physical_name))
    if len(ordered) > 1:
        return StructuredClarification(
            "字段名称存在歧义，请选择一个字段",
            tuple(column.physical_name for column in ordered),
        )
    return ordered


def _column_names(column: StructuredColumnSchema) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (column.physical_name, column.display_name, column.original_name, *column.aliases)
        )
    )


def _resolution_names(column: StructuredColumnSchema) -> tuple[tuple[int, str], ...]:
    return tuple(
        dict.fromkeys(
            (
                (0, column.physical_name),
                (1, column.display_name),
                (1, column.original_name),
                *((2, alias) for alias in column.aliases),
            )
        )
    )


def _field_pattern(name: str, *, normalized: bool) -> str:
    if not normalized:
        return re.escape(name)
    normalized_name = _normalize(name)
    return r"[\s_-]*".join(re.escape(character) for character in normalized_name)


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _normalize_with_positions(value: str) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(value.casefold()):
        if re.fullmatch(r"[0-9a-z\u4e00-\u9fff]", character):
            characters.append(character)
            positions.append(index)
    return "".join(characters), tuple(positions)


def _find_normalized_spans(value: str, name: str) -> tuple[_TextSpan, ...]:
    normalized, positions = _normalize_with_positions(value)
    normalized_name = _normalize(name)
    if not normalized_name:
        return ()
    spans: list[_TextSpan] = []
    start = normalized.find(normalized_name)
    while start >= 0:
        end = start + len(normalized_name)
        spans.append(_TextSpan(positions[start], positions[end - 1] + 1))
        start = normalized.find(normalized_name, start + 1)
    return tuple(spans)


def _contains(container: _TextSpan, candidate: _TextSpan) -> bool:
    return container.start <= candidate.start and candidate.end <= container.end


def _merge_spans(spans: Iterable[_TextSpan]) -> tuple[_TextSpan, ...]:
    ordered = sorted(set(spans))
    if not ordered:
        return ()
    merged = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        if span.start <= previous.end:
            merged[-1] = _TextSpan(previous.start, max(previous.end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def _mask_spans(value: str, spans: Iterable[_TextSpan]) -> str:
    masked = list(value)
    for span in _merge_spans(spans):
        masked[span.start : span.end] = " " * (span.end - span.start)
    return "".join(masked)


def _require_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise UnsafeStructuredQueryError(f"untrusted ClickHouse identifier: {value!r}")
    return value


def _clickhouse_parameter_type(column_type: StructuredColumnType) -> str:
    return {
        StructuredColumnType.STRING: "String",
        StructuredColumnType.INTEGER: "Int64",
        StructuredColumnType.DECIMAL: "Decimal(38, 9)",
        StructuredColumnType.DATE: "Date",
        StructuredColumnType.DATETIME: "DateTime64(3)",
        StructuredColumnType.BOOLEAN: "UInt8",
    }[column_type]


def _convert_parameter(value: str, column_type: StructuredColumnType) -> object:
    try:
        if column_type is StructuredColumnType.INTEGER:
            return int(value)
        if column_type is StructuredColumnType.DECIMAL:
            return Decimal(value)
        if column_type is StructuredColumnType.DATE:
            return date.fromisoformat(value)
        if column_type is StructuredColumnType.DATETIME:
            return datetime.fromisoformat(value)
        if column_type is StructuredColumnType.BOOLEAN:
            normalized = value.strip().casefold()
            if normalized in {"1", "true", "yes", "是"}:
                return 1
            if normalized in {"0", "false", "no", "否"}:
                return 0
            raise ValueError
    except (InvalidOperation, ValueError) as error:
        raise UnsafeStructuredQueryError(
            "filter value does not match the confirmed column type"
        ) from error
    return value


def _validate_generated_select(
    sql: str,
    *,
    table_name: str,
    allowed_columns: frozenset[str],
) -> None:
    try:
        statements = sqlglot.parse(sql, read="clickhouse")
    except sqlglot.errors.ParseError as error:
        raise UnsafeStructuredQueryError("generated ClickHouse SQL could not be parsed") from error
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise UnsafeStructuredQueryError("structured query must be exactly one SELECT")
    parsed = statements[0]
    if any(parsed.find_all(exp.Join)) or any(parsed.find_all(exp.Subquery)):
        raise UnsafeStructuredQueryError("joins and subqueries are forbidden")
    if any(isinstance(node, (exp.Union, exp.Intersect, exp.Except)) for node in parsed.walk()):
        raise UnsafeStructuredQueryError("set operations are forbidden")
    tables = tuple(parsed.find_all(exp.Table))
    if len(tables) != 1 or tables[0].name != table_name:
        raise UnsafeStructuredQueryError("query table is outside the active publication")
    for function in parsed.find_all(exp.AggFunc):
        if function.sql_name().upper() not in _ALLOWED_SQL_FUNCTIONS:
            raise UnsafeStructuredQueryError("query contains a non-whitelisted function")
    if any(parsed.find_all(exp.Anonymous)):
        raise UnsafeStructuredQueryError("query contains a non-whitelisted function")
    for column in parsed.find_all(exp.Column):
        if column.name not in allowed_columns:
            raise UnsafeStructuredQueryError("query contains an unknown column")


def _aggregate_row(result: object) -> Mapping[str, object]:
    if isinstance(result, Mapping):
        return result
    named_results = getattr(result, "named_results", None)
    if named_results is not None:
        rows = list(named_results())
        if rows and isinstance(rows[0], Mapping):
            return rows[0]
    column_names = getattr(result, "column_names", None)
    result_rows = getattr(result, "result_rows", None)
    if column_names and result_rows:
        return dict(zip(column_names, result_rows[0], strict=True))
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        rows = list(result)
        if not rows:
            raise ValueError("empty aggregate result")
        first = rows[0]
        if isinstance(first, Mapping):
            return first
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
            return dict(
                zip(
                    ("aggregate_value", "total_count", "valid_count", "null_count"),
                    first,
                    strict=True,
                )
            )
    raise TypeError("unsupported ClickHouse aggregate result shape")


def _aggregate_value(value: object, aggregate: Literal["avg", "sum", "count", "min", "max"]):
    if value is None:
        return None
    if aggregate == "count":
        return int(value)
    if isinstance(value, (int, Decimal)):
        return value
    return Decimal(str(value))
