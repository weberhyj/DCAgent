from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime
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
            if metric is None or not metric.allow_aggregate:
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
                parameters[upper_name] = _convert_parameter(item.upper_value, column.data_type)
                upper_placeholder = f"{{{upper_name}:{parameter_type}}}"
                predicates.append(f"({name} >= {placeholder} AND {name} <= {upper_placeholder})")
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
    aggregate = _resolve_aggregate(question)
    if aggregate is None:
        return StructuredUnavailable("未识别到受支持的聚合意图")

    dataset = _resolve_dataset(question, catalog)
    if not isinstance(dataset, StructuredDatasetCatalog):
        return dataset

    filters = _resolve_filters(question, dataset.schema.columns)
    if isinstance(filters, StructuredClarification):
        return filters

    metric = _resolve_metric(question, dataset.schema.columns, aggregate)
    if isinstance(metric, StructuredClarification):
        return metric
    if isinstance(metric, StructuredUnavailable):
        return metric

    return StructuredIntent(
        dataset_id=dataset.schema.dataset_id,
        aggregate=aggregate,
        metric_physical_name=None if metric is None else metric.physical_name,
        filters=filters,
    )


def _resolve_aggregate(question: str) -> str | None:
    normalized = _normalize(question)
    matches = [
        (normalized.rfind(_normalize(word)), aggregate)
        for aggregate, words in _AGGREGATE_WORDS
        for word in words
        if _normalize(word) in normalized
    ]
    return max(matches, default=(-1, None))[1]


def _resolve_dataset(
    question: str, catalog: StructuredCatalog
) -> StructuredDatasetCatalog | StructuredClarification | StructuredUnavailable:
    normalized = _normalize(question)
    mentioned = []
    for dataset in catalog.datasets:
        names = {
            dataset.schema.dataset_id,
            dataset.schema.worksheet_name,
            dataset.source_name,
            PurePath(dataset.source_name).stem,
        }
        if any(_normalize(name) and _normalize(name) in normalized for name in names):
            mentioned.append(dataset)

    if len(mentioned) > 1:
        return StructuredClarification(
            "问题同时匹配多个数据集，请选择一个数据集",
            tuple(sorted(dataset.schema.dataset_id for dataset in mentioned)),
        )
    if mentioned:
        selected = mentioned[0]
        if selected.active_publication is None:
            return StructuredUnavailable("指定数据集尚未确认并发布")
        return selected

    published = [dataset for dataset in catalog.datasets if dataset.active_publication is not None]
    if len(published) == 1:
        return published[0]
    if not published:
        return StructuredUnavailable("没有已确认并发布的结构化数据集")
    return StructuredClarification(
        "请指定要查询的数据集",
        tuple(sorted(dataset.schema.dataset_id for dataset in published)),
    )


def _resolve_metric(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
    aggregate: str,
) -> StructuredColumnSchema | StructuredClarification | StructuredUnavailable | None:
    aggregate_columns = tuple(column for column in columns if column.allow_aggregate)
    matches = _resolve_columns(question, aggregate_columns)
    if isinstance(matches, StructuredClarification):
        return matches
    if matches:
        return matches[0]
    if aggregate == "count":
        return None
    return StructuredUnavailable("未识别到可聚合的指标字段")


def _resolve_filters(
    question: str, columns: tuple[StructuredColumnSchema, ...]
) -> tuple[StructuredFilter, ...] | StructuredClarification:
    filters: list[StructuredFilter] = []
    filter_columns = tuple(column for column in columns if column.allow_filter)

    for column in filter_columns:
        for name in _column_names(column):
            escaped = re.escape(name)
            comparison = re.search(
                rf"{escaped}\s*(大于|不少于|小于|不超过)\s*({_NUMBER_RE})",
                question,
                re.IGNORECASE,
            )
            if comparison:
                filters.append(
                    StructuredFilter(
                        column.physical_name,
                        _COMPARISON_OPERATORS[comparison.group(1)],
                        comparison.group(2),
                    )
                )
                break
            equality = re.search(
                rf"{escaped}\s*(?:为|=)\s*([^\s，,。的]+)",
                question,
                re.IGNORECASE,
            )
            if equality:
                filters.append(
                    StructuredFilter(column.physical_name, "eq", equality.group(1).strip())
                )
                break

    date_range = _DATE_RANGE_RE.search(question)
    if date_range:
        date_columns = tuple(
            column
            for column in filter_columns
            if column.data_type in {StructuredColumnType.DATE, StructuredColumnType.DATETIME}
        )
        explicit = _resolve_columns(question, date_columns)
        if isinstance(explicit, StructuredClarification):
            return explicit
        if explicit:
            date_column = explicit[0]
        elif len(date_columns) == 1:
            date_column = date_columns[0]
        elif len(date_columns) > 1:
            return StructuredClarification(
                "日期范围匹配多个日期字段，请选择一个字段",
                tuple(sorted(column.physical_name for column in date_columns)),
            )
        else:
            date_column = None
        if date_column is not None:
            filters.append(
                StructuredFilter(
                    date_column.physical_name,
                    "between",
                    date_range.group(1),
                    date_range.group(2),
                )
            )

    explicitly_filtered = {item.physical_name for item in filters}
    for column in filter_columns:
        if (
            column.physical_name in explicitly_filtered
            or column.data_type is not StructuredColumnType.STRING
        ):
            continue
        for name in sorted(_column_names(column), key=len, reverse=True):
            implicit = re.search(rf"([A-Za-z0-9\u4e00-\u9fff]{{1,20}}){re.escape(name)}", question)
            if implicit is None:
                continue
            value = re.sub(r"^(?:请问|请|统计|计算|查询|求)", "", implicit.group(1))
            if value and not any(word in value for _, words in _AGGREGATE_WORDS for word in words):
                filters.append(StructuredFilter(column.physical_name, "eq", value))
                explicitly_filtered.add(column.physical_name)
                break

    return tuple(filters)


def _resolve_columns(
    question: str, columns: Iterable[StructuredColumnSchema]
) -> tuple[StructuredColumnSchema, ...] | StructuredClarification:
    candidates = tuple(columns)
    normalized_question = _normalize(question)
    priority_groups = (
        tuple((column.physical_name, column) for column in candidates),
        tuple(
            (name, column)
            for column in candidates
            for name in (column.display_name, column.original_name)
        ),
    )
    for names_and_columns in priority_groups:
        matched = [
            (len(_normalize(name)), column)
            for name, column in names_and_columns
            if _normalize(name) and _normalize(name) in normalized_question
        ]
        if not matched:
            continue
        longest = max(length for length, _ in matched)
        matches = {column for length, column in matched if length == longest}
        if matches:
            return _unique_or_clarification(matches)

    alias_matches: list[tuple[int, StructuredColumnSchema]] = []
    for column in candidates:
        for alias in column.aliases:
            if _normalize(alias) and _normalize(alias) in normalized_question:
                alias_matches.append((len(_normalize(alias)), column))
    if not alias_matches:
        return ()
    longest = max(length for length, _ in alias_matches)
    return _unique_or_clarification(
        {column for length, column in alias_matches if length == longest}
    )


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


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


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
