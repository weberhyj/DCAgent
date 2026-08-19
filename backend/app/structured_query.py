from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import PurePath

import sqlglot
from sqlglot import exp

from .clickhouse_compatibility import (
    ClickHouseCompatibilityMode,
    ClickHouseCompatibilityProfile,
)
from .structured_models import (
    MAX_STRUCTURED_ROUTE_FIELDS,
    StructuredAggregateResult,
    StructuredCatalog,
    StructuredClarification,
    StructuredColumnSchema,
    StructuredColumnType,
    StructuredDatasetCatalog,
    StructuredFilter,
    StructuredGroupedAggregateRow,
    StructuredIntent,
    StructuredMetricIntent,
    StructuredMetricResult,
    StructuredMultiAggregateIntent,
    StructuredMultiAggregatePlan,
    StructuredMultiAggregateResult,
    StructuredPublication,
    StructuredQueryPlan,
    StructuredRowLookupIntent,
    StructuredRowLookupPlan,
    StructuredRowLookupResult,
    StructuredUnavailable,
)

StructuredIntentResolution = (
    StructuredIntent
    | StructuredMultiAggregateIntent
    | StructuredRowLookupIntent
    | StructuredClarification
    | StructuredUnavailable
)


@dataclass(frozen=True, order=True)
class _TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class _ClauseParseResult[T]:
    value: T | None = None
    consumed_spans: tuple[_TextSpan, ...] = ()
    shared_columns: tuple[StructuredColumnSchema, ...] = ()
    count_all_hint: bool = False
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
    ("count_distinct", ("去重数量", "去重数", "不同数量", "唯一数量", "不重复数量")),
    ("max", ("最大", "最高")),
    ("min", ("最小", "最低")),
    ("median", ("中位数", "中位值")),
    ("stddev", ("标准差", "标准偏差")),
    ("variance", ("方差", "变异数")),
    ("percentile", ("分位数", "百分位", "百分位数")),
)
_SUMMARY_WORDS = ("汇总", "统计")
_NUMERIC_TYPES = frozenset({StructuredColumnType.INTEGER, StructuredColumnType.DECIMAL})
_COMPARISON_OPERATORS = {"大于": "gt", "不少于": "gte", "小于": "lt", "不超过": "lte"}
_DATE_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*至\s*(\d{4}-\d{2}-\d{2})")
_NATURAL_DATETIME_RANGE_RE = re.compile(
    r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日?\s*"
    r"(?P<start_hour>\d{1,2}):(?P<start_minute>\d{2})\s*"
    r"(?:到|至)\s*"
    r"(?:(?P<end_year>\d{4})年(?P<end_month>\d{1,2})月(?P<end_day>\d{1,2})日?\s*)?"
    r"(?P<end_hour>\d{1,2}):(?P<end_minute>\d{2})"
)
_DATE_LITERAL_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_NUMBER_RE = r"-?\d+(?:\.\d+)?"
_PERCENTILE_RE = re.compile(
    r"(?:P\s*(?P<p_short>\d{1,3}(?:\.\d+)?)|"
    r"(?P<p_long>\d{1,3}(?:\.\d+)?)\s*(?:%|百分位|百分位数|分位数))",
    re.IGNORECASE,
)
_TOP_N_RE = re.compile(
    r"(?P<direction>前|后|最高|最低|TOP|BOTTOM)\s*(?P<limit>\d{1,4})\s*(?:名|个|条|组)?",
    re.IGNORECASE,
)
_ORDER_DIRECTION_RE = re.compile(r"(?P<direction>升序|降序|从低到高|从高到低|由小到大|由大到小)")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]+$")
_ALLOWED_AGGREGATES = frozenset(
    {
        "avg",
        "sum",
        "count",
        "count_distinct",
        "min",
        "max",
        "median",
        "percentile",
        "stddev",
        "variance",
    }
)
_ALLOWED_SQL_FUNCTIONS = frozenset(
    {
        "AVG",
        "SUM",
        "COUNT",
        "MIN",
        "MAX",
        "UNIQEXACT",
        "QUANTILE",
        "QUANTILEEXACT",
        "STDDEVPOP",
        "VARPOP",
    }
)
_ROW_LOOKUP_MARKERS = (
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


class UnsafeStructuredQueryError(ValueError):
    pass


class InconsistentStructuredResultError(ValueError):
    pass


def _aggregate_sql_expression(
    aggregate: str,
    column_name: str,
    *,
    percentile: float | None = None,
) -> str:
    """Build one of the finite, application-owned ClickHouse aggregate expressions."""

    name = _require_identifier(column_name)
    if aggregate == "count_distinct":
        return f"uniqExact({name})"
    if aggregate == "median":
        return f"quantileExact(0.5)({name})"
    if aggregate == "percentile":
        if percentile is None or not 0 < percentile <= 100:
            raise UnsafeStructuredQueryError("percentile must be between 0 and 100")
        return f"quantileExact({percentile / 100:g})({name})"
    if aggregate == "stddev":
        return f"stddevPop({name})"
    if aggregate == "variance":
        return f"varPop({name})"
    if aggregate in {"avg", "sum", "count", "min", "max"}:
        return f"{aggregate}({name})"
    raise UnsafeStructuredQueryError("unsupported aggregate")


class StructuredQueryPlanner:
    def __init__(
        self,
        catalog: StructuredCatalog,
        compatibility: ClickHouseCompatibilityProfile | None = None,
        *,
        implicit_summary_max_metrics: int = 12,
    ) -> None:
        self._catalog = catalog
        self._compatibility = compatibility or ClickHouseCompatibilityProfile.for_mode(
            ClickHouseCompatibilityMode.MODERN
        )
        self._implicit_summary_max_metrics = implicit_summary_max_metrics

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
            if metric is None or (
                intent.aggregate not in {"count", "count_distinct"}
                and not metric.allow_aggregate
            ):
                raise UnsafeStructuredQueryError("unknown or disallowed aggregate column")
            _require_identifier(metric.physical_name)
        elif intent.aggregate != "count":
            raise UnsafeStructuredQueryError("aggregate requires a confirmed metric")

        aggregate_expression = (
            "count()"
            if metric is None
            else _aggregate_sql_expression(
                intent.aggregate,
                metric.physical_name,
                percentile=intent.percentile,
            )
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
            parameter_type = self._compatibility.parameter_type(column.data_type)
            parameters[parameter_name] = _convert_parameter(
                item.value, column.data_type, self._compatibility
            )
            placeholder = f"{{{parameter_name}:{parameter_type}}}"
            if item.operator == "between":
                if item.upper_value is None:
                    raise UnsafeStructuredQueryError("between filter requires an upper value")
                upper_name = f"filter_{index}_upper"
                upper_value = _convert_parameter(
                    item.upper_value, column.data_type, self._compatibility
                )
                upper_value, upper_operator = _between_upper_bound(
                    column.data_type,
                    item.upper_value,
                    upper_value,
                )
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
            percentile=intent.percentile,
        )

    def plan_multi(
        self,
        intent: StructuredMultiAggregateIntent,
        publication: StructuredPublication,
    ) -> StructuredMultiAggregatePlan:
        dataset = self._require_dataset(intent.dataset_id)
        active = dataset.active_publication
        if active is None:
            raise UnsafeStructuredQueryError("structured dataset is not published")
        if publication != active:
            raise UnsafeStructuredQueryError("publication is not the active catalog publication")
        if not intent.metrics:
            raise UnsafeStructuredQueryError("multi-metric summary requires at least one metric")
        if intent.implicit and len(intent.metrics) > self._implicit_summary_max_metrics:
            raise UnsafeStructuredQueryError("implicit metric count exceeds configured limit")

        table_name = _require_identifier(publication.physical_table_name)
        columns = {column.physical_name: column for column in dataset.schema.columns}
        projections = []
        seen_metrics: set[tuple[str, str, float | None]] = set()
        columns_for_query = {column.physical_name: column for column in dataset.schema.columns}
        group_names: list[str] = []
        for group_name in intent.group_by:
            group_column = columns_for_query.get(group_name)
            if group_column is None or not group_column.allow_filter:
                raise UnsafeStructuredQueryError("unknown or disallowed group-by column")
            if group_name in group_names:
                raise UnsafeStructuredQueryError("duplicate group-by column")
            group_names.append(_require_identifier(group_name))
            projections.append(f"{group_names[-1]} AS group_{len(group_names) - 1}")
        projections.append("count() AS total_count")
        for index, metric_intent in enumerate(intent.metrics):
            aggregate = metric_intent.aggregate
            if aggregate not in _ALLOWED_AGGREGATES:
                raise UnsafeStructuredQueryError("unsupported aggregate")
            name = _require_identifier(metric_intent.metric_physical_name)
            metric_key = (aggregate, name, metric_intent.percentile or intent.percentile)
            if metric_key in seen_metrics:
                raise UnsafeStructuredQueryError("duplicate metric")
            seen_metrics.add(metric_key)
            metric = columns.get(name)
            if metric is None:
                raise UnsafeStructuredQueryError("unknown metric column")
            if aggregate not in {"count", "count_distinct"} and not metric.allow_aggregate:
                raise UnsafeStructuredQueryError("disallowed aggregate column")
            aggregate_expression = _aggregate_sql_expression(
                aggregate,
                name,
                percentile=metric_intent.percentile or intent.percentile,
            )
            projections.extend(
                (
                    f"{aggregate_expression} AS metric_{index}_value",
                    f"count({name}) AS metric_{index}_valid_count",
                    f"count() - count({name}) AS metric_{index}_null_count",
                )
            )

        parameters: dict[str, object] = {}
        predicates = []
        for index, item in enumerate(intent.filters):
            column = columns.get(item.physical_name)
            if column is None or not column.allow_filter:
                raise UnsafeStructuredQueryError("unknown or disallowed filter column")
            name = _require_identifier(column.physical_name)
            parameter_name = f"filter_{index}"
            parameter_type = self._compatibility.parameter_type(column.data_type)
            parameters[parameter_name] = _convert_parameter(
                item.value, column.data_type, self._compatibility
            )
            placeholder = f"{{{parameter_name}:{parameter_type}}}"
            if item.operator == "between":
                if item.upper_value is None:
                    raise UnsafeStructuredQueryError("between filter requires an upper value")
                upper_name = f"filter_{index}_upper"
                upper_value = _convert_parameter(
                    item.upper_value, column.data_type, self._compatibility
                )
                upper_value, upper_operator = _between_upper_bound(
                    column.data_type,
                    item.upper_value,
                    upper_value,
                )
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
        if group_names:
            sql += " GROUP BY " + ", ".join(group_names)
            if intent.order_by is not None:
                order_column = _require_identifier(intent.order_by)
                if order_column not in columns or order_column not in group_names:
                    metric_alias = next(
                        (
                            f"metric_{index}_value"
                            for index, metric in enumerate(intent.metrics)
                            if metric.metric_physical_name == order_column
                        ),
                        None,
                    )
                    if metric_alias is None:
                        raise UnsafeStructuredQueryError("order-by column is not in the analysis")
                    order_column = metric_alias
                sql += f" ORDER BY {order_column} {'DESC' if intent.order_desc else 'ASC'}"
            if intent.limit is not None:
                if not 1 <= intent.limit <= 1000:
                    raise UnsafeStructuredQueryError("analysis limit must be between 1 and 1000")
                sql += f" LIMIT {intent.limit}"
        _validate_generated_select(
            sql,
            table_name=table_name,
            allowed_columns=frozenset((*columns, *(f"group_{i}" for i in range(len(group_names))), *(f"metric_{i}_value" for i in range(len(intent.metrics))))),
        )
        return StructuredMultiAggregatePlan(
            publication_id=publication.publication_id,
            dataset_id=intent.dataset_id,
            metrics=intent.metrics,
            sql=sql,
            parameters=parameters,
            filters=intent.filters,
            implicit=intent.implicit,
            group_by=tuple(group_names),
            percentile=intent.percentile,
            order_by=intent.order_by,
            order_desc=intent.order_desc,
            limit=intent.limit,
        )

    def plan_row_lookup(
        self,
        intent: StructuredRowLookupIntent,
        publication: StructuredPublication,
    ) -> StructuredRowLookupPlan:
        dataset = self._require_dataset(intent.dataset_id)
        if dataset.active_publication != publication:
            raise UnsafeStructuredQueryError("publication is not the active catalog publication")
        if not 1 <= intent.limit <= 1000:
            raise UnsafeStructuredQueryError("row lookup limit must be between 1 and 1000")
        columns = {column.physical_name: column for column in dataset.schema.columns}
        selected = []
        for name in intent.selected_physical_names:
            column = columns.get(name)
            if column is None:
                raise UnsafeStructuredQueryError("unknown row lookup column")
            selected.append(column)
        if not selected:
            raise UnsafeStructuredQueryError("row lookup requires selected columns")
        parameters: dict[str, object] = {}
        predicates: list[str] = []
        for index, item in enumerate(intent.filters):
            column = columns.get(item.physical_name)
            if column is None or not column.allow_filter:
                raise UnsafeStructuredQueryError("unknown or disallowed filter column")
            name = _require_identifier(column.physical_name)
            parameter_name = f"filter_{index}"
            parameter_type = self._compatibility.parameter_type(column.data_type)
            if item.operator == "eq" and item.upper_value is None:
                parameters[parameter_name] = _convert_parameter(
                    item.value, column.data_type, self._compatibility
                )
                predicates.append(f"{name} = {{{parameter_name}:{parameter_type}}}")
                continue
            if item.operator != "between" or item.upper_value is None:
                raise UnsafeStructuredQueryError(
                    "row lookup supports equality and bounded date filters only"
                )
            upper_name = f"filter_{index}_upper"
            upper_value = _convert_parameter(
                item.upper_value, column.data_type, self._compatibility
            )
            upper_value, upper_operator = _between_upper_bound(
                column.data_type,
                item.upper_value,
                upper_value,
            )
            parameters[parameter_name] = _convert_parameter(
                item.value, column.data_type, self._compatibility
            )
            parameters[upper_name] = upper_value
            predicates.append(
                f"({name} >= {{{parameter_name}:{parameter_type}}} AND "
                f"{name} {upper_operator} {{{upper_name}:{parameter_type}}})"
            )
        table_name = _require_identifier(publication.physical_table_name)
        sql = f"SELECT {', '.join(_require_identifier(column.physical_name) for column in selected)} FROM {table_name}"
        if predicates:
            sql += " WHERE " + " AND ".join(predicates)
        sql += f" LIMIT {intent.limit + 1}"
        _validate_generated_select(sql, table_name=table_name, allowed_columns=frozenset(columns))
        return StructuredRowLookupPlan(
            publication_id=publication.publication_id,
            dataset_id=intent.dataset_id,
            selected_physical_names=tuple(column.physical_name for column in selected),
            sql=sql,
            parameters=parameters,
            filters=intent.filters,
            limit=intent.limit,
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
        compatibility: ClickHouseCompatibilityProfile | None = None,
        implicit_summary_max_metrics: int = 12,
        clock: Callable[[], float] = time.perf_counter,
        audit_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._catalog = catalog
        self._clickhouse = clickhouse_gateway
        self._compatibility = compatibility or ClickHouseCompatibilityProfile.for_mode(
            ClickHouseCompatibilityMode.MODERN
        )
        self._implicit_summary_max_metrics = implicit_summary_max_metrics
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
            expected = StructuredQueryPlanner(self._catalog, self._compatibility).plan(
                StructuredIntent(
                    dataset_id=plan.dataset_id,
                    aggregate=plan.aggregate,
                    metric_physical_name=plan.metric_physical_name,
                    filters=plan.filters,
                    percentile=plan.percentile,
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
            _require_exact_result_aliases(
                row,
                ("aggregate_value", "total_count", "valid_count", "null_count"),
            )
            total_count = _strict_integer(row["total_count"])
            valid_count = _strict_integer(row["valid_count"])
            null_count = _strict_integer(row["null_count"])
            value = _aggregate_value(
                row["aggregate_value"],
                plan.aggregate,
                valid_count=valid_count,
            )
        except (KeyError, TypeError, ValueError, IndexError, ArithmeticError):
            return StructuredUnavailable("结构化查询返回了无效结果")
        if min(total_count, valid_count, null_count) < 0 or valid_count + null_count != total_count:
            return StructuredUnavailable("结构化查询返回了不一致的计数")
        if (
            plan.aggregate == "count_distinct"
            and value is not None
            and (not isinstance(value, int) or value < 0 or value > valid_count)
        ):
            return StructuredUnavailable("结构化查询返回了不一致的去重计数")

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
            source_id=dataset.schema.source_id,
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
            percentile=plan.percentile,
        )

    def execute_multi(
        self, plan: StructuredMultiAggregatePlan
    ) -> StructuredMultiAggregateResult | StructuredUnavailable:
        dataset = self._require_active_dataset(plan)
        if isinstance(dataset, StructuredUnavailable):
            return dataset
        publication = dataset.active_publication
        assert publication is not None

        try:
            expected = StructuredQueryPlanner(
                self._catalog,
                self._compatibility,
                implicit_summary_max_metrics=self._implicit_summary_max_metrics,
            ).plan_multi(
                StructuredMultiAggregateIntent(
                    dataset_id=plan.dataset_id,
                    metrics=plan.metrics,
                    filters=plan.filters,
                    implicit=plan.implicit,
                    group_by=plan.group_by,
                    percentile=plan.percentile,
                    order_by=plan.order_by,
                    order_desc=plan.order_desc,
                    limit=plan.limit,
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
            rows = _multi_aggregate_rows(raw_result, len(plan.metrics), len(plan.group_by))
            if not plan.group_by and len(rows) != 1:
                raise ValueError("ungrouped aggregate query must return one row")
            grouped_results = tuple(
                StructuredGroupedAggregateRow(
                    group_values=tuple(
                        _format_row_value(row[f"group_{index}"])
                        for index in range(len(plan.group_by))
                    ),
                    metrics=self._parse_metric_results(row, plan, dataset),
                    total_count=_strict_integer(row["total_count"]),
                )
                for row in rows
            )
            if plan.group_by:
                total_count = sum(group.total_count for group in grouped_results)
                metric_results: tuple[StructuredMetricResult, ...] = ()
            else:
                total_count = grouped_results[0].total_count
                metric_results = grouped_results[0].metrics
        except InconsistentStructuredResultError:
            return StructuredUnavailable("结构化查询返回了不一致的计数")
        except (KeyError, TypeError, ValueError, IndexError, ArithmeticError):
            return StructuredUnavailable("结构化查询返回了无效结果")

        return StructuredMultiAggregateResult(
            dataset_id=dataset.schema.dataset_id,
            source_id=dataset.schema.source_id,
            schema_version=dataset.schema.schema_version,
            metrics=metric_results,
            total_count=total_count,
            source_name=dataset.source_name,
            worksheet_name=dataset.schema.worksheet_name,
            publication_id=publication.publication_id,
            filters=plan.filters,
            elapsed_ms=elapsed_ms,
            audit_id=self._audit_id_factory(),
            group_by=plan.group_by,
            groups=grouped_results if plan.group_by else (),
        )

    def _parse_metric_results(
        self,
        row: Mapping[str, object],
        plan: StructuredMultiAggregatePlan,
        dataset: StructuredDatasetCatalog,
    ) -> tuple[StructuredMetricResult, ...]:
        _require_exact_result_aliases(
            row,
            _multi_aggregate_aliases(len(plan.metrics), len(plan.group_by)),
        )
        total_count = _strict_integer(row["total_count"])
        results: list[StructuredMetricResult] = []
        for index, metric_intent in enumerate(plan.metrics):
            valid_count = _strict_integer(row[f"metric_{index}_valid_count"])
            null_count = _strict_integer(row[f"metric_{index}_null_count"])
            if min(total_count, valid_count, null_count) < 0 or valid_count + null_count != total_count:
                raise InconsistentStructuredResultError("inconsistent aggregate counts")
            metric = next(
                (
                    column
                    for column in dataset.schema.columns
                    if column.physical_name == metric_intent.metric_physical_name
                ),
                None,
            )
            if metric is None:
                raise ValueError("unknown metric")
            raw_value = row[f"metric_{index}_value"]
            if metric_intent.aggregate == "count":
                value = _strict_integer(raw_value)
                if value < 0 or value != valid_count:
                    raise InconsistentStructuredResultError("inconsistent count")
            elif metric_intent.aggregate == "count_distinct":
                value = _strict_integer(raw_value)
                if value < 0 or value > valid_count:
                    raise InconsistentStructuredResultError("inconsistent distinct count")
            else:
                value = _aggregate_value(
                    raw_value,
                    metric_intent.aggregate,
                    valid_count=valid_count,
                )
            results.append(
                StructuredMetricResult(
                    aggregate=metric_intent.aggregate,
                    metric_physical_name=metric_intent.metric_physical_name,
                    metric_display_name=metric.display_name,
                    value=value,
                    valid_count=valid_count,
                    null_count=null_count,
                    percentile=metric_intent.percentile or plan.percentile,
                )
            )
        return tuple(results)

    def execute_row_lookup(
        self, plan: StructuredRowLookupPlan
    ) -> StructuredRowLookupResult | StructuredUnavailable:
        dataset = self._require_active_dataset(plan)
        if isinstance(dataset, StructuredUnavailable):
            return dataset
        publication = dataset.active_publication
        assert publication is not None
        try:
            expected = StructuredQueryPlanner(self._catalog, self._compatibility).plan_row_lookup(
                StructuredRowLookupIntent(
                    dataset_id=plan.dataset_id,
                    filters=plan.filters,
                    selected_physical_names=plan.selected_physical_names,
                    limit=plan.limit,
                ),
                publication,
            )
        except UnsafeStructuredQueryError:
            return StructuredUnavailable("row lookup plan is no longer valid")
        if plan.sql != expected.sql or dict(plan.parameters) != dict(expected.parameters):
            return StructuredUnavailable("row lookup plan failed safety validation")
        query = getattr(self._clickhouse, "query", None)
        if query is None:
            return StructuredUnavailable("structured query service is unavailable")
        started = self._clock()
        try:
            raw_result = query(plan.sql, plan.parameters)
            raw_rows = [raw_result] if isinstance(raw_result, Mapping) else list(raw_result)
        except Exception:
            return StructuredUnavailable("structured query service is unavailable")
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
        rows: list[tuple[str, ...]] = []
        for raw_row in raw_rows[: plan.limit + 1]:
            if isinstance(raw_row, Mapping):
                try:
                    rows.append(
                        tuple(
                            _format_row_value(raw_row[name])
                            for name in plan.selected_physical_names
                        )
                    )
                except KeyError:
                    return StructuredUnavailable("structured query returned invalid columns")
            elif isinstance(raw_row, (list, tuple)) and len(raw_row) == len(
                plan.selected_physical_names
            ):
                rows.append(tuple(_format_row_value(value) for value in raw_row))
            else:
                return StructuredUnavailable("structured query returned invalid rows")
        truncated = len(rows) > plan.limit
        rows = rows[: plan.limit]
        selected_display_names = tuple(
            next(
                column.display_name
                for column in dataset.schema.columns
                if column.physical_name == name
            )
            for name in plan.selected_physical_names
        )
        return StructuredRowLookupResult(
            dataset_id=dataset.schema.dataset_id,
            source_id=dataset.schema.source_id,
            schema_version=dataset.schema.schema_version,
            selected_physical_names=plan.selected_physical_names,
            selected_display_names=selected_display_names,
            rows=tuple(rows),
            total_count=len(rows),
            truncated=truncated,
            source_name=dataset.source_name,
            worksheet_name=dataset.schema.worksheet_name,
            publication_id=publication.publication_id,
            filters=plan.filters,
            elapsed_ms=elapsed_ms,
            audit_id=self._audit_id_factory(),
        )

    def _require_active_dataset(
        self, plan: StructuredQueryPlan | StructuredMultiAggregatePlan | StructuredRowLookupPlan
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
    question: str,
    catalog: StructuredCatalog,
    *,
    implicit_summary_max_metrics: int = 12,
) -> StructuredIntentResolution:
    row_lookup = _parse_row_lookup_intent(question, catalog)
    if row_lookup is not None:
        return row_lookup
    dataset_result = _parse_dataset_clause(question, catalog)
    if dataset_result.issue is not None:
        return _with_dataset_issue_context(question, catalog, dataset_result.issue)
    assert dataset_result.value is not None
    dataset = dataset_result.value

    group_result = _parse_group_by_clause(
        question,
        dataset.schema.columns,
        dataset_result.consumed_spans,
    )
    if group_result.issue is not None:
        return _with_clarification_context(
            group_result.issue,
            dataset,
            origin_route="excel_multi_aggregate",
        )
    group_columns = group_result.value or ()
    filter_result = _parse_filter_clause(
        question,
        dataset.schema.columns,
        (*dataset_result.consumed_spans, *group_result.consumed_spans),
    )
    if filter_result.issue is not None:
        return _with_route_context(
            filter_result.issue, dataset, origin_route="excel_filtered_aggregate"
        )
    assert filter_result.value is not None

    base_consumed = (
        *dataset_result.consumed_spans,
        *group_result.consumed_spans,
        *filter_result.consumed_spans,
    )
    top_limit, top_desc, top_spans = _parse_top_n_clause(question)
    order_explicit, order_desc, order_spans = _parse_order_direction_clause(question)
    if not order_explicit:
        order_desc = top_desc
    if top_limit == 0:
        return _with_route_context(
            StructuredUnavailable("Top N 数量必须在 1 到 1000 之间"),
            dataset,
            origin_route="excel_multi_aggregate",
        )
    if (top_limit is not None or order_explicit) and not group_columns:
        return _with_route_context(
            StructuredUnavailable("Top N 汇总必须明确分组字段，例如“按地区统计销售额前 10 名”"),
            dataset,
            origin_route="excel_multi_aggregate",
        )
    consumed = (*base_consumed, *top_spans, *order_spans)
    aggregate_result = _parse_aggregate_clause(
        question,
        dataset.schema.columns,
        consumed,
        allow_missing=True,
    )
    if aggregate_result.issue is not None:
        combined = _parse_combined_aggregate_intent(
            question,
            dataset,
            filter_result.value,
            consumed,
            group_columns,
            top_limit,
            order_desc,
            order_explicit,
        )
        if combined is not None:
            return combined
        return _with_route_context(
            aggregate_result.issue, dataset, origin_route="excel_filtered_aggregate"
        )

    consumed = (*consumed, *aggregate_result.consumed_spans)
    available = _mask_spans(question, consumed)
    field_spans = _column_name_spans(available, dataset.schema.columns)
    has_summary_word = any(
        not any(_contains(field_span, span) for field_span in field_spans)
        for word in _SUMMARY_WORDS
        for span in _find_normalized_spans(available, word)
    )
    if aggregate_result.value is None and not has_summary_word:
        return _with_route_context(
            StructuredUnavailable("未识别到受支持的聚合意图"),
            dataset,
            origin_route="excel_filtered_aggregate",
        )
    aggregate = aggregate_result.value or "sum"
    percentile = _extract_percentile(question) if aggregate == "percentile" else None
    if aggregate == "percentile" and percentile is None:
        return _with_route_context(
            StructuredUnavailable("请明确分位点，例如 P90 或 90 分位数"),
            dataset,
            origin_route="excel_filtered_aggregate",
        )

    metric_list_result = _parse_metric_list(
        question,
        dataset.schema.columns,
        aggregate,
        consumed,
    )
    if metric_list_result.issue is not None:
        return _with_clarification_context(
            metric_list_result.issue,
            dataset,
            origin_route="excel_filtered_aggregate",
        )
    assert metric_list_result.value is not None
    metrics = metric_list_result.value

    if metrics:
        consumed = (*consumed, *metric_list_result.consumed_spans)
        if len(metrics) > MAX_STRUCTURED_ROUTE_FIELDS:
            return _with_clarification_context(
                StructuredClarification(
                    f"一次最多只能汇总 {MAX_STRUCTURED_ROUTE_FIELDS} 个指标，请减少选择",
                    tuple(column.display_name for column in metrics[:MAX_STRUCTURED_ROUTE_FIELDS]),
                ),
                dataset,
                origin_route="excel_multi_aggregate",
                target_fields=tuple(
                    column.display_name for column in metrics[:MAX_STRUCTURED_ROUTE_FIELDS]
                ),
            )
        remaining = _mask_spans(question, consumed)
        if _DATE_RANGE_RE.search(remaining) or re.search(
            r"大于|不少于|小于|不超过|为|=", remaining
        ):
            return _with_route_context(
                StructuredUnavailable("结构化查询包含未识别的筛选条件"),
                dataset,
                origin_route=(
                    "excel_multi_aggregate" if len(metrics) > 1 else "excel_filtered_aggregate"
                ),
                target_fields=tuple(column.display_name for column in metrics),
            )
        if len(metrics) == 1 and not group_columns:
            return StructuredIntent(
                dataset_id=dataset.schema.dataset_id,
                aggregate=aggregate,
                metric_physical_name=metrics[0].physical_name,
                filters=filter_result.value,
                percentile=percentile,
            )
        return StructuredMultiAggregateIntent(
            dataset_id=dataset.schema.dataset_id,
            metrics=tuple(
                StructuredMetricIntent(aggregate, column.physical_name, percentile)
                for column in metrics
            ),
            filters=filter_result.value,
            implicit=False,
            group_by=tuple(column.physical_name for column in group_columns),
            percentile=percentile,
            order_by=metrics[0].physical_name if top_limit is not None or order_explicit else None,
            order_desc=order_desc,
            limit=top_limit,
        )

    if aggregate_result.value is not None:
        metric_result = _parse_metric_clause(
            question,
            dataset.schema.columns,
            aggregate,
            filter_result.value,
            filter_result.shared_columns,
            aggregate_result.count_all_hint,
            consumed,
        )
        if metric_result.issue is not None:
            return _with_clarification_context(
                metric_result.issue,
                dataset,
                origin_route="excel_filtered_aggregate",
            )
        metric = metric_result.value

        consumed = (*consumed, *metric_result.consumed_spans)
        remaining = _mask_spans(question, consumed)
        if _DATE_RANGE_RE.search(remaining) or re.search(
            r"大于|不少于|小于|不超过|为|=", remaining
        ):
            return _with_route_context(
                StructuredUnavailable("结构化查询包含未识别的筛选条件"),
                dataset,
                origin_route="excel_filtered_aggregate",
                target_fields=(() if metric is None else (metric.display_name,)),
            )

        if group_columns or top_limit is not None:
            if metric is None:
                return _with_route_context(
                    StructuredUnavailable("分组聚合必须指定指标列"),
                    dataset,
                    origin_route="excel_multi_aggregate",
                )
            return StructuredMultiAggregateIntent(
                dataset_id=dataset.schema.dataset_id,
                metrics=(StructuredMetricIntent(aggregate, metric.physical_name, percentile),),
                filters=filter_result.value,
                implicit=False,
                group_by=tuple(column.physical_name for column in group_columns),
                percentile=percentile,
                order_by=metric.physical_name if top_limit is not None or order_explicit else None,
                order_desc=order_desc,
                limit=top_limit,
            )
        return StructuredIntent(
            dataset_id=dataset.schema.dataset_id,
            aggregate=aggregate,
            metric_physical_name=None if metric is None else metric.physical_name,
            filters=filter_result.value,
            percentile=percentile,
        )

    if has_summary_word:
        implicit_columns = tuple(
            column
            for column in dataset.schema.columns
            if column.allow_aggregate and column.data_type in _NUMERIC_TYPES
        )
        if not implicit_columns:
            return StructuredUnavailable(
                "没有可汇总的已授权数值列",
                dataset.schema.dataset_id,
                (),
                (dataset.schema.source_id,),
                "excel_multi_aggregate",
            )
        capped_columns = implicit_columns[:MAX_STRUCTURED_ROUTE_FIELDS]
        if len(implicit_columns) > implicit_summary_max_metrics:
            return StructuredClarification(
                f"可汇总指标超过上限，最多可汇总 {implicit_summary_max_metrics} 个指标，请选择",
                tuple(column.display_name for column in capped_columns),
                dataset_id=dataset.schema.dataset_id,
                target_fields=tuple(column.display_name for column in capped_columns),
                candidate_source_ids=(dataset.schema.source_id,),
                origin_route="excel_multi_aggregate",
            )
        return StructuredMultiAggregateIntent(
            dataset_id=dataset.schema.dataset_id,
            metrics=tuple(
                StructuredMetricIntent("sum", column.physical_name) for column in implicit_columns
            ),
            filters=filter_result.value,
            implicit=True,
            group_by=tuple(column.physical_name for column in group_columns),
            order_by=(
                implicit_columns[0].physical_name
                if top_limit is not None or order_explicit
                else None
            ),
            order_desc=order_desc,
            limit=top_limit,
        )

    return StructuredUnavailable("未识别到受支持的聚合意图")


def _parse_combined_aggregate_intent(
    question: str,
    dataset: StructuredDatasetCatalog,
    filters: tuple[StructuredFilter, ...],
    excluded_spans: tuple[_TextSpan, ...],
    group_columns: tuple[StructuredColumnSchema, ...],
    top_limit: int | None,
    order_desc: bool,
    order_explicit: bool,
) -> StructuredMultiAggregateIntent | None:
    """Support one metric requested with several explicit statistics in one query."""

    available = _mask_spans(question, excluded_spans)
    field_spans = _column_name_spans(available, dataset.schema.columns)
    occurrences: list[tuple[int, str, _TextSpan]] = []
    for aggregate, words in _AGGREGATE_WORDS:
        for word in words:
            for span in _find_normalized_spans(available, word):
                if any(_contains(field_span, span) for field_span in field_spans):
                    continue
                occurrences.append((span.start, aggregate, span))
    if len({aggregate for _, aggregate, _ in occurrences}) < 2:
        return None
    longest = [
        item
        for item in occurrences
        if not any(
            other[1] != item[1]
            and other[2].start <= item[2].start
            and item[2].end <= other[2].end
            and (other[2].end - other[2].start) > (item[2].end - item[2].start)
            for other in occurrences
        )
    ]
    aggregates = tuple(dict.fromkeys(aggregate for _, aggregate, _ in sorted(longest)))
    metric_text = _mask_spans(available, tuple(span for _, _, span in longest))
    metrics = _resolve_multiple_columns(metric_text, dataset.schema.columns)
    if isinstance(metrics, StructuredClarification) or len(metrics) != 1:
        return None
    metric = metrics[0]
    if any(
        aggregate not in {"count", "count_distinct"} and not metric.allow_aggregate
        for aggregate in aggregates
    ):
        return None
    metric_intents = tuple(
        StructuredMetricIntent(
            aggregate,
            metric.physical_name,
            _extract_percentile(question) if aggregate == "percentile" else None,
        )
        for aggregate in aggregates
    )
    return StructuredMultiAggregateIntent(
        dataset_id=dataset.schema.dataset_id,
        metrics=metric_intents,
        filters=filters,
        implicit=False,
        group_by=tuple(column.physical_name for column in group_columns),
        order_by=metric.physical_name if top_limit is not None or order_explicit else None,
        order_desc=order_desc,
        limit=top_limit,
    )


def _parse_group_by_clause(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
    excluded_spans: tuple[_TextSpan, ...],
) -> _ClauseParseResult[tuple[StructuredColumnSchema, ...]]:
    """Resolve governed group dimensions from phrases such as ``按地区`` or ``各地区``."""

    available = _mask_spans(question, excluded_spans)
    marker_matches = list(
        re.finditer(
            r"(?P<marker>按照|分别按|分组按|每个|每种|按|各)(?P<body>.+?)"
            r"(?=分别|分组|统计|汇总|求和|合计|总和|平均|均值|计数|去重|最大|最小|"
            r"最高|最低|中位|标准差|方差|P\s*\d|\d+\s*(?:%|分位)|[，,。；;]|$)",
            available,
            re.IGNORECASE,
        )
    )
    if not marker_matches:
        return _ClauseParseResult(value=())
    selected: list[StructuredColumnSchema] = []
    spans: list[_TextSpan] = []
    for marker in marker_matches:
        body_start = marker.start("body")
        body = marker.group("body")
        resolved = _resolve_multiple_columns(body, columns)
        if isinstance(resolved, StructuredClarification):
            return _ClauseParseResult(issue=resolved)
        resolved_columns = (
            resolved[:1]
            if marker.group("marker") in {"各", "每个", "每种"}
            or re.search(r"[、,，和与及]", body) is None
            else resolved
        )
        for column in resolved_columns:
            if column.physical_name not in {item.physical_name for item in selected}:
                selected.append(column)
                spans.extend(_column_name_spans(available, (column,)))
        spans.append(_TextSpan(body_start, marker.end("body")))
    if not selected:
        return _ClauseParseResult(issue=StructuredClarification("未识别到分组字段", ()))
    return _ClauseParseResult(value=tuple(selected), consumed_spans=_merge_spans(spans))


def _parse_top_n_clause(question: str) -> tuple[int | None, bool, tuple[_TextSpan, ...]]:
    match = _TOP_N_RE.search(question)
    if match is None:
        return None, True, ()
    limit = int(match.group("limit"))
    if not 1 <= limit <= 1000:
        return 0, True, (_TextSpan(match.start(), match.end()),)
    direction = match.group("direction")
    return (
        limit,
        direction.casefold() not in {"后", "最低", "bottom"},
        (_TextSpan(match.start(), match.end()),),
    )


def _parse_order_direction_clause(question: str) -> tuple[bool, bool, tuple[_TextSpan, ...]]:
    match = _ORDER_DIRECTION_RE.search(question)
    if match is None:
        return False, True, ()
    direction = match.group("direction")
    descending = direction in {"降序", "从高到低", "由大到小"}
    return True, descending, (_TextSpan(match.start(), match.end()),)


def _extract_percentile(question: str) -> float | None:
    match = _PERCENTILE_RE.search(question)
    if match is None:
        return None
    raw = match.group("p_short") or match.group("p_long")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if 0 < value <= 100 else None


def _parse_row_lookup_intent(
    question: str,
    catalog: StructuredCatalog,
) -> StructuredRowLookupIntent | StructuredClarification | StructuredUnavailable | None:
    if not any(marker in question for marker in _ROW_LOOKUP_MARKERS):
        return None
    if any(term in question for _, words in _AGGREGATE_WORDS for term in words) or any(
        term in question for term in _SUMMARY_WORDS
    ):
        return None
    dataset_result = _parse_dataset_clause(question, catalog)
    if dataset_result.issue is not None:
        return None
    assert dataset_result.value is not None
    dataset = dataset_result.value
    filter_result = _parse_filter_clause(
        question,
        dataset.schema.columns,
        dataset_result.consumed_spans,
    )
    if filter_result.issue is not None:
        return _with_route_context(
            filter_result.issue,
            dataset,
            origin_route="excel_row_lookup",
        )
    marker_positions = [
        question.find(marker) for marker in _ROW_LOOKUP_MARKERS if question.find(marker) >= 0
    ]
    if not marker_positions:
        return None
    marker_start = min(marker_positions)
    selected_text = question[marker_start:]
    selected = _resolve_multiple_columns(selected_text, dataset.schema.columns)
    if isinstance(selected, StructuredClarification):
        return _with_clarification_context(
            selected,
            dataset,
            origin_route="excel_row_lookup",
        )
    selected = tuple(
        column
        for column in selected
        if column.physical_name not in {item.physical_name for item in filter_result.value or ()}
    )
    if not selected:
        return _with_route_context(
            StructuredUnavailable(
                "未识别到要返回的列",
                dataset.schema.dataset_id,
                (),
                (dataset.schema.source_id,),
                "excel_row_lookup",
            ),
            dataset,
            origin_route="excel_row_lookup",
        )
    if len(selected) > MAX_STRUCTURED_ROUTE_FIELDS:
        return _with_clarification_context(
            StructuredClarification(
                f"一次最多返回 {MAX_STRUCTURED_ROUTE_FIELDS} 个列",
                tuple(column.display_name for column in selected[:MAX_STRUCTURED_ROUTE_FIELDS]),
            ),
            dataset,
            origin_route="excel_row_lookup",
        )
    return StructuredRowLookupIntent(
        dataset_id=dataset.schema.dataset_id,
        filters=filter_result.value or (),
        selected_physical_names=tuple(column.physical_name for column in selected),
        limit=100,
    )


def _parse_aggregate_clause(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
    excluded_spans: tuple[_TextSpan, ...],
    *,
    allow_missing: bool = False,
) -> _ClauseParseResult[str]:
    available = _mask_spans(question, excluded_spans)
    field_spans = _column_name_spans(available, columns)
    matches: dict[str, list[_TextSpan]] = {}
    count_all_hint = False
    for aggregate, words in _AGGREGATE_WORDS:
        for word in words:
            for span in _find_normalized_spans(available, word):
                if any(_contains(field_span, span) for field_span in field_spans):
                    continue
                matches.setdefault(aggregate, []).append(span)
                if aggregate == "count" and word == "多少条":
                    count_all_hint = True
    # Prefer a governed compound phrase such as ``去重数量`` over the
    # shorter ``数量`` token contained inside it.
    all_spans = [
        (aggregate, span)
        for aggregate, spans in matches.items()
        for span in spans
    ]
    for aggregate, span in all_spans:
        if any(
            other_aggregate != aggregate
            and other_span.start <= span.start
            and span.end <= other_span.end
            and (other_span.end - other_span.start) > (span.end - span.start)
            for other_aggregate, other_span in all_spans
        ):
            matches[aggregate].remove(span)
    matches = {aggregate: spans for aggregate, spans in matches.items() if spans}
    if len(matches) > 1:
        return _ClauseParseResult(
            issue=StructuredClarification(
                "问题包含多个不同的聚合意图，请选择一个",
                tuple(sorted(matches)),
            )
        )
    if not matches:
        if allow_missing:
            return _ClauseParseResult()
        return _ClauseParseResult(issue=StructuredUnavailable("未识别到受支持的聚合意图"))
    aggregate = next(iter(matches))
    return _ClauseParseResult(
        value=aggregate,
        consumed_spans=_merge_spans(matches[aggregate]),
        count_all_hint=count_all_hint,
    )


def _with_route_context(
    issue: StructuredClarification | StructuredUnavailable,
    dataset: StructuredDatasetCatalog,
    *,
    origin_route: str,
    target_fields: tuple[str, ...] = (),
) -> StructuredClarification | StructuredUnavailable:
    candidates = issue.candidates if isinstance(issue, StructuredClarification) else ()
    fields = tuple(
        next(
            (
                column.display_name
                for column in dataset.schema.columns
                if column.physical_name == candidate
            ),
            candidate,
        )
        for candidate in candidates
    )
    fields = target_fields or fields
    if isinstance(issue, StructuredClarification):
        return StructuredClarification(
            issue.message,
            issue.candidates,
            dataset.schema.dataset_id,
            fields,
            (dataset.schema.source_id,),
            origin_route,
        )
    return StructuredUnavailable(
        issue.message, dataset.schema.dataset_id, fields, (dataset.schema.source_id,), origin_route
    )


_with_clarification_context = _with_route_context


def _with_dataset_issue_context(
    question: str,
    catalog: StructuredCatalog,
    issue: StructuredClarification | StructuredUnavailable,
) -> StructuredClarification | StructuredUnavailable:
    candidates = issue.candidates if isinstance(issue, StructuredClarification) else ()
    datasets = tuple(
        dataset for dataset in catalog.datasets if dataset.schema.dataset_id in candidates
    )
    source_ids = tuple(
        dataset.schema.source_id for dataset in datasets[:MAX_STRUCTURED_ROUTE_FIELDS]
    )
    is_multi = any(word in _normalize(question) for word in _SUMMARY_WORDS) or "、" in question
    origin_route = "excel_multi_aggregate" if is_multi else "excel_filtered_aggregate"
    if isinstance(issue, StructuredClarification):
        return StructuredClarification(
            issue.message,
            issue.candidates[:MAX_STRUCTURED_ROUTE_FIELDS],
            candidate_source_ids=source_ids,
            origin_route=origin_route,
        )
    return StructuredUnavailable(issue.message, origin_route=origin_route)


def _parse_metric_list(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
    aggregate: str,
    excluded_spans: tuple[_TextSpan, ...],
) -> _ClauseParseResult[tuple[StructuredColumnSchema, ...]]:
    available = _mask_spans(question, excluded_spans)
    raw_matches: list[tuple[_TextSpan, int, StructuredColumnSchema]] = []
    for column in columns:
        for _, name in _resolution_names(column):
            normalized_name = _normalize(name)
            if not normalized_name:
                continue
            for span in _find_normalized_spans(available, name):
                raw_matches.append((span, len(normalized_name), column))

    by_span: dict[_TextSpan, list[tuple[int, StructuredColumnSchema]]] = {}
    for span, match_length, column in raw_matches:
        by_span.setdefault(span, []).append((match_length, column))

    span_groups: list[tuple[_TextSpan, int, dict[str, StructuredColumnSchema]]] = []
    for span, span_matches in by_span.items():
        finalists = {column.physical_name: column for _, column in span_matches}
        span_groups.append(
            (
                span,
                max(match_length for match_length, _ in span_matches),
                finalists,
            )
        )

    non_overlapping: list[tuple[_TextSpan, int, dict[str, StructuredColumnSchema]]] = []
    for match in sorted(
        span_groups,
        key=lambda item: (-item[1], item[0].start, item[0].end),
    ):
        span = match[0]
        if any(
            span.start < selected_span.end and selected_span.start < span.end
            for selected_span, _, _ in non_overlapping
        ):
            continue
        non_overlapping.append(match)

    selected: list[StructuredColumnSchema] = []
    selected_names: set[str] = set()
    selected_spans: list[_TextSpan] = []
    for span, _, finalists in sorted(non_overlapping, key=lambda item: item[0]):
        if len(finalists) > 1:
            return _ClauseParseResult(
                issue=StructuredClarification(
                    "字段名称存在歧义，请选择一个",
                    tuple(sorted(finalists)),
                )
            )
        column = next(iter(finalists.values()))
        if column.physical_name not in selected_names:
            selected.append(column)
            selected_names.add(column.physical_name)
        selected_spans.append(span)

    disallowed = tuple(
        column
        for column in selected
        if aggregate not in {"count", "count_distinct"} and not column.allow_aggregate
    )
    if disallowed:
        return _ClauseParseResult(
            issue=StructuredUnavailable(
                "指标列未授权用于聚合: " + "、".join(column.display_name for column in disallowed)
            )
        )

    return _ClauseParseResult(
        value=tuple(selected),
        consumed_spans=tuple(selected_spans),
    )


def _parse_dataset_clause(
    question: str, catalog: StructuredCatalog
) -> _ClauseParseResult[StructuredDatasetCatalog]:
    column_spans = {
        span
        for dataset in catalog.datasets
        for span in _column_name_spans(question, dataset.schema.columns)
    }
    implicit_filter_field_starts = {
        span.start
        for dataset in catalog.datasets
        for column in dataset.schema.columns
        if column.allow_filter and column.data_type is StructuredColumnType.STRING
        for span in _column_name_spans(question, (column,))
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
                if priority == 2 and span.end in implicit_filter_field_starts:
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
    shared_columns: tuple[StructuredColumnSchema, ...],
    count_all_hint: bool,
    excluded_spans: tuple[_TextSpan, ...],
) -> _ClauseParseResult[StructuredColumnSchema | None]:
    available = _mask_spans(question, excluded_spans)
    aggregate_columns = (
        columns
        if aggregate in {"count", "count_distinct"}
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
    if aggregate in {"count", "count_distinct"}:
        if count_all_hint:
            return _ClauseParseResult(value=None)
        reusable = tuple(dict.fromkeys(shared_columns))
        if len(reusable) == 1:
            return _ClauseParseResult(value=reusable[0])
        if len(reusable) > 1:
            return _ClauseParseResult(
                issue=StructuredClarification(
                    "多个筛选字段都可作为计数指标，请明确指标字段",
                    tuple(sorted(column.physical_name for column in reusable)),
                )
            )
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
    shared_columns: list[StructuredColumnSchema] = []

    natural_datetime_ranges = tuple(_NATURAL_DATETIME_RANGE_RE.finditer(available))
    if len(natural_datetime_ranges) > 1:
        return _ClauseParseResult(issue=StructuredUnavailable("结构化筛选暂不支持多个日期范围"))
    if natural_datetime_ranges:
        datetime_columns = tuple(
            column
            for column in filter_columns
            if column.data_type is StructuredColumnType.DATETIME
        )
        date_columns = tuple(
            column for column in filter_columns if column.data_type is StructuredColumnType.DATE
        )
        candidate_columns = datetime_columns or date_columns
        if len(candidate_columns) > 1:
            return _ClauseParseResult(
                issue=StructuredClarification(
                    "日期范围匹配多个日期字段，请选择一个字段",
                    tuple(sorted(column.display_name for column in candidate_columns)),
                )
            )
        if not candidate_columns:
            return _ClauseParseResult(
                issue=StructuredUnavailable("问题包含日期范围，但数据集没有可筛选的日期字段")
            )
        range_match = natural_datetime_ranges[0]
        try:
            start_value = datetime(
                int(range_match.group("year")),
                int(range_match.group("month")),
                int(range_match.group("day")),
                int(range_match.group("start_hour")),
                int(range_match.group("start_minute")),
            )
            end_value = datetime(
                int(range_match.group("end_year") or range_match.group("year")),
                int(range_match.group("end_month") or range_match.group("month")),
                int(range_match.group("end_day") or range_match.group("day")),
                int(range_match.group("end_hour")),
                int(range_match.group("end_minute")),
            )
        except ValueError:
            return _ClauseParseResult(issue=StructuredUnavailable("日期时间范围格式无效"))
        if end_value < start_value:
            return _ClauseParseResult(issue=StructuredUnavailable("日期时间范围的结束时间早于开始时间"))
        date_column = candidate_columns[0]
        if date_column.data_type is StructuredColumnType.DATETIME:
            start_text = start_value.isoformat(timespec="seconds")
            end_text = end_value.isoformat(timespec="seconds")
        else:
            start_text = start_value.date().isoformat()
            end_text = end_value.date().isoformat()
        range_span = _TextSpan(range_match.start(), range_match.end())
        all_matches.append(
            _FilterMatch(
                StructuredFilter(date_column.physical_name, "between", start_text, end_text),
                range_span,
                (range_span,),
            )
        )
        consumed.append(range_span)

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
            shared_columns.append(date_column)
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

    natural = _parse_natural_date_and_value_filters(
        _mask_spans(available, consumed),
        filter_columns,
        existing_columns={match.item.physical_name for match in all_matches},
    )
    if natural.issue is not None:
        return _ClauseParseResult(issue=natural.issue)
    natural_matches = natural.value or ()
    all_matches.extend(natural_matches)
    consumed.extend(span for match in natural_matches for span in match.consumed_spans)

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
        shared_columns=tuple(dict.fromkeys(shared_columns)),
    )


def _parse_natural_date_and_value_filters(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
    *,
    existing_columns: set[str],
) -> _ClauseParseResult[tuple[_FilterMatch, ...]]:
    """Parse common user phrasing without requiring ``field=value`` syntax.

    For example, when a dataset has one filterable text column and one date
    column, ``华东在 2025-01-01 的所有销售额`` becomes region=华东 and
    date=2025-01-01. We only infer an omitted text field when it is
    unambiguous; otherwise the caller receives a clarification instead of
    silently querying the wrong column.
    """
    matches: list[_FilterMatch] = []
    date_columns = tuple(
        column
        for column in columns
        if column.physical_name not in existing_columns
        and column.data_type in {StructuredColumnType.DATE, StructuredColumnType.DATETIME}
    )
    date_match = _DATE_LITERAL_RE.search(question)
    if date_match is not None and date_columns:
        if len(date_columns) > 1:
            return _ClauseParseResult(
                issue=StructuredClarification(
                    "日期匹配多个字段，请明确日期字段",
                    tuple(sorted(column.display_name for column in date_columns)),
                )
            )
        date_column = date_columns[0]
        date_value = date_match.group(1)
        matches.append(
            _FilterMatch(
                StructuredFilter(date_column.physical_name, "between", date_value, date_value),
                _TextSpan(date_match.start(), date_match.end()),
                (_TextSpan(date_match.start(), date_match.end()),),
            )
        )

    string_columns = tuple(
        column
        for column in columns
        if column.physical_name not in existing_columns
        and column.data_type is StructuredColumnType.STRING
    )
    value_match = re.search(
        r"(?P<value>[A-Za-z0-9\u4e00-\u9fff]{1,20})\s*(?:在|于)\s*"
        r"\d{4}-\d{2}-\d{2}\s*(?:中|内|当天|这一天)?",
        question,
    )
    if value_match is not None and string_columns:
        if len(string_columns) > 1:
            return _ClauseParseResult(
                issue=StructuredClarification(
                    "自然语言中的筛选值匹配多个字段，请明确筛选字段",
                    tuple(sorted(column.display_name for column in string_columns)),
                )
            )
        value = value_match.group("value")
        matches.append(
            _FilterMatch(
                StructuredFilter(string_columns[0].physical_name, "eq", value),
                _TextSpan(value_match.start("value"), value_match.end("value")),
                (_TextSpan(value_match.start(), value_match.end()),),
            )
        )
    return _ClauseParseResult(value=tuple(matches))


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


def _candidate_equality_value_spans(
    question: str,
    columns: tuple[StructuredColumnSchema, ...],
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for operator in re.finditer(r"为|=", question):
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
        if not isinstance(resolved, tuple):
            continue
        value_match = re.match(
            r"\s*(?P<value>[^\s，,。的且或；;]+)",
            question[operator.end() :],
        )
        if value_match is None:
            continue
        spans.append(
            (
                operator.end() + value_match.start("value"),
                operator.end() + value_match.end("value"),
            )
        )
    return tuple(spans)


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


def _resolve_multiple_columns(
    question: str,
    columns: Iterable[StructuredColumnSchema],
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
                    (priority, len(normalized_name), start, start + len(normalized_name), column)
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
    selected: list[tuple[int, StructuredColumnSchema]] = []
    for span, span_matches in by_span.items():
        best_priority = min(priority for priority, _ in span_matches)
        finalists = tuple(
            {
                column.physical_name: column
                for priority, column in span_matches
                if priority == best_priority
            }.values()
        )
        if len(finalists) > 1:
            return StructuredClarification(
                "字段名称存在歧义，请选择一个字段",
                tuple(sorted(column.physical_name for column in finalists)),
            )
        selected.append((span[0], finalists[0]))
    by_name: dict[str, tuple[int, StructuredColumnSchema]] = {}
    for start, column in selected:
        by_name.setdefault(column.physical_name, (start, column))
    return tuple(column for _, column in sorted(by_name.values(), key=lambda item: item[0]))


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
    names = [
        (0, column.physical_name),
        (1, column.display_name),
        (1, column.original_name),
        *((2, alias) for alias in column.aliases),
    ]
    for priority, name in tuple(names):
        simplified = _strip_parenthetical_suffix(name)
        # Unit-stripped aliases are primarily for metric phrases such as
        # “平均温度”; generic filter/date columns like “时间（时区）” must not
        # match incidental words such as “这段时间”.
        if simplified != name and column.allow_aggregate:
            names.append((priority + 1, simplified))
    return tuple(dict.fromkeys(names))


def _strip_parenthetical_suffix(value: str) -> str:
    simplified = re.sub(r"\s*[（(][^（）()]{1,40}[）)]\s*$", "", value).strip()
    return simplified or value


def _field_pattern(name: str, *, normalized: bool) -> str:
    if not normalized:
        return re.escape(name)
    normalized_name = _normalize(name)
    return r"[\s_-]*".join(re.escape(character) for character in normalized_name)


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _format_row_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    return str(value)


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


def _convert_parameter(
    value: str,
    column_type: StructuredColumnType,
    compatibility: ClickHouseCompatibilityProfile,
) -> object:
    try:
        if column_type is StructuredColumnType.INTEGER:
            return int(value)
        if column_type is StructuredColumnType.DECIMAL:
            return Decimal(value)
        if column_type is StructuredColumnType.DATE:
            return date.fromisoformat(value)
        if column_type is StructuredColumnType.DATETIME:
            # Structured DateTime columns are stored with an explicit UTC
            # timezone.  Pass an aware UTC value to clickhouse-connect so a
            # host-local timezone cannot shift the filter by several hours.
            normalized = compatibility.normalize_datetime(datetime.fromisoformat(value))
            return normalized.replace(tzinfo=timezone.utc)
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


def _between_upper_bound(
    column_type: StructuredColumnType,
    upper_text: str,
    upper_value: object,
) -> tuple[object, str]:
    """Return the normalized upper bound and its comparison operator.

    Date-only ranges represent calendar days, so a DateTime column is expanded
    to the beginning of the following day and compared exclusively.  A
    DateTime range that contains an explicit time is also left-closed/right-
    open.  This prevents a query such as ``00:00 到 01:00`` over minute data
    from counting the 01:00 row as part of the first hour.
    """
    if column_type is StructuredColumnType.DATETIME:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", upper_text):
            if not isinstance(upper_value, datetime):
                raise UnsafeStructuredQueryError("datetime upper bound is invalid")
            return upper_value + timedelta(days=1), "<"
        return upper_value, "<"
    return upper_value, "<="


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
        function_name = function.sql_name().upper()
        if isinstance(function, exp.Anonymous) or function.sql_name() in {
            "ANONYMOUS_AGG_FUNC",
            "PARAMETERIZED_AGG",
        }:
            function_name = getattr(function, "name", "").upper()
        if function_name not in _ALLOWED_SQL_FUNCTIONS:
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
        if len(rows) == 1 and isinstance(rows[0], Mapping):
            return rows[0]
        raise ValueError("aggregate query must return exactly one named row")
    column_names = getattr(result, "column_names", None)
    result_rows = getattr(result, "result_rows", None)
    if column_names is not None and result_rows is not None:
        rows = list(result_rows)
        if len(rows) != 1:
            raise ValueError("aggregate query must return exactly one result row")
        return dict(zip(column_names, rows[0], strict=True))
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        rows = list(result)
        if len(rows) != 1:
            raise ValueError("aggregate query must return exactly one result row")
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


def _strict_integer(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean is not an integer result")
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError("non-integral decimal result")
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("non-integral float result")
        return int(value)
    if isinstance(value, str):
        if re.fullmatch(r"[+-]?\d+", value) is None:
            raise ValueError("non-integral string result")
        return int(value)
    raise TypeError("unsupported integer result type")


def _multi_aggregate_row(
    result: object,
    metric_count: int,
) -> Mapping[str, object]:
    rows = _multi_aggregate_rows(result, metric_count, 0)
    if len(rows) != 1:
        raise ValueError("multi-aggregate query must return exactly one row")
    return rows[0]


def _multi_aggregate_rows(
    result: object,
    metric_count: int,
    group_count: int,
) -> tuple[Mapping[str, object], ...]:
    if isinstance(result, Mapping):
        return (result,)
    named_results = getattr(result, "named_results", None)
    if named_results is not None:
        rows = list(named_results())
        if not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("multi-aggregate query returned an invalid named row")
        return tuple(rows)
    column_names = getattr(result, "column_names", None)
    result_rows = getattr(result, "result_rows", None)
    if column_names is not None and result_rows is not None:
        rows = list(result_rows)
        return tuple(dict(zip(column_names, row, strict=True)) for row in rows)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        rows = list(result)
        if all(isinstance(row, Mapping) for row in rows):
            return tuple(rows)
        if all(
            isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray))
            for row in rows
        ):
            aliases = _multi_aggregate_aliases(metric_count, group_count)
            return tuple(dict(zip(aliases, row, strict=True)) for row in rows)
    raise TypeError("unsupported ClickHouse multi-aggregate result shape")


def _multi_aggregate_aliases(metric_count: int, group_count: int = 0) -> tuple[str, ...]:
    aliases = [*(f"group_{index}" for index in range(group_count)), "total_count"]
    for index in range(metric_count):
        aliases.extend(
            (
                f"metric_{index}_value",
                f"metric_{index}_valid_count",
                f"metric_{index}_null_count",
            )
        )
    return tuple(aliases)


def _require_exact_result_aliases(
    row: Mapping[str, object],
    expected_aliases: tuple[str, ...],
) -> None:
    actual_aliases = tuple(row)
    if len(actual_aliases) != len(expected_aliases) or set(actual_aliases) != set(expected_aliases):
        raise ValueError("aggregate result aliases do not match the generated projection")


def _aggregate_value(
    value: object,
    aggregate: str,
    *,
    valid_count: int,
):
    if aggregate in {"count", "count_distinct"}:
        return int(value)
    if value is None:
        if valid_count == 0:
            return None
        raise ValueError("non-empty aggregate result cannot be null")
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric aggregate result")
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        numeric = value
    elif isinstance(value, (float, str)):
        numeric = Decimal(str(value))
    else:
        raise TypeError("unsupported aggregate result type")
    if not numeric.is_finite():
        raise ValueError("aggregate result must be finite")
    return numeric
