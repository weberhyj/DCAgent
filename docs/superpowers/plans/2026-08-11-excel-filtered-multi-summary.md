# Excel Filtered Multi-Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the published Excel/CSV structured-query path so a question can filter by one or more confirmed columns and return one or many exact aggregates, including the default Chinese “汇总/统计” behavior, without loading source rows into Python or falling back to document RAG.

**Architecture:** Keep the existing single-metric contracts and execution path compatible, and add separate multi-metric intent, plan, and result contracts. The rule-based parser resolves datasets, filters, aggregate wording, and metric columns only from the active structured catalog; the planner emits one parameterized ClickHouse `SELECT` containing all requested metrics; the answer service renders immutable numeric results and a table artifact. Parsing, publication, or ClickHouse failures terminate inside the structured path and never invoke the Agent or LLM.

**Tech Stack:** Python 3.12, dataclasses, FastAPI service wiring, SQLGlot, ClickHouse 18.16.1 and 26.7 compatibility profiles, Python `unittest`, uv.

## Global Constraints

- Existing published Excel/CSV data must remain queryable without re-uploading or republishing it.
- `StructuredIntent`, `StructuredQueryPlan`, `StructuredAggregateResult`, `StructuredQueryPlanner.plan()`, and `StructuredQueryExecutor.execute()` remain valid for existing single-metric callers.
- “汇总” and “统计” default to `sum` over every integer/decimal column with `allow_aggregate=True`, and also return the matched row count.
- Explicit “平均/平均值/均值”, “最大/最高”, “最小/最低”, “总和/合计/求和”, and “数量/计数/多少条” select the named aggregate.
- Implicit summary is limited by `STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS=12`; exceeding the limit returns clarification candidates and executes no SQL.
- Dataset, worksheet, filter columns, and metric columns resolve only from the active published catalog. Filter columns require `allow_filter=True`; non-count metrics require `allow_aggregate=True`.
- The server generates exactly one parameterized, join-free, subquery-free ClickHouse `SELECT`. The user and LLM never supply SQL.
- All aggregate values, matched counts, valid counts, and null counts come from ClickHouse and must not be recalculated or rewritten by an LLM.
- ClickHouse timeout, unavailable publication, ambiguous fields, or invalid structured results must not fall back to Qdrant, Word chunks, `inspect_document`, or the LLM.
- Generated SQL must use functions supported by ClickHouse 18.16.1; do not use `toDecimalString`, window functions, CTEs, JSON functions, or modern-only aggregate combinators.

---

## File map

- `backend/app/structured_models.py`: add multi-metric intent, plan, metric-result, and aggregate-result contracts while retaining single-metric contracts.
- `backend/app/structured_query.py`: recognize “汇总/统计”, resolve ordered metric lists, enforce the implicit cap, build one multi-metric SQL statement, and decode its result.
- `backend/app/structured_answer.py`: classify summary questions, execute the single or multi path, and render deterministic paragraph/table answers without an LLM.
- `backend/app/models.py`: reuse the existing `TableArtifactModel`; no new user-facing artifact type is required.
- `backend/app/main.py`: pass the configured implicit-summary limit into `StructuredAnswerService`.
- `backend/app/offline_settings.py`: parse and validate `STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS`.
- `.env.example`, `backend/.env.example`, `deploy/offline/.env.example`: document the default limit.
- `backend/tests/support/structured_fakes.py`: provide a catalog with sales, cost, profit, region, and more-than-limit metric variants.
- `backend/tests/test_structured_query.py`: parser, cap, SQL safety, result validation, and backward-compatibility coverage.
- `backend/tests/test_structured_answer.py`: routing, no-fallback, immutable answer, artifact, and LLM-isolation coverage.
- `backend/tests/test_offline_settings.py`: configuration default and bounds.
- `backend/tests/integration/test_structured_query_clickhouse.py`: real ClickHouse multi-projection result coverage.
- `backend/tests/integration/test_clickhouse_legacy_18_16.py`: legacy-server SQL compatibility coverage.
- `backend/tests/integration/test_structured_aggregation_e2e.py`: filtered single/multi summary and 100,000-row acceptance.

### Task 1: Add compatible multi-metric contracts and configuration

**Files:**
- Modify: `backend/app/structured_models.py`
- Modify: `backend/app/offline_settings.py`
- Modify: `backend/app/main.py`
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `deploy/offline/.env.example`
- Modify: `backend/tests/test_structured_query.py`
- Modify: `backend/tests/test_offline_settings.py`

**Interfaces:**
- Consumes: existing `StructuredFilter`, `StructuredIntent`, `StructuredQueryPlan`, `StructuredAggregateResult`, and `OfflineSettings.from_environ()`.
- Produces: `StructuredMetricIntent`, `StructuredMultiAggregateIntent`, `StructuredMultiAggregatePlan`, `StructuredMetricResult`, `StructuredMultiAggregateResult`, and `OfflineSettings.structured_implicit_summary_max_metrics: int`.

- [ ] **Step 1: Write failing contract and configuration tests**

Add these tests with the existing imports and `unittest.TestCase` style:

```python
def test_multi_metric_contracts_preserve_metric_order(self) -> None:
    intent = StructuredMultiAggregateIntent(
        dataset_id="ds-sales",
        metrics=(
            StructuredMetricIntent("sum", "sales_amount"),
            StructuredMetricIntent("sum", "cost_amount"),
        ),
        filters=(StructuredFilter("region", "eq", "华东"),),
        implicit=False,
    )
    self.assertEqual(
        [(item.aggregate, item.metric_physical_name) for item in intent.metrics],
        [("sum", "sales_amount"), ("sum", "cost_amount")],
    )

def test_implicit_summary_limit_defaults_to_twelve(self) -> None:
    settings = OfflineSettings.from_environ({})
    self.assertEqual(settings.structured_implicit_summary_max_metrics, 12)

def test_implicit_summary_limit_is_bounded(self) -> None:
    for value in ("0", "51", "not-an-integer"):
        with self.subTest(value=value), self.assertRaises(OfflineSettingsError):
            OfflineSettings.from_environ(
                {"STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS": value}
            )
    settings = OfflineSettings.from_environ(
        {"STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS": "20"}
    )
    self.assertEqual(settings.structured_implicit_summary_max_metrics, 20)
```

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_query tests.test_offline_settings -v
Pop-Location
```

Expected: FAIL because the multi-metric dataclasses and setting do not exist.

- [ ] **Step 3: Add the exact contracts and setting**

Use one aggregate alias and keep all existing dataclasses callable:

```python
StructuredAggregateName = Literal["avg", "sum", "count", "min", "max"]

@dataclass(frozen=True, slots=True)
class StructuredMetricIntent:
    aggregate: StructuredAggregateName
    metric_physical_name: str

@dataclass(frozen=True, slots=True)
class StructuredMultiAggregateIntent:
    dataset_id: str
    metrics: tuple[StructuredMetricIntent, ...]
    filters: tuple[StructuredFilter, ...]
    implicit: bool

@dataclass(frozen=True, slots=True)
class StructuredMultiAggregatePlan:
    publication_id: str
    dataset_id: str
    metrics: tuple[StructuredMetricIntent, ...]
    sql: str
    parameters: Mapping[str, object]
    filters: tuple[StructuredFilter, ...]
    implicit: bool

@dataclass(frozen=True, slots=True)
class StructuredMetricResult:
    aggregate: StructuredAggregateName
    metric_physical_name: str
    metric_display_name: str
    value: Decimal | int | None
    valid_count: int
    null_count: int

@dataclass(frozen=True, slots=True)
class StructuredMultiAggregateResult:
    dataset_id: str
    source_id: str
    schema_version: int
    metrics: tuple[StructuredMetricResult, ...]
    total_count: int
    source_name: str
    worksheet_name: str
    publication_id: str
    filters: tuple[StructuredFilter, ...]
    elapsed_ms: float
    audit_id: str
```

Also add `source_id: str` to `StructuredAggregateResult` and populate it from `dataset.schema.source_id`; this corrects the existing structured audit source identity. Replace repeated aggregate `Literal` declarations in existing contracts with `StructuredAggregateName` without renaming fields or changing their order other than the new `source_id` field.

Extend `OfflineSettings` and `from_environ()` exactly as follows:

```python
try:
    structured_implicit_summary_max_metrics = int(
        environ.get("STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS", "12")
    )
except ValueError as error:
    raise OfflineSettingsError(
        "STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS must be between 1 and 50"
    ) from error
if not 1 <= structured_implicit_summary_max_metrics <= 50:
    raise OfflineSettingsError(
        "STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS must be between 1 and 50"
    )
```

Construct the service in `_create_structured_answer_service()` with `StructuredAnswerService(structured_repository.get_catalog, gateway, compatibility=compatibility, implicit_summary_max_metrics=settings.structured_implicit_summary_max_metrics)`. Add `STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS=12` to all three environment examples.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_query tests.test_offline_settings tests.test_structured_answer -v
Pop-Location
git add backend/app/structured_models.py backend/app/offline_settings.py backend/app/main.py .env.example backend/.env.example deploy/offline/.env.example backend/tests/test_structured_query.py backend/tests/test_offline_settings.py
git commit -m "feat: define multi-metric structured summaries"
```

### Task 2: Parse filtered explicit and implicit summaries

**Files:**
- Modify: `backend/app/structured_query.py`
- Modify: `backend/tests/support/structured_fakes.py`
- Modify: `backend/tests/test_structured_query.py`

**Interfaces:**
- Consumes: `resolve_structured_intent(question, catalog, *, implicit_summary_max_metrics=12)` and existing dataset/filter parsing helpers.
- Produces: `StructuredIntentResolution = StructuredIntent | StructuredMultiAggregateIntent | StructuredClarification | StructuredUnavailable` and ordered metric resolution through `_parse_metric_list()`.

- [ ] **Step 1: Write failing parser tests**

```python
def test_huizong_without_metric_selects_all_governed_numeric_columns(self) -> None:
    result = resolve_structured_intent(
        "地区为华东的汇总",
        sample_multi_metric_catalog(),
        implicit_summary_max_metrics=12,
    )
    self.assertIsInstance(result, StructuredMultiAggregateIntent)
    assert isinstance(result, StructuredMultiAggregateIntent)
    self.assertTrue(result.implicit)
    self.assertEqual(
        [(item.aggregate, item.metric_physical_name) for item in result.metrics],
        [("sum", "sales_amount"), ("sum", "cost_amount"), ("sum", "profit_amount")],
    )
    self.assertEqual(result.filters, (StructuredFilter("region", "eq", "华东"),))

def test_explicit_multi_metric_summary_preserves_question_order(self) -> None:
    result = resolve_structured_intent(
        "地区为华东的利润、销售额、成本汇总",
        sample_multi_metric_catalog(),
    )
    self.assertIsInstance(result, StructuredMultiAggregateIntent)
    assert isinstance(result, StructuredMultiAggregateIntent)
    self.assertFalse(result.implicit)
    self.assertEqual(
        [item.metric_physical_name for item in result.metrics],
        ["profit_amount", "sales_amount", "cost_amount"],
    )

def test_explicit_average_applies_to_every_named_metric(self) -> None:
    result = resolve_structured_intent(
        "华东地区销售额和成本平均值",
        sample_multi_metric_catalog(),
    )
    self.assertIsInstance(result, StructuredMultiAggregateIntent)
    assert isinstance(result, StructuredMultiAggregateIntent)
    self.assertEqual([item.aggregate for item in result.metrics], ["avg", "avg"])

def test_single_metric_huizong_keeps_single_metric_contract(self) -> None:
    result = resolve_structured_intent("华东地区销售额汇总", sample_multi_metric_catalog())
    self.assertEqual(
        result,
        StructuredIntent(
            dataset_id="ds-sales",
            aggregate="sum",
            metric_physical_name="sales_amount",
            filters=(StructuredFilter("region", "eq", "华东"),),
        ),
    )

def test_implicit_summary_over_limit_clarifies_without_selecting_first_columns(self) -> None:
    result = resolve_structured_intent(
        "汇总",
        sample_multi_metric_catalog(metric_count=13),
        implicit_summary_max_metrics=12,
    )
    self.assertIsInstance(result, StructuredClarification)
    assert isinstance(result, StructuredClarification)
    self.assertEqual(len(result.candidates), 13)
    self.assertIn("最多可汇总 12 个指标", result.message)

def test_summary_does_not_include_string_or_disallowed_numeric_columns(self) -> None:
    result = resolve_structured_intent("汇总", sample_multi_metric_catalog())
    assert isinstance(result, StructuredMultiAggregateIntent)
    self.assertNotIn("region", [item.metric_physical_name for item in result.metrics])
    self.assertNotIn("internal_score", [item.metric_physical_name for item in result.metrics])
```

- [ ] **Step 2: Run parser tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_query.StructuredIntentParsingTest -v
Pop-Location
```

Expected: FAIL because “汇总/统计” and multiple metrics are not supported.

- [ ] **Step 3: Implement deterministic summary parsing**

Add summary wording separately from the existing explicit aggregate words:

```python
_SUMMARY_WORDS = ("汇总", "统计")
_NUMERIC_TYPES = frozenset(
    {StructuredColumnType.INTEGER, StructuredColumnType.DECIMAL}
)

StructuredIntentResolution = (
    StructuredIntent
    | StructuredMultiAggregateIntent
    | StructuredClarification
    | StructuredUnavailable
)
```

Change the public resolver signature to:

```python
def resolve_structured_intent(
    question: str,
    catalog: StructuredCatalog,
    *,
    implicit_summary_max_metrics: int = 12,
) -> StructuredIntentResolution:
```

The parsing order remains dataset → filters → aggregate wording → metrics. Apply these exact rules:

```python
explicit_aggregate = _parse_aggregate_clause(
    question,
    dataset.schema.columns,
    consumed,
    allow_missing=True,
)
has_summary_word = any(
    _find_normalized_spans(_mask_spans(question, consumed), word)
    for word in _SUMMARY_WORDS
)
if explicit_aggregate.value is None and not has_summary_word:
    return StructuredUnavailable("未识别到受支持的聚合意图")
aggregate = explicit_aggregate.value or "sum"
```

Implement `_parse_metric_list(question, columns, excluded_spans)` by collecting maximal, non-overlapping matches for confirmed metric names and aliases, sorting by raw question span, deduplicating by `physical_name`, and returning `StructuredClarification` when one span maps to more than one column. For non-count aggregates, candidates are only `allow_aggregate=True`; for `count`, candidates may be any confirmed column.

If the list contains one metric, return the existing `StructuredIntent`. If it contains more than one, return `StructuredMultiAggregateIntent(dataset_id=dataset.schema.dataset_id, metrics=tuple(StructuredMetricIntent(aggregate, item.physical_name) for item in metrics), filters=filter_result.value, implicit=False)`. If it is empty and a summary word is present, select catalog columns in schema order where `allow_aggregate=True` and `data_type` is integer or decimal, create `sum` metric intents, and mark `implicit=True`. Return `StructuredUnavailable("没有可汇总的已授权数值列")` when none exist. If the implicit list exceeds the configured limit, return:

```python
StructuredClarification(
    f"可汇总指标超过上限，最多可汇总 {implicit_summary_max_metrics} 个指标，请选择",
    tuple(column.display_name for column in implicit_columns),
)
```

Explicit aggregate wording always overrides the default: “统计销售额平均值” is `avg`, not `sum`. Do not treat summary words inside a column name as an aggregate span unless there is a separate independent occurrence, matching the current aggregate-name masking behavior.

Extend `sample_multi_metric_catalog(metric_count: int = 3)` with `sales_amount`, `cost_amount`, `profit_amount`, string `region`, and numeric `internal_score` with `allow_aggregate=False`; generate additional `metric_04`… columns only when `metric_count > 3`.

- [ ] **Step 4: Run parser tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_query -v
Pop-Location
git add backend/app/structured_query.py backend/tests/support/structured_fakes.py backend/tests/test_structured_query.py
git commit -m "feat: parse filtered multi-metric summaries"
```

### Task 3: Plan and execute all metrics in one safe ClickHouse query

**Files:**
- Modify: `backend/app/structured_query.py`
- Modify: `backend/tests/test_structured_query.py`
- Modify: `backend/tests/integration/test_structured_query_clickhouse.py`
- Modify: `backend/tests/integration/test_clickhouse_legacy_18_16.py`

**Interfaces:**
- Consumes: `StructuredMultiAggregateIntent`, active `StructuredPublication`, existing filter conversion and `_validate_generated_select()`.
- Produces: `StructuredQueryPlanner.plan_multi(intent, publication) -> StructuredMultiAggregatePlan` and `StructuredQueryExecutor.execute_multi(plan) -> StructuredMultiAggregateResult | StructuredUnavailable`.

- [ ] **Step 1: Write failing SQL and executor tests**

```python
def test_multi_plan_uses_one_select_with_stable_projection_aliases(self) -> None:
    catalog = sample_multi_metric_catalog()
    intent = StructuredMultiAggregateIntent(
        dataset_id="ds-sales",
        metrics=(
            StructuredMetricIntent("sum", "sales_amount"),
            StructuredMetricIntent("sum", "cost_amount"),
            StructuredMetricIntent("sum", "profit_amount"),
        ),
        filters=(StructuredFilter("region", "eq", "华东"),),
        implicit=False,
    )
    plan = StructuredQueryPlanner(catalog).plan_multi(intent, sample_publication())
    self.assertEqual(plan.sql.count("SELECT"), 1)
    self.assertIn("sum(sales_amount) AS metric_0_value", plan.sql)
    self.assertIn("count(cost_amount) AS metric_1_valid_count", plan.sql)
    self.assertIn("count() - count(profit_amount) AS metric_2_null_count", plan.sql)
    self.assertEqual(plan.parameters, {"filter_0": "华东"})

def test_multi_executor_returns_clickhouse_values_without_python_recalculation(self) -> None:
    gateway = FakeClickHouse(
        aggregate_rows=[{
            "total_count": 2,
            "metric_0_value": "300.25",
            "metric_0_valid_count": 2,
            "metric_0_null_count": 0,
            "metric_1_value": "120.10",
            "metric_1_valid_count": 2,
            "metric_1_null_count": 0,
        }]
    )
    catalog = sample_multi_metric_catalog()
    intent = StructuredMultiAggregateIntent(
        "ds-sales",
        (
            StructuredMetricIntent("sum", "sales_amount"),
            StructuredMetricIntent("sum", "cost_amount"),
        ),
        (StructuredFilter("region", "eq", "华东"),),
        False,
    )
    plan = StructuredQueryPlanner(catalog).plan_multi(intent, sample_publication())
    result = StructuredQueryExecutor(catalog, gateway).execute_multi(plan)
    self.assertIsInstance(result, StructuredMultiAggregateResult)
    assert isinstance(result, StructuredMultiAggregateResult)
    self.assertEqual(result.total_count, 2)
    self.assertEqual([item.value for item in result.metrics], [Decimal("300.25"), Decimal("120.10")])
    self.assertEqual(len(gateway.queries), 1)

def test_multi_executor_rejects_inconsistent_counts(self) -> None:
    gateway = FakeClickHouse(aggregate_rows=[{
        "total_count": 2,
        "metric_0_value": "10",
        "metric_0_valid_count": 2,
        "metric_0_null_count": 1,
    }])
    plan = StructuredQueryPlanner(sample_multi_metric_catalog()).plan_multi(
        StructuredMultiAggregateIntent(
            "ds-sales", (StructuredMetricIntent("sum", "sales_amount"),), (), False
        ),
        sample_publication(),
    )
    result = StructuredQueryExecutor(
        sample_multi_metric_catalog(), gateway
    ).execute_multi(plan)
    self.assertEqual(result, StructuredUnavailable("结构化查询返回了不一致的计数"))
```

Add integration assertions that the generated statement succeeds on modern ClickHouse and on the `legacy_18_16` profile without `toDecimalString` or any modern-only function.

- [ ] **Step 2: Run query tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest tests.test_structured_query -v
Pop-Location
```

Expected: FAIL because `plan_multi()` and `execute_multi()` do not exist.

- [ ] **Step 3: Implement one-query planning and decoding**

Add public method `StructuredQueryPlanner.plan_multi(intent: StructuredMultiAggregateIntent, publication: StructuredPublication) -> StructuredMultiAggregatePlan` and public method `StructuredQueryExecutor.execute_multi(plan: StructuredMultiAggregatePlan) -> StructuredMultiAggregateResult | StructuredUnavailable` without changing the existing single-metric methods.

Extend both constructors with a keyword-only cap while preserving all existing call sites:

```python
StructuredQueryPlanner(
    catalog,
    compatibility=None,
    *,
    implicit_summary_max_metrics=12,
)
StructuredQueryExecutor(
    catalog,
    clickhouse_gateway,
    *,
    compatibility=None,
    implicit_summary_max_metrics=12,
)
```

The single-metric `plan()` and `execute()` behavior is unchanged; only `plan_multi()` and `execute_multi()` enforce the implicit cap.

`plan_multi()` must reject an empty metric tuple, duplicate metric columns, unknown columns, disallowed aggregates, non-active publications, and implicit metric counts above the configured limit. Reuse the existing identifier, filter parameter, compatibility-profile, and SQLGlot validation helpers. Build projections exactly in this order:

```python
projections = ["count() AS total_count"]
for index, metric_intent in enumerate(intent.metrics):
    name = _require_identifier(metric_intent.metric_physical_name)
    aggregate = metric_intent.aggregate
    aggregate_expression = f"{aggregate}({name})"
    projections.extend(
        (
            f"{aggregate_expression} AS metric_{index}_value",
            f"count({name}) AS metric_{index}_valid_count",
            f"count() - count({name}) AS metric_{index}_null_count",
        )
    )
```

For `count`, `metric_N_value` is `count(column)` and therefore an integer. For `avg/sum/min/max`, decode with the existing `_aggregate_value()` so Decimal precision is preserved. `execute_multi()` regenerates the expected plan before execution, compares SQL and parameters exactly, makes one gateway call, requires exactly one result row, verifies every `valid_count + null_count == total_count`, and returns metrics in plan order with display names from the active schema. Populate `source_id`, source file, worksheet, schema version, publication ID, elapsed time, filters, and one audit ID.

Do not loop over metrics with separate gateway calls. Do not use Python to sum, average, minimize, maximize, or count source data.

- [ ] **Step 4: Run unit and integration tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest tests.test_structured_query -v
$env:RUN_OFFLINE_INTEGRATION = "1"
uv run --project . --group dev --group offline python -m unittest tests.integration.test_structured_query_clickhouse -v
Remove-Item Env:RUN_OFFLINE_INTEGRATION -ErrorAction SilentlyContinue
uv run --project . --group dev --group offline python -m pytest tests/integration/test_clickhouse_legacy_18_16.py -v
Pop-Location
git add backend/app/structured_query.py backend/tests/test_structured_query.py backend/tests/integration/test_structured_query_clickhouse.py backend/tests/integration/test_clickhouse_legacy_18_16.py
git commit -m "feat: execute multi-metric ClickHouse summaries"
```

The pytest command must collect the legacy module; `python -m unittest` is not a valid runner for
this pytest-style file and a `Ran 0 tests` result is a failed gate. Without a configured target, the
guard tests pass and the three live acceptance cases skip accurately. A release claim for legacy
compatibility requires rerunning the same pytest command on an explicitly opted-in private
ClickHouse 18.16.1 target with `RUN_CLICKHOUSE_18_16=1`,
`CLICKHOUSE_COMPATIBILITY_MODE=legacy_18_16`, `CLICKHOUSE_URL`, distinct ingest/query users, and both
password-file variables configured; the live acceptance cases must report zero skips.

### Task 4: Render deterministic multi-metric answers and block RAG fallback

**Files:**
- Modify: `backend/app/structured_answer.py`
- Modify: `backend/app/models.py`
- Modify: `backend/tests/test_structured_answer.py`
- Modify: `backend/tests/test_llm_provider.py`

**Interfaces:**
- Consumes: single or multi structured resolution and result contracts.
- Produces: `StructuredAnswerService.try_answer()` returning a normal `AgentRunResult` with `query_structured_data`, a deterministic paragraph, and a `TableArtifactModel` for multi-metric results.

- [ ] **Step 1: Write failing routing and answer tests**

```python
def test_filtered_multi_summary_uses_clickhouse_once_and_never_calls_llm(self) -> None:
    service, gateway, llm = self.build_multi_summary_service()
    result = service.try_answer(
        "conv-1", "地区为华东的销售额、成本、利润汇总", "deep", []
    )
    self.assertIsNotNone(result)
    assert result is not None
    self.assertEqual(len(gateway.queries), 1)
    self.assertEqual(llm.generation_calls, 0)
    self.assertEqual(result.steps[0].tool_name, "query_structured_data")
    self.assertEqual(result.steps[0].source_ids, ["kb-sales"])
    self.assertEqual(result.reply.artifacts[0].type, "table")
    self.assertEqual(
        result.reply.artifacts[0].columns,
        ["指标", "聚合", "值", "匹配行数", "有效值", "空值"],
    )

def test_clickhouse_failure_after_excel_route_never_searches_word_documents(self) -> None:
    repository, rag_search = self.build_repository_with_failing_multi_summary()
    _, _, messages = repository.send_message(
        "conv-1", "地区为华东的销售额、成本汇总", "deep"
    )
    self.assertIn("结构化查询服务不可用", messages[-1].paragraphs[0].text)
    self.assertEqual(rag_search.calls, 0)

def test_implicit_limit_clarification_executes_neither_clickhouse_nor_llm(self) -> None:
    service, gateway, llm = self.build_over_limit_service()
    result = service.try_answer("conv-1", "汇总", "quick", [])
    self.assertIsNotNone(result)
    assert result is not None
    self.assertIn("最多可汇总 12 个指标", result.reply.paragraphs[0].text)
    self.assertEqual(gateway.queries, [])
    self.assertEqual(llm.generation_calls, 0)

def test_single_metric_existing_answer_format_remains_supported(self) -> None:
    result = self.single_metric_service.try_answer(
        "conv-1", "华东地区销售额总和", "quick", []
    )
    self.assertIsNotNone(result)
    assert result is not None
    self.assertIn("aggregate=sum", result.reply.paragraphs[0].text)
    self.assertEqual(result.reply.artifacts, [])
```

- [ ] **Step 2: Run answer tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_answer tests.test_llm_provider -v
Pop-Location
```

Expected: FAIL because the service only handles `StructuredIntent` and does not emit a table artifact.

- [ ] **Step 3: Implement deterministic answer construction**

Add `implicit_summary_max_metrics: int = 12` to `StructuredAnswerService.__init__()` and pass it to `resolve_structured_intent()`. Extend `_CHINESE_AGGREGATE_TERMS` with `汇总` and `统计`, while retaining the existing concept-question exclusions.

Branch on the resolution type:

```python
if isinstance(resolution, StructuredMultiAggregateIntent):
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
        catalog, self._clickhouse_gateway, compatibility=self._compatibility
    ).execute(plan)
```

For a successful multi result, construct:

```python
paragraph = (
    f"结构化汇总结果：{result.source_name} / {result.worksheet_name}，"
    f"匹配 {result.total_count} 行，筛选条件：{_format_filters(result.filters)}。"
)
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
```

Extend `_structured_run()` with `artifacts: list[ArtifactModel] | None = None` and attach the artifact to the assistant message. Use `source_ids=[result.source_id]` for both single and multi results. A clarification, unavailable catalog, invalid plan, ClickHouse timeout, invalid result, or over-limit summary returns a non-`None` structured `AgentRunResult`, preserving the current no-fallback contract. Never call `LLMProvider.generate_reply()` in this service.

- [ ] **Step 4: Run answer regressions and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_answer tests.test_llm_provider tests.test_agent tests.test_sql_repository -v
Pop-Location
git add backend/app/structured_answer.py backend/app/models.py backend/tests/test_structured_answer.py backend/tests/test_llm_provider.py
git commit -m "feat: render exact Excel multi-summary answers"
```

### Task 5: Prove large-sheet correctness and no full-row fallback

**Files:**
- Modify: `backend/tests/integration/test_structured_aggregation_e2e.py`
- Modify: `backend/tests/test_rag_acceptance.py`
- Modify: `README.md`
- Modify: `deploy/offline/README.md`

**Interfaces:**
- Consumes: an already published structured dataset, the chat endpoint, and the existing ClickHouse integration fixture.
- Produces: acceptance evidence that filtered single/multi results match a `Decimal` baseline, 100,000+ rows remain in ClickHouse, and failures cannot enter Word RAG.

- [ ] **Step 1: Add failing end-to-end acceptance cases**

Generate a deterministic workbook with 100,001 data rows containing `地区`, `销售额`, `成本`, `利润`, and nullable numeric cells. Publish through the existing structured ingestion path, then assert:

```python
def test_filtered_multi_summary_matches_decimal_reference(self) -> None:
    expected_rows = [row for row in self.rows if row.region == "华东"]
    response = self.ask("地区为华东的销售额、成本、利润汇总")
    table = response["artifacts"][0]
    values = {row[0]: Decimal(row[2].replace(",", "")) for row in table["rows"]}
    self.assertEqual(values["销售额"], sum((row.sales for row in expected_rows), Decimal(0)))
    self.assertEqual(values["成本"], sum((row.cost for row in expected_rows), Decimal(0)))
    self.assertEqual(values["利润"], sum((row.profit for row in expected_rows), Decimal(0)))
    self.assertTrue(all(row[3] == str(len(expected_rows)) for row in table["rows"]))

def test_large_summary_does_not_load_clickhouse_rows_into_python_or_llm(self) -> None:
    response = self.ask("地区为华东的汇总")
    self.assertEqual(self.clickhouse.query_calls, 1)
    self.assertEqual(self.clickhouse.returned_row_count, 1)
    self.assertEqual(self.llm.generation_calls, 0)
    self.assertNotIn("word-policy.docx", response["paragraphs"][0]["text"])

def test_clickhouse_unavailable_has_no_word_citations(self) -> None:
    self.clickhouse.fail_with = TimeoutError("timed out")
    response = self.ask("地区为华东的销售额汇总")
    self.assertIn("结构化查询超时", response["paragraphs"][0]["text"])
    self.assertEqual(response["paragraphs"][0]["citations"], [])
    self.assertEqual(self.rag_search.calls, 0)
```

- [ ] **Step 2: Run the acceptance tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest tests.integration.test_structured_aggregation_e2e tests.test_rag_acceptance -v
Pop-Location
```

Expected: FAIL until multi-summary routing, result artifacts, and no-fallback assertions are implemented.

- [ ] **Step 3: Document the operational behavior**

Add a “Filtered Excel summaries” section to both runbooks with these exact operator facts:

```text
- Existing published Excel/CSV datasets do not need to be uploaded or published again.
- “汇总/统计” sums all allowAggregate integer/decimal columns and returns matched/valid/null counts.
- STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS defaults to 12; over-limit questions ask the user to choose fields.
- All metrics in one answer are calculated by one ClickHouse SELECT.
- ClickHouse or parsing failures return a structured error and never search Word/PDF chunks.
```

Do not add a deployment step that rebuilds Qdrant or reindexes Word documents for this Excel-only change.

- [ ] **Step 4: Run the complete Excel gate and commit**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest tests.test_structured_query tests.test_structured_answer tests.test_offline_settings tests.integration.test_structured_aggregation_e2e tests.integration.test_structured_query_clickhouse -v
uv run --project . --group dev --group offline python -m pytest tests/integration/test_clickhouse_legacy_18_16.py -v
uv run --project . --group dev ruff format --check app tests
uv run --project . --group dev ruff check app tests
Pop-Location
git diff --check
git add backend/tests/integration/test_structured_aggregation_e2e.py backend/tests/test_rag_acceptance.py README.md deploy/offline/README.md
git commit -m "test: verify filtered Excel multi-summaries"
```

Expected: all focused tests pass; pytest collects the legacy module instead of reporting `Ran 0
tests`; unconfigured modern, 18.16.1, and target-host cases are recorded as real skips rather than
live green evidence. The in-process 100,001-row fake verifies structure/resource behavior and one
returned aggregate row only. Exact ClickHouse filtering and Decimal arithmetic require the opt-in
target-host test documented in `deploy/offline/README.md`. No LLM, Qdrant, or Word inspection call
occurs for any recognized Excel query.
