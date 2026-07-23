# Excel/CSV Structured Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a governed Excel/CSV structured-query path that computes exact aggregates over published rows while keeping the existing document RAG and Physoc answer path unchanged for non-structured questions.

**Architecture:** Uploaded `.xlsx` and `.csv` sources first produce a bounded schema preview. An administrator confirms field types, aliases, and query capabilities. A worker streams confirmed rows through Parquet staging into versioned ClickHouse tables. A rule-based structured router creates a whitelisted aggregate plan; the service executes it with ClickHouse and returns deterministic facts, while ambiguous or non-structured questions never silently fall back to chunk-based arithmetic.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/Alembic, `openpyxl`/CSV streaming, Polars/PyArrow, ClickHouse, SQLGlot, Vue 3/TypeScript admin UI, Vitest, Python `unittest`, uv.

---

## File map

- `backend/app/structured_models.py`: typed schema-preview, publication, intent, plan, and result contracts.
- `backend/app/database.py` and `backend/alembic/versions/20260721_01_structured_aggregation.py`: durable dataset, column, job, and publication metadata.
- `backend/app/structured_repository.py`: PostgreSQL CRUD and state transitions for previews, confirmations, jobs, and publications.
- `backend/app/spreadsheet_schema.py`: bounded XLSX/CSV sampling, type inference, header normalization, aliases, and validation diagnostics.
- `backend/app/structured_ingestion.py`: Parquet staging and ClickHouse batch publication.
- `backend/app/clickhouse_gateway.py`: identifier-safe DDL, bounded inserts, aggregate execution, and timeout handling.
- `backend/app/structured_query.py`: aggregate intent parsing, column/dataset matching, SQLGlot validation, and deterministic response construction.
- `backend/app/repository.py` and `backend/app/sql_repository.py`: optional structured-answer service injection before the legacy Agent.
- `backend/app/ingestion.py`, `backend/app/main.py`, `backend/app/routes.py`, and `backend/app/offline_settings.py`: feature-gated preview/publication job wiring and API endpoints.
- `backend/app/schemas.py` and `backend/app/models.py`: source status and structured response contracts.
- `admin-frontend/src/types/chat.ts`, `services/api.ts`, `composables/useChatKnowledgeManagement.ts`, `views/KnowledgeSourceDetailPage.vue`: schema preview/confirmation workflow.
- `backend/tests/` and `admin-frontend/src/**/__tests__/`: unit, integration, route, and UI regression tests.
- `.env.example`, `backend/.env.example`, `deploy/offline/compose.yaml`, `deploy/offline/.env.example`, `README.md`, and `tools/structured_aggregation_benchmark.py`: feature flag, ClickHouse credentials, worker, rollout documentation, and target-host acceptance measurement.

All Python tests use `unittest`. To keep snippets readable, method definitions shown below are indented inside the named `unittest.TestCase` class in the actual test file. Reusable fixtures come only from `backend/tests/support/structured_fakes.py`; do not create duplicate per-test fake implementations.

### Task 1: Define structured metadata contracts and migration

**Files:**
- Create: `backend/app/structured_models.py`
- Modify: `backend/app/models.py`
- Modify: `backend/app/database.py`
- Create: `backend/alembic/versions/20260721_01_structured_aggregation.py`
- Create: `backend/tests/support/structured_fakes.py`
- Create: `backend/tests/test_structured_schema.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_structured_column_and_dataset_contracts_are_typed(self) -> None:
    column = StructuredColumnSchema(
        physical_name="order_amount",
        original_name="订单金额",
        display_name="订单金额",
        data_type=StructuredColumnType.DECIMAL,
        aliases=("金额",),
        allow_aggregate=True,
        allow_filter=True,
    )
    dataset = StructuredDatasetSchema(
        dataset_id="ds-sales",
        source_id="kb-sales",
        worksheet_name="明细",
        schema_version=1,
        columns=(column,),
        schema_hash="a" * 64,
    )
    assert dataset.columns[0].data_type is StructuredColumnType.DECIMAL
    assert dataset.schema_hash == "a" * 64

def test_database_creates_structured_tables(self) -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()
    tables = set(inspect(database.engine).get_table_names())
    assert {
        "structured_datasets",
        "structured_columns",
        "structured_ingestion_jobs",
        "structured_publications",
    }.issubset(tables)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_schema -v
Pop-Location
```

Expected: FAIL because the structured contracts and tables do not exist.

- [ ] **Step 3: Implement the contracts and migration**

Define these exact enum values and dataclasses:

```python
class StructuredColumnType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"

class StructuredDatasetStatus(StrEnum):
    PREVIEW = "preview"
    CONFIRMED = "confirmed"
    IMPORTING = "importing"
    PUBLISHED = "published"
    FAILED = "failed"

@dataclass(frozen=True, slots=True)
class StructuredDiagnostic:
    code: str
    message: str
    worksheet_name: str
    column_name: str | None = None
    row_number: int | None = None

@dataclass(frozen=True, slots=True)
class StructuredColumnPreview:
    physical_name: str
    original_name: str
    display_name: str
    data_type: StructuredColumnType
    aliases: tuple[str, ...]
    examples: tuple[str, ...]
    sampled_rows: int
    null_count: int

@dataclass(frozen=True, slots=True)
class StructuredDatasetPreview:
    dataset_id: str
    source_id: str
    worksheet_name: str
    columns: tuple[StructuredColumnPreview, ...]
    sampled_rows: int
    schema_hash: str

@dataclass(frozen=True, slots=True)
class SpreadsheetPreview:
    source_id: str
    datasets: tuple[StructuredDatasetPreview, ...]
    diagnostics: tuple[StructuredDiagnostic, ...]

@dataclass(frozen=True, slots=True)
class StructuredColumnSchema:
    physical_name: str
    original_name: str
    display_name: str
    data_type: StructuredColumnType
    aliases: tuple[str, ...]
    allow_aggregate: bool
    allow_filter: bool
    null_policy: str = "ignore"

@dataclass(frozen=True, slots=True)
class StructuredDatasetSchema:
    dataset_id: str
    source_id: str
    worksheet_name: str
    schema_version: int
    columns: tuple[StructuredColumnSchema, ...]
    schema_hash: str

@dataclass(frozen=True, slots=True)
class StructuredPublication:
    publication_id: str
    dataset_id: str
    schema_version: int
    physical_table_name: str
    row_count: int
    content_hash: str

@dataclass(frozen=True, slots=True)
class StructuredPublicationResult:
    publication_id: str
    physical_table_name: str
    row_count: int
    column_count: int
    null_counts: Mapping[str, int]
    content_hash: str

@dataclass(frozen=True, slots=True)
class StructuredDatasetCatalog:
    schema: StructuredDatasetSchema
    source_name: str
    active_publication: StructuredPublication | None

@dataclass(frozen=True, slots=True)
class StructuredCatalog:
    datasets: tuple[StructuredDatasetCatalog, ...]
```

Add SQLAlchemy records for datasets, columns, ingestion jobs, and publications. The ingestion-job record includes `lease_token`, `lease_expires_at`, `checkpoint_row`, `attempt`, `next_attempt_at`, and `error_message` from its first migration. The migration must use `down_revision = "20260715_00"`, keep all existing tables untouched, add unique `(source_id, worksheet_name, schema_version)` constraints, and index `status`, `source_id`, and `dataset_id`. Extend `KnowledgeStatus` and the Pydantic source status union with `待确认表结构` and `结构化导入中`.

`backend/tests/support/structured_fakes.py` exports `write_xlsx`, `write_formula_xlsx`, `write_csv`, `ConfirmedSpreadsheetFixture`, `sample_confirmed_schema`, `sample_columns`, `sample_catalog`, `sample_publication`, `RecordingParquetSink`, `FakeClickHouse`, and `RecordingLLMProvider`. `write_xlsx` creates exactly the supplied worksheets/rows with openpyxl. `write_formula_xlsx` saves a formula without a cached value so formula-cache validation is reproducible. `sample_confirmed_schema(temp_dir: Path, row_count: int = 3) -> ConfirmedSpreadsheetFixture` creates the source file and returns its `path` plus a matching `StructuredDatasetSchema`; `sample_columns()` returns that schema's columns; `sample_catalog()` and `sample_publication()` return the exact catalog/publication contracts consumed by Task 7. `RecordingParquetSink` stores only received batch sizes and output paths; `FakeClickHouse` records DDL/inserts/queries and returns configured aggregate rows; `RecordingLLMProvider` counts generation calls.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_schema -v
Pop-Location
git add backend/app/structured_models.py backend/app/models.py backend/app/database.py backend/alembic/versions/20260721_01_structured_aggregation.py backend/tests/support/structured_fakes.py backend/tests/test_structured_schema.py
git commit -m "feat: add structured aggregation metadata"
```

### Task 2: Infer XLSX/CSV schemas with bounded memory

**Files:**
- Create: `backend/app/spreadsheet_schema.py`
- Create: `backend/tests/test_spreadsheet_schema.py`
- Modify: `backend/app/text_parser.py` only to reuse encoding/header helpers without changing legacy document parsing.

- [ ] **Step 1: Write failing inference tests**

```python
def test_infers_numeric_column_and_aliases_from_xlsx(self) -> None:
    path = write_xlsx(self.temp_dir / "sales.xlsx", "明细", [
        ["订单金额", "地区", "日期"],
        ["10.5", "华东", "2026-01-01"],
        ["20", "华南", "2026-01-02"],
    ])
    preview = infer_spreadsheet_schema(path, source_id="kb-sales")
    column = preview.datasets[0].columns[0]
    assert column.data_type is StructuredColumnType.DECIMAL
    assert "订单金额" in column.aliases

def test_mixed_numeric_values_upgrade_to_string_and_report_rows(self) -> None:
    path = write_xlsx(self.temp_dir / "mixed.xlsx", "Sheet1", [["金额"], ["10"], ["未知"]])
    preview = infer_spreadsheet_schema(path, source_id="kb-mixed")
    assert preview.datasets[0].columns[0].data_type is StructuredColumnType.STRING
    assert preview.diagnostics[0].code == "mixed_type"

def test_duplicate_headers_are_stable_and_require_confirmation(self) -> None:
    path = write_csv(self.temp_dir / "duplicate.csv", [["amount", "amount"], ["10", "20"]])
    preview = infer_spreadsheet_schema(path, source_id="kb-duplicate")
    assert [column.physical_name for column in preview.datasets[0].columns] == ["amount", "amount_2"]
    assert any(item.code == "duplicate_header" for item in preview.diagnostics)

def test_formula_without_cached_value_is_reported(self) -> None:
    path = write_formula_xlsx(self.temp_dir / "formula.xlsx", header="合计", formula="=SUM(1,2)")
    preview = infer_spreadsheet_schema(path, source_id="kb-formula")
    assert any(item.code == "formula_cache_missing" for item in preview.diagnostics)

def test_empty_sheet_and_unsupported_csv_encoding_are_blocking(self) -> None:
    empty = write_xlsx(self.temp_dir / "empty.xlsx", "Sheet1", [])
    empty_preview = infer_spreadsheet_schema(empty, source_id="kb-empty")
    assert empty_preview.datasets == ()
    assert any(item.code == "empty_sheet" for item in empty_preview.diagnostics)

    unsupported = self.temp_dir / "unsupported.csv"
    unsupported.write_bytes(b"\xff\xfe\x00\x00")
    csv_preview = infer_spreadsheet_schema(unsupported, source_id="kb-encoding")
    assert csv_preview.datasets == ()
    assert any(item.code == "unsupported_encoding" for item in csv_preview.diagnostics)
```

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_spreadsheet_schema -v
Pop-Location
```

Expected: FAIL because the streaming preview service does not exist.

- [ ] **Step 3: Implement bounded sampling and normalization**

Implement `infer_spreadsheet_schema(path: Path, source_id: str) -> SpreadsheetPreview` with these rules:

- XLSX uses `openpyxl.load_workbook(read_only=True, data_only=True)` and iterates at most 10,000 non-empty rows per sheet.
- CSV uses `utf-8-sig`, `utf-8`, then `gb18030` decoding and `csv.reader` without loading the whole file.
- The first non-empty row is the header; blank headers become `column_1`, `column_2`, and so on.
- Normalize physical identifiers to lowercase ASCII `[a-z0-9_]`, prefix a digit with `col_`, and append `_2`, `_3` for duplicates.
- Infer `integer -> decimal -> string`; detect ISO date/datetime and strict booleans only when all sampled non-null values match.
- Record examples (maximum five), null count, sampled row count, duplicate-header diagnostics, and schema SHA-256.
- Inspect formulas through a second bounded read-only workbook view with `data_only=False`; record `formula_cache_missing` when the corresponding `data_only=True` value is empty, without evaluating formulas in the service.
- Return one dataset preview per non-empty XLSX worksheet and one dataset preview per non-empty CSV. Empty sheets/files, missing headers, and unsupported encodings return blocking diagnostics and no publishable dataset.

Do not retain raw workbook rows in the preview object. Store only bounded examples and diagnostics.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_spreadsheet_schema -v
Pop-Location
git add backend/app/spreadsheet_schema.py backend/app/text_parser.py backend/tests/test_spreadsheet_schema.py
git commit -m "feat: infer bounded spreadsheet schemas"
```

### Task 3: Add preview, confirmation, and status APIs

**Files:**
- Create: `backend/app/structured_repository.py`
- Modify: `backend/app/ingestion.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/offline_settings.py`
- Create: `backend/tests/test_structured_api.py`

- [ ] **Step 1: Write failing API tests**

```python
def test_table_upload_enters_schema_preview_state(self) -> None:
    response = self.client.post("/api/knowledge/uploads", files={"files": self.xlsx_upload})
    assert response.status_code == 200
    source_id = response.json()[0]["id"]
    sources = self.client.get("/api/knowledge/sources").json()
    source = next(item for item in sources if item["id"] == source_id)
    assert source["status"] == "待确认表结构"
    preview = self.client.get(f"/api/knowledge/sources/{source_id}/structured-preview")
    assert preview.status_code == 200
    assert preview.json()["datasets"][0]["columns"][0]["dataType"] == "decimal"

def test_confirm_schema_requires_explicit_aggregate_capability(self) -> None:
    response = self.client.put(
        f"/api/knowledge/sources/{self.preview.source_id}/structured-schema",
        json={"datasets": [{"datasetId": self.preview.datasets[0].dataset_id, "columns": [
            {"physicalName": "amount", "displayName": "订单金额", "dataType": "decimal",
             "aliases": ["金额"], "allowAggregate": True, "allowFilter": False,
             "nullPolicy": "ignore"}
        ]}]},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "confirmed"

def test_disabled_feature_preserves_legacy_spreadsheet_ingestion(self) -> None:
    client = self.build_client(structured_query_enabled=False)
    response = client.post("/api/knowledge/uploads", files={"files": self.xlsx_upload})
    assert response.status_code == 200
    assert response.json()[0]["status"] == "解析中"
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_api -v
Pop-Location
```

Expected: FAIL because structured preview routes and repository state transitions do not exist.

- [ ] **Step 3: Implement state transitions and routes**

Add these routes in this task:

```text
GET  /api/knowledge/sources/{source_id}/structured-preview
PUT  /api/knowledge/sources/{source_id}/structured-schema
```

Add `structured_query_enabled: bool = False` to offline settings and inject it into `KnowledgeIngestionQueue`. When enabled, `KnowledgeIngestionQueue._process()` must call `infer_spreadsheet_schema()` for `.xlsx`/`.csv`, save the bounded preview, set the source to `待确认表结构`, and skip legacy text chunks. When disabled, spreadsheet ingestion keeps the current text-parser/chunk path so the default deployment is backward compatible. Non-table files always keep the current parser path. Confirmation validates every physical column, aliases, capability flags, and readable display names for generated blank headers before setting the dataset to `confirmed`; only integer/decimal fields may enable `avg/sum/min/max`, while `count` remains available for any confirmed field or total rows. Reconfirmation creates a new schema version and never mutates an already published version. Return API aliases in camelCase matching existing admin contracts.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_api tests.test_knowledge_upload -v
Pop-Location
git add backend/app/structured_repository.py backend/app/ingestion.py backend/app/main.py backend/app/routes.py backend/app/schemas.py backend/app/offline_settings.py backend/tests/test_structured_api.py
git commit -m "feat: expose spreadsheet schema confirmation"
```

### Task 4: Add admin schema preview and confirmation UI

**Files:**
- Create: `admin-frontend/src/components/knowledge/StructuredSchemaPanel.vue`
- Create: `admin-frontend/src/components/knowledge/__tests__/StructuredSchemaPanel.spec.ts`
- Modify: `admin-frontend/src/types/chat.ts`
- Modify: `admin-frontend/src/services/api.ts`
- Modify: `admin-frontend/src/composables/useChatKnowledgeManagement.ts`
- Modify: `admin-frontend/src/views/KnowledgeSourceDetailPage.vue`
- Modify: `admin-frontend/src/views/__tests__/KnowledgeSourceDetailPage.spec.ts`
- Modify: `admin-frontend/src/composables/useChatKnowledgeManagement.spec.ts`

- [ ] **Step 1: Add failing UI and service tests**

Tests must assert that a table source loads preview rows, exposes type/alias controls, disables confirmation until the schema is valid, and calls `PUT /structured-schema` with camelCase fields. Non-table sources must continue showing the existing chunk panel.

```ts
it('shows schema confirmation controls for a spreadsheet source', async () => {
  const wrapper = mount(KnowledgeSourceDetailPage, {
    global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
  })
  await flushPromises()
  expect(wrapper.get('[data-testid="structured-schema-panel"]').exists()).toBe(true)
  expect(wrapper.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()
})
```

- [ ] **Step 2: Run admin focused tests and verify RED**

```powershell
Push-Location admin-frontend
npm.cmd exec -- vitest run src/components/knowledge/__tests__/StructuredSchemaPanel.spec.ts src/views/__tests__/KnowledgeSourceDetailPage.spec.ts src/composables/useChatKnowledgeManagement.spec.ts
Pop-Location
```

Expected: FAIL because the preview API types, composable methods, and panel do not exist.

- [ ] **Step 3: Implement the confirmation panel**

Add typed interfaces for `StructuredPreview`, `StructuredDatasetPreview`, `StructuredColumnPreview`, and `StructuredSchemaSubmission`. Add API methods `fetchStructuredPreview` and `confirmStructuredSchema`. `StructuredSchemaPanel.vue` renders one table per worksheet with editable display name, aliases, type select, aggregate/filter toggles, diagnostics, and a confirmation button. The detail page remains a composition surface and chooses between the new panel and the existing chunk preview. The button is enabled only when every dataset has a valid schema and no blocking diagnostic.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location admin-frontend
npm.cmd exec -- vitest run src/components/knowledge/__tests__/StructuredSchemaPanel.spec.ts src/views/__tests__/KnowledgeSourceDetailPage.spec.ts src/composables/useChatKnowledgeManagement.spec.ts
npm.cmd run test:run
npm.cmd run build
Pop-Location
git add admin-frontend/src/components/knowledge/StructuredSchemaPanel.vue admin-frontend/src/components/knowledge/__tests__/StructuredSchemaPanel.spec.ts admin-frontend/src/types/chat.ts admin-frontend/src/services/api.ts admin-frontend/src/composables/useChatKnowledgeManagement.ts admin-frontend/src/views/KnowledgeSourceDetailPage.vue admin-frontend/src/views/__tests__/KnowledgeSourceDetailPage.spec.ts admin-frontend/src/composables/useChatKnowledgeManagement.spec.ts
git commit -m "feat: add spreadsheet schema confirmation UI"
```

### Task 5: Stream confirmed rows to Parquet and ClickHouse

**Files:**
- Create: `backend/app/structured_ingestion.py`
- Create: `backend/app/clickhouse_gateway.py`
- Create: `backend/tests/test_structured_ingestion.py`
- Create: `backend/tests/test_clickhouse_gateway.py`
- Modify: `backend/app/offline_settings.py`
- Modify: `backend/pyproject.toml` only if a runtime dependency is missing from the `offline` group.

- [ ] **Step 1: Write failing bounded-ingestion and gateway tests**

```python
def test_ingestion_writes_bounded_batches_and_counts_rows(self) -> None:
    confirmed_schema = sample_confirmed_schema(self.temp_dir, row_count=100_000)
    sink = RecordingParquetSink(self.temp_dir)
    result = SpreadsheetPublisher(sink=sink, clickhouse=FakeClickHouse()).publish(
        path=confirmed_schema.path,
        schema=confirmed_schema.schema,
        publication_id="pub-1",
    )
    assert result.row_count == 100_000
    assert max(sink.batch_rows) <= 50_000
    assert result.null_counts["amount"] == 3

def test_gateway_rejects_untrusted_identifiers(self) -> None:
    gateway = ClickHouseGateway(FakeClickHouse())
    with self.assertRaises(StructuredStorageError):
        gateway.create_table("sales; DROP TABLE users", sample_columns())
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest tests.test_structured_ingestion tests.test_clickhouse_gateway -v
Pop-Location
```

Expected: FAIL because the bounded publisher and ClickHouse gateway do not exist.

- [ ] **Step 3: Implement streaming publication**

`SpreadsheetPublisher.publish()` must read the confirmed schema, stream rows in batches of `STRUCTURED_INGEST_BATCH_ROWS` (default `50000`), write Parquet parts under `PARQUET_ROOT/<source>/<dataset>/<version>/`, and call the gateway with typed Arrow batches. It must never build a list containing all rows. Every row receives source, dataset, publication version, worksheet, and stable row number fields. Conversion failures include row number, physical column, and a redacted sample, stop the publication, and leave the old publication active. A confirmed aggregate column containing a formula without a cached value fails publication with `formula_cache_missing` and instructs the administrator to recalculate and save the workbook in an office application.

`ClickHouseGateway` must generate identifiers only from `[a-z0-9_]`, create an immutable staging table, insert Arrow batches, validate row/column counts, schema, content hash, and basic `count/min/max` results for numeric columns against the streaming publication statistics, then atomically rename/promote the versioned table. Use a separate read-only query client. All ClickHouse calls set `max_execution_time`, `max_memory_usage`, `max_result_rows`, and `overflow_mode=break`.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest tests.test_structured_ingestion tests.test_clickhouse_gateway -v
Pop-Location
git add backend/app/structured_ingestion.py backend/app/clickhouse_gateway.py backend/app/offline_settings.py backend/pyproject.toml backend/tests/test_structured_ingestion.py backend/tests/test_clickhouse_gateway.py
git commit -m "feat: publish confirmed spreadsheets to ClickHouse"
```

### Task 6: Add durable publication jobs and status polling

**Files:**
- Modify: `backend/app/database.py`
- Create: `backend/app/structured_worker.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes.py`
- Create: `backend/tests/test_structured_worker.py`
- Modify: `admin-frontend/src/types/chat.ts`
- Modify: `admin-frontend/src/services/api.ts`
- Modify: `admin-frontend/src/composables/useChatKnowledgeManagement.ts`
- Modify: `admin-frontend/src/components/knowledge/StructuredSchemaPanel.vue`
- Modify: `admin-frontend/src/components/knowledge/__tests__/StructuredSchemaPanel.spec.ts`

- [ ] **Step 1: Write failing job state tests**

Cover queued → running → published, retry after failure, stale lease rejection, and old-publication preservation:

```python
def test_only_one_worker_claims_a_publication_job(self) -> None:
    job = self.repository.enqueue_publication("kb-sales", "ds-sales", "pub-1")
    first = self.repository.claim_publication("worker-1", lease_seconds=60)
    second = self.repository.claim_publication("worker-2", lease_seconds=60)
    assert first.id == job.id
    assert second is None

def test_failed_new_version_keeps_old_publication(self) -> None:
    self.repository.set_active_publication("ds-sales", "pub-old")
    self.repository.enqueue_publication("kb-sales", "ds-sales", "pub-new")
    claimed = self.repository.claim_publication("worker-1", 60)
    self.repository.fail_publication(claimed.id, claimed.lease_token, "type_conversion_failed")
    assert self.repository.get_active_publication("ds-sales").publication_id == "pub-old"
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_worker -v
Pop-Location
```

Expected: FAIL because durable structured jobs and the worker loop do not exist.

- [ ] **Step 3: Implement bounded worker execution**

Use PostgreSQL row locking with a lease token, `checkpoint_row`, `attempt`, `next_attempt_at`, and `error_message`. Add `POST /api/knowledge/sources/{source_id}/structured-publications` and `GET /api/knowledge/sources/{source_id}/structured-status`. The POST route only enqueues a job and returns `202` with a job ID. `StructuredIngestionWorker.run_once()` claims one job, calls `SpreadsheetPublisher`, validates the result, updates the active publication pointer, and marks the source `已索引` only after promotion. A failure leaves the prior publication queryable and sets the attempted publication to `failed`; the source uses `解析失败` only when no older active publication exists. The worker handles SIGTERM and does not claim a second job while one is active. Add admin API/composable methods and a publish button that polls status while importing.

Expose a runnable module:

```python
def main() -> None:
    worker = build_structured_worker()
    worker.run_forever()

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_worker tests.test_structured_api -v
Pop-Location
git add backend/app/database.py backend/app/structured_worker.py backend/app/main.py backend/app/routes.py backend/tests/test_structured_worker.py admin-frontend/src/types/chat.ts admin-frontend/src/services/api.ts admin-frontend/src/composables/useChatKnowledgeManagement.ts admin-frontend/src/components/knowledge/StructuredSchemaPanel.vue admin-frontend/src/components/knowledge/__tests__/StructuredSchemaPanel.spec.ts
git commit -m "feat: add durable spreadsheet publication jobs"
```

### Task 7: Implement governed aggregate intent and SQL plans

**Files:**
- Create: `backend/app/structured_query.py`
- Create: `backend/tests/test_structured_query.py`
- Create: `backend/tests/integration/test_structured_query_clickhouse.py`

- [ ] **Step 1: Write failing intent and safety tests**

```python
def test_parses_average_with_alias_and_filter(self) -> None:
    intent = parse_structured_intent("统计华东地区订单金额的平均值", sample_catalog())
    assert intent.aggregate == "avg"
    assert intent.metric_physical_name == "order_amount"
    assert intent.filters == (StructuredFilter("region", "eq", "华东"),)

def test_ambiguous_metric_never_selects_first_column(self) -> None:
    result = resolve_structured_intent("平均金额", sample_catalog(ambiguous=True))
    assert isinstance(result, StructuredClarification)
    assert len(result.candidates) == 2

def test_plan_is_select_only_and_whitelisted(self) -> None:
    plan = StructuredQueryPlanner(sample_catalog()).plan(
        StructuredIntent("ds-sales", "avg", "order_amount", ()), sample_publication()
    )
    parsed = sqlglot.parse_one(plan.sql, read="clickhouse")
    assert parsed.key == "select"
    assert plan.aggregate == "avg"

def test_parses_numeric_and_date_range_filters(self) -> None:
    intent = parse_structured_intent(
        "统计2026-01-01至2026-01-31订单金额大于100的总和",
        sample_catalog(),
    )
    assert intent.aggregate == "sum"
    assert StructuredFilter("order_amount", "gt", "100") in intent.filters
    assert StructuredFilter("order_date", "between", "2026-01-01", "2026-01-31") in intent.filters
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_query -v
Pop-Location
```

Expected: FAIL because structured intent parsing and SQL planning do not exist.

- [ ] **Step 3: Implement the constrained query layer**

Add these query contracts to `structured_models.py` and import them from `structured_query.py`:

```python
@dataclass(frozen=True, slots=True)
class StructuredFilter:
    physical_name: str
    operator: Literal["eq", "gt", "gte", "lt", "lte", "between"]
    value: str
    upper_value: str | None = None

@dataclass(frozen=True, slots=True)
class StructuredIntent:
    dataset_id: str
    aggregate: Literal["avg", "sum", "count", "min", "max"]
    metric_physical_name: str | None
    filters: tuple[StructuredFilter, ...]

@dataclass(frozen=True, slots=True)
class StructuredClarification:
    message: str
    candidates: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class StructuredUnavailable:
    message: str

@dataclass(frozen=True, slots=True)
class StructuredQueryPlan:
    publication_id: str
    dataset_id: str
    metric_physical_name: str | None
    sql: str
    parameters: Mapping[str, object]
    aggregate: Literal["avg", "sum", "count", "min", "max"]
    filters: tuple[StructuredFilter, ...]

@dataclass(frozen=True, slots=True)
class StructuredAggregateResult:
    dataset_id: str
    schema_version: int
    aggregate: Literal["avg", "sum", "count", "min", "max"]
    metric_physical_name: str | None
    metric_display_name: str | None
    value: Decimal | int | None
    total_count: int
    valid_count: int
    null_count: int
    source_name: str
    worksheet_name: str
    publication_id: str
    filters: tuple[StructuredFilter, ...]
    elapsed_ms: float
    audit_id: str
```

Support these Chinese intent words: `平均/平均值/均值`, `总和/合计/求和`, `数量/多少条/计数`, `最大/最高`, and `最小/最低`. Resolve datasets and columns by exact normalized name, display name, then longest alias; ties return `StructuredClarification` with candidates. Support equality filters (`字段为值`, `字段=值`), numeric comparisons (`大于/不少于/小于/不超过`), and ISO date ranges (`YYYY-MM-DD 至 YYYY-MM-DD`). Do not use an LLM to select a field.

`StructuredQueryPlanner` builds a single parameterized ClickHouse `SELECT` using only catalog-approved identifiers and functions. It rejects unknown columns, unconfirmed datasets, multiple datasets, arbitrary SQL, subqueries, and joins. `count` with no metric counts all rows; `count` with a confirmed field counts non-null values. `StructuredQueryExecutor` returns dataset/schema version, aggregate and metric identity, aggregate value, total/valid/null row counts, filters, source/worksheet, publication ID, elapsed milliseconds, and query audit ID. A ClickHouse timeout becomes `StructuredUnavailable`, not a chunk fallback.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest tests.test_structured_query -v
$env:RUN_OFFLINE_INTEGRATION = "1"
uv run --project . --group dev --group offline python -m unittest tests.integration.test_structured_query_clickhouse -v
Remove-Item Env:RUN_OFFLINE_INTEGRATION
Pop-Location
git add backend/app/structured_models.py backend/app/structured_query.py backend/tests/test_structured_query.py backend/tests/integration/test_structured_query_clickhouse.py
git commit -m "feat: add safe spreadsheet aggregate queries"
```

### Task 8: Route structured answers before legacy RAG

**Files:**
- Create: `backend/app/structured_answer.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_structured_answer.py`
- Modify: `backend/tests/test_llm_provider.py`

- [ ] **Step 1: Write failing routing tests**

```python
def test_average_uses_clickhouse_and_does_not_call_physoc(self) -> None:
    _, _, messages = self.repository.send_message("conv-1", "订单金额平均值", "deep")
    assert "平均值" in messages[-1].paragraphs[0].text
    assert "12,345.67" in messages[-1].paragraphs[0].text
    assert self.repository.physoc_calls == 0
    assert self.repository.structured_calls == 1

def test_non_structured_question_keeps_legacy_physoc_path(self) -> None:
    self.repository.send_message("conv-1", "合同的付款条款是什么", "deep")
    assert self.repository.physoc_calls == 1

def test_ambiguous_structured_question_returns_clarification_not_chunk_math(self) -> None:
    _, _, messages = self.repository.send_message("conv-1", "平均金额", "deep")
    assert "请选择" in messages[-1].paragraphs[0].text
    assert self.repository.physoc_calls == 0
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_answer tests.test_llm_provider -v
Pop-Location
```

Expected: FAIL because repositories always invoke the legacy Agent.

- [ ] **Step 3: Implement optional structured service injection**

Define `StructuredAnswerService.try_answer(conversation_id, content, mode, previous_messages) -> AgentRunResult | None`. A structured result creates a normal user/assistant message pair and one audit step named `query_structured_data`; it does not call `PhysocDeepSeekLLMProvider`. Format the answer deterministically from `StructuredAggregateResult`, including source file, worksheet, aggregate/metric, immutable numeric value, total/valid/null counts, filters, schema/publication version, elapsed time, and audit ID. `None` means the question is not structured and preserves the current Agent path. `StructuredClarification` is a non-`None` deterministic answer. `StructuredUnavailable` is a non-`None` explicit service-unavailable answer.

Inject the optional service into both `ChatRepository` and `SqlChatRepository`, preserving existing constructors by defaulting it to `None`. Update `create_default_repository()` and `create_production_app()` to build it only when `STRUCTURED_QUERY_ENABLED=true` and ClickHouse settings are present. Preserve existing message persistence and audit behavior exactly once.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_answer tests.test_llm_provider tests.test_sql_repository -v
Pop-Location
git add backend/app/structured_answer.py backend/app/repository.py backend/app/sql_repository.py backend/app/main.py backend/tests/test_structured_answer.py backend/tests/test_llm_provider.py
git commit -m "feat: route structured answers before RAG"
```

### Task 9: Add feature flags, Compose wiring, and operational documentation

**Files:**
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `deploy/offline/.env.example`
- Modify: `deploy/offline/compose.yaml`
- Modify: `README.md`
- Modify: `deploy/offline/README.md`
- Create: `tools/tests/test_structured_deployment_contract.py`

- [ ] **Step 1: Write failing deployment contract tests**

Assert that examples define `STRUCTURED_QUERY_ENABLED`, `CLICKHOUSE_URL`, `PARQUET_ROOT`, `STRUCTURED_QUERY_TIMEOUT_SECONDS`, and `STRUCTURED_INGEST_BATCH_ROWS`; Compose passes the feature flag and ClickHouse credentials; the old `template`/legacy path remains the default.

- [ ] **Step 2: Run tests and verify RED**

```powershell
uv run --project backend --group dev python -m unittest tools.tests.test_structured_deployment_contract -v
```

Expected: FAIL because the new flag and worker/query wiring are absent.

- [ ] **Step 3: Implement safe rollout configuration**

Use `STRUCTURED_QUERY_ENABLED=false` by default. Add `CLICKHOUSE_QUERY_USER`, `CLICKHOUSE_QUERY_PASSWORD_FILE`, `CLICKHOUSE_INGEST_USER`, `CLICKHOUSE_INGEST_PASSWORD_FILE`, `STRUCTURED_QUERY_TIMEOUT_SECONDS=4`, and `STRUCTURED_INGEST_BATCH_ROWS=50000`. Pass only the required variables to `api` and `ingestion-worker`; never place passwords directly in `.env.example`. Document that the feature requires confirmed schema and the indexing worker profile. Update the runbook with migration, worker start, smoke aggregate, rollback to `false`, and the fact that ClickHouse unavailability must not fall back to slice arithmetic.

- [ ] **Step 4: Run tests and commit**

```powershell
uv run --project backend --group dev python -m unittest tools.tests.test_structured_deployment_contract -v
git add .env.example backend/.env.example deploy/offline/.env.example deploy/offline/compose.yaml README.md deploy/offline/README.md tools/tests/test_structured_deployment_contract.py
git commit -m "docs: add structured aggregation deployment contract"
```

### Task 10: Run end-to-end acceptance and finalize

**Files:**
- Create: `backend/tests/integration/test_structured_aggregation_e2e.py`
- Create: `tools/structured_aggregation_benchmark.py`
- Create: `tools/tests/test_structured_aggregation_benchmark.py`
- Modify: `tools/tests/test_benchmark_report.py` only when the acceptance runner records structured-query timings.

- [ ] **Step 1: Add the end-to-end acceptance fixture**

Generate a deterministic XLSX fixture with 100,000 rows, numeric values, nulls, a region column, and dates. Upload it, confirm the schema, publish it, and execute `avg`, `sum`, `count`, `min`, and `max` plus equality, numeric-range, and date-range filters against a Python `Decimal` reference. Also assert that a document question still reaches the existing Physoc fake, ambiguous fields return clarification, and a ClickHouse timeout returns unavailable without invoking Physoc.

Add `tools/structured_aggregation_benchmark.py` with explicit `--rows`, `--concurrency`, `--requests`, `--p95-seconds`, and `--max-rss-growth-mb` arguments. It must generate and publish 1,000,000 rows, record process RSS before/after each bounded ingestion batch, execute a fixed aggregate mix with 15 concurrent clients, emit JSON containing row count, peak RSS growth, success count, error count, and p50/p95 latency, and exit non-zero when the configured bounds are exceeded. Unit-test percentile calculation, report schema, and non-zero threshold behavior without requiring ClickHouse.

- [ ] **Step 2: Run the local acceptance suite**

```powershell
Set-Location backend
uv sync --project . --group dev --group offline
uv run --project . --group dev --group offline python -m unittest discover -s tests -p "test_*.py" -v
uv run --project . --group dev ruff format --check app tests
uv run --project . --group dev ruff check app tests
Set-Location ..
uv run --project backend --group dev --group offline python -m unittest tools.tests.test_structured_aggregation_benchmark tools.tests.test_benchmark_report -v
Set-Location frontend
npm.cmd run test:run
npm.cmd run build
Set-Location ..
Set-Location admin-frontend
npm.cmd run test:run
npm.cmd run build
Set-Location ..
git diff --check
```

Expected: all backend/admin/frontend tests pass and both frontend builds succeed. Docker-only ClickHouse integration is a target-host gate when Docker is unavailable locally.

On the deployment target with ClickHouse available, run:

```powershell
uv run --project backend --group benchmark python tools/structured_aggregation_benchmark.py --rows 1000000 --concurrency 15 --requests 150 --p95-seconds 5 --max-rss-growth-mb 512
```

Expected: exit code `0`, `errorCount=0`, bounded RSS growth, and `p95Seconds <= 5`.

- [ ] **Step 3: Verify scope and commit the acceptance test**

```powershell
git status --short --branch
git diff --stat main...HEAD
git add backend/tests/integration/test_structured_aggregation_e2e.py tools/structured_aggregation_benchmark.py tools/tests/test_structured_aggregation_benchmark.py tools/tests/test_benchmark_report.py
git commit -m "test: verify structured spreadsheet aggregation"
```

Expected scope: only structured metadata, spreadsheet inference, API/UI confirmation, ingestion/storage, query/routing, deployment docs, and acceptance tests. The existing document RAG and Physoc configuration remain backward compatible.
