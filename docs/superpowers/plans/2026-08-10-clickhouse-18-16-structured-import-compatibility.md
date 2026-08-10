# ClickHouse 18.16 Structured Import Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make structured XLSX publication and structured queries work against the fixed Ubuntu ClickHouse 18.16.1 server while preserving modern ClickHouse behavior.

**Architecture:** Add one immutable compatibility profile selected by `CLICKHOUSE_COMPATIBILITY_MODE`, inject it into ingestion, publication validation, query planning, and API/worker gateways, and fail startup before queue claim when the selected server profile is incompatible. Keep legacy Ubuntu user provisioning separate from the modern SQL-RBAC initialization script.

**Tech Stack:** Python 3.12, FastAPI, `clickhouse-connect==1.6.0`, PyArrow, openpyxl, SQLAlchemy, unittest/pytest, Bash, Supervisor, ClickHouse HTTP SQL.

## Global Constraints

- The production database is fixed at ClickHouse `18.16.1` and cannot be upgraded.
- `CLICKHOUSE_COMPATIBILITY_MODE` accepts exactly `modern` and `legacy_18_16`; the default is `modern`.
- Legacy datetime storage and query parameters use `DateTime`; milliseconds are intentionally discarded.
- Legacy decimal canonicalization uses `toString(decimal_column)` and must match the Python scale-9 digest test.
- Legacy bounded settings are exactly `max_execution_time`, `max_memory_usage`, `max_result_rows`, and `result_overflow_mode=break`; never send `overflow_mode`.
- Worker preflight must complete before `claim_publication`; a preflight failure leaves queued jobs unclaimed.
- Every successfully inserted ClickHouse batch persists cumulative `checkpointRow`; checkpoint is diagnostic progress, not row-level resume.
- Legacy Ubuntu accounts are documented through ClickHouse users configuration; application scripts must not overwrite `/etc/clickhouse-server/users.xml`.
- Pin the Python dependency to `clickhouse-connect==1.6.0`.
- Do not change embedding, reranker, narrative RAG, or unrelated Supervisor behavior.

---

## File Map

| File | Responsibility in this plan |
| --- | --- |
| `backend/app/clickhouse_compatibility.py` | Mode enum, immutable profile, type mappings, datetime normalization, canonical expressions, settings, version validation. |
| `backend/app/offline_settings.py` | Parse and expose the selected compatibility mode. |
| `backend/app/clickhouse_gateway.py` | Consume the profile for DDL, validation SQL, query settings, and server preflight. |
| `backend/app/structured_query.py` | Use the profile for typed placeholders and datetime conversion. |
| `backend/app/structured_answer.py` | Pass one profile through plan creation and execution-time plan verification. |
| `backend/app/main.py` | Inject the profile into the lazy query gateway and run query preflight before first query. |
| `backend/app/structured_worker.py` | Inject the profile into the worker gateway and run worker preflight before queue claim. |
| `backend/app/structured_ingestion.py` | Use legacy Arrow seconds timestamps, pass the profile into conversion, and report insert-batch progress. |
| `backend/pyproject.toml`, `backend/uv.lock` | Pin `clickhouse-connect` to `1.6.0`. |
| `backend/tests/test_clickhouse_compatibility.py` | Profile and preflight contract tests. |
| `backend/tests/test_offline_settings.py` | Environment parsing and invalid-mode tests. |
| `backend/tests/test_clickhouse_gateway.py` | Legacy/modern DDL, settings, canonical SQL, and preflight tests. |
| `backend/tests/test_structured_query.py` | Legacy/modern datetime placeholder and conversion tests. |
| `backend/tests/test_structured_answer.py` | Profile propagation and lazy gateway preflight tests. |
| `backend/tests/test_structured_ingestion.py` | Seconds timestamp conversion and batch checkpoint tests. |
| `backend/tests/test_structured_worker.py` | Preflight-before-claim and checkpoint persistence tests. |
| `backend/tests/integration/test_clickhouse_legacy_18_16.py` | Opt-in acceptance tests against a real 18.16.1 server. |
| `deploy/ubuntu/clickhouse-18.16-users.xml.example` | Reviewed legacy query/ingest account XML example without secrets. |
| `deploy/ubuntu/CLICKHOUSE_18_16.md` | Ubuntu apt configuration, preflight, Supervisor environment, rollout, and rollback instructions. |
| `.env.example`, `deploy/offline/.env.example`, `deploy/offline/compose.yaml` | Explicit compatibility-mode environment wiring for local/offline deployments. |

---

### Task 1: Add the compatibility profile and environment parsing

**Files:**
- Create: `backend/app/clickhouse_compatibility.py`
- Modify: `backend/app/offline_settings.py`
- Test: `backend/tests/test_clickhouse_compatibility.py`
- Test: `backend/tests/test_offline_settings.py`

**Interfaces:**
- Produces `ClickHouseCompatibilityMode` with values `modern` and `legacy_18_16`.
- Produces `ClickHouseCompatibilityProfile` with methods:
  `storage_type(StructuredColumnType) -> str`,
  `parameter_type(StructuredColumnType) -> str`,
  `canonical_value_expression(name: str, column_type: StructuredColumnType) -> str`,
  `normalize_datetime(value: datetime) -> datetime`,
  `command_settings() -> dict[str, object]`,
  `query_settings() -> dict[str, object]`, and
  `validate_server_version(version: str) -> None`.
- `OfflineSettings.clickhouse_compatibility_mode` exposes the parsed enum.

- [ ] **Step 1: Write failing profile tests.**

```python
def test_legacy_profile_uses_second_precision_and_legacy_decimal_expression():
    profile = ClickHouseCompatibilityProfile.for_mode(
        ClickHouseCompatibilityMode.LEGACY_18_16
    )
    assert profile.storage_type(StructuredColumnType.DATETIME) == "Nullable(DateTime)"
    assert profile.parameter_type(StructuredColumnType.DATETIME) == "DateTime"
    assert profile.canonical_value_expression("amount", StructuredColumnType.DECIMAL) == "toString(amount)"
    assert profile.normalize_datetime(datetime(2026, 8, 10, 12, 30, 1, 999999)) == datetime(2026, 8, 10, 12, 30, 1)
```

- [ ] **Step 2: Add failing settings tests.**

```python
def test_settings_parse_legacy_clickhouse_mode():
    settings = OfflineSettings.from_environ({"CLICKHOUSE_COMPATIBILITY_MODE": "legacy_18_16"})
    assert settings.clickhouse_compatibility_mode is ClickHouseCompatibilityMode.LEGACY_18_16

def test_settings_reject_unknown_clickhouse_mode():
    with pytest.raises(OfflineSettingsError, match="CLICKHOUSE_COMPATIBILITY_MODE"):
        OfflineSettings.from_environ({"CLICKHOUSE_COMPATIBILITY_MODE": "18.16"})
```

- [ ] **Step 3: Run the focused tests and verify they fail.**

Run: `uv run --project backend --group dev pytest backend/tests/test_clickhouse_compatibility.py backend/tests/test_offline_settings.py -q`

Expected: FAIL because the module, enum, profile methods, and settings field do not exist.

- [ ] **Step 4: Implement the profile and parser.**

Use a `StrEnum` or `str, Enum` compatible with Python 3.12. Keep all mappings in the profile. Use
`datetime.replace(microsecond=0)` for legacy normalization and preserve the input unchanged in
modern mode. Return fresh settings dictionaries so callers cannot mutate the profile. Legacy
`validate_server_version` accepts only `18.16.x`; modern mode accepts any syntactically valid
version and leaves newer-version policy to a warning logger.

- [ ] **Step 5: Run the focused tests and verify they pass.**

Run: `uv run --project backend --group dev pytest backend/tests/test_clickhouse_compatibility.py backend/tests/test_offline_settings.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the profile boundary.**

```bash
git add backend/app/clickhouse_compatibility.py backend/app/offline_settings.py backend/tests/test_clickhouse_compatibility.py backend/tests/test_offline_settings.py
git commit -m "feat: add ClickHouse compatibility profiles"
```

### Task 2: Integrate the profile into ClickHouseGateway and preflight

**Files:**
- Modify: `backend/app/clickhouse_gateway.py`
- Modify: `backend/tests/test_clickhouse_gateway.py`

**Interfaces:**
- `ClickHouseGateway.__init__(..., compatibility: ClickHouseCompatibilityProfile | None = None)` defaults to the modern profile.
- `ClickHouseGateway.preflight() -> str` executes version and settings/expression probes and returns the server version.
- `_clickhouse_type` and `_canonical_row_expression` consume the injected profile.

- [ ] **Step 1: Write failing gateway contract tests.**

```python
def test_legacy_gateway_uses_datetime_and_never_emits_forbidden_tokens():
    ingest = RecordingIngestClient()
    query = RecordingQueryClient([[("18.16.1",)], [(1,)]])
    gateway = ClickHouseGateway(
        ingest,
        query_client=query,
        compatibility=ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16),
    )
    gateway.preflight()
    schema = sample_confirmed_schema_pathless()
    gateway.create_table("structured_sales", schema.columns)
    ddl = "\n".join(statement for statement, _settings in ingest.ddl)
    assert "Nullable(DateTime)" in ddl
    assert "DateTime64" not in ddl
    validation = _validation_query(
        ClickHousePublicationTarget(schema, "structured_sales_staging", "structured_sales", "a" * 64),
        "structured_sales",
        ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16),
    )
    assert "toDecimalString" not in validation
```

```python
def test_preflight_rejects_non_18_16_in_legacy_mode_before_work():
    query = RecordingQueryClient([[('22.8.1',)]])
    gateway = ClickHouseGateway(
        RecordingIngestClient(),
        query_client=query,
        compatibility=ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16),
    )
    with pytest.raises(StructuredStorageError, match="18.16"):
        gateway.preflight()
```

- [ ] **Step 2: Run the gateway tests and verify they fail.**

Run: `uv run --project backend --group dev pytest backend/tests/test_clickhouse_gateway.py -q`

Expected: FAIL because gateway construction ignores the profile and has no preflight method.

- [ ] **Step 3: Implement profile injection and preflight.**

Replace direct `_clickhouse_type` and decimal expression branches with profile calls. Keep the
existing safe identifier checks and staging logic unchanged. `preflight()` must:

1. query `SELECT version()` with the read-only settings;
2. extract the first scalar from tuple, mapping, or `result_rows` shapes already supported by the
   gateway test helpers;
3. call `profile.validate_server_version(version)`;
4. execute `SELECT 1` with the exact profile settings;
5. execute a Decimal string probe using the profile's canonical expression;
6. raise `StructuredStorageError` with a sanitized capability name on failure.

Use the profile's `command_settings()` for command/insert calls and `query_settings()` plus
`readonly=1` for read-only validation and query calls. Update `_validation_query` to receive the
profile so its digest expression matches the ingest profile.

- [ ] **Step 4: Run gateway tests and inspect generated SQL.**

Run: `uv run --project backend --group dev pytest backend/tests/test_clickhouse_gateway.py -q`

Expected: PASS; the legacy assertions must show `DateTime`, `toString`, and no `overflow_mode`.

- [ ] **Step 5: Commit the gateway integration.**

```bash
git add backend/app/clickhouse_gateway.py backend/tests/test_clickhouse_gateway.py
git commit -m "feat: make ClickHouse gateway 18.16 compatible"
```

### Task 3: Propagate the profile through structured query and API lazy gateway

**Files:**
- Modify: `backend/app/structured_query.py`
- Modify: `backend/app/structured_answer.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_structured_query.py`
- Modify: `backend/tests/test_structured_answer.py`

**Interfaces:**
- `StructuredQueryPlanner(catalog, compatibility: ClickHouseCompatibilityProfile | None = None)`.
- `StructuredQueryExecutor(catalog, clickhouse_gateway, compatibility: ClickHouseCompatibilityProfile | None = None)`.
- `StructuredAnswerService(catalog_provider, clickhouse_gateway, compatibility: ClickHouseCompatibilityProfile | None = None)`.
- `_LazyStructuredQueryGateway(..., compatibility=profile)` passes the same profile to every `ClickHouseGateway` instance.

- [ ] **Step 1: Write failing planner tests.**

```python
def test_legacy_datetime_filter_uses_datetime_placeholder():
    profile = ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16)
    catalog = sample_catalog()
    dataset = catalog.datasets[0]
    datetime_column = replace(dataset.schema.columns[2], data_type=StructuredColumnType.DATETIME)
    catalog = replace(
        catalog,
        datasets=(replace(dataset, schema=replace(dataset.schema, columns=(*dataset.schema.columns[:2], datetime_column))),),
    )
    plan = StructuredQueryPlanner(catalog, compatibility=profile).plan(
        StructuredIntent(
            "ds-sales", "sum", "order_amount",
            (StructuredFilter("order_date", "between", "2026-01-01", "2026-01-31"),),
        ),
        sample_publication(),
    )
    assert "DateTime64(3)" not in plan.sql
    assert "{filter_0:DateTime}" in plan.sql
    assert plan.parameters["filter_0"].microsecond == 0
```

- [ ] **Step 2: Run the focused planner and answer tests to verify failure.**

Run: `uv run --project backend --group dev pytest backend/tests/test_structured_query.py backend/tests/test_structured_answer.py -q`

Expected: FAIL because the planner and service constructors do not accept or use a profile.

- [ ] **Step 3: Implement profile propagation.**

Replace the local `_clickhouse_parameter_type` mapping with `profile.parameter_type`. Pass the
profile into executor re-planning so the SQL and parameter dictionaries are compared under the same
dialect. In `main.py`, build the profile from `OfflineSettings`, pass it to the service and lazy
gateway, and call `gateway.preflight()` during lazy gateway construction before publishing it as
ready. Keep existing client identity separation and close-on-failure behavior.

- [ ] **Step 4: Update regression expectations and run tests.**

Keep existing modern assertions explicitly constructing the default/modern profile. Add legacy
assertions for datetime equality, upper-bound date expansion, and Decimal parameters.

Run: `uv run --project backend --group dev pytest backend/tests/test_structured_query.py backend/tests/test_structured_answer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the query propagation.**

```bash
git add backend/app/structured_query.py backend/app/structured_answer.py backend/app/main.py backend/tests/test_structured_query.py backend/tests/test_structured_answer.py
git commit -m "feat: route structured queries through ClickHouse dialect"
```

### Task 4: Normalize legacy datetimes and publish batch checkpoints

**Files:**
- Modify: `backend/app/structured_ingestion.py`
- Modify: `backend/app/structured_worker.py`
- Modify: `backend/tests/test_structured_ingestion.py`
- Modify: `backend/tests/test_structured_worker.py`

**Interfaces:**
- `SpreadsheetPublisher(..., compatibility: ClickHouseCompatibilityProfile | None = None)`.
- `StructuredPublisher.publish(..., lease_guard: Callable[..., None] | None = None, ...)` accepts an optional `checkpoint_row` keyword on the guard.
- `_check_lease(lease_guard, checkpoint_row: int | None = None)` calls the guard with the cumulative row count when provided.

- [ ] **Step 1: Write failing ingestion tests.**

```python
def test_legacy_arrow_schema_uses_seconds_and_truncates_datetime_microseconds():
    confirmed = sample_confirmed_schema(tmp_path, row_count=1)
    datetime_column = replace(confirmed.schema.columns[2], data_type=StructuredColumnType.DATETIME)
    schema = replace(confirmed.schema, columns=(*confirmed.schema.columns[:2], datetime_column))
    path = write_csv(tmp_path / "legacy.csv", [[column.original_name for column in schema.columns], ["1", "east", "2026-01-01T12:30:01.999999"]])
    sink = RecordingParquetSink(tmp_path / "parquet")
    profile = ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16)
    publisher = SpreadsheetPublisher(sink=sink, clickhouse=RecordingPublicationGateway(), compatibility=profile)
    publisher.publish(path, schema, "pub-legacy")
    batch = next(iter(sink.iter_batches(sink.output_paths)))
    assert batch.schema.field("order_date").type == pa.timestamp("s")
    assert batch.column("order_date")[0].as_py().microsecond == 0
```

```python
def test_insert_batches_report_cumulative_checkpoint_rows():
    checkpoints = []
    confirmed = sample_confirmed_schema(tmp_path, row_count=5)
    profile = ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16)
    publisher = SpreadsheetPublisher(
        sink=RecordingParquetSink(tmp_path / "parquet-progress"),
        clickhouse=RecordingPublicationGateway(),
        batch_rows=2,
        compatibility=profile,
    )
    publisher.publish(
        confirmed.path, confirmed.schema, "pub-progress",
        lease_guard=lambda **kwargs: checkpoints.append(kwargs.get("checkpoint_row")),
    )
    assert max(value for value in checkpoints if value is not None) == 5
```

- [ ] **Step 2: Run ingestion tests and verify failure.**

Run: `uv run --project backend --group dev --group offline pytest backend/tests/test_structured_ingestion.py -q`

Expected: FAIL because Arrow always uses millisecond timestamps and the publisher never reports insert progress.

- [ ] **Step 3: Implement profile-aware conversion and progress.**

Pass the profile into `_arrow_schema` and `_convert_value`; use `pa.timestamp("s")` in legacy mode
and `pa.timestamp("ms")` in modern mode. After each `insert_batch`, increment an `inserted_rows`
counter by `batch.num_rows` and call `_check_lease(lease_guard, checkpoint_row=inserted_rows)`.
Update the worker callback type and repository renewal call so `checkpoint_row` is forwarded. Keep
the final result checkpoint renewal as a safety fence. Update test publishers/guards to accept the
optional keyword without weakening lease-loss behavior.

- [ ] **Step 4: Add worker preflight-before-claim test.**

Inject a fake gateway whose `preflight()` raises and assert `worker.run_once()` is never reached by
the repository claim spy. Add a success test that records `checkpoint_row` values and verifies the
status endpoint exposes a positive checkpoint during a multi-batch publication.

- [ ] **Step 5: Run ingestion and worker tests.**

Run: `uv run --project backend --group dev --group offline pytest backend/tests/test_structured_ingestion.py backend/tests/test_structured_worker.py -q`

Expected: PASS.

- [ ] **Step 6: Commit progress and datetime behavior.**

```bash
git add backend/app/structured_ingestion.py backend/app/structured_worker.py backend/tests/test_structured_ingestion.py backend/tests/test_structured_worker.py
git commit -m "feat: report structured import progress in legacy mode"
```

### Task 5: Pin the client and document Ubuntu 18.16 account provisioning

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`
- Modify: `.env.example`
- Modify: `deploy/offline/.env.example`
- Modify: `deploy/offline/compose.yaml`
- Create: `deploy/ubuntu/clickhouse-18.16-users.xml.example`
- Create: `deploy/ubuntu/CLICKHOUSE_18_16.md`
- Test: `tools/tests/test_structured_deployment_contract.py`

**Interfaces:**
- Offline and local examples explicitly define `CLICKHOUSE_COMPATIBILITY_MODE=modern`.
- Ubuntu instructions require `CLICKHOUSE_COMPATIBILITY_MODE=legacy_18_16` for both API and worker processes.
- The XML example contains no real password and uses a documented placeholder token that operators replace with a generated SHA-256 password hash.

- [ ] **Step 1: Write failing deployment contract tests.**

```python
def test_examples_wire_clickhouse_compatibility_mode():
    root = Path(__file__).resolve().parents[2]
    assert "CLICKHOUSE_COMPATIBILITY_MODE" in (root / ".env.example").read_text()
    assert "CLICKHOUSE_COMPATIBILITY_MODE" in (root / "deploy/offline/.env.example").read_text()
    compose = (root / "deploy/offline/compose.yaml").read_text()
    assert compose.count("CLICKHOUSE_COMPATIBILITY_MODE") >= 2

def test_legacy_users_example_is_xml_without_modern_rbac_sql():
    root = Path(__file__).resolve().parents[2]
    xml = (root / "deploy/ubuntu/clickhouse-18.16-users.xml.example").read_text()
    assert "<users>" in xml and "dc_agent_query" in xml and "dc_agent_ingest" in xml
    assert "CREATE USER" not in xml and "GRANT" not in xml
```

- [ ] **Step 2: Run the deployment tests and verify failure.**

Run: `uv run --project backend --group dev pytest tools/tests/test_structured_deployment_contract.py -q`

Expected: FAIL because the Ubuntu directory, XML example, and environment wiring do not exist.

- [ ] **Step 3: Pin the dependency and update the lock metadata.**

Change `clickhouse-connect>=0.8` to `clickhouse-connect==1.6.0`. Run
`uv lock --project backend --offline` and verify the lock still resolves the same 1.6.0 artifact.

- [ ] **Step 4: Add environment wiring.**

Add `CLICKHOUSE_COMPATIBILITY_MODE=modern` to both example env files. Add a required Compose
interpolation to the API and indexing-worker environment blocks so the value cannot be silently
omitted.

- [ ] **Step 5: Add the legacy users XML example.**

Provide two users under `<users>`: query with `readonly=1` and database allow-list for `default`,
and ingest with `readonly=0` and the same database allow-list. Include profile/quota defaults and a
clearly marked hash replacement instruction; do not include plaintext secrets or SQL RBAC commands.

- [ ] **Step 6: Write Ubuntu rollout documentation.**

Document apt service checks, backup of `/etc/clickhouse-server/users.xml`, merging the XML example,
ClickHouse restart, password hash generation, API/worker environment entries, preflight curl/query,
Supervisor restart, status polling, existing queued-job recovery, and rollback. State that the
modern `deploy/offline/clickhouse-init.sh` is not used on 18.16.

- [ ] **Step 7: Run contract tests and commit deployment materials.**

Run: `uv run --project backend --group dev pytest tools/tests/test_structured_deployment_contract.py -q`

Expected: PASS.

```bash
git add backend/pyproject.toml backend/uv.lock .env.example deploy/offline/.env.example deploy/offline/compose.yaml deploy/ubuntu/clickhouse-18.16-users.xml.example deploy/ubuntu/CLICKHOUSE_18_16.md tools/tests/test_structured_deployment_contract.py
git commit -m "docs: document Ubuntu ClickHouse 18.16 deployment"
```

### Task 6: Add opt-in ClickHouse 18.16.1 integration acceptance coverage

**Files:**
- Create: `backend/tests/integration/test_clickhouse_legacy_18_16.py`
- Modify: `backend/tests/integration/test_structured_query_clickhouse.py`
- Modify: `backend/tests/integration/test_structured_aggregation_e2e.py`

**Interfaces:**
- Tests run only when `RUN_CLICKHOUSE_18_16=1` and an explicit private `CLICKHOUSE_URL`/credential set is present.
- Tests construct the legacy profile and use the same `ClickHouseGateway`, `SpreadsheetPublisher`, and `StructuredQueryPlanner` used in production.

- [ ] **Step 1: Add an opt-in guard and failing acceptance skeleton.**

```python
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_CLICKHOUSE_18_16") != "1",
    reason="set RUN_CLICKHOUSE_18_16=1 for an explicit ClickHouse 18.16.1 target",
)

def build_legacy_gateway_from_environment():
    import clickhouse_connect
    profile = ClickHouseCompatibilityProfile.for_mode(ClickHouseCompatibilityMode.LEGACY_18_16)
    common = {
        "dsn": os.environ["CLICKHOUSE_URL"],
        "username": os.environ["CLICKHOUSE_INGEST_USER"],
        "password": Path(os.environ["CLICKHOUSE_INGEST_PASSWORD_FILE"]).read_text().strip(),
    }
    ingest = clickhouse_connect.get_client(**common)
    query = clickhouse_connect.get_client(**common, autogenerate_session_id=False)
    return ClickHouseGateway(ingest, query_client=query, compatibility=profile)

def test_server_is_18_16_and_legacy_preflight_passes():
    gateway = build_legacy_gateway_from_environment()
    assert gateway.preflight().startswith("18.16.")
```

- [ ] **Step 2: Add end-to-end publication/query assertions.**

Use an isolated table prefix and cleanup in `finally`. Publish a workbook containing all supported
types, assert `DateTime` in `DESCRIBE`, verify row/content/null statistics, run datetime range and
Decimal aggregate queries, and publish a generated 100,000-row workbook with bounded batch size.

- [ ] **Step 3: Run the acceptance tests against the actual intranet server.**

Run from the backend environment:

```bash
RUN_CLICKHOUSE_18_16=1 CLICKHOUSE_COMPATIBILITY_MODE=legacy_18_16 \
uv run --project backend --group dev --group offline pytest backend/tests/integration/test_clickhouse_legacy_18_16.py -v
```

Expected: PASS against ClickHouse 18.16.1; SKIP without the explicit opt-in variable.

- [ ] **Step 4: Commit the acceptance coverage.**

```bash
git add backend/tests/integration/test_clickhouse_legacy_18_16.py backend/tests/integration/test_structured_query_clickhouse.py backend/tests/integration/test_structured_aggregation_e2e.py
git commit -m "test: cover ClickHouse 18.16 structured publication"
```

### Task 7: Full verification and production handoff

**Files:**
- Read: `docs/superpowers/specs/2026-08-10-clickhouse-18-16-structured-import-compatibility-design.md`
- Verify: all files from Tasks 1–6

- [ ] **Step 1: Run focused backend regression tests.**

```bash
uv run --project backend --group dev --group offline pytest \
  backend/tests/test_clickhouse_compatibility.py \
  backend/tests/test_offline_settings.py \
  backend/tests/test_clickhouse_gateway.py \
  backend/tests/test_structured_query.py \
  backend/tests/test_structured_answer.py \
  backend/tests/test_structured_ingestion.py \
  backend/tests/test_structured_worker.py \
  tools/tests/test_structured_deployment_contract.py -q
```

Expected: PASS with no new skips except optional integration tests.

- [ ] **Step 2: Run backend lint/type and existing contract checks.**

Run: `uv run --project backend --group dev ruff check backend/app backend/tests tools/tests`

Expected: PASS.

- [ ] **Step 3: Run the real-server acceptance command from Task 6.**

Record the ClickHouse version, preflight result, published row count, checkpoint progression, and
query results without recording passwords or full DSNs.

- [ ] **Step 4: Verify the existing queued job.**

After deploying `legacy_18_16` to API and Worker, query the structured-status endpoint. Confirm the
job leaves `queued`, `checkpointRow` becomes positive during multi-batch insertion, and the final
status is `published` with non-zero records/rows. If it fails, capture the sanitized `errorMessage`
and stop before re-uploading.

- [ ] **Step 5: Review each commit and report the safe rollout.**

Confirm no modern RAG/Reranker files changed, no secret values entered documentation, and no remote
push occurs without explicit user approval.
