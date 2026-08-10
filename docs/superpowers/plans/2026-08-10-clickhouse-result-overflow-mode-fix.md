# ClickHouse Result Overflow Setting Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow structured XLSX publication against ClickHouse servers that reject the invalid `overflow_mode` request setting.

**Architecture:** Keep all bounded ClickHouse request settings centralized in `ClickHouseGateway`, but use ClickHouse's supported `result_overflow_mode` key. Update the focused gateway and structured query contract tests so the obsolete key cannot regress.

**Tech Stack:** Python 3.12, ClickHouse Connect, `unittest`/pytest, uv

## Global Constraints

- Do not change XLSX parsing, PostgreSQL schema, ClickHouse table schemas, retry behavior, or public API models.
- Preserve `max_execution_time`, `max_memory_usage`, `max_result_rows`, `readonly`, and the `break` overflow behavior.
- Do not modify unrelated pnpm migration files already present in the worktree.

---

### Task 1: Correct the ClickHouse request setting

**Files:**
- Modify: `backend/tests/test_clickhouse_gateway.py:104-120,607-609`
- Modify: `backend/tests/test_structured_query.py:1216-1224`
- Modify: `backend/app/clickhouse_gateway.py:69-75`

**Interfaces:**
- Consumes: `ClickHouseGateway` constructor arguments and existing client `settings` mappings.
- Produces: all gateway requests use `result_overflow_mode="break"` and omit `overflow_mode`.

- [ ] **Step 1: Write the failing contract assertions**

Update the focused assertions to require the supported setting and reject the obsolete one:

```python
self.assertEqual(settings["result_overflow_mode"], "break")
self.assertNotIn("overflow_mode", settings)
```

Update the exact structured query settings mapping to:

```python
{
    "max_execution_time": 4,
    "max_memory_usage": 1024,
    "max_result_rows": 1,
    "result_overflow_mode": "break",
    "readonly": 1,
}
```

- [ ] **Step 2: Run the focused tests and verify the old implementation fails**

Run:

```bash
uv run --project backend --group dev python -m pytest \
  backend/tests/test_clickhouse_gateway.py \
  backend/tests/test_structured_query.py -q
```

Expected: FAIL because the gateway still returns `overflow_mode` and does not return `result_overflow_mode`.

- [ ] **Step 3: Implement the minimal compatibility fix**

Change the centralized settings mapping in `ClickHouseGateway.__init__` to:

```python
self._settings = {
    "max_execution_time": max_execution_time,
    "max_memory_usage": max_memory_usage,
    "max_result_rows": max_result_rows,
    "result_overflow_mode": "break",
}
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run the same focused pytest command. Expected: all selected tests pass.

- [ ] **Step 5: Run structured ingestion regression tests**

Run:

```bash
uv run --project backend --group dev --group offline python -m pytest \
  backend/tests/test_structured_ingestion.py \
  backend/tests/test_structured_worker.py \
  backend/tests/test_structured_api.py -q
```

Expected: all selected tests pass with no XLSX publication or worker regressions.

- [ ] **Step 6: Review the final diff**

Run:

```bash
git diff -- backend/app/clickhouse_gateway.py \
  backend/tests/test_clickhouse_gateway.py \
  backend/tests/test_structured_query.py
```

Expected: only the ClickHouse setting key and its assertions changed.
