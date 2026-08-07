# Structured Publication FK Order Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保证结构化发布父记录先于引用它的发布任务写入数据库，修复 Excel 字段确认后点击 Publish data 触发的 PostgreSQL 外键错误。

**Architecture:** 在现有单事务入队流程中，对新建 `StructuredPublicationRecord` 执行显式 `session.flush()`，再添加 `StructuredIngestionJobRecord`。使用开启 SQLite 外键约束的 API 回归测试复现 PostgreSQL 的即时外键校验，不新增数据库迁移或 ORM relationship。

**Tech Stack:** Python 3.12、FastAPI、SQLAlchemy 2、PostgreSQL、SQLite、pytest、uv、Ruff、ty

---

## 文件结构

- Modify: `backend/tests/test_structured_api.py` — 让现有发布入队 API 测试启用 SQLite 外键约束，直接覆盖用户操作路径。
- Modify: `backend/app/structured_repository.py` — 在添加发布父记录后显式 flush，再添加任务记录。
- Modify: `backend/app/__init__.py` — 后端版本从 `0.1.11` 升级到 `0.1.12`。
- Refresh if changed: `backend/uv.lock` — 执行锁文件刷新命令。

### Task 1: 用外键约束复现发布入队失败

**Files:**
- Modify: `backend/tests/test_structured_api.py`

- [ ] **Step 1: 在现有发布 API 测试中启用 SQLite 外键约束**

修改 `StructuredApiTest.test_publication_post_only_enqueues_and_status_reports_job`，在构建客户端前加入：

```python
def test_publication_post_only_enqueues_and_status_reports_job(self) -> None:
    with self.database.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")

    client = self.build_client()
```

只在现有测试开头插入上述外键启用步骤；其余上传、字段确认、POST 入队和 GET 状态代码保持原样。该测试已有的关键断言继续保留：

```python
self.assertEqual(enqueue.status_code, 202, enqueue.text)
self.assertTrue(body["jobId"])
self.assertEqual(body["status"], "queued")
self.assertEqual(status_body["job"]["status"], "queued")
self.assertIsNone(status_body["activePublication"])
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```bash
cd backend
uv run --project . python -m pytest \
  tests/test_structured_api.py::StructuredApiTest::test_publication_post_only_enqueues_and_status_reports_job \
  -q
```

Expected: FAIL/ERROR，异常包含：

```text
sqlite3.IntegrityError: FOREIGN KEY constraint failed
INSERT INTO structured_ingestion_jobs
```

该失败证明测试覆盖了与 PostgreSQL `structured_ingestion_jobs_publication_id_fkey` 相同的写入顺序问题。

### Task 2: 显式持久化发布父记录

**Files:**
- Modify: `backend/app/structured_repository.py`

- [ ] **Step 1: 在创建任务前 flush 发布记录**

在 `StructuredRepository.enqueue_publication()` 中把新增记录的写入顺序改为：

```python
session.add(publication)
session.flush()
session.add(job)
self._refresh_source_publication_state(session, source)
session.flush()
return _job_from_record(job)
```

第一次 `session.flush()` 只负责确保 `structured_publications.publication_id` 已经存在。Session 上下文仍在方法返回后统一 commit；如果任务写入或状态刷新失败，现有异常处理会 rollback 整个事务，发布父记录不会孤立保留。

- [ ] **Step 2: 运行外键回归测试**

Run: Task 1 Step 2 的命令。

Expected: PASS，POST 返回 202，任务状态为 `queued`。

- [ ] **Step 3: 运行结构化 API 与 Worker 测试**

Run:

```bash
cd backend
uv run --project . python -m pytest tests/test_structured_api.py tests/test_structured_worker.py -q
```

Expected: 全部 PASS，现有任务复用、租约、失败重试和发布完成逻辑不受影响。

### Task 3: 升级后端版本并完成验证

**Files:**
- Modify: `backend/app/__init__.py`
- Refresh if changed: `backend/uv.lock`

- [ ] **Step 1: 独立提升后端 patch 版本**

Run:

```bash
uv run --project backend python tools/bump_version.py backend patch
uv lock --project backend
```

Expected:

```text
backend version: 0.1.11 -> 0.1.12
```

用户端和管理端版本保持 `0.1.1`。

- [ ] **Step 2: 运行版本契约和完整后端测试**

Run:

```bash
cd backend
uv run --project . python -m pytest ../tools/tests/test_version_contract.py -q
uv run --project . python -m pytest tests -q
```

Expected: 版本契约与全部后端测试通过，允许仓库既有 skip 和弃用警告。

- [ ] **Step 3: 运行静态检查和差异检查**

Run:

```bash
cd ..
fast lint --ty
git diff --check
```

Expected: Ruff、ty 和空白检查全部通过。

- [ ] **Step 4: 检查最终范围**

Run:

```bash
git status --short
git diff -- backend/app/structured_repository.py backend/app/__init__.py backend/tests/test_structured_api.py backend/uv.lock
```

Expected: 只包含显式父记录 flush、外键回归测试和后端版本变更；无数据库迁移、前端代码或前端版本改动。
