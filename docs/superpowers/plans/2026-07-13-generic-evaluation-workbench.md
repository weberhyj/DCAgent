# 通用质量评测工作台实施计划

> **供执行代理使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按照任务逐项实施。计划使用复选框（`- [ ]`）跟踪进度。

**目标：** 在不依赖真实公司业务数据、也不向正式数据库写入演示数据的前提下，为 DC-Agent 管理端实现评测问题批量导入、分类标签、后台批次运行、质量报告和批次对比。

**架构：** 后端继续以 `ChatRepository` 作为持久化边界，新建独立的导入解析模块和批次指标模块；FastAPI 路由只做请求校验与任务编排。管理端将现有 `/quality` 单页拆分为评测集、报告列表和报告详情三个页面，分别使用 `useQualityCases` 与 `useEvaluationBatches` 管理接口状态。

**技术栈：** Python 3、FastAPI、SQLAlchemy、PostgreSQL、openpyxl、Vue 3、TypeScript、Vue Router、Axios、Vitest、Vue Test Utils。

---

## 执行约束

- 当前工作区根目录不是 Git 仓库，不能执行提交步骤；每个任务以指定测试命令作为检查点。
- 所有新增后端测试使用现有 `unittest` 风格，不引入 pytest。
- 所有 Vue 文件继续使用 `<script setup lang="ts">`、Composition API、明确的 Props/Emits。
- 正式数据库默认保持空白，任何测试数据只能出现在测试文件中。
- 评测批次的阈值只影响评测检索，不修改环境变量，也不改变用户侧 DCAgent 问答链路。

## 文件结构

### 后端新增文件

- `backend/app/evaluation_import.py`：解析 XLSX、CSV、JSON，校验行数据，生成短期预览令牌。
- `backend/app/evaluation_batches.py`：批次汇总指标、状态变化和批次比较纯函数。
- `backend/tests/test_evaluation_import.py`：导入解析、预览、确认和重复数据测试。
- `backend/tests/test_evaluation_batches.py`：阈值隔离、批次运行、汇总和比较测试。

### 后端修改文件

- `backend/app/database.py`：增加评测元数据、导入批次、运行批次及兼容迁移。
- `backend/app/evaluation.py`：扩展评测问题与运行模型，增加批次模型。
- `backend/app/retrieval.py`：保持现有阈值解析，并支持显式评测阈值。
- `backend/app/repository.py`：扩展仓储协议和内存实现。
- `backend/app/sql_repository.py`：实现 PostgreSQL/SQLite 持久化与事务导入。
- `backend/app/schemas.py`：增加导入、筛选、批次和报告接口模型。
- `backend/app/routes.py`：增加导入、筛选、批次和比较接口。
- `backend/app/main.py`：初始化预览存储和批次执行依赖。
- `backend/tests/test_quality_evaluation.py`：保护原有单条评测接口兼容性。

### 管理端新增文件

- `admin-frontend/src/components/evaluation/EvaluationCaseToolbar.vue`
- `admin-frontend/src/components/evaluation/EvaluationImportDialog.vue`
- `admin-frontend/src/components/evaluation/EvaluationBatchDialog.vue`
- `admin-frontend/src/components/evaluation/EvaluationBatchList.vue`
- `admin-frontend/src/components/evaluation/EvaluationReportSummary.vue`
- `admin-frontend/src/components/evaluation/EvaluationFailureList.vue`
- `admin-frontend/src/composables/useQualityCases.ts`
- `admin-frontend/src/composables/useQualityCases.spec.ts`
- `admin-frontend/src/composables/useEvaluationBatches.ts`
- `admin-frontend/src/composables/useEvaluationBatches.spec.ts`
- `admin-frontend/src/views/QualityModuleLayout.vue`
- `admin-frontend/src/views/QualityCasesPage.vue`
- `admin-frontend/src/views/QualityReportsPage.vue`
- `admin-frontend/src/views/QualityReportDetailPage.vue`
- `admin-frontend/src/views/__tests__/QualityCasesPage.spec.ts`
- `admin-frontend/src/views/__tests__/QualityReportsPage.spec.ts`
- `admin-frontend/src/views/__tests__/QualityReportDetailPage.spec.ts`

### 管理端修改或删除文件

- 修改 `admin-frontend/src/types/chat.ts`：增加评测元数据、导入和批次类型。
- 修改 `admin-frontend/src/services/api.ts`：增加导入与批次接口。
- 修改 `admin-frontend/src/router/index.ts`：拆分质量评测子路由。
- 修改 `admin-frontend/src/components/layout/AdminLayout.vue`：质量评测子路由保持侧栏激活。
- 修改 `admin-frontend/src/components/evaluation/EvaluationCaseList.vue`：增加多选、分类和标签。
- 修改 `admin-frontend/src/components/evaluation/EvaluationCaseFormDialog.vue`：支持分类和标签。
- 删除 `admin-frontend/src/views/QualityEvaluationPage.vue`。
- 删除 `admin-frontend/src/views/__tests__/QualityEvaluationPage.spec.ts`。
- 删除 `admin-frontend/src/composables/useQualityEvaluation.ts`。
- 删除 `admin-frontend/src/composables/useQualityEvaluation.spec.ts`。

---

### Task 1：扩展评测问题数据模型与兼容迁移

**Files：**

- Modify: `backend/app/database.py`
- Modify: `backend/app/evaluation.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Test: `backend/tests/test_quality_evaluation.py`

- [ ] **Step 1：先编写元数据持久化失败测试**

在 `QualityEvaluationSqlRepositoryTest` 中增加：

```python
def test_persists_business_neutral_case_metadata(self) -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()
    repository = SqlChatRepository(database)

    created = repository.create_evaluation_case(
        question="资料归档要求是什么",
        expected_source_ids=[],
        expected_terms=["归档"],
        top_k=5,
        category="制度",
        tags=["归档", "流程", "归档"],
        external_key="policy-archive-001",
    )

    persisted = SqlChatRepository(database).list_evaluation_cases()[0]

    self.assertEqual(persisted.id, created.id)
    self.assertEqual(persisted.category, "制度")
    self.assertEqual(persisted.tags, ["归档", "流程"])
    self.assertEqual(persisted.external_key, "policy-archive-001")
    self.assertIsNone(persisted.import_batch_id)
```

- [ ] **Step 2：运行测试并确认因接口缺少字段而失败**

Run: `py -m unittest tests.test_quality_evaluation.QualityEvaluationSqlRepositoryTest.test_persists_business_neutral_case_metadata -v`

Expected: FAIL，错误指出 `create_evaluation_case()` 不接受 `category`、`tags` 或 `external_key`。

- [ ] **Step 3：扩展模型和仓储接口**

将 `EvaluationCaseModel` 扩展为：

```python
@dataclass(slots=True)
class EvaluationCaseModel:
    id: str
    question: str
    expected_source_ids: list[str]
    expected_terms: list[str]
    expect_answer: bool
    top_k: int
    category: str | None
    tags: list[str]
    external_key: str | None
    import_batch_id: str | None
    created_at: str
    updated_at: str
```

将 `ChatRepository.create_evaluation_case()`、内存仓储和 SQL 仓储统一扩展为：

```python
def create_evaluation_case(
    self,
    question: str,
    expected_source_ids: list[str],
    expected_terms: list[str],
    top_k: int,
    expect_answer: bool = True,
    category: str | None = None,
    tags: list[str] | None = None,
    external_key: str | None = None,
    import_batch_id: str | None = None,
) -> EvaluationCaseModel:
    ...
```

标签使用 `normalized_unique(tags or [])`；分类和外部标识使用 `strip()` 后的值，空字符串转换为 `None`。

- [ ] **Step 4：增加数据库字段和兼容迁移**

在 `EvaluationCaseRecord` 中增加：

```python
category: Mapped[str | None] = mapped_column(String(80), index=True)
tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
external_key: Mapped[str | None] = mapped_column(String(120), index=True)
import_batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
```

在 `_ensure_evaluation_columns()` 中为已有 PostgreSQL 表补充 `category`、`tags`、`external_key`、`import_batch_id`；`tags` 使用非空 JSON 默认空数组。SQLite 内存测试由 `Base.metadata.create_all()` 直接创建完整结构。

- [ ] **Step 5：扩展 API Schema 并保持旧请求兼容**

在 `EvaluationCaseRequest` 与 `EvaluationCase` 增加：

```python
category: str | None = Field(default=None, max_length=80)
tags: list[str] = Field(default_factory=list)
external_key: str | None = Field(default=None, alias="externalKey", max_length=120)
import_batch_id: str | None = Field(default=None, alias="importBatchId")
```

旧请求不传这些字段时仍然创建成功。更新 `EvaluationCase.from_model()` 和路由调用。

- [ ] **Step 6：运行元数据测试与原有评测测试**

Run: `py -m unittest tests.test_quality_evaluation -v`

Expected: 原有测试与新增元数据测试全部 PASS。

---

### Task 2：实现 XLSX、CSV、JSON 导入解析与预览令牌

**Files：**

- Create: `backend/app/evaluation_import.py`
- Create: `backend/tests/test_evaluation_import.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1：编写 CSV 和 JSON 解析失败测试**

创建 `backend/tests/test_evaluation_import.py`：

```python
from __future__ import annotations

import json
import unittest

from app.evaluation_import import EvaluationImportService
from app.models import KnowledgeSourceModel


class EvaluationImportServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = EvaluationImportService(ttl_seconds=1800)
        self.sources = [
            KnowledgeSourceModel(
                id="kb-policy",
                name="制度汇编.pdf",
                source_type="PDF",
                records=1,
                status="已索引",
                updated_at="2026-07-13 10:00:00",
                classification="内部",
            )
        ]

    def test_previews_csv_without_persisting_rows(self) -> None:
        content = (
            "question,expect_answer,expected_sources,expected_terms,category,tags,top_k,external_key\n"
            "资料如何归档,true,制度汇编.pdf,归档|保管,制度,归档|流程,5,policy-001\n"
            "是否提供火星补贴,false,,,福利,无答案,5,no-answer-001\n"
        ).encode("utf-8-sig")

        preview = self.service.preview("cases.csv", content, self.sources)

        self.assertEqual(preview.total_rows, 2)
        self.assertEqual(preview.valid_rows, 2)
        self.assertEqual(preview.rows[0].expected_source_ids, ["kb-policy"])
        self.assertEqual(preview.rows[0].tags, ["归档", "流程"])
        self.assertFalse(preview.rows[1].expect_answer)

    def test_reports_json_row_validation_errors(self) -> None:
        content = json.dumps([
            {"question": "", "expect_answer": True, "expected_terms": []},
            {"question": "正常问题", "expect_answer": True, "expected_sources": ["不存在.pdf"]},
        ], ensure_ascii=False).encode("utf-8")

        preview = self.service.preview("cases.json", content, self.sources)

        self.assertEqual(preview.valid_rows, 0)
        self.assertEqual(preview.invalid_rows, 2)
        self.assertEqual(preview.errors[0].row_number, 2)
        self.assertIn("问题不能为空", preview.errors[0].message)
        self.assertIn("未找到资料", preview.errors[1].message)
```

- [ ] **Step 2：运行测试并确认模块不存在**

Run: `py -m unittest tests.test_evaluation_import -v`

Expected: FAIL，错误为 `ModuleNotFoundError: app.evaluation_import`。

- [ ] **Step 3：实现导入数据结构和字段标准化**

在 `evaluation_import.py` 定义：

```python
class EvaluationImportFileError(ValueError):
    pass


class EvaluationImportTokenError(ValueError):
    pass


@dataclass(slots=True)
class EvaluationImportRow:
    row_number: int
    question: str
    expect_answer: bool
    expected_source_ids: list[str]
    expected_terms: list[str]
    category: str | None
    tags: list[str]
    top_k: int
    external_key: str | None


@dataclass(slots=True)
class EvaluationImportError:
    row_number: int
    field: str
    message: str


@dataclass(slots=True)
class EvaluationImportPreview:
    token: str
    file_name: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_rows: int
    rows: list[EvaluationImportRow]
    errors: list[EvaluationImportError]
    duplicate_keys: list[str]
    expires_at: float
```

实现 `parse_boolean()`，接受 `true/false`、`1/0`、`是/否`、`应有答案/应无答案`；实现 `split_values()`，同时接受列表或使用 `|` 分隔的字符串。

- [ ] **Step 4：实现三种文件解析器和限制**

实现：

```python
MAX_IMPORT_BYTES = 5 * 1024 * 1024
MAX_IMPORT_ROWS = 2000

def read_import_records(file_name: str, content: bytes) -> list[dict[str, object]]:
    suffix = Path(file_name).suffix.lower()
    if len(content) > MAX_IMPORT_BYTES:
        raise EvaluationImportFileError("文件不能超过 5MB")
    if suffix == ".csv":
        return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if suffix == ".json":
        payload = json.loads(content.decode("utf-8-sig"))
        if not isinstance(payload, list):
            raise EvaluationImportFileError("JSON 顶层必须是数组")
        return payload
    if suffix == ".xlsx":
        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        worksheet = workbook.worksheets[0]
        rows = list(worksheet.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [str(value or "").strip() for value in rows[0]]
        return [dict(zip(headers, values, strict=False)) for values in rows[1:]]
    raise EvaluationImportFileError("仅支持 XLSX、CSV 和 JSON 文件")
```

解析后检查最多 2,000 行。

- [ ] **Step 5：实现资料名称解析和短期预览存储**

`EvaluationImportService.preview()` 使用资料名称建立 `name -> source ids` 映射；不存在或同名多条都返回行错误。方法签名为 `preview(file_name, content, sources, existing_cases=[])`，并使用现有问题完成重复检测。预览令牌使用 `secrets.token_urlsafe(24)`，并以线程锁保护内存字典。

实现：

```python
def consume(self, token: str) -> EvaluationImportPreview:
    with self._lock:
        preview = self._previews.pop(token, None)
    if preview is None or preview.expires_at < time.time():
        raise EvaluationImportTokenError("导入预览已过期，请重新上传文件")
    return preview
```

- [ ] **Step 6：增加 XLSX 和过期令牌测试**

测试使用 `openpyxl.Workbook()` 在内存中创建工作簿，并验证第一张工作表被读取；另一个测试将 `ttl_seconds=0`，验证 `consume()` 抛出 `EvaluationImportTokenError`。

- [ ] **Step 7：初始化服务并运行解析测试**

在 `create_app()` 中增加：

```python
app.state.evaluation_import_service = EvaluationImportService(ttl_seconds=1800)
```

Run: `py -m unittest tests.test_evaluation_import -v`

Expected: CSV、JSON、XLSX、限制和令牌测试全部 PASS。

---

### Task 3：实现导入预览、确认接口和事务化批量写入

**Files：**

- Modify: `backend/app/schemas.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/tests/test_evaluation_import.py`

- [ ] **Step 1：编写 API 预览和确认失败测试**

在 `test_evaluation_import.py` 增加：

```python
class EvaluationImportApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        self.client = TestClient(create_app(repository=self.repository))

    def test_previews_then_confirms_valid_rows(self) -> None:
        csv_content = (
            "question,expect_answer,expected_terms,category,tags,external_key\n"
            "资料保管多久,true,保管期限,制度,档案|保管,case-001\n"
            "是否有火星津贴,false,,福利,无答案,case-002\n"
        ).encode("utf-8")

        preview_response = self.client.post(
            "/api/admin/evaluations/import/preview",
            files={"file": ("cases.csv", csv_content, "text/csv")},
        )
        self.assertEqual(preview_response.status_code, 200)
        preview = preview_response.json()
        self.assertEqual(preview["validRows"], 2)
        self.assertEqual(self.repository.list_evaluation_cases(), [])

        confirm_response = self.client.post(
            "/api/admin/evaluations/import/confirm",
            json={"previewToken": preview["previewToken"]},
        )
        self.assertEqual(confirm_response.status_code, 200)
        self.assertEqual(confirm_response.json()["createdCount"], 2)
        self.assertEqual(len(self.repository.list_evaluation_cases()), 2)
```

- [ ] **Step 2：运行 API 测试并确认路由不存在**

Run: `py -m unittest tests.test_evaluation_import.EvaluationImportApiTest -v`

Expected: FAIL，预览接口返回 404。

- [ ] **Step 3：增加导入 Schema**

在 `schemas.py` 增加 `EvaluationImportRowResponse`、`EvaluationImportErrorResponse`、`EvaluationImportPreviewResponse`、`EvaluationImportConfirmRequest` 和 `EvaluationImportConfirmResponse`。后缀 `Response` 用于避免与内部导入数据类重名。返回字段统一使用 camelCase，例如：

```python
class EvaluationImportConfirmRequest(ApiModel):
    preview_token: str = Field(alias="previewToken", min_length=1)


class EvaluationImportConfirmResponse(ApiModel):
    import_batch_id: str = Field(alias="importBatchId")
    created_count: int = Field(alias="createdCount")
    duplicate_count: int = Field(alias="duplicateCount")
    dashboard: EvaluationDashboard
```

- [ ] **Step 4：增加导入批次表和事务化仓储方法**

在 `database.py` 增加 `EvaluationImportBatchRecord`，字段与设计规格保持一致：`id`、`file_name`、`status`、`total_rows`、`valid_rows`、`invalid_rows`、`duplicate_rows`、`created_at`、`completed_at`。确认导入时在同一事务中先写入状态为 `completed` 的导入批次，再写入全部有效评测问题。

扩展协议：

```python
def create_evaluation_cases(
    self,
    rows: list[EvaluationImportRow],
    import_batch_id: str,
) -> list[EvaluationCaseModel]:
    ...
```

SQL 实现必须在一个 `Database.session()` 中完成重复检查和全部插入。重复判断优先使用 `external_key`，否则使用 `question.strip().casefold()`。内存实现先计算新列表，成功后一次性替换，避免半写入。

- [ ] **Step 5：实现两个导入路由**

路由依赖：

```python
def get_evaluation_import_service(request: Request) -> EvaluationImportService:
    return request.app.state.evaluation_import_service
```

预览路由读取 `UploadFile` 内容，并调用 `service.preview(file_name, content, repository.list_knowledge_sources(), repository.list_evaluation_cases())`；确认路由调用 `service.consume()`，生成 `eval-import-<12 hex>`，再调用仓储批量写入。文件类型、大小、令牌和行错误转换为可读的 400 或 422 响应。

- [ ] **Step 6：增加重复跳过和事务回滚测试**

先导入 `external_key=case-001`，再次预览相同文件时断言 `duplicateRows == 1`；模拟第二行写入异常时断言 SQL 仓储中仍然没有新行。

- [ ] **Step 7：运行导入模块全量测试**

Run: `py -m unittest tests.test_evaluation_import -v`

Expected: 解析、API、重复和事务测试全部 PASS。

---

### Task 4：实现评测问题筛选接口

**Files：**

- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/tests/test_quality_evaluation.py`

- [ ] **Step 1：编写分类、标签和答案预期筛选失败测试**

```python
def test_filters_evaluation_cases_by_metadata(self) -> None:
    self.repository.create_evaluation_case(
        "合同归档要求",
        [],
        ["归档"],
        5,
        category="合同",
        tags=["法务"],
    )
    self.repository.create_evaluation_case(
        "是否提供火星补贴",
        [],
        [],
        5,
        expect_answer=False,
        category="福利",
        tags=["无答案"],
    )

    response = self.client.get(
        "/api/admin/evaluations/cases",
        params={"category": "合同", "tag": "法务", "expectAnswer": "true"},
    )

    self.assertEqual(response.status_code, 200)
    self.assertEqual([item["question"] for item in response.json()], ["合同归档要求"])
```

- [ ] **Step 2：运行测试并确认接口返回 404**

Run: `py -m unittest tests.test_quality_evaluation.QualityEvaluationApiTest.test_filters_evaluation_cases_by_metadata -v`

Expected: FAIL，接口返回 404。

- [ ] **Step 3：扩展仓储查询签名**

```python
def list_evaluation_cases(
    self,
    category: str | None = None,
    tag: str | None = None,
    expect_answer: bool | None = None,
    status: EvaluationRunStatus | None = None,
) -> list[EvaluationCaseModel]:
    ...
```

SQL 端对分类和答案预期使用数据库条件；标签和最新运行状态在读取后进行明确过滤。内存实现使用同一组纯过滤函数，保证两种仓储行为一致。

- [ ] **Step 4：增加筛选路由和分类标签摘要**

增加 `GET /api/admin/evaluations/cases`。响应头不承担元数据，另在 `EvaluationCaseCollection` 返回：

```python
class EvaluationCaseCollection(ApiModel):
    items: list[EvaluationCase]
    categories: list[str]
    tags: list[str]
    total: int
```

分类和标签按中文文本排序并去重。

- [ ] **Step 5：运行质量评测测试**

Run: `py -m unittest tests.test_quality_evaluation -v`

Expected: 新筛选测试和旧 Dashboard 接口测试全部 PASS。

---

### Task 5：支持仅用于评测的检索阈值覆盖

**Files：**

- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/retrieval.py`
- Create: `backend/tests/test_evaluation_batches.py`

- [ ] **Step 1：编写阈值隔离失败测试**

创建 `test_evaluation_batches.py`：

```python
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.models import ChatState, KnowledgeChunkModel
from app.repository import InMemoryChatRepository


class EvaluationThresholdIsolationTest(unittest.TestCase):
    def test_explicit_threshold_changes_only_one_search(self) -> None:
        repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        repository.add_uploaded_knowledge_source(
            "kb-1", "制度.txt", "文档", "内部", 0, "制度.txt", 10, "text/plain"
        )
        repository.complete_knowledge_source_indexing(
            "kb-1",
            [KnowledgeChunkModel("chunk-1", "kb-1", 0, "归档制度要求", 6)],
        )

        with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": "100"}):
            self.assertEqual(repository.search_knowledge_chunks("归档制度", 5), [])
            self.assertGreater(
                len(repository.search_knowledge_chunks("归档制度", 5, minimum_score=0)),
                0,
            )
            self.assertEqual(os.environ["RETRIEVAL_MIN_SCORE"], "100")
```

- [ ] **Step 2：运行测试并确认签名不支持 `minimum_score`**

Run: `py -m unittest tests.test_evaluation_batches.EvaluationThresholdIsolationTest -v`

Expected: FAIL，`search_knowledge_chunks()` 不接受 `minimum_score`。

- [ ] **Step 3：扩展检索签名并传递阈值**

协议、内存仓储和 SQL 仓储统一使用：

```python
def search_knowledge_chunks(
    self,
    query: str,
    limit: int = KNOWLEDGE_SEARCH_LIMIT,
    minimum_score: float | None = None,
) -> list[KnowledgeSearchHitModel]:
    ...
```

过滤时调用：

```python
is_reliable_knowledge_score(
    keyword_score,
    vector_score,
    total_score,
    minimum_score=minimum_score,
)
```

用户侧调用不传该参数，继续使用环境变量解析结果。

- [ ] **Step 4：运行阈值测试和现有检索测试**

Run: `py -m unittest tests.test_evaluation_batches.EvaluationThresholdIsolationTest tests.test_retrieval_threshold tests.test_sql_repository -v`

Expected: 全部 PASS，环境变量未被修改。

---

### Task 6：实现评测批次、后台运行和汇总指标

**Files：**

- Modify: `backend/app/database.py`
- Modify: `backend/app/evaluation.py`
- Create: `backend/app/evaluation_batches.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/tests/test_evaluation_batches.py`

- [ ] **Step 1：编写批次运行和汇总失败测试**

```python
class EvaluationBatchApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryChatRepository(
            ChatState(conversations=[], messages_by_conversation={}, knowledge_sources=[])
        )
        self.client = TestClient(create_app(repository=self.repository))

    def test_runs_named_batch_and_returns_summary(self) -> None:
        answer_case = self.repository.create_evaluation_case(
            "归档制度是什么", [], ["归档"], 5, category="制度"
        )
        no_answer_case = self.repository.create_evaluation_case(
            "是否提供火星补贴", [], [], 5, expect_answer=False, category="福利"
        )

        response = self.client.post(
            "/api/admin/evaluations/batches",
            json={
                "name": "首轮通用评测",
                "caseIds": [answer_case.id, no_answer_case.id],
                "retrievalMinScore": 2.2,
            },
        )

        self.assertEqual(response.status_code, 200)
        batch_id = response.json()["id"]
        detail = self.client.get(f"/api/admin/evaluations/batches/{batch_id}").json()
        self.assertEqual(detail["name"], "首轮通用评测")
        self.assertEqual(detail["caseCount"], 2)
        self.assertEqual(detail["completedCount"], 2)
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["summary"]["noAnswerAccuracy"], 1.0)
```

- [ ] **Step 2：运行测试并确认批次接口不存在**

Run: `py -m unittest tests.test_evaluation_batches.EvaluationBatchApiTest.test_runs_named_batch_and_returns_summary -v`

Expected: FAIL，批次接口返回 404。

- [ ] **Step 3：增加批次数据模型和数据库表**

在 `evaluation.py` 定义 `EvaluationBatchStatus = Literal["queued", "running", "completed", "failed"]`，以及：

```python
@dataclass(slots=True)
class EvaluationBatchModel:
    id: str
    name: str
    status: EvaluationBatchStatus
    case_ids: list[str]
    retrieval_min_score: float
    case_count: int
    completed_count: int
    passed_count: int
    failed_count: int
    false_positive_count: int
    started_at: str
    completed_at: str | None
    error_message: str | None


@dataclass(slots=True)
class EvaluationMetricGroupModel:
    name: str
    total: int
    passed: int
    pass_rate: float


@dataclass(slots=True)
class EvaluationBatchSummaryModel:
    total: int
    passed: int
    failed: int
    pass_rate: float
    answer_pass_rate: float
    no_answer_accuracy: float
    false_positive_count: int
    false_positive_rate: float
    average_source_recall: float
    average_term_recall: float
    average_top_score: float
    maximum_top_score: float
    category_breakdown: list[EvaluationMetricGroupModel]
    tag_breakdown: list[EvaluationMetricGroupModel]
```

数据库增加 `EvaluationBatchRecord`，其中 `case_ids` 使用 JSON；`EvaluationRunRecord.batch_id` 增加可空外键和索引。兼容迁移只补充缺失的 `batch_id`。

- [ ] **Step 4：实现纯汇总函数**

在 `evaluation_batches.py` 实现：

```python
def summarize_evaluation_runs(
    runs: list[EvaluationRunModel],
    cases_by_id: dict[str, EvaluationCaseModel],
) -> EvaluationBatchSummaryModel:
    answer_runs = [run for run in runs if run.expect_answer]
    no_answer_runs = [run for run in runs if not run.expect_answer]
    source_runs = [run for run in answer_runs if run.expected_source_ids]
    term_runs = [run for run in answer_runs if run.expected_terms]
    return EvaluationBatchSummaryModel(
        total=len(runs),
        passed=sum(run.status == "passed" for run in runs),
        failed=sum(run.status == "failed" for run in runs),
        pass_rate=ratio(sum(run.status == "passed" for run in runs), len(runs)),
        answer_pass_rate=ratio(sum(run.status == "passed" for run in answer_runs), len(answer_runs)),
        no_answer_accuracy=ratio(sum(run.status == "passed" for run in no_answer_runs), len(no_answer_runs)),
        false_positive_count=sum(run.false_positive for run in runs),
        false_positive_rate=ratio(sum(run.false_positive for run in runs), len(no_answer_runs)),
        average_source_recall=average(run.source_recall for run in source_runs),
        average_term_recall=average(run.term_recall for run in term_runs),
        average_top_score=average(run.top_score for run in runs),
        maximum_top_score=max((run.top_score for run in runs), default=0.0),
        category_breakdown=build_category_breakdown(runs, cases_by_id),
        tag_breakdown=build_tag_breakdown(runs, cases_by_id),
    )
```

`ratio()` 和 `average()` 在分母或数据为空时返回 `0.0`。`build_category_breakdown()` 与 `build_tag_breakdown()` 按问题元数据统计总数、通过数和通过率；没有分类的问题归入“未分类”，没有标签的问题不进入标签汇总。

- [ ] **Step 5：实现仓储批次方法**

协议增加：

```python
def create_evaluation_batch(
    self,
    name: str,
    case_ids: list[str],
    retrieval_min_score: float,
) -> EvaluationBatchModel: ...

def run_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel: ...
def list_evaluation_batches(self) -> list[EvaluationBatchModel]: ...
def get_evaluation_batch(self, batch_id: str) -> EvaluationBatchModel: ...
def list_evaluation_runs_for_batch(self, batch_id: str) -> list[EvaluationRunModel]: ...
```

批次创建时保存不可变的 `case_ids` 快照。运行时状态依次为 `queued -> running -> completed`，每条运行记录写入 `batch_id`。单条异常生成失败运行并继续，批次级异常写入 `error_message` 并标记为 `failed`。

- [ ] **Step 6：实现后台路由**

在 `schemas.py` 增加 `EvaluationBatchRequest`、`EvaluationBatch`、`EvaluationBatchSummary` 和 `EvaluationBatchDetail`，所有对外字段继续使用 camelCase。`EvaluationBatchRequest` 要求非空名称、至少一个问题 ID，并限制 `retrievalMinScore >= 0`。

使用 FastAPI `BackgroundTasks`：

```python
@router.post("/admin/evaluations/batches", response_model=EvaluationBatch)
def create_evaluation_batch(
    request: EvaluationBatchRequest,
    background_tasks: BackgroundTasks,
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationBatch:
    batch = repository.create_evaluation_batch(
        request.name,
        request.case_ids,
        request.retrieval_min_score,
    )
    background_tasks.add_task(repository.run_evaluation_batch, batch.id)
    return EvaluationBatch.from_model(batch, summary=None)
```

增加批次列表和详情接口。详情接口读取批次运行记录并调用汇总函数。

- [ ] **Step 7：增加进度、失败继续和 SQL 持久化测试**

测试覆盖：空 `caseIds` 拒绝、未知问题 ID 返回 404、单条异常后 `completedCount` 仍等于总数、重建 `SqlChatRepository` 后批次和运行记录仍存在。

- [ ] **Step 8：运行批次测试**

Run: `py -m unittest tests.test_evaluation_batches -v`

Expected: 阈值隔离、批次状态、指标和持久化测试全部 PASS。

---

### Task 7：实现批次比较和失败案例分类

**Files：**

- Modify: `backend/app/evaluation_batches.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/tests/test_evaluation_batches.py`

- [ ] **Step 1：编写批次比较失败测试**

```python
from app.evaluation import EvaluationBatchModel, EvaluationRunModel, EvaluationRunStatus
from app.evaluation_batches import compare_evaluation_batches


def make_run(case_id: str, status: EvaluationRunStatus) -> EvaluationRunModel:
    return EvaluationRunModel(
        id=f"run-{case_id}-{status}",
        case_id=case_id,
        question=case_id,
        status=status,
        expect_answer=True,
        answerable=status == "passed",
        false_positive=False,
        expected_source_ids=[],
        matched_source_ids=[],
        missing_source_ids=[],
        expected_terms=["命中"],
        found_terms=["命中"] if status == "passed" else [],
        missing_terms=[] if status == "passed" else ["命中"],
        source_recall=1.0,
        term_recall=1.0 if status == "passed" else 0.0,
        top_score=3.0 if status == "passed" else 0.0,
        hit_count=1 if status == "passed" else 0,
        started_at="2026-07-13 10:00:00",
        completed_at="2026-07-13 10:00:01",
        hits=[],
    )


def make_batch(batch_id: str, case_ids: list[str]) -> EvaluationBatchModel:
    return EvaluationBatchModel(
        id=batch_id,
        name=batch_id,
        status="completed",
        case_ids=case_ids,
        retrieval_min_score=2.2,
        case_count=len(case_ids),
        completed_count=len(case_ids),
        passed_count=0,
        failed_count=0,
        false_positive_count=0,
        started_at="2026-07-13 10:00:00",
        completed_at="2026-07-13 10:00:01",
        error_message=None,
    )


def test_compares_shared_case_status_changes(self) -> None:
    left_runs = [make_run("case-1", "failed"), make_run("case-2", "passed")]
    right_runs = [make_run("case-1", "passed"), make_run("case-3", "passed")]
    left = make_batch("batch-left", ["case-1", "case-2"])
    right = make_batch("batch-right", ["case-1", "case-3"])

    comparison = compare_evaluation_batches(left, left_runs, right, right_runs)

    self.assertEqual(comparison.shared_case_count, 1)
    self.assertEqual(comparison.improved_case_ids, ["case-1"])
    self.assertEqual(comparison.regressed_case_ids, [])
    self.assertEqual(comparison.left_only_case_ids, ["case-2"])
    self.assertEqual(comparison.right_only_case_ids, ["case-3"])
```

- [ ] **Step 2：运行测试并确认比较函数不存在**

Run: `py -m unittest tests.test_evaluation_batches.EvaluationBatchComparisonTest -v`

Expected: FAIL，`compare_evaluation_batches` 未定义。

- [ ] **Step 3：实现比较和失败原因分类纯函数**

定义 `EvaluationBatchComparisonModel`，字段包含左右批次 ID、整体指标差值、共享问题数量、改善、退化、仅左侧和仅右侧问题 ID。实现 `compare_evaluation_batches(left_batch, left_runs, right_batch, right_runs)`，按 `case_id` 建立映射并生成该模型。

实现：

```python
def evaluation_failure_reasons(run: EvaluationRunModel) -> list[str]:
    reasons: list[str] = []
    if run.false_positive:
        reasons.append("false_positive")
    if run.expect_answer and not run.answerable:
        reasons.append("no_hit")
    if run.missing_source_ids:
        reasons.append("missing_source")
    if run.missing_terms:
        reasons.append("missing_term")
    return reasons
```

- [ ] **Step 4：增加比较接口**

增加 `GET /api/admin/evaluations/batches/compare?left=...&right=...`。两个批次必须存在且状态为 `completed`，否则返回 409，并给出中文错误原因。

- [ ] **Step 5：运行批次测试与后端全量测试**

Run: `py -m unittest tests.test_evaluation_batches -v`

Expected: 比较和失败原因测试全部 PASS。

Run: `py -m unittest discover -s tests -v`

Expected: 后端全量测试 0 failures、0 errors。

---

### Task 8：增加管理端类型、API 和组合式函数

**Files：**

- Modify: `admin-frontend/src/types/chat.ts`
- Modify: `admin-frontend/src/services/api.ts`
- Create: `admin-frontend/src/composables/useQualityCases.ts`
- Create: `admin-frontend/src/composables/useQualityCases.spec.ts`
- Create: `admin-frontend/src/composables/useEvaluationBatches.ts`
- Create: `admin-frontend/src/composables/useEvaluationBatches.spec.ts`
- Delete: `admin-frontend/src/composables/useQualityEvaluation.ts`
- Delete: `admin-frontend/src/composables/useQualityEvaluation.spec.ts`

- [ ] **Step 1：编写评测集组合式函数失败测试**

```typescript
it('previews and confirms an evaluation import', async () => {
  vi.mocked(previewEvaluationImport).mockResolvedValue({
    previewToken: 'preview-1',
    fileName: 'cases.csv',
    totalRows: 2,
    validRows: 2,
    invalidRows: 0,
    duplicateRows: 0,
    rows: [],
    errors: [],
    duplicateKeys: [],
  })
  vi.mocked(confirmEvaluationImport).mockResolvedValue({
    importBatchId: 'eval-import-1',
    createdCount: 2,
    duplicateCount: 0,
    dashboard: { cases: [], runs: [] },
  })
  const quality = useQualityCases()
  const file = new File(['content'], 'cases.csv', { type: 'text/csv' })

  await quality.previewImport(file)
  await quality.confirmImport()

  expect(previewEvaluationImport).toHaveBeenCalledWith(file)
  expect(confirmEvaluationImport).toHaveBeenCalledWith('preview-1')
  expect(quality.importPreview.value).toBeNull()
})
```

- [ ] **Step 2：编写批次轮询失败测试**

使用 `vi.useFakeTimers()`，让 `fetchEvaluationBatch()` 依次返回 `running` 和 `completed`，验证 `useEvaluationBatches.startPolling()` 在完成后停止定时器并更新详情。

- [ ] **Step 3：运行测试并确认模块不存在**

Run: `npm.cmd run test:run -- src/composables/useQualityCases.spec.ts src/composables/useEvaluationBatches.spec.ts`

Expected: FAIL，两个组合式函数模块不存在。

- [ ] **Step 4：增加 TypeScript 类型**

扩展 `EvaluationCase`：

```typescript
category: string | null
tags: readonly string[]
externalKey: string | null
importBatchId: string | null
```

新增 `EvaluationCaseCollection`、`EvaluationImportPreview`、`EvaluationImportConfirmResult`、`EvaluationBatch`、`EvaluationBatchSummary`、`EvaluationBatchDetail`、`EvaluationBatchComparison` 和失败原因联合类型。

- [ ] **Step 5：增加 Axios API 函数**

实现：

```typescript
export async function previewEvaluationImport(file: File) {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await http.post<EvaluationImportPreview>(
    '/admin/evaluations/import/preview',
    formData,
  )
  return data
}

export async function confirmEvaluationImport(previewToken: string) {
  const { data } = await http.post<EvaluationImportConfirmResult>(
    '/admin/evaluations/import/confirm',
    { previewToken },
  )
  return data
}
```

同时增加筛选问题、创建批次、批次列表、批次详情和批次比较函数。

- [ ] **Step 6：实现两个组合式函数**

`useQualityCases` 管理问题、资料、筛选、选择、导入预览、确认、创建和删除；`useEvaluationBatches` 管理批次列表、详情、比较和轮询。所有公开状态使用 `readonly()`，可派生状态使用 `computed()`，轮询在 `onScopeDispose()` 中清理。

- [ ] **Step 7：运行组合式函数测试**

Run: `npm.cmd run test:run -- src/composables/useQualityCases.spec.ts src/composables/useEvaluationBatches.spec.ts`

Expected: 新增组合式函数测试全部 PASS。

---

### Task 9：拆分质量评测路由并实现评测集导入页面

**Files：**

- Modify: `admin-frontend/src/router/index.ts`
- Modify: `admin-frontend/src/components/layout/AdminLayout.vue`
- Create: `admin-frontend/src/views/QualityModuleLayout.vue`
- Create: `admin-frontend/src/views/QualityCasesPage.vue`
- Create: `admin-frontend/src/components/evaluation/EvaluationCaseToolbar.vue`
- Create: `admin-frontend/src/components/evaluation/EvaluationImportDialog.vue`
- Create: `admin-frontend/src/components/evaluation/EvaluationBatchDialog.vue`
- Modify: `admin-frontend/src/components/evaluation/EvaluationCaseList.vue`
- Modify: `admin-frontend/src/components/evaluation/EvaluationCaseFormDialog.vue`
- Create: `admin-frontend/src/views/__tests__/QualityCasesPage.spec.ts`
- Delete: `admin-frontend/src/views/QualityEvaluationPage.vue`
- Delete: `admin-frontend/src/views/__tests__/QualityEvaluationPage.spec.ts`

- [ ] **Step 1：编写路由和空状态失败测试**

`QualityCasesPage.spec.ts` 验证：

```typescript
expect(wrapper.find('[data-testid="quality-cases-page"]').exists()).toBe(true)
expect(wrapper.text()).toContain('评测集')
expect(wrapper.text()).toContain('手工创建')
expect(wrapper.text()).toContain('导入文件')
expect(wrapper.text()).not.toContain('差旅票据材料需要什么')
```

另外验证没有问题时“运行评测”按钮禁用。

- [ ] **Step 2：运行页面测试并确认页面不存在**

Run: `npm.cmd run test:run -- src/views/__tests__/QualityCasesPage.spec.ts`

Expected: FAIL，`QualityCasesPage.vue` 不存在。

- [ ] **Step 3：拆分 Vue Router 路由**

将质量评测路由改为：

```typescript
{
  path: 'quality',
  component: () => import('@/views/QualityModuleLayout.vue'),
  children: [
    { path: '', redirect: { name: 'quality-cases' } },
    {
      path: 'cases',
      name: 'quality-cases',
      component: () => import('@/views/QualityCasesPage.vue'),
      meta: { title: '质量评测' },
    },
    {
      path: 'reports',
      name: 'quality-reports',
      component: () => import('@/views/QualityReportsPage.vue'),
      meta: { title: '评测报告' },
    },
    {
      path: 'reports/:batchId',
      name: 'quality-report-detail',
      component: () => import('@/views/QualityReportDetailPage.vue'),
      meta: { title: '评测报告详情' },
    },
  ],
}
```

`AdminLayout.vue` 通过 `$route.path.startsWith('/quality')` 保持质量评测导航项激活。

- [ ] **Step 4：实现质量评测模块布局**

`QualityModuleLayout.vue` 使用两个文本标签页“评测集”和“评测报告”，内部只放 `<RouterView />`。标签页为页面导航，不使用卡片嵌套，也不增加英文副标题。

- [ ] **Step 5：实现工具栏和问题多选**

`EvaluationCaseToolbar.vue` 使用下拉选择分类、标签、答案预期和状态；提供“导入文件”“手工创建”“运行评测”按钮。`EvaluationCaseList.vue` 增加复选框和 `selection-change` 事件，列表展示分类和标签但不展示不存在的空占位。

- [ ] **Step 6：实现导入弹窗**

弹窗状态依次为文件选择、解析中、预览、确认中和完成。预览表格显示行号、问题、答案预期、分类、标签、状态；错误行显示字段和中文原因。确认按钮仅在 `validRows > 0` 时可用。

文件选择限制：

```html
<input type="file" accept=".xlsx,.csv,.json" data-testid="evaluation-import-file">
```

空白模板下载只生成表头和说明，不包含模拟问题。

- [ ] **Step 7：实现评测集页面组合**

`QualityCasesPage.vue` 只负责组合工具栏、列表、现有诊断组件和三个弹窗。筛选与导入状态来自 `useQualityCases`，批次创建来自 `useEvaluationBatches`。

- [ ] **Step 8：运行评测集页面和已有 UI 测试**

Run: `npm.cmd run test:run -- src/views/__tests__/QualityCasesPage.spec.ts src/components/ui/__tests__/base-ui.spec.ts`

Expected: 页面、空状态、导入和基础组件测试全部 PASS。

---

### Task 10：实现报告列表、报告详情和批次对比页面

**Files：**

- Create: `admin-frontend/src/views/QualityReportsPage.vue`
- Create: `admin-frontend/src/views/QualityReportDetailPage.vue`
- Create: `admin-frontend/src/components/evaluation/EvaluationBatchList.vue`
- Create: `admin-frontend/src/components/evaluation/EvaluationReportSummary.vue`
- Create: `admin-frontend/src/components/evaluation/EvaluationFailureList.vue`
- Create: `admin-frontend/src/views/__tests__/QualityReportsPage.spec.ts`
- Create: `admin-frontend/src/views/__tests__/QualityReportDetailPage.spec.ts`

- [ ] **Step 1：编写报告列表和详情失败测试**

`QualityReportsPage.spec.ts` 验证批次名称、状态、进度、阈值和时间；`QualityReportDetailPage.spec.ts` 验证整体通过率、无答案准确率、误召回、资料召回、关键词召回，以及关键词、向量、综合三类命中分数。

```typescript
expect(wrapper.text()).toContain('首轮通用评测')
expect(wrapper.text()).toContain('无答案准确率')
expect(wrapper.text()).toContain('误召回')
expect(wrapper.text()).toContain('关键词 10.00')
expect(wrapper.text()).toContain('向量 0.61')
expect(wrapper.text()).toContain('综合 12.42')
```

- [ ] **Step 2：运行测试并确认页面不存在**

Run: `npm.cmd run test:run -- src/views/__tests__/QualityReportsPage.spec.ts src/views/__tests__/QualityReportDetailPage.spec.ts`

Expected: FAIL，两个页面模块不存在。

- [ ] **Step 3：实现批次列表**

`EvaluationBatchList.vue` 使用表格式列表展示名称、状态、进度、通过率、误召回、阈值和完成时间。操作列使用查看图标并保持单元格居中；移动端改为纵向信息行，不产生页面级横向溢出。

- [ ] **Step 4：实现汇总和失败案例组件**

`EvaluationReportSummary.vue` 使用紧凑指标条展示报告指标，不使用嵌套卡片。`EvaluationFailureList.vue` 提供“全部、资料缺失、关键词缺失、无命中、误召回”筛选，点击案例发出 `select-run` 事件。

- [ ] **Step 5：实现报告列表和批次对比**

`QualityReportsPage.vue` 支持选择左右两个已完成批次。选择完成后调用比较接口，展示整体指标差值，以及改善、退化、仅左侧、仅右侧数量。没有批次时显示空状态，不创建示例批次。

- [ ] **Step 6：实现报告详情组合**

`QualityReportDetailPage.vue` 根据 `route.params.batchId` 加载详情，组合汇总、失败案例和 `EvaluationRunDetail.vue`。轮询中的批次显示实际进度，完成或失败后停止轮询。

- [ ] **Step 7：运行报告页面测试**

Run: `npm.cmd run test:run -- src/views/__tests__/QualityReportsPage.spec.ts src/views/__tests__/QualityReportDetailPage.spec.ts`

Expected: 报告列表、比较、失败筛选和详情测试全部 PASS。

---

### Task 11：全量回归、构建和浏览器验证

**Files：**

- Modify only when verification exposes a confirmed defect.

- [ ] **Step 1：运行后端全量测试**

Run: `py -m unittest discover -s tests -v`

Workdir: `D:\project\DC-Agent\backend`

Expected: 0 failures、0 errors；原有单条评测、问答、知识库上传和 Agent 测试保持通过。

- [ ] **Step 2：运行管理端全量测试**

Run: `npm.cmd run test:run`

Workdir: `D:\project\DC-Agent\admin-frontend`

Expected: 0 failed test files、0 failed tests。

- [ ] **Step 3：运行管理端生产构建**

Run: `npm.cmd run build`

Workdir: `D:\project\DC-Agent\admin-frontend`

Expected: `vue-tsc` 0 errors，Vite build exit code 0。

- [ ] **Step 4：确认空环境没有死数据**

Run:

```powershell
$evaluations = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/api/admin/evaluations' -Method Get
[pscustomobject]@{
  cases = @($evaluations.cases).Count
  runs = @($evaluations.runs).Count
}
```

Expected: 在未手工导入的干净环境中 `cases = 0`、`runs = 0`。

- [ ] **Step 5：桌面端浏览器检查**

打开 `http://127.0.0.1:5174/quality/cases`，验证空状态、手工创建、文件导入、筛选和批次弹窗；导入一个只含测试夹具数据的临时文件，确认预览阶段数据库仍为空，确认后数据正确出现。

- [ ] **Step 6：报告链路浏览器检查**

运行临时评测批次，验证进度、完成指标、失败案例筛选、分数诊断和批次比较。测试结束后通过管理接口删除临时评测问题和批次记录。

- [ ] **Step 7：移动端和文字尺寸检查**

在 390x844 视口验证三个质量评测页面无页面级横向溢出，弹窗可滚动，操作按钮不互相遮挡；读取可见元素计算样式，确认常规文字不低于 12px。

- [ ] **Step 8：最终服务状态检查**

Run:

```powershell
$health = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/api/health' -Method Get
$health.status
```

Expected: `ok`。

---

## 完成标准

- 正式和本地业务环境不会自动出现评测演示数据。
- 管理员可以预览并导入 XLSX、CSV、JSON 评测集。
- 分类、标签、答案预期和运行状态可以筛选。
- 批次在后台运行并展示可靠进度。
- 评测阈值被记录且只影响该批次。
- 报告展示通过率、无答案准确率、误召回率、资料召回、关键词召回和分数诊断。
- 两个批次可以比较整体指标和共享问题状态变化。
- 用户侧应用不出现资料原文、检索诊断或评测入口。
- 后端全量测试、管理端全量测试、生产构建和桌面/移动端浏览器验证全部通过。
