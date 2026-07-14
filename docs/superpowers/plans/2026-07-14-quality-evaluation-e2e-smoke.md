# 质量评测工作台 E2E 冒烟验收实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在完全隔离的临时 SQLite 环境中，用 Playwright 自动验证评测集导入、批次运行、报告详情、批次比较和移动端布局，并把该流程纳入现有 UI 冒烟脚本。

**Architecture:** 继续使用 `8015/5177/5178` 三个临时服务端口和内存 SQLite。`tools/ui_smoke.py` 通过页面完成真实管理员操作，并通过只读 HTTP 请求验证“预览不落库”和批次最终状态；所有测试数据随临时后端进程退出而销毁，不访问本地 PostgreSQL。新增一个轻量 `unittest` 契约测试，先锁定 CSV fixture 和 `main()` 必须调用质量验收流程，再实现 Playwright 流程。

**Tech Stack:** Python 3、`unittest`、Playwright Sync API、httpx、FastAPI、Vue 3、临时 SQLite。

---

## 文件结构

- Create: `tools/tests/__init__.py`：使工具测试可通过 `unittest` 模块路径运行。
- Create: `tools/tests/test_ui_smoke.py`：锁定评测 CSV fixture 和冒烟入口契约。
- Modify: `tools/ui_smoke.py`：增加评测工作台桌面端、报告链路、批次比较和移动端验证。
- Modify: `README.md`：记录新增冒烟覆盖范围和临时数据边界。

不修改 `backend/app`、`frontend/src` 或 `admin-frontend/src`。若浏览器验收暴露产品缺陷，停止本计划并为该缺陷单独执行系统化调试和 TDD。

### Task 1：先锁定质量冒烟测试契约

**Files:**

- Create: `tools/tests/__init__.py`
- Create: `tools/tests/test_ui_smoke.py`
- Test: `tools/tests/test_ui_smoke.py`

- [ ] **Step 1：创建测试包标记**

创建空文件：

```python
# tools/tests/__init__.py
```

- [ ] **Step 2：编写失败的契约测试**

创建 `tools/tests/test_ui_smoke.py`：

```python
from __future__ import annotations

import csv
import io
import unittest
from unittest.mock import patch

from tools import ui_smoke


class QualityEvaluationSmokeContractTest(unittest.TestCase):
    def test_builds_business_neutral_two_row_import_fixture(self) -> None:
        payload = ui_smoke.build_evaluation_import_csv()
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["expect_answer"], "true")
        self.assertEqual(rows[0]["expected_sources"], "travel-policy.txt")
        self.assertEqual(rows[0]["expected_terms"], "发票|行程单")
        self.assertEqual(rows[1]["expect_answer"], "false")
        self.assertEqual(rows[1]["expected_sources"], "")
        self.assertEqual(rows[1]["expected_terms"], "")
        self.assertNotEqual(rows[0]["external_key"], rows[1]["external_key"])

    def test_main_runs_quality_evaluation_verification(self) -> None:
        with (
            patch.object(ui_smoke, "verify_user_app") as verify_user,
            patch.object(ui_smoke, "verify_admin_app") as verify_admin,
            patch.object(ui_smoke, "verify_quality_app") as verify_quality,
            patch.object(ui_smoke, "SCREENSHOT_DIR") as screenshot_dir,
        ):
            ui_smoke.main()

        verify_user.assert_called_once_with()
        verify_admin.assert_called_once_with()
        verify_quality.assert_called_once_with()
        screenshot_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3：运行测试并确认按预期失败**

Run:

```powershell
py -m unittest tools.tests.test_ui_smoke -v
```

Expected: FAIL，错误指出 `tools.ui_smoke` 缺少 `build_evaluation_import_csv` 或 `verify_quality_app`。如果测试因导入错误失败，先修正测试环境，直到它因缺少目标行为而失败。

### Task 2：实现隔离的质量评测浏览器流程

**Files:**

- Modify: `tools/ui_smoke.py`
- Test: `tools/tests/test_ui_smoke.py`

- [ ] **Step 1：增加确定性 CSV fixture**

在 `tools/ui_smoke.py` 顶部增加：

```python
import csv
import io
```

在 URL 常量之后增加：

```python
QUALITY_BATCH_BASELINE = "E2E 基线批次"
QUALITY_BATCH_STRICT = "E2E 严格阈值批次"


def build_evaluation_import_csv() -> bytes:
    fieldnames = [
        "question",
        "expect_answer",
        "expected_sources",
        "expected_terms",
        "category",
        "tags",
        "top_k",
        "external_key",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows([
        {
            "question": "差旅票据材料需要什么",
            "expect_answer": "true",
            "expected_sources": "travel-policy.txt",
            "expected_terms": "发票|行程单",
            "category": "流程",
            "tags": "差旅|票据",
            "top_k": "5",
            "external_key": "e2e-answerable-001",
        },
        {
            "question": "量子咖啡机审批制度是什么",
            "expect_answer": "false",
            "expected_sources": "",
            "expected_terms": "",
            "category": "边界",
            "tags": "无答案",
            "top_k": "5",
            "external_key": "e2e-no-answer-001",
        },
    ])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")
```

- [ ] **Step 2：运行契约测试，确认 fixture 测试转绿而入口测试仍为红色**

Run:

```powershell
py -m unittest tools.tests.test_ui_smoke -v
```

Expected: fixture 测试 PASS；入口测试仍因缺少 `verify_quality_app` 而 FAIL。

- [ ] **Step 3：实现质量评测工作台真实流程**

在 `tools/ui_smoke.py` 的 `verify_admin_app()` 之后增加 `verify_quality_app()`。实现必须包含以下完整行为：

```python
def verify_quality_app() -> None:
    def wait_for_batch(client: httpx.Client, name: str) -> dict:
        for _ in range(100):
            batches = client.get(f"{BACKEND_URL}/api/admin/evaluations/batches").json()
            batch = next((item for item in batches if item["name"] == name), None)
            if batch and batch["status"] == "completed":
                return batch
            if batch and batch["status"] == "failed":
                raise AssertionError(f"Evaluation batch failed: {batch.get('errorMessage')}")
            sleep(0.1)
        raise AssertionError(f"Evaluation batch did not complete: {name}")

    def start_batch(page: Page, name: str, minimum_score: str) -> None:
        page.locator('[data-testid="run-evaluation-batch"]').click()
        expect(page.locator('[data-testid="evaluation-batch-form"]')).to_be_visible()
        page.locator('[data-testid="evaluation-batch-name"]').fill(name)
        page.locator('[data-testid="evaluation-retrieval-min-score"]').fill(minimum_score)
        page.locator('[data-testid="submit-evaluation-batch"]').click()
        expect(page.get_by_text(f"批次“{name}”已开始运行，进度将自动刷新。")).to_be_visible(timeout=10_000)

    with httpx.Client(timeout=10.0) as client:
        dashboard = client.get(f"{BACKEND_URL}/api/admin/evaluations").json()
        batches = client.get(f"{BACKEND_URL}/api/admin/evaluations/batches").json()
        if dashboard["cases"] or dashboard["runs"] or batches:
            raise AssertionError("Temporary evaluation database was not empty before smoke flow")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors = watch_errors(page, "quality")
            page.goto(f"{ADMIN_URL}/quality/cases", wait_until="networkidle")
            expect(page.locator('[data-testid="quality-cases-page"]')).to_be_visible(timeout=20_000)
            expect(page.get_by_text("评测集为空", exact=True)).to_be_visible()

            page.locator('[data-testid="open-evaluation-import"]').click()
            expect(page.locator('[data-testid="evaluation-import-dialog"]')).to_be_visible()
            page.locator('[data-testid="evaluation-import-file"]').set_input_files({
                "name": "quality-e2e.csv",
                "mimeType": "text/csv",
                "buffer": build_evaluation_import_csv(),
            })
            expect(page.locator(".preview-summary")).to_contain_text("有效 2 行", timeout=10_000)

            preview_dashboard = client.get(f"{BACKEND_URL}/api/admin/evaluations").json()
            if preview_dashboard["cases"] or preview_dashboard["runs"]:
                raise AssertionError("Evaluation preview persisted data before confirmation")

            page.locator('[data-testid="confirm-evaluation-import"]').click()
            expect(page.get_by_text("导入完成", exact=True)).to_be_visible(timeout=10_000)
            expect(page.get_by_text("成功创建 2 条，重复 0 条。", exact=True)).to_be_visible()
            page.get_by_role("button", name="完成", exact=True).click()
            expect(page.locator('[data-testid="evaluation-case-counts"]')).to_contain_text("共 2 项")

            page.locator('[data-testid="select-visible-evaluation-cases"]').check()
            start_batch(page, QUALITY_BATCH_BASELINE, "0")
            baseline = wait_for_batch(client, QUALITY_BATCH_BASELINE)
            start_batch(page, QUALITY_BATCH_STRICT, "100")
            strict = wait_for_batch(client, QUALITY_BATCH_STRICT)

            page.goto(f"{ADMIN_URL}/quality/reports", wait_until="networkidle")
            expect(page.locator('[data-testid="quality-reports-page"]')).to_be_visible(timeout=20_000)
            expect(page.get_by_text(QUALITY_BATCH_BASELINE, exact=True).first).to_be_visible()
            expect(page.get_by_text(QUALITY_BATCH_STRICT, exact=True).first).to_be_visible()

            page.locator(f'[data-testid="view-evaluation-batch-{strict["id"]}"]').click()
            expect(page.locator('[data-testid="quality-report-detail-page"]')).to_be_visible()
            expect(page.get_by_text("整体通过率", exact=True)).to_be_visible()
            expect(page.get_by_text("无答案准确率", exact=True)).to_be_visible()
            expect(page.get_by_text("差旅票据材料需要什么", exact=True)).to_be_visible()

            page.goto(f"{ADMIN_URL}/quality/reports", wait_until="networkidle")
            page.locator('[aria-label="选择左侧评测批次"]').click()
            page.locator(f'[data-testid="base-select-option-{baseline["id"]}"]').click()
            page.locator('[aria-label="选择右侧评测批次"]').click()
            page.locator(f'[data-testid="base-select-option-{strict["id"]}"]').click()
            expect(page.locator(".comparison-results")).to_be_visible(timeout=10_000)
            expect(page.get_by_text("共享案例 2", exact=True)).to_be_visible()
            page.screenshot(path=str(SCREENSHOT_DIR / "runtime-quality-reports.png"), full_page=True)

            mobile = browser.new_page(viewport={"width": 390, "height": 844})
            mobile_errors = watch_errors(mobile, "quality-mobile")
            for path, test_id, screenshot in (
                ("/quality/cases", "quality-cases-page", "runtime-quality-mobile-cases.png"),
                ("/quality/reports", "quality-reports-page", "runtime-quality-mobile-reports.png"),
            ):
                mobile.goto(f"{ADMIN_URL}{path}", wait_until="networkidle")
                expect(mobile.locator(f'[data-testid="{test_id}"]')).to_be_visible(timeout=20_000)
                width = mobile.evaluate("""() => ({
                    client: document.documentElement.clientWidth,
                    scroll: document.documentElement.scrollWidth,
                })""")
                if width["scroll"] > width["client"] + 1:
                    raise AssertionError(f"{path} has horizontal overflow: {width}")
                mobile.screenshot(path=str(SCREENSHOT_DIR / screenshot), full_page=True)

            if errors or mobile_errors:
                raise AssertionError("\n".join([*errors, *mobile_errors]))
            mobile.close()
            page.close()
            browser.close()
```

- [ ] **Step 4：把质量流程接入主入口**

将 `main()` 调整为：

```python
def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    verify_user_app()
    verify_admin_app()
    verify_quality_app()
    print("UI smoke passed")
```

- [ ] **Step 5：运行契约测试并确认全部通过**

Run:

```powershell
py -m unittest tools.tests.test_ui_smoke -v
```

Expected: 2 tests PASS。

- [ ] **Step 6：提交测试与冒烟实现**

```powershell
git add tools/tests tools/ui_smoke.py
git commit -m "test: cover quality evaluation browser flow"
```

### Task 3：运行临时环境全链路验收

**Files:**

- Modify only if the browser run exposes a confirmed defect.

- [ ] **Step 1：确认浏览器测试辅助脚本用法**

Run:

```powershell
py "C:\Users\56252\.codex\skills\webapp-testing\scripts\with_server.py" --help
```

Expected: 显示 `--server`、`--port` 和后续测试命令用法。

- [ ] **Step 2：用临时端口运行完整冒烟**

优先使用测试辅助脚本启动三个现有命令：

```powershell
py "C:\Users\56252\.codex\skills\webapp-testing\scripts\with_server.py" `
  --server "tools\start_smoke_backend.cmd" --port 8015 `
  --server "tools\start_smoke_frontend.cmd" --port 5177 `
  --server "tools\start_smoke_admin.cmd" --port 5178 `
  -- py tools\ui_smoke.py
```

如果辅助脚本在 Windows 上不接受 `.cmd`，按 README 现有方式分别启动三个脚本，再运行 `py tools\ui_smoke.py`。不得把目标改为 `8000/5173/5174`，以免写入开发数据库。

Expected: 输出 `UI smoke passed`；生成桌面端质量报告和两个移动端质量页面截图；临时后端结束后所有评测数据自动销毁。

- [ ] **Step 3：运行全量回归**

Run:

```powershell
cd backend
py -m unittest discover -s tests -p "test_*.py"

cd ..\admin-frontend
npm.cmd run test:run -- --maxWorkers=2
npm.cmd run build

cd ..\frontend
npm.cmd run test:run -- --maxWorkers=2
npm.cmd run build
```

Expected: 后端、管理端和用户端测试 0 failures；两个生产构建 exit code 0。

### Task 4：记录验收范围并提交

**Files:**

- Modify: `README.md`
- Verify: `git diff --check`

- [ ] **Step 1：更新 README 冒烟说明**

在“页面级冒烟”段落补充：

```markdown
冒烟流程还会在临时 SQLite 环境中验证质量评测工作台：导入预览不会提前落库，确认后可创建评测案例，两个不同阈值的批次能够完成，并可查看报告详情、批次比较以及桌面端和 390×844 移动端布局。临时服务退出后测试数据自动销毁。
```

- [ ] **Step 2：检查差异和仓库状态**

Run:

```powershell
git diff --check
git status --short --branch
```

Expected: `git diff --check` 无输出；状态只包含本计划预期文件。

- [ ] **Step 3：提交文档**

```powershell
git add README.md
git commit -m "docs: document quality evaluation smoke coverage"
```

## 完成标准

- 评测导入预览阶段经 API 验证 `cases=0`、`runs=0`。
- 确认导入后页面和 API 均看到 2 条案例。
- 两个不同检索阈值的批次均完成，不出现永久 `queued/running`。
- 报告详情展示核心指标和失败案例。
- 批次比较展示完整批次与共享案例口径。
- `/quality/cases` 和 `/quality/reports` 在 390×844 下无页面级横向溢出。
- 页面无 console error 或 uncaught page error。
- 流程只访问 `8015/5177/5178`，不写入本地 PostgreSQL。
- 契约测试、全量测试和两个生产构建通过。
