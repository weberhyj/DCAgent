# fast lint --ty 全量修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让仓库根目录执行 `fast lint --ty` 时，Ruff 格式化、Ruff `I/B/SIM` 检查和 `ty` 类型检查全部以退出码 0 完成。

**Architecture:** 保留后端 `backend/pyproject.toml` 作为后端依赖与基础 Ruff 配置，同时增加仓库根目录的静态检查入口配置，使 `fast` 从根目录扫描时使用统一规则。第三方库解析通过后端 `.venv` 和项目开发依赖完成；类型问题按模块分组修复，测试与工具代码也纳入检查范围。

**Tech Stack:** Python 3.12、Fast Dev CLI、Ruff、ty、uv、pytest、FastAPI、LangGraph。

---

### Task 1: 建立可重复的 lint 契约测试

**Files:**
- Create: `tools/tests/test_fast_lint_contract.py`
- Modify: `docs/superpowers/plans/2026-08-07-fast-lint-ty.md`

- [ ] **Step 1: 写失败测试**

断言仓库根目录存在根级静态检查配置，并声明后端 Python 3.12 环境路径与 Ruff 规则入口。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run --project backend python -m pytest tools/tests/test_fast_lint_contract.py -q`

Expected: FAIL，因为根级配置和契约测试尚不存在。

- [ ] **Step 3: 实现最小配置**

添加根级 `pyproject.toml` 的 Ruff/ty 配置，并让测试只验证配置存在、路径可解析和规则集合明确。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run --project backend python -m pytest tools/tests/test_fast_lint_contract.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml tools/tests/test_fast_lint_contract.py
git commit -m "test: add fast lint contract"
```

### Task 2: 统一 Ruff 根目录检查范围并自动修复机械问题

**Files:**
- Create: `pyproject.toml`
- Modify: Python files reported by `ruff check --extend-select=I,B,SIM --fix .`

- [ ] **Step 1: 运行失败基线**

Run: `fast lint --ty --check-only --skip-mypy`

Expected: FAIL，记录 Ruff 的全部规则和文件清单。

- [ ] **Step 2: 写配置回归断言**

在 `tools/tests/test_fast_lint_contract.py` 中断言 Ruff 排除生成目录、保留 `I/B/SIM` 检查，并使用 Python 3.12。

- [ ] **Step 3: 执行最小机械修复**

运行 `ruff format .` 和 `ruff check --extend-select=I,B,SIM --fix .`，只保留自动修复结果；无法自动修复的规则逐项修改。

- [ ] **Step 4: 验证 Ruff**

Run: `fast lint --ty --check-only --skip-mypy`

Expected: Ruff 阶段退出码 0，并进入 ty 阶段或因显式跳过而正常结束。

### Task 3: 修复 ty 的项目环境与第三方导入解析

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`
- Modify: `tools/tests/test_fast_lint_contract.py`

- [ ] **Step 1: 写失败测试**

断言 `ty` 运行时使用 `backend/.venv`/Python 3.12，且开发依赖包含 `ty`。

- [ ] **Step 2: 验证当前失败**

Run: `ty check --python backend/.venv .`

Expected: FAIL，至少包含第三方导入无法解析或类型环境未配置诊断。

- [ ] **Step 3: 添加依赖与配置**

把 `ty` 放入后端 dev 依赖，重新生成锁文件；配置类型检查的 Python 版本、源码根目录和明确排除目录。

- [ ] **Step 4: 验证导入解析**

Run: `uv sync --project backend --group dev`；然后 `ty check --python backend/.venv backend/app`。

Expected: 不再出现因环境缺失导致的 `unresolved-import`。

### Task 4: 按模块修复 ty 类型诊断

**Files:**
- Modify: `backend/app/*.py` 报告文件
- Modify: `backend/tests/*.py` 报告文件
- Modify: `tools/**/*.py` 报告文件

- [ ] **Step 1: 生成机器可读诊断清单**

Run: `ty check --python backend/.venv --output-format json .`，按 `unresolved-*`、`invalid-*`、`not-*` 和测试窄化问题分类。

- [ ] **Step 2: 每个模块先补回归测试**

对生产代码的每类类型错误增加最小测试或使用现有测试覆盖的失败断言，确保修复不是单纯压制诊断。

- [ ] **Step 3: 修复生产代码类型契约**

优先修复公共协议、可选值窄化、泛型返回值、回调签名和第三方调用边界；只在第三方库确实缺少类型信息时使用最窄范围的 `# type: ignore[rule]`。

- [ ] **Step 4: 修复测试和工具代码类型契约**

为闭包绑定循环变量、补充 fake 工厂返回类型、修正 `Mapping`/联合类型窄化，并保持测试行为不变。

- [ ] **Step 5: 分组验证**

每完成一个模块运行对应 pytest 文件和 `ty check --python backend/.venv <module>`；所有模块完成后运行整套 Python 测试。

### Task 5: 原始命令验收与文档

**Files:**
- Modify: `README.md`
- Modify: `docs/intranet-deployment-configuration.md`

- [ ] **Step 1: 运行原始命令**

Run: `fast lint --ty`

Expected: Ruff format、Ruff check 和 ty check 全部成功，退出码 0。

- [ ] **Step 2: 运行完整验证**

Run: `uv run --project backend python -m pytest -q`、`git diff --check`。

Expected: 测试无新增失败，补充说明已有环境门禁。

- [ ] **Step 3: 更新使用说明**

在 README 中记录 `fast lint --ty` 的 Windows 开发机用法、`uv sync --group dev` 前置条件和 Ubuntu 上的等价 `uv run` 命令。

- [ ] **Step 4: 提交**

```bash
git add pyproject.toml backend/pyproject.toml backend/uv.lock backend tools README.md docs
git commit -m "fix: make fast lint ty pass"
```
