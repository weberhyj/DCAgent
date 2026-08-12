# Yarn 4 前端工作区迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将两个 Vue 前端从 pnpm workspace 迁移到 Yarn 4 的 `node_modules` linker，修复 Ubuntu 上旧 Yarn `fsevents`/peer 依赖兼容问题。

**Architecture:** 根目录保留一个 Yarn workspace，工作区为 `frontend` 和 `admin-frontend`。Yarn 4.9.2 通过 `.yarnrc.yml` 与项目内 release 文件固定，使用 `node_modules` linker；前端依赖和业务代码不改，只迁移包管理入口、锁文件、契约测试和部署文档。

**Tech Stack:** Node.js >=20.19, Yarn 4.9.2, Vue 3, Vite 7, TypeScript 5, Vitest 4, Ubuntu + Corepack。

## Global Constraints

- 使用 Yarn，不再使用 pnpm 管理前端。
- 固定 Yarn 4.9.2 和 `nodeLinker: node-modules`。
- 不修改前端业务逻辑。
- 不提交 `node_modules/`、`dist/`、`.env` 或 Yarn cache。
- Ubuntu 构建必须支持 `/admin/` 子路径。

---

### Task 1: 固定 Yarn 4 workspace 基础配置

**Files:**
- Create: `.yarnrc.yml`
- Create: `.yarn/releases/yarn-4.9.2.cjs`
- Modify: `package.json`
- Delete: `pnpm-workspace.yaml`

- [ ] **Step 1: Write the failing contract test**

更新 `tools/tests/test_frontend_pnpm_contract.py` 为 Yarn 契约测试，断言根清单包含 `packageManager: yarn@4.9.2`，`.yarnrc.yml` 使用 `nodeLinker: node-modules`，两个工作区存在，并且 pnpm 配置不存在。

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tools/tests/test_frontend_pnpm_contract.py -q`
Expected: FAIL，因为当前仓库仍声明 pnpm 并存在 pnpm workspace 文件。

- [ ] **Step 3: Implement minimal configuration**

将根脚本改为：

```json
"build": "yarn workspaces foreach run build",
"test": "yarn workspaces foreach run test:run"
```

新增：

```yaml
nodeLinker: node-modules
yarnPath: .yarn/releases/yarn-4.9.2.cjs
packageExtensions:
  "@floating-ui/vue@*":
    peerDependencies:
      vue: "*"
```

用 Corepack 获取 Yarn 4.9.2 release 文件，删除 pnpm workspace 配置。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tools/tests/test_frontend_pnpm_contract.py -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add package.json .yarnrc.yml .yarn pnpm-workspace.yaml tools/tests/test_frontend_pnpm_contract.py
git commit -m "chore: configure yarn 4 frontend workspace"
```

### Task 2: 生成 Yarn 锁文件并修正 peer 依赖契约

**Files:**
- Create: `yarn.lock`
- Delete: `pnpm-lock.yaml`
- Modify: `frontend/package.json`
- Modify: `admin-frontend/package.json`
- Modify: `tools/tests/test_version_contract.py`

- [ ] **Step 1: Add failing peer contract assertions**

在包管理器契约测试中断言两个前端 workspace 显式提供 `vue`，并断言 Yarn lock 文件存在、pnpm lock 文件不存在。

- [ ] **Step 2: Run focused tests to verify RED**

Run: `python -m pytest tools/tests/test_frontend_pnpm_contract.py tools/tests/test_version_contract.py -q`
Expected: FAIL，因为当前仍使用 pnpm lock，且 root peer 供给尚未声明。

- [ ] **Step 3: Implement lock and peer declarations**

在两个前端的 `devDependencies` 增加与当前 Vue 兼容的 `vue` peer 供给声明（保持现有 `dependencies.vue` 不变），执行 `corepack yarn install` 生成 Yarn 4 lock 文件，确认 `fsevents` 在 Linux 为 optional Darwin-only 包。

- [ ] **Step 4: Run focused tests and dependency inspection**

Run:

```bash
python -m pytest tools/tests/test_frontend_pnpm_contract.py tools/tests/test_version_contract.py -q
corepack yarn why vue
corepack yarn why vue-demi
```

Expected: PASS，且依赖链解析到单一 Vue 版本。

- [ ] **Step 5: Commit**

```bash
git add yarn.lock frontend/package.json admin-frontend/package.json pnpm-lock.yaml tools/tests/test_frontend_pnpm_contract.py tools/tests/test_version_contract.py
git commit -m "chore: migrate frontend dependencies to yarn lockfile"
```

### Task 3: 迁移脚本、部署文档与 smoke 入口

**Files:**
- Modify: `README.md`
- Modify: `deploy/ubuntu/ADMIN_FRONTEND_SUBPATH.md`
- Modify: `docs/offline-platform-runbook.md`
- Modify: `tools/start_smoke_frontend.cmd`
- Modify: `tools/start_smoke_admin.cmd`
- Modify: `tools/tests/test_frontend_pnpm_contract.py`

- [ ] **Step 1: Extend failing command contract tests**

断言 README、部署文档和两个 smoke 脚本只使用 Yarn 命令，不能出现 pnpm 构建/启动命令。

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tools/tests/test_frontend_pnpm_contract.py -q`
Expected: FAIL，因为当前文档和 smoke 脚本仍使用 pnpm。

- [ ] **Step 3: Replace commands**

使用以下命令格式：

```bash
corepack yarn install --immutable
corepack yarn workspace dc-agent-frontend dev
corepack yarn workspace dc-agent-admin-frontend dev
corepack yarn workspace dc-agent-frontend build
VITE_ADMIN_BASE_PATH=/admin/ corepack yarn workspace dc-agent-admin-frontend build
```

Windows smoke 脚本使用 `corepack yarn.cmd workspace ...`，不使用 pnpm 或 npm。

- [ ] **Step 4: Run command contract tests**

Run: `python -m pytest tools/tests/test_frontend_pnpm_contract.py -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add README.md deploy/ubuntu/ADMIN_FRONTEND_SUBPATH.md docs/offline-platform-runbook.md tools/start_smoke_frontend.cmd tools/start_smoke_admin.cmd tools/tests/test_frontend_pnpm_contract.py
git commit -m "docs: switch frontend deployment commands to yarn"
```

### Task 4: 完整安装、构建和测试验证

**Files:** None (verification only)

- [ ] **Step 1: Clean generated dependencies**

```bash
rm -rf node_modules frontend/node_modules admin-frontend/node_modules
```

- [ ] **Step 2: Install from immutable Yarn lock**

```bash
corepack yarn install --immutable
```

- [ ] **Step 3: Run all frontend tests**

```bash
corepack yarn workspaces foreach run test:run
```

- [ ] **Step 4: Build user frontend and admin subpath frontend**

```bash
corepack yarn workspace dc-agent-frontend build
VITE_ADMIN_BASE_PATH=/admin/ corepack yarn workspace dc-agent-admin-frontend build
```

- [ ] **Step 5: Run repository contracts and diff checks**

```bash
python -m pytest tools/tests/test_frontend_pnpm_contract.py tools/tests/test_version_contract.py -q
git diff --check
git status --short
```

- [ ] **Step 6: Commit final verification adjustments if needed**

Only commit if a test or documentation correction was required; do not commit generated `node_modules` or `dist` output.
