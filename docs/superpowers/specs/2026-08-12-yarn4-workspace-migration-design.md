# Yarn 4 前端工作区迁移设计

## 目标

将两个 Vue 前端从 pnpm workspace 迁移到 Yarn 4，统一 Ubuntu 与本地开发/构建方式，避免 Yarn 旧版 PnP/fsevents 兼容补丁导致的构建异常。

## 现状与根因

- 根目录通过 `package.json` 固定 `pnpm@11.16.0`，并使用 `pnpm-lock.yaml` 与 `pnpm-workspace.yaml`。
- Ubuntu 实际使用 Yarn Berry，错误来自 `fsevents@patch:...builtin<compat/fsevent>`，说明 Yarn 版本/安装模式与项目依赖树不匹配。
- `@floating-ui/vue` 通过 `vue-demi` 使用 Vue。Yarn 的 peer 依赖解析需要在根 workspace 明确提供 Vue，避免 `doesn't provide vue` 警告升级为安装/构建问题。

## 方案

采用 Yarn 4.9.2，并在 `.yarnrc.yml` 固定：

```yaml
nodeLinker: node-modules
yarnPath: .yarn/releases/yarn-4.9.2.cjs
packageExtensions:
  "@floating-ui/vue@*":
    peerDependencies:
      vue: "*"
```

根目录继续作为单一 workspace，工作区包含 `frontend` 与 `admin-frontend`。两个前端的 Vue、Vite 和测试依赖保持现有版本范围；只增加必要的 peer 供给声明，不修改业务代码。

迁移后：

- `package.json` 的 `packageManager` 改为 `yarn@4.9.2`。
- 新增 `yarn.lock`、`.yarnrc.yml` 和 Yarn release 文件。
- 删除 `pnpm-lock.yaml`、`pnpm-workspace.yaml` 及 pnpm 专用契约/命令引用。
- 根脚本改为 `yarn workspaces foreach run build/test:run`，Ubuntu 文档改为 `corepack yarn install --immutable` 和 Yarn workspace 命令。
- 安装使用 `node_modules` linker，Ubuntu 不使用 PnP；`fsevents` 仅作为 Darwin 可选依赖，不在 Linux 构建中执行。

## 验证

在干净依赖目录执行：

```bash
corepack yarn install --immutable
corepack yarn workspace dc-agent-frontend build
VITE_ADMIN_BASE_PATH=/admin/ corepack yarn workspace dc-agent-admin-frontend build
corepack yarn workspaces foreach run test:run
```

同时运行包管理器契约测试，确认仓库不再依赖 pnpm 配置或锁文件，且不存在 `package-lock.json`、Yarn PnP 产物不会被生成。

## 回滚

若迁移后构建失败，可在单个提交上回滚，恢复 `package.json` 的 pnpm 声明、pnpm workspace/锁文件和部署命令；不涉及数据库、后端或运行时数据。
