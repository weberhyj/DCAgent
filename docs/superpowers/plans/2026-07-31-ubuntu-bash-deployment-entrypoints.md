# Ubuntu Bash 双入口实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 Ubuntu 20.04 Bash 生产部署入口，同时保留现有 PowerShell 开发兼容入口，并保持两条路线的安全契约一致。

**Architecture:** `prepare_offline_env.sh` 和 `invoke_offline_compose.sh` 只负责定位仓库并启动 Python 3。环境准备和 Compose 预检分别由 `tools/offline_env.py`、`tools/offline_compose.py` 实现；`compose_smoke.py` 根据操作系统选择 `.sh` 或 `.ps1`。

**Tech Stack:** Bash、Python 3.12 标准库、Docker Compose v2、unittest、Ruff、Git executable mode。

---

### Task 1：Bash 入口和文件契约

**Files:**
- Create: `.gitattributes`
- Create: `tools/prepare_offline_env.sh`
- Create: `tools/invoke_offline_compose.sh`
- Create: `tools/tests/test_ubuntu_deployment_entrypoints.py`

- [ ] 先写测试，断言两个脚本使用 LF、`#!/usr/bin/env bash`、`set -Eeuo pipefail`、不包含 `pwsh`，并调用对应 Python 文件。
- [ ] 运行测试，确认因为脚本不存在而失败。
- [ ] 创建两个最小 Bash 入口，并使用 `git update-index --chmod=+x` 记录 `100755`。
- [ ] 运行测试，确认通过。

入口固定内容：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "$SCRIPT_DIR/offline_env.py" "$@"
```

Compose 入口只把最后一行替换为：

```bash
exec python3 "$SCRIPT_DIR/offline_compose.py" "$@"
```

### Task 2：Ubuntu 环境准备核心

**Files:**
- Create: `tools/offline_env.py`
- Create: `tools/tests/test_offline_env.py`

- [ ] 先写环境解析、重复键、UID/GID、路径、secret 成对创建、轮换回滚和权限测试。
- [ ] 运行测试，确认模块缺失。
- [ ] 实现固定公开接口：`DeploymentError`、`load_env(path)`、`set_env_value(path, name, value)`、`resolve_env_path(env_path, name, raw_value, environ=os.environ)`、`prepare_environment(repo_root, rotate_secrets=False)` 和 `main(argv=None)`。

- [ ] 保持 `.env` 只在缺失时复制，已有 UID/GID 不匹配时失败。
- [ ] secret 固定在 `artifacts/secrets`，目录 `0700`、文件 `0600`，失败时恢复原集合。
- [ ] 拒绝 root、远程 Docker、非默认 context、符号链接和路径重定向。
- [ ] 运行单元测试和 Ruff。

### Task 3：Ubuntu Compose 安全核心

**Files:**
- Create: `tools/offline_compose.py`
- Create: `tools/tests/test_offline_compose.py`

- [ ] 先写危险参数、Docker endpoint、镜像 digest、网络、端口、bind 和 secret 渲染测试。
- [ ] 运行测试，确认模块缺失。
- [ ] 实现固定公开接口：`validate_compose_arguments(arguments)`、`assert_local_docker_environment(environ)`、`assert_rendered_compose(rendered, repo_root, environment)`、`run_compose(arguments, repo_root)` 和 `main(argv=None)`。

- [ ] 完整移植 PowerShell wrapper 的禁止参数、固定 context、全 profile JSON 预检、内部镜像、网络、端口、bind 和 secret 校验。
- [ ] 实际执行固定使用 `docker --context default compose --env-file deploy/offline/.env -f deploy/offline/compose.yaml`，通过独立环境副本清除 override。
- [ ] 运行新测试和原 PowerShell 契约测试。

### Task 4：跨平台 smoke wrapper 选择

**Files:**
- Modify: `tools/compose_smoke.py`
- Modify: `tools/tests/test_compose_smoke.py`

- [ ] 先写 POSIX 选择 `.sh`、Windows 选择 `.ps1` 的失败测试。
- [ ] 修改 `_wrapper_prefix()`：`.sh` 直接执行，`.ps1` 继续使用 `pwsh -NoProfile -NonInteractive -File`。
- [ ] 保留显式 wrapper path 覆盖。
- [ ] 运行完整 smoke 工具测试。

### Task 5：Ubuntu 文档和文档契约

**Files:**
- Modify: `docs/intranet-deployment-configuration.md`
- Modify: `docs/offline-platform-runbook.md`
- Modify: `deploy/offline/README.md`
- Modify: `README.md`
- Modify: `tools/tests/test_compose_contract.py`
- Modify: `tools/tests/test_physoc_llm_contract.py`
- Modify: `tools/tests/test_structured_deployment_contract.py`

- [ ] 先修改文档契约测试，要求公司内网清单不包含 `.ps1`、`Copy-Item`、`$LASTEXITCODE`、`New-Item` 和 PowerShell 续行符。
- [ ] 运行测试，确认当前 PowerShell 文档失败。
- [ ] 把 Ubuntu 生产命令统一改成：

```bash
./tools/prepare_offline_env.sh
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh up -d
./tools/invoke_offline_compose.sh --profile indexing up -d
```

- [ ] README 仅保留一段 Windows 开发机 `.ps1` 兼容说明。
- [ ] 运行 Physoc、结构化和 Compose 文档契约测试。

### Task 6：回归和目标机门禁

**Files:**
- Verify all files changed by Tasks 1-5.

- [ ] 运行 `bash -n` 和 Git executable mode 检查。
- [ ] 运行 `tools/tests` 全量测试。
- [ ] 运行 `backend/tests` 全量测试。
- [ ] 运行 Ruff、格式检查和 `git diff --check`。
- [ ] Ubuntu 20.04 目标机执行：

```bash
./tools/prepare_offline_env.sh
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh build \
  schema-migration embedding-service reranker-service api ingestion-worker
./tools/invoke_offline_compose.sh up -d
curl --fail --silent --show-error http://127.0.0.1:8000/api/readyz
```

本开发机没有目标 Docker 拓扑时，只记录“目标机门禁未运行”，不得声称通过。
