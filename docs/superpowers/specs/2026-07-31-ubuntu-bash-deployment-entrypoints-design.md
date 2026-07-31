# Ubuntu Bash 部署入口设计

## 目标

为 DC-Agent 增加 Ubuntu 20.04 Bash 部署入口，将 Ubuntu 作为生产部署的主要运行环境，同时保留现有 PowerShell 入口供 Windows 开发机兼容使用。

## 范围

本次修改包括：

- 新增 `tools/prepare_offline_env.sh`；
- 新增 `tools/invoke_offline_compose.sh`；
- 保留 `tools/prepare_offline_env.ps1` 和 `tools/invoke_offline_compose.ps1`；
- Ubuntu 20.04 生产部署文档默认使用 Bash；
- 从 `docs/intranet-deployment-configuration.md` 中移除所有 PowerShell 命令；
- 更新面向 Linux 的部署文档和 smoke 工具，使其在 POSIX 系统上选择 Bash wrapper；
- 保留当前 PowerShell 部署路线的安全校验和失败关闭行为。

本次修改不会删除 Windows 开发支持，不会放宽 Compose 校验，不会把直接运行 `docker compose` 作为受支持的生产方式，也不会改变系统部署拓扑。

## 架构

Bash 文件使用 `#!/usr/bin/env bash` 和 `set -Eeuo pipefail`，作为小型可执行入口。脚本必须能够在任意当前目录下正确定位仓库根目录，并调用 Python 3 辅助程序完成复杂校验。

Ubuntu 路线包含两个 Python 辅助程序：

- `tools/offline_env.py`：创建或校验 `deploy/offline/.env`，记录当前非 root Linux UID/GID，原子创建受管 secret 文件，校验所有者和权限，校验 bind 根目录，并支持显式轮换 secret。
- `tools/offline_compose.py`：校验 Compose 参数，强制使用本地 `default` Docker context，清除冲突的环境变量覆盖，渲染所有 profile 的 Compose JSON，校验镜像、网络、bind mount 和 secret 路径，通过后才执行 Docker Compose。

现有 PowerShell 脚本继续作为兼容入口。Ubuntu 不需要安装 PowerShell，Bash 路线内部也不得调用 `pwsh`。

## 命令约定

Ubuntu 环境准备命令：

```bash
./tools/prepare_offline_env.sh
./tools/prepare_offline_env.sh --rotate-secrets
```

Ubuntu 受控 Compose 命令：

```bash
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh up -d
./tools/invoke_offline_compose.sh --profile indexing up -d
./tools/invoke_offline_compose.sh exec -T api python -m app.physoc_probe
```

环境准备脚本收到未知参数时必须失败。Compose wrapper 没有收到 Compose 参数时必须失败。Compose wrapper 必须完整保留参数边界，并拒绝 PowerShell 实现当前已经禁止的危险命令和选项。

## 环境准备行为

Ubuntu 环境准备路线必须保留以下约定：

1. 只有 `deploy/offline/.env` 不存在时，才复制 `deploy/offline/.env.example`。
2. 不得自动覆盖已有且有效的 `.env`。
3. 第一次准备时记录 `id -u` 和 `id -g`，拒绝 root UID/GID，并在后续执行时拒绝身份不一致。
4. 要求使用本地 rootful Docker 和现有部署规定的 `default` context。
5. secret 必须位于仓库管理的 `artifacts/secrets`，拒绝重定向、引号、未解析变量或符号链接路径。
6. Linux 上 secret 目录权限保持 `0700`，secret 文件权限保持 `0600`。
7. PostgreSQL 密码和数据库 URL 必须成对原子创建，缺少其中一个时必须拒绝继续。
8. 分别创建 ClickHouse 查询账号和写入账号密码，且不得输出密码内容。
9. 除非明确传入 `--rotate-secrets`，否则必须保留已有且有效的 secret。
10. 校验数据和模型 bind 根目录，只创建允许写入的 `raw` 和 `parquet` 目录。

密码生成使用 Python `secrets` 模块，Ubuntu 不需要为了生成密码额外依赖 OpenSSL。

## Compose Wrapper 行为

Ubuntu Compose wrapper 必须保留以下约定：

1. 只允许使用 `docker --context default compose`，并固定使用 `deploy/offline/.env` 和 `deploy/offline/compose.yaml`。
2. 拒绝远程 Docker endpoint、非默认 context，以及通过环境变量覆盖 Compose project、file 或 profile。
3. 拒绝不受支持或危险的命令和参数，包括一次性 `run`、直接 `create`、`start`、`restart`、修改 scale，以及跳过 build、依赖或重建的参数。
4. 执行实际命令前，必须使用 `config --format json` 渲染全部 profile。
5. 校验固定 project 名称、使用 digest 固定的内部镜像、内部网络隔离、API 暴露范围、允许的 bind source 和仓库管理的 secret 路径。
6. 执行结束后恢复调用者环境，并返回真实的 Docker Compose 退出码。
7. 不得打印 secret 内容或解析后的数据库凭据。

## Smoke 工具选择

`tools/compose_smoke.py` 根据操作系统选择 wrapper：

- POSIX/Linux：`tools/invoke_offline_compose.sh`；
- Windows：`tools/invoke_offline_compose.ps1`。

显式传入的 wrapper 路径继续拥有最高优先级。现有 Windows 进程处理逻辑继续保留，Linux 则直接执行 Bash wrapper。

## 文档规则

以下文档以 Ubuntu Bash 命令作为主要部署示例：

- `docs/intranet-deployment-configuration.md`；
- `docs/offline-platform-runbook.md`；
- `deploy/offline/README.md`；
- `README.md` 中的生产部署部分。

`docs/intranet-deployment-configuration.md` 不得包含 `.ps1`、`Copy-Item`、PowerShell 反引号续行符或 `& tools/...` 调用。多行命令统一使用 Bash `\` 续行符。

项目 README 和 Compose README 可以保留一段简短的 Windows 开发兼容说明，指向 `.ps1` 脚本，但 Ubuntu 是唯一的生产服务器部署路线。

## 测试

测试采用契约优先方式：

1. 先添加失败测试，检查两个 `.sh` 入口、LF 换行、Git 可执行权限和预期的 Python 辅助程序调用。
2. 先添加失败单元测试，覆盖环境变量解析、重复键、secret 路径限制、UID/GID 校验、secret 成对原子创建、轮换和权限。
3. 先添加失败单元测试，覆盖 Compose 参数拦截、环境清理、渲染 JSON 校验、内部镜像 digest、网络、bind 和 secret。
4. 添加失败测试，证明 `compose_smoke.py` 在 POSIX 上选择 `.sh`，在 Windows 上选择 `.ps1`。
5. 添加文档契约测试，证明公司内网部署清单只使用 Bash，并包含受支持的 Ubuntu 命令。
6. 运行现有 Compose、Physoc、结构化部署和 smoke 测试，防止 PowerShell 兼容路线回归。
7. 运行 Ruff 和 `git diff --check`。

真实 Docker Compose 执行仍属于目标 Ubuntu 服务器验收项，因为当前开发机不具备生产 Docker 拓扑。

## 兼容和上线

- Windows 开发人员可以继续使用 `.ps1` 脚本。
- Ubuntu 运维人员不需要安装 PowerShell 7。
- 现有 `.env`、bind 数据和 secret 文件保持兼容，Bash 环境准备命令负责校验并保留它们。
- 第一次 Ubuntu 部署必须运行 Bash 环境准备命令、渲染 Compose 配置、执行服务 smoke，并验证 Physoc/Ollama 路线后才能接入生产流量。
- Bash 和 PowerShell 路线出现行为差异时视为测试失败，任何一条受支持路线都不能静默省略安全校验。

## 验收标准

满足以下条件后视为完成：

- 两个 Bash 入口可以在 Ubuntu 20.04 上运行，且不依赖 `pwsh`；
- Bash 和 PowerShell 入口保持相同的部署安全契约；
- 公司内网部署清单完全使用 Bash；
- POSIX smoke 工具默认使用 Bash Compose wrapper；
- 所有受影响的自动化测试、Ruff 和 Markdown/diff 检查通过；
- 文档明确要求在 Ubuntu 目标服务器执行 Compose 和网络连通性验收。
