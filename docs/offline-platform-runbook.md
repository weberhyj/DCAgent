# DC-Agent 离线平台运行手册

本文是 Phase 1 离线单机部署和容量门禁的操作记录。它只描述本地、可审计的开源组件和内部镜像；运行时不得访问公共模型 API、公共镜像仓库或其他外部服务。

## 1. 当前状态与适用范围

- 目标环境是 Linux、PowerShell 7（`pwsh`）、rootful Docker Engine、Docker Compose v2、本地 `default` Docker context。
- Python 3.12 是支持的基线；Node.js 20 用于两个前端的本地开发和 UI smoke。
- 本阶段启动 API、PostgreSQL、schema migration、ClickHouse、Qdrant、Redis、ClamAV 和一个私有 Embedding 服务。`indexing` 与 `generation` profile 默认关闭。
- 生产 Compose 的唯一宿主端口是 `127.0.0.1:8000:8000/tcp`。内部服务只能通过 Compose 网络和 `exec` 检查访问。
- 当前开发机没有 Docker，也没有目标 Linux 的内部 wheelhouse、模型和镜像，因此本机只能运行单元测试，不能提供真实 Compose smoke、镜像构建或容量结果。

## 2. 离线依赖与 Python 3.12

在目标主机准备 Python 3.12 virtualenv，并从已审核的内部 wheelhouse 生成带 hash 的锁文件。锁文件必须在与目标主机相同的 Python 3.12 环境中生成，不能把未审核的公共索引地址写入仓库。

```powershell
python3.12 -m venv .venv
& ./.venv/bin/python -m pip install --no-index --find-links artifacts/wheels pip-tools
$env:PIP_NO_INDEX = "1"
& ./.venv/bin/pip-compile `
  --no-index `
  --generate-hashes `
  --no-emit-index-url `
  --no-emit-trusted-host `
  --find-links artifacts/wheels `
  --output-file backend/requirements-offline.txt `
  backend/requirements-offline.in
& ./.venv/bin/pip-compile `
  --no-index `
  --generate-hashes `
  --no-emit-index-url `
  --no-emit-trusted-host `
  --find-links artifacts/wheels `
  --output-file backend/requirements-benchmark.txt `
  backend/requirements-benchmark.in
& ./.venv/bin/python -m pip install `
  --no-index --find-links artifacts/wheels --require-hashes `
  -r backend/requirements-offline.txt `
  -r backend/requirements-benchmark.txt
```

提交或部署前检查：

1. 两个 lock 文件中的每个发行版都有 hash，且所有 wheel 来自内部 wheelhouse。
2. `pip install` 使用 `--no-index --find-links ... --require-hashes`，不允许 fallback 到网络。
3. `backend/requirements-offline.txt` 和 `backend/requirements-benchmark.txt` 的生成日志、Python 版本、wheelhouse 清单和 wheel SHA-256 一起归档。

当前仓库只保留 `.in` 输入文件；上述 lock 文件尚未在本开发机生成，不能把“可安装”写成已验证事实。

## 3. Artifact manifest 与许可证审核

把 Docling/PaddleOCR/LibreOffice/Poppler、本地模型和其他运行时文件放入目标主机的 `artifacts/`，不要把模型或许可证不明的二进制加入 Git。受控清单路径为 `deploy/offline/artifacts.lock.json`，字段必须严格符合 [`deploy/offline/artifacts.schema.json`](../deploy/offline/artifacts.schema.json)：

```json
{
  "artifacts": [
    {
      "name": "example-artifact",
      "kind": "docling-artifact",
      "version": "2.40.0",
      "sha256": "<64 lowercase hex characters>",
      "license": "Apache-2.0",
      "localPath": "artifacts/vendor/example-artifact"
    }
  ]
}
```

对文件或无符号链接目录使用仓库已有的确定性 artifact 哈希实现，禁止用下载 URL 代替本地路径：

```powershell
& ./.venv/bin/python -c "from pathlib import Path; from tools.benchmarks.model_probe import sha256_artifact; print(sha256_artifact(Path(r'artifacts/vendor/example-artifact')))"
```

每个条目在进入清单前必须完成：

- 上游项目、版本、许可证和许可证文本归档；传递依赖也要逐项记录。
- 本地路径、SHA-256、大小和解压后的目录结构复核；目录不得含符号链接。
- 与 `backend/app/offline_artifacts.py` 相同的字段闭集和小写 SHA-256 校验。
- 许可证不兼容、缺少 NOTICE、无法解释的预编译库或无法复现的二进制直接阻断部署。

可用下面的只读检查验证清单形状（不会访问网络）：

```powershell
$env:PYTHONPATH = "backend"
& ./.venv/bin/python -c "import json; from pathlib import Path; from app.offline_artifacts import validate_artifact_manifest; validate_artifact_manifest(json.loads(Path('deploy/offline/artifacts.lock.json').read_text(encoding='utf-8'))); print('artifact manifest valid')"
```

## 4. Compose profile、内存预算与准备

先复制并填写 `deploy/offline/.env.example`：

```powershell
Copy-Item deploy/offline/.env.example deploy/offline/.env
& tools/prepare_offline_env.ps1
```

必须替换全部占位 digest、Embedding checksum、模型文件和模型名。默认资源预算记录如下；它们是 Compose 配置值，不是已经测得的性能结果：

| 配置项 | 默认值 | 说明 |
| --- | ---: | --- |
| `CLICKHOUSE_MEMORY_LIMIT` / ClickHouse `mem_limit` | 4g | 结构化查询服务 |
| `QDRANT_MEMORY_LIMIT` / Qdrant `mem_limit` | 2g | 向量检索服务 |
| `MODEL_SLOTS` | 2 | 生成 profile 的并发槽位；需按本地模型实测调整 |
| API 宿主端口 | 127.0.0.1:8000 | 唯一发布端口 |

profile 约定：

- 默认：PostgreSQL、migration、ClickHouse、Qdrant、Redis、ClamAV、Embedding、API。
- `--profile indexing`：Phase 2 ingestion worker；当前不要启用。
- `--profile generation`：本地 llama.cpp 和 GGUF 模型；只有完成模型锁、许可证审核和目标主机内存评估后才启用。

所有 Compose 操作必须经由受信 wrapper：

```powershell
& tools/invoke_offline_compose.ps1 config --quiet
& tools/invoke_offline_compose.ps1 up -d
```

不要直接运行 `docker compose`，不要使用远程 context、rootless Docker、userns remapping、NFS root-squash 或 Windows container UID 语义。wrapper 会在执行前清理环境覆盖、渲染所有 profile，并拒绝非内部 digest、异常 bind/secret、非 loopback API 端口和其他绕过参数。

## 5. Compose smoke 与容量门禁

### 5.1 Real-service smoke

在目标主机完成 artifact、`.env`、镜像和目录准备后，从仓库根目录运行：

```powershell
$HardwareClass = "32gb" # 64GB 主机必须改为 "64gb"
& ./.venv/bin/python tools/compose_smoke.py `
  --report "artifacts/benchmarks/$HardwareClass/compose-smoke.json"
```

命令会依次执行 wrapper `config`、仅启动 `api` 及其核心依赖、通过 `exec` 检查 PostgreSQL/Alembic、ClickHouse、Qdrant、Redis、ClamAV、Embedding，并请求宿主 `127.0.0.1:8000/api/readyz`。默认 `down` 保留 volume；只有明确传入 `--remove-volumes` 才删除 volume。该选项是破坏性清理，不属于常规验收命令，执行前必须确认 volume 可删除并继续写入独立报告：

```powershell
& ./.venv/bin/python tools/compose_smoke.py `
  --report "artifacts/benchmarks/$HardwareClass/compose-smoke-remove-volumes.json" `
  --remove-volumes
```

成功报告必须包含 component versions、硬件、`composeYamlSha256`、`wrapperSha256`、migration head、每个 readiness 结果和 `offlineOnly: true`。省略 `--report` 时默认写入 `artifacts/benchmarks/compose-smoke.json`，但正式验收必须使用硬件规格子目录。本机无 Docker 时，该命令必须失败并且不能输出 “compose smoke passed”；这不是失败的实现，而是目标主机门禁尚未执行。

### 5.2 Phase 1 service-round-trip report

容量报告工具会 fail-closed：它要求 benchmark 子命令实际写出 `BENCHMARK_METRICS_PATH`，缺少 metrics、命令非零或门禁未满足都会失败。仓库目前提供 report/gate runner 和 `smoke.json`，没有伪造一个服务 round-trip producer；目标主机需要提供经过审核的内部 benchmark command。

命令形状如下（将最后的 benchmark command 替换为目标主机已审核的本地命令）：

```powershell
$HardwareClass = "32gb" # 64GB 主机必须改为 "64gb"
& ./.venv/bin/python -m tools.benchmarks.run_capacity_benchmark `
  --manifest tools/benchmarks/manifests/smoke.json `
  --metrics "artifacts/benchmarks/$HardwareClass/service-metrics.json" `
  --report "artifacts/benchmarks/$HardwareClass/service-report.json" `
  --profile service-round-trip `
  --mode phase1-smoke `
  --vector-dimension 32 `
  --model-slots 1 `
  --benchmark-command <approved-local-service-round-trip-command>
```

该命令不能在没有真实 metrics producer 的情况下被改成“写入固定数字”的脚本；那会制造假通过。

### 5.3 Phase 4 acceptance benchmark

30M ClickHouse rows、5M Qdrant points、15 virtual users、30 分钟运行时间的 acceptance manifest 是 `tools/benchmarks/manifests/acceptance-30m-5m.json`。只有 Phase 2/3 的 ingestion/query endpoints 已经在目标主机可用时才执行；Phase 1 不应提前宣称通过。

冷缓存示例（768 维、2 个模型槽位）如下。外层 capacity runner 会把绝对路径注入 `BENCHMARK_METRICS_PATH`，仓库的 `tools/benchmarks/locustfile.py` 在 Locust `test_stop` 事件中汇总请求并原子写出该 JSON；普通 stock Locust 单独运行不会生成 capacity metrics：

```powershell
$HardwareClass = "32gb" # 64GB 主机必须改为 "64gb"
& ./.venv/bin/python -m tools.benchmarks.run_capacity_benchmark `
  --manifest tools/benchmarks/manifests/acceptance-30m-5m.json `
  --metrics "artifacts/benchmarks/$HardwareClass/online-cold-metrics.json" `
  --report "artifacts/benchmarks/$HardwareClass/online-cold-report.json" `
  --profile online-cold `
  --mode phase4-online `
  --cache-label cold `
  --vector-dimension 768 `
  --model-slots 2 `
  --benchmark-command ./.venv/bin/python -m locust -f tools/benchmarks/locustfile.py --headless -u 15 -r 1 --run-time 30m --host http://127.0.0.1:8000
```

`--benchmark-command` 使用 remainder 解析：它之后的全部 token 都属于内部 Locust 命令，不需要额外的 `--` 分隔符。`-u 15` 表示 15 个并发虚拟用户；实际同时在途请求数仍由任务执行时间和 5 秒 think time 决定。`--model-slots 2` 是报告字段，必须与本次运行的 `deploy/offline/.env` 中 `MODEL_SLOTS=2` 完全一致，不得只改报告参数。

暖缓存使用 `--profile online-warm --cache-label warm` 和独立的 `warm` metrics/report 路径。批处理 profile 使用 `batch-initial`、`batch-daily` 或 `batch-weekly`，并把 report 放入同一硬件目录；不要覆盖冷/暖缓存结果。

## 6. 32GB 与 64GB 结果记录

两种主机规格必须分开记录，不能把一个规格的结果复制到另一个规格：

```text
artifacts/benchmarks/32gb/
  compose-smoke.json
  service-metrics.json
  service-report.json
  online-cold-metrics.json
  online-cold-report.json
  online-warm-metrics.json
  online-warm-report.json

artifacts/benchmarks/64gb/
  compose-smoke.json
  service-metrics.json
  service-report.json
  online-cold-metrics.json
  online-cold-report.json
  online-warm-metrics.json
  online-warm-report.json
```

每份报告应同时保留 manifest/profile SHA-256、硬件总内存、CPU、向量维度、模型槽位、模型/Embedding 身份、服务版本、命令退出码、错误率和 P95 指标。没有真实运行结果时，报告状态应为 `not_run` 或不存在；不要提交生成的 benchmark JSON 到 Git。

## 7. 本地回归与目标主机验收清单

目标 Linux 主机本地、无需 Docker 的回归（Windows 开发机可将 `./.venv/bin/python` 替换为 `py`）：

```powershell
Set-Location backend
& ../.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..
& ./.venv/bin/python -m unittest discover -s tools/tests -p "test_*.py" -v
& ./.venv/bin/python -m compileall -q tools
git diff --check
```

目标主机 gate：

```powershell
& ./.venv/bin/python tools/compose_smoke.py
& ./.venv/bin/python -m tools.benchmarks.run_capacity_benchmark --help
```

必须人工确认：本地 rootful daemon、镜像 digest、模型和 parser artifact checksum、许可证、secret 文件权限、PostgreSQL 备份/恢复演练、migration head、API loopback 绑定、Compose volume 保留策略以及 32GB/64GB 报告目录。当前开发机已完成单元回归，但没有 Docker，因此“目标主机 gate”仍未完成。

## 8. 故障处理边界

- smoke 的任何 command、HTTP status、JSON、版本或 checksum 失败都按失败处理；不要手工编辑 report 把 `passed` 改成 true。
- Compose 配置失败时不要删除数据目录；默认 cleanup 不移除 volume。
- 第一次 PostgreSQL baseline stamp 之前必须有可恢复备份；baseline drift 应停止启动，不要用 downgrade 代替恢复。
- secret rotation 仅适用于 PostgreSQL 初始化前；初始化后必须走受控 `ALTER ROLE`、双文件切换和连通性验证流程。
- 发现公共 endpoint、公共镜像、未审核 license、符号链接 bind source、远程 Docker context 或缺少 hash 时，立即停止并记录为 gate failure。
