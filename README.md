# DC-Agent

## 版本管理

后端、用户端和管理端分别维护版本号，只升级实际发生代码修改的应用。后端版本源是 `backend/app/__init__.py`，两个前端各自使用自己的 `package.json`。完成修改后，默认执行对应应用的 patch 升级：

```bash
python tools/bump_version.py backend patch
python tools/bump_version.py frontend patch
python tools/bump_version.py admin-frontend patch
cd backend && uv lock
```

只执行本次涉及的应用命令；仅后端版本变化时才需要重新运行 `uv lock`。前端版本号不写入 Yarn 锁文件，因此单纯升级前端版本无需重新生成 `yarn.lock`。需要发布新功能或不兼容变更时，分别使用 `minor`、`major`，也可以传入明确的 `X.Y.Z` 版本号。提交前运行 `uv run --project backend pytest tools/tests/test_version_contract.py -q` 检查各应用版本契约。

DC-Agent 是一个公司内部只读知识 Agent。管理员把制度、合同、会议纪要、经营数据等资料上传到知识库后，DCAgent 会围绕用户问题执行有界的多步检索、资料深挖和证据对比，再基于已索引资料生成回答。

## 项目结构

- `frontend`：用户检索端，面向普通用户提问和查看 DCAgent 生成的答案。
- `admin-frontend`：知识库管理端，面向管理员上传、筛选、重新解析、删除资料源，查看解析片段和 Agent 执行审计。
- `backend`：Python + FastAPI + LangGraph 服务，提供只读 Agent、问答、资料上传、解析、知识库索引和审计接口。

## 本地环境

建议环境：

- Python 3.12.x（不支持 3.13）
- Node.js 20.19 或更高版本
- Yarn 4.9.2（通过根目录 `packageManager` 和 `.yarnrc.yml` 固定）
- PostgreSQL，本地默认库名为 `dc_agent`

两个前端由根目录 Yarn workspace 统一管理。首次安装依赖和常用命令：

```bash
corepack enable
corepack yarn install --immutable
corepack yarn dev:frontend
corepack yarn dev:admin
corepack yarn build
corepack yarn test
```

后端默认数据库连接：

```text
postgresql+psycopg://postgres:123456@127.0.0.1:5432/dc_agent
```

也可以用 `DATABASE_URL` 覆盖默认连接串。管理员上传的文件默认保存在 `backend/uploads/knowledge`，该目录只用于本地运行数据，不进入版本管理。

## 环境变量

后端启动时会自动读取项目根目录 `.env` 和 `backend/.env`。读取顺序为根目录 `.env` 后读取 `backend/.env`，但系统环境变量优先级最高，不会被文件覆盖。

本地开发时，按所用开发环境为项目根目录和 `backend` 创建各自的 `.env` 文件。

本地兜底模式：

```text
LLM_PROVIDER=template
```

真实模型模式使用 OpenAI-compatible Chat Completions 接口：

```text
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://your-llm-host.example/v1
LLM_API_KEY=replace-with-your-api-key
LLM_MODEL=your-model-name
```

### 原生 Ubuntu 本地 Ollama DeepSeek LLM

原生 Ubuntu + Supervisor 内网部署不需要 Physoc。由本机 Ollama 加载 `deepseek-llm:7b`，
DC-Agent 直接调用 Ollama 提供的 OpenAI-compatible `/v1/chat/completions`：

```text
LLM_PROVIDER=openai_compatible
LLM_API_BASE=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama-local
LLM_MODEL=deepseek-llm:7b
LLAMA_SERVER_URL=http://127.0.0.1:11434
LLM_HEALTH_PATH=/api/version
OFFLINE_MODE=true
```

完整的模型命令、Supervisor 配置和探针见
[`deploy/ubuntu/LLAMA_CPP_EMBEDDING.md`](deploy/ubuntu/LLAMA_CPP_EMBEDDING.md)。这里的
`LLM_API_KEY` 是无鉴权 loopback Ollama 服务的非空客户端占位值，不应替换为公司真实密钥。
知识库生成请求固定使用 `temperature=0`、`top_p=1`、`seed=42`。独立问题不会携带
历史回答；只有“继续”、“那他呢”、“刚才提到的”等明确追问才会使用最近会话。
检索步骤的问答审计会记录候选片段 ID 和分数，可用于对比重复问题是否获得了相同依据。

生产环境的 `/api/knowledge/uploads` 只负责接收并落盘文件、创建状态为“解析中”的资料源，随后
以 HTTP `202` 返回；Word、PDF、TXT 等解析、切片和检索索引由独立的
`python -m app.ingestion_worker` 进程完成。Worker 会从 PostgreSQL 重新发现尚未完成的资料源，
因此 API 或 Worker 重启不会丢失已经落盘的上传任务。Excel/CSV 完成表结构预览后，后续正式
结构化发布仍由 `python -m app.structured_worker` 执行，这两个 Worker 都需要由 Supervisor 常驻。

### Physoc DeepSeek 模式

Physoc DeepSeek 流式接口可以按以下方式配置，示例使用本机 loopback 地址：

```text
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://127.0.0.1:8090
LLM_STREAM_PATH=/api/physoc/deepseeks/stream
LLM_MODEL=my_deepseek_r1_7b
```

该 loopback 示例适用于后端直接运行在同一主机的开发场景，不表示当前 offline Compose 拓扑可以直接启用 Physoc。Compose 内的 `127.0.0.1` 指向 API 容器自身，部署前必须完成隔离网络和变量接线审核。

Physoc 模式无需 LLM_API_KEY。后端向 `LLM_API_BASE` 与 `LLM_STREAM_PATH` 组成的地址发送 `POST` 请求，请求 JSON 包含 `"query"` 和 `"model"`。其中 `query` 是完整 RAG 提示词，不是原始用户问题：

```json
{"query":"完整 RAG 提示词（系统约束、检索证据、Agent 摘要和近期会话）","model":"my_deepseek_r1_7b"}
```

响应类型为 `text/event-stream`。后端读取 SSE `message` 事件中的 `"response"` 内容，直到收到 `"done": true`。现阶段后端会缓冲完整结果后再返回；前端对话 API 保持不变，模拟逐字显示保持不变。真实私有 IP 应在部署环境中设置，不要把实际地址或凭据写入示例文件。

开发机尚未执行真实私有 Physoc POST/SSE 互操作验证。目标环境 smoke gate 必须核验 body/query/model、Content-Type、message/response/done，以及 timeout and interrupted-stream behavior。

当知识库没有命中资料时，DCAgent 会返回“未检索到足够依据”，不会调用真实模型编造答案。

两个前端项目都支持通过 `VITE_API_PROXY_TARGET` 覆盖本地 API 代理目标，默认代理到 `http://127.0.0.1:8000`。

生产入口禁止 template 和 mock；它们仅用于本地开发和固定测试数据。公司内网部署应设置：

```text
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://172.16.0.10:8090
LLM_STREAM_PATH=/api/physoc/deepseeks/stream
LLM_MODEL=my_deepseek_r1_7b
```

上面的 `172.16.0.10` 是不含凭据的私网示例，实际容器必须使用容器可达的批准 private address。核心 `offline` 网络仍为 `internal`；只有 API 同时连接 `physoc-egress` 以访问 Physoc。`physoc-egress` 是受控出口，不使其他核心服务获得外部网络能力，也不表示 `internal` 网络可访问外部。目标主机和防火墙必须把该出口限制到批准的 private Physoc 地址。前面的 loopback 配置仅保留用于后端和 Physoc 在同一主机、且后端不在容器内运行的开发场景；容器内不得把 `127.0.0.1` 当作宿主机 Physoc 地址。

部署后必须从仓库根目录通过受支持的 Compose wrapper 在 API 容器内执行。probe 成功后再把
脱敏报告复制到 host：

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh up -d
if ! ./tools/invoke_offline_compose.sh exec -T api \
  python -m app.physoc_probe --report /tmp/physoc-probe.json
then
  echo "Physoc probe failed; do not persist evidence." >&2
  exit 1
fi
mkdir -p artifacts/benchmarks
./tools/invoke_offline_compose.sh cp api:/tmp/physoc-probe.json artifacts/benchmarks/physoc-probe.json
```

探针成功报告只记录 provider、model、streamPath、elapsedMs、answerChars 和 citationCount 等运行元数据，不会输出提示词、证据正文或模型回答正文。只有容器内 probe exit 0 后才创建 host 目录并执行 `cp`；将 `artifacts/benchmarks/physoc-probe.json` 作为切换门禁证据保存。探针失败时不得复制旧报告或启用该生产路由。

## Ubuntu 20.04 公司内网事务部署

生产主路径只支持 Ubuntu 20.04、Bash、rootful Docker Compose v2 和仓库根目录下的
`prepare_offline_env.sh`、`invoke_offline_compose.sh`、`recover_offline_deployment.sh`。`DEPLOYMENT_STATE_ROOT`
必须是 `DATA_ROOT/.dcagent-deployment-state`，由
初始化或接管操作写入并与 data/model/secret roots 绑定。普通 prepare/Compose 不隐式创建 identity；
更换 `DATA_ROOT` 视为新部署。

新部署先由 Ubuntu 管理员预创建固定数据目录，并把 owner 设置为实际的非 root 部署账号。下面的
`dcagent` 只是示例账号；部署前必须改成实际账号和组：

```bash
set -Eeuo pipefail
deployment_user=dcagent
deployment_group=dcagent
sudo install -d -o "$deployment_user" -g "$deployment_group" -m 0700 \
  /srv/dcagent/data /srv/dcagent/models
```

随后以该非 root 部署账号登录并进入仓库根目录。不要手工复制或改写 `.env.example`；首次不存在
`deploy/offline/.env` 时，由 `--initialize-state` 自动创建 `.env`、写入当前 UID/GID，并把下面成对
提供的 HOST roots 固化为绝对 `DATA_ROOT`/`MODEL_ROOT`。已有 `deploy/offline/.env` 不得覆盖，脚本会
校验其 roots、UID/GID 和 deployment identity，不匹配时直接失败。

公共前置完成后，手工路径与推荐 gate 路径二选一。

### 手工路径

固定核心顺序是 prepare → config → 单次 build → up：

```bash
set -Eeuo pipefail
export HOST_DATA_ROOT=/srv/dcagent/data
export HOST_MODEL_ROOT=/srv/dcagent/models
./tools/prepare_offline_env.sh --initialize-state
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh build schema-migration embedding-service api ingestion-worker
./tools/invoke_offline_compose.sh up -d
```

### 推荐 gate 路径

gate 自身执行上述 prepare/config/build/up 固定序列及验收；不要先运行手工路径，否则会重复 build/up。config 60 秒、build 1800 秒、up/readyz 300 秒、每个 probe 60 秒、recovery drill 120 秒。

```bash
set -Eeuo pipefail
export HOST_DATA_ROOT=/srv/dcagent/data
export HOST_MODEL_ROOT=/srv/dcagent/models
python3 tools/intranet_deployment_gate.py --mode fresh --report artifacts/benchmarks/intranet-deployment-gate.json
```

已有数据的旧部署必须先接管，不能跳过 state root：

```bash
set -Eeuo pipefail
export HOST_DATA_ROOT=/absolute/data/root
export HOST_MODEL_ROOT=/absolute/model/root
./tools/recover_offline_deployment.sh adopt-existing --state-root /absolute/data/root/.dcagent-deployment-state
./tools/prepare_offline_env.sh
```

部署锁等待上限为 30 秒。六个 Compose verb：config/build/up/down/exec/cp；只有
`./tools/invoke_offline_compose.sh up`、`./tools/invoke_offline_compose.sh exec` 和
`./tools/invoke_offline_compose.sh cp` 会在执行前 durable 写入 `deployment-started.json`，失败后保留它。
`./tools/invoke_offline_compose.sh config`、`./tools/invoke_offline_compose.sh build` 和
`./tools/invoke_offline_compose.sh down` 不写 marker。marker 存在时普通 `--rotate-secrets` 拒绝；只有经
`recover_offline_deployment.sh clear-start-marker` 且确认无 `PG_VERSION`、无未完成事务后，才可能恢复
pre-init rotation。任意形态 `PG_VERSION` 存在后永久拒绝；不提供在线 PostgreSQL role 密码修改，也不提供单行删除 marker 的命令。

故障先运行 `./tools/recover_offline_deployment.sh inspect --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>`。
自动回滚成功后可继续；`rollback_failed` 使用
`./tools/recover_offline_deployment.sh resume-rollback --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>`；
`committed_cleanup_required` 使用
`./tools/recover_offline_deployment.sh finalize-cleanup --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>`；
损坏 journal/quarantine 经人工修复后使用
`./tools/recover_offline_deployment.sh acknowledge-repaired --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id> --evidence /absolute/path/sanitized-repair-evidence.json`。
人工执行 `./tools/recover_offline_deployment.sh clear-start-marker --state-root /absolute/data/root/.dcagent-deployment-state` 前，逐项确认无 DC-Agent 容器、无 `PG_VERSION`、PostgreSQL 目录不存在或未初始化、且无未完成事务。
日志和 evidence receipt 不含 secret、数据库 URL、模型正文或原始 SSE。

开发机本地测试不是Ubuntu live gate通过；没有真实 Ubuntu Docker、Physoc 和 Ollama 拓扑时，只能记录 live gate 未运行。

## Structured spreadsheet aggregation

Exact Excel/CSV aggregation is enabled in the shipped environment templates with
`STRUCTURED_QUERY_ENABLED=true`. This does not add an external API: published rows stay in local
Parquet staging and the private ClickHouse service. Set the flag to `false` only when intentionally
rolling back to the legacy document RAG path.

The query API uses the `CLICKHOUSE_QUERY_USER` account and its password file; the indexing worker
uses the separate `CLICKHOUSE_INGEST_USER` account and password file. Password values must not be
placed in `.env` or an example file. The 4-second timeout applies only to API aggregate connection
and query execution. Structured publication keeps the storage gateway's independent 30-second
execution default, and the default bounded ingestion batch is 50,000 rows.

The feature is usable only after an administrator has approved a confirmed schema for the XLSX/CSV
dataset and the offline `--profile indexing` worker has published that schema version. See
[`deploy/offline/README.md`](deploy/offline/README.md) for migration, worker startup, smoke aggregate,
rollback, ClickHouse failure handling, and the mandatory live 100,001-row filtered-summary gate.
The in-process large-sheet fake only proves batching, deterministic response structure, one returned
aggregate row, and LLM isolation; it does not prove ClickHouse filtering or Decimal arithmetic.

## 当前真实架构与能力边界

> 本节描述当前功能分支已经接入的运行链路。生产切换仍必须完成目标内网服务器上的模型、
> 数据、权限和 15 用户并发验收。

```mermaid
flowchart TD
    A["管理端上传文件"] --> B["FastAPI API"]
    B --> C["本地原始文件目录"]
    B --> D{"是否启用结构化 Excel/CSV"}

    D -->|"否或普通文档"| E["解析、切片与发布元数据"]
    E --> F["PostgreSQL 权威文档与切片"]
    F --> G["Qdrant 版本化索引"]

    D -->|"是"| H["推断并人工确认表结构"]
    H --> I["结构化 indexing worker"]
    I --> J["Parquet 暂存"]
    I --> K["ClickHouse 不可变发布表"]

    L["用户提问"] --> M{"是否为已发布数据集的统计问题"}
    M -->|"avg / sum / count / min / max"| K
    K --> N["确定性统计答案，不调用大模型"]
    K -.->|"不可用或超时"| X["显式失败，不回退到切片计算"]

    M -->|"普通文档问题"| O["RetrievalRouter"]
    O -->|"legacy"| P0["PostgreSQL Legacy 检索"]
    O -->|"shadow / qwen3"| P1["BGE Chinese Embedding adapter + BM25"]
    P1 --> P2["Qdrant Dense + Sparse"]
    P2 --> P3["RRF 融合 + BGE-Reranker-v2-M3"]
    P3 --> P["Top 8 授权证据"]
    P0 --> P
    P -.->|"无可靠证据"| Y["拒绝回答，不调用模型"]
    P --> Q["Physoc DeepSeek POST/SSE"]
    Q --> R["基于证据归纳的最终答案"]
    Q -.->|"超时、非 2xx 或异常 SSE"| Z["HTTP 502"]
```

### 当前生产：llama.cpp BGE-M3 + BGE-Reranker-v2-M3

生产默认配置为 `RETRIEVAL_MODE=qwen3`、`RERANKER_ENABLED=true`：

```env
EMBEDDING_RUNTIME=llama_cpp
EMBEDDING_MODEL_NAME=bge-m3-Q4_K_M.gguf
RERANKER_RUNTIME=llama_cpp
RERANKER_MODEL_NAME=bge-reranker-v2-m3-Q4_K_M.gguf
```

llama.cpp 分别提供 `/v1/embeddings` 和 `/v1/rerank`。Excel 行查询与 Word 字段事实问答走确定性路由，
不调用 Embedding、Reranker 或 LLM；普通文档问题才走 Dense + BM25 + RRF + Reranker + Physoc。

旧 Ollama/RRF-only 内容仅保留作兼容回滚参考。

### Ollama BGE 中文混合检索链路（兼容回滚）

普通文档问答默认采用 `Dense + BM25 + RRF + BGE-Reranker-v2-M3`，完整配置如下：

```env
RETRIEVAL_MODE=qwen3
RERANKER_ENABLED=true
RETRIEVAL_FINAL_TOP_K=8
OLLAMA_BASE_URL=http://ollama.inner:11434
OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
```

该路线只启动 BGE Embedding adapter，不创建、不探测也不调用 Reranker。Qdrant Dense 与 BM25
Sparse 各自召回候选，RRF 融合后按稳定顺序取 Top 8，再执行邻接片段扩展，最后由 Physoc
DeepSeek 基于证据归纳答案。

1. PostgreSQL 保存权威文档、切片、权限标签和发布状态；Qdrant 保存版本化检索点。
2. 轻量 Embedding adapter 通过公司内网 Ollama 的 `bge-large-zh-v1.5:latest` 生成 Dense vector，并按
   固定 profile 做归一化；仅 query 文本添加前缀 `为这个句子生成表示以用于检索相关文章：`，document
   文本保持原文。维度必须以目标 `/api/embed` 的 `len(embeddings[0])` 实测值为准；本项目示例为 1024。
   本地 BM25 生成 Sparse query vector。
3. 两路检索都先应用 knowledge-base、permission-tag 和 publication filters，各取 Top 50。
4. RRF（`k=60`）融合候选；默认启用 BGE-Reranker-v2-M3，异常时才按受控策略降级到 RRF 顺序并取
   `RETRIEVAL_FINAL_TOP_K=8`，最终只给 Agent 提供有界授权证据。
5. Agent 把授权证据、调查摘要和近期会话组成完整 RAG 提示词，再通过
   `POST /api/physoc/deepseeks/stream` 交给私有 Physoc DeepSeek 归纳。

`RETRIEVAL_MODE=legacy|shadow|qwen3` 的语义如下：

- `legacy`：只运行 PostgreSQL Legacy 检索，不构造混合检索/Qdrant 查询资源。
- `shadow`：前台仍返回 Legacy 结果；按 `RETRIEVAL_SHADOW_PERCENT` 在后台运行混合检索对比，
  只存 case/chunk ID、排名指标、耗时和脱敏失败码，不保存原始问题或证据正文。
- `qwen3`：为保持 collection、数据库记录和环境变量兼容而保留的路由名；按稳定会话哈希和
  `RETRIEVAL_CANARY_PERCENT` 选择 Ollama-backed 混合检索；未命中 canary 或
  Embedding、Qdrant、Alias、超时/熔断异常时安全回退 Legacy；只有显式启用 Reranker 时才检查其状态。

新上传知识库文档的默认分类为“公开”，默认部署模板使用：

```env
RETRIEVAL_PERMISSION_TAGS=公开
```

显式提交的其他分类仍会保留。升级不会修改数据库中的历史分类，现有文档不会自动迁移；如需采用新默认值，请删除后重新导入并等待索引完成。

DC-Agent 本身不运行 Embedding/Reranker 权重；默认只有 Embedding adapter 调用公司内网可达的
Ollama，不依赖外部 API。服务器现有的 `kopens/bge-reranker-large:latest` 经 Ollama
`/api/embed` 返回的是 1024 维向量，不是 cross-encoder 的单一相关性分数，因此不能作为当前
`/v1/rerank` 的替代实现。

如以后有经过验证、能够输出每个 query/passage 相关性分数的 Reranker，再显式启用：

```env
RERANKER_ENABLED=true
RERANKER_SERVICE_URL=http://reranker-service:8082
```

llama.cpp `bge-reranker-v2-m3-Q4_K_M.gguf` 的部署配置与原生 `/v1/rerank`
适配说明见 [deploy/offline/LLAMA_CPP_RERANKER.md](deploy/offline/LLAMA_CPP_RERANKER.md)。

并使用 `./tools/invoke_offline_compose.sh up -d reranker-service api` 启动核心服务。
旧的 `qwen2.5:3b` `/api/generate` 生成式 rerank 仅为兼容模式，不等价于专用 cross-encoder；
重新启用前必须完成目标服务器 15 并发容量测试并观测 429/503、延迟和 controlled fallback。
`/v1/rerank` 的 wire contract 始终接受 1–32 个 passages；`RERANKER_BATCH_MAX_ITEMS=32`
保证单个合法请求可以进入服务，`OLLAMA_RERANK_BATCH_MAX_ITEMS=8` 则只限制一次
`/api/generate` 的候选数，较大请求按连续分块执行并恢复原顺序。输出预算必须满足
`OLLAMA_RERANK_NUM_PREDICT >= 64 * OLLAMA_RERANK_BATCH_MAX_ITEMS`，默认 8 项使用 512；
修改分块大小或预算后必须重新执行目标机容量 gate。`RETRIEVAL_RERANK_TOP_K=8` 是当前检索
策略，不是 HTTP passages 上限。

没有可靠证据时不会调用模型。Physoc 超时、返回非 2xx、SSE 格式异常或答案为空时，API
返回 HTTP 502；`no raw-chunk answer on Physoc failure` 是强制规则，系统不得把检索切片、
Legacy 结果或模板文本伪装成模型答案。

这条链路是“检索少量相关证据后回答”，不是“读取整个知识库后进行全库总结”。跨章节、
跨文档和全库 Map-Reduce 汇总仍属于后续阶段。

### Excel/CSV 的两条处理路线

- `STRUCTURED_QUERY_ENABLED=false`（回滚/兼容模式）：XLSX/CSV 被展开为普通文本并进入切片 RAG。
  这种模式适合查找某行或某项说明，不适合计算整列平均值、总和等全量统计。
- `STRUCTURED_QUERY_ENABLED=true`（环境模板默认）：管理员必须确认字段类型、别名和统计权限，随后由
  `--profile indexing` worker 分批写入 Parquet 并发布到 ClickHouse。对于已成功发布且已通过
  行数和内容校验的 ClickHouse publication，`avg`、`sum`、`count`、`min`、`max` 由
  ClickHouse 对该 publication 的数据确定性计算，不调用 Physoc，也不会从文档切片估算结果。

这条路线称为 `ClickHouse complete-data aggregation`。Spreadsheet averages must not be
calculated from RAG chunks；Qdrant 中只保存表结构、字段和安全摘要，绝不把局部切片平均值当作
完整数据结果。

### Filtered Excel summaries

- Existing published Excel/CSV datasets do not need to be uploaded or published again.
- “汇总”/“统计” sums all `allowAggregate` integer/decimal columns and returns matched/valid/null counts.
- `STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS` defaults to 12; over-limit questions ask the user to choose fields.
- All metrics in one answer are calculated by one ClickHouse `SELECT`.
- ClickHouse or parsing failures return a structured error and never search Word/PDF chunks.

This Excel-only behavior does not require rebuilding Qdrant indexes or reindexing Word documents.

### Word factual-answer reindexing

Word factual answers use extracted `knowledge_facts`; existing Word sources must be reindexed, while
published Excel tables do not require re-uploading.

The routing rollout is disabled by default with
`UNIFIED_KNOWLEDGE_ROUTING_ENABLED=false` and `WORD_FACTUAL_QA_ENABLED=false`. Unified routing may
be enabled first without enabling Word facts; Word factual QA must never be enabled while unified
routing is disabled. For the Ubuntu/Supervisor deployment sequence, route-audit smoke checks, and
configuration-only rollback, follow
[`deploy/ubuntu/KNOWLEDGE_ROUTING_ROLLOUT.md`](deploy/ubuntu/KNOWLEDGE_ROUTING_ROLLOUT.md).

1. Apply Alembic revision 20260811_07.
2. Confirm every existing Word source still has a readable file_path.
3. POST /api/knowledge/sources/{source_id}/reindex once for each Word source.
4. Keep the old retrieval publication active until the normal Qdrant publication fence completes.
5. Verify records > 0, knowledge_facts contains rows, and “张三几岁” returns only age.
6. If conflicts appear, disable the factual route and inspect extracted facts; do not route the question to unrelated Word RAG.

这里的“精确统计”只承诺覆盖通过校验的 publication，不等同于无条件覆盖原工作簿中的每个
单元格。空值、错误值、公式结果、隐藏行、多 Sheet 合并方式和类型转换规则仍需在目标业务
数据集上冻结口径并验收。

如果 ClickHouse 不可用或查询超时，结构化问题必须显式失败，不得回退到普通 RAG 做切片
算术。

### 当前文件支持情况

| 文件类型 | 当前处理方式 | 当前限制 |
| --- | --- | --- |
| `.txt`、`.md` | UTF-8/GB18030 文本读取后切片 | 加密、损坏或超大文件未形成生产验收口径；不保留复杂结构 |
| `.docx` | 提取段落和表格文本 | 密码保护、损坏或超大文件未形成生产验收口径；不保留完整章节层级、页码和复杂布局 |
| 文本型 `.pdf` | 使用 `pypdf` 提取页面文字 | 密码保护、损坏、扫描页、字体映射异常和超大文件未形成生产验收口径 |
| `.xlsx`、`.csv` | 默认展开为文本；启用结构化功能后可发布到 ClickHouse | XLSX 多 Sheet、公式、合并单元格，以及 CSV 编码和分隔符规则需按数据集验收 |
| `.doc`、`.xls` | 上传入口兼容，但当前生产解析不支持 | 解析器会退化为不可靠的二进制文本读取，生产环境不得使用 |
| `.ppt`、`.pptx` | 当前上传入口和主解析链路不支持 | 计划在统一 Docling/PaddleOCR 阶段实现 |
| 图片 | 当前上传入口会拒绝 | 图片 OCR 尚未接入主解析链路 |

当前主解析器尚未接入 Docling 和 PaddleOCR。扫描 PDF、图片文字、PPTX、页码、章节、幻灯片、
表格范围、单元格范围及 OCR 置信度等结构化定位信息尚未进入普通问答链路。

当前回答中的引用主要用于标识资料来源和证据切片，并不保证定位到页码、章节、幻灯片或
单元格范围；这些细粒度引用尚未进入 API 和前端展示契约。

### Compose 服务与实际接入状态

离线 Compose 声明 PostgreSQL、ClickHouse、Qdrant、Redis、ClamAV、Embedding Service、
Reranker Service、API，以及可选的 indexing worker 和 llama.cpp profile。当前实际状态如下：

- PostgreSQL 已用于会话、文档、切片、Agent 审计和结构化元数据。
- ClickHouse 已用于启用后的 Excel/CSV 精确统计。
- Physoc 是独立部署的公司内网模型服务，不包含在本 Compose 项目中。
- Qdrant、Ollama BGE 中文 Embedding adapter、BM25、RRF 和生成式 Reranker adapter 已接入
  普通文档检索，并由
  `RETRIEVAL_MODE` 控制 Legacy、Shadow 和 Qwen3 路由。
- Redis 保留给后台任务和后续队列扩展；当前 Shadow 比较使用进程内有界队列。
- ClamAV 服务当前进入了部署健康检查，但文件上传路由尚未调用病毒扫描。

API readiness 是 mode-aware：`qwen3` 下 Embedding、Reranker、Qdrant、Alias 或模型元数据不匹配
会使 `/api/readyz` 返回 503；`shadow` 下 API 保持可用但把 Qwen3 依赖报告为 degraded；
`legacy` 不要求构造这些查询资源。Compose 容器健康只证明进程可用，不能替代索引质量、权限、
15 用户并发和业务答案验收。

### 安全与开放范围

当前 FastAPI 路由没有接入身份认证、用户主体或角色映射，CORS 目前也配置为
`allow_origins=["*"]`。混合检索会对部署配置中的 knowledge-base 和 permission tags 做
fail-closed 过滤，但这些静态标签还不能替代真实用户身份到部门/文档权限的映射。

因此，当前版本只能部署在网络和人员范围均受控的隔离验收环境，不得直接面向公司全员或其他
不受信任客户端开放。身份认证、管理员隔离、知识库授权、审计闭环和安全加固属于升级路线的
Phase 6；在这些门禁完成前，反向代理和网络 ACL 不能替代应用层权限控制。

### 搬入公司内网前的条件

项目运行时可以不调用公共 API，但不能只复制仓库目录后直接开放使用。目标环境至少需要：

1. 一台符合运行手册要求的 Linux 主机、rootful Docker 和 Compose v2。
2. 已审核并预置的内部基础镜像、Python wheelhouse、解析/Sparse artifact 和镜像 digest。
   两个前端还需要预构建静态产物，或可用的 Yarn 离线 cache/公司内部 npm 源；仓库本身不包含
   所有运行所需 artifact。
3. PostgreSQL、ClickHouse、Qdrant、Redis、ClamAV、Embedding Service 和 Reranker Service
   所需的数据目录、权限、校验和和 Secret 文件；Embedding/Reranker 权重由公司内网 Ollama
   承载，不再放入 DC-Agent 容器。
4. 两个 adapter 容器可以访问的私有 Ollama 地址，以及 API 容器可以访问的私有 Physoc 地址；
   Physoc 还需要正确的 `LLM_PROVIDER`、`LLM_API_BASE`、
   `LLM_STREAM_PATH` 和 `LLM_MODEL`。
5. 同机反向代理。离线 Compose 只将 API 发布到 `127.0.0.1:8000`，内网客户端不应直接访问
   容器端口。
6. 在目标服务器完成 Compose smoke、Ollama/Physoc probe、新索引发布、Shadow/Canary、真实文档
   问答、模型故障 HTTP 502、Alias/Legacy 回滚、结构化统计和 15 用户并发验收。

Ubuntu 20.04 新部署必须使用上文的事务部署固定顺序（包括 `--initialize-state`、五个服务的单次 build 和基础 `up`）；不得以普通 prepare 代替初始化。状态已初始化并且基础服务启动后，才可按需启动结构化 indexing worker：

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh --profile indexing up -d
```

Windows 开发机兼容：本地开发者可以继续使用 `tools/prepare_offline_env.ps1` 和
`tools/invoke_offline_compose.ps1`；这两个入口不是 Ubuntu 生产部署路径。

### Ollama BGE 中文部署准备与上线门禁

如果目标环境已经部署过旧版 DC-Agent，本次升级不能只替换后端代码。DC-Agent 本身不再运行
Embedding/Reranker 权重；公司内网 Ollama 必须可达，且不依赖外部 API。先在批准的 Ollama
主机拉取并探测两个模型：

#### 本次 Qwen3 混合检索修改的部署准备（历史兼容说明）

数据库迁移名 `20260727_04_qwen3_retrieval`、`20260728_05_shadow_evaluation_labels`，以及
`qwen3` route/collection 命名继续保留。旧手册中的
`artifacts/models/qwen3-embedding-0.6b`、`artifacts/models/qwen3-reranker-0.6b` 不再承载
运行权重，不得按旧方案恢复；`artifacts/models/qdrant-bm25` 仅用于批准的本地 Sparse artifact。
原有文档切片不会自动变成新的 Qdrant 索引，必须按下文全量重建。首次构建保持
`RETRIEVAL_MODE=shadow`、`RETRIEVAL_CANARY_PERCENT=0`，最终答案路由仍使用
`/api/physoc/deepseeks/stream`。

#### Linux (Bash)

```bash
set -Eeuo pipefail
ollama pull bge-large-zh-v1.5:latest
ollama pull qwen2.5:3b
ollama_url='http://127.0.0.1:11434'

embed_json="$(curl --fail-with-body --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"bge-large-zh-v1.5:latest","input":["dimension-probe"],"truncate":true,"keep_alive":"30m"}' \
  "$ollama_url/api/embed")"
dimensions="$(python3 -c 'import json,sys; body=json.load(sys.stdin); value=len(body["embeddings"][0]); assert value > 0; print(value)' <<<"$embed_json")"
printf 'EMBEDDING_MODEL_DIMENSIONS=%s\n' "$dimensions"

generate_body="$(python3 - <<'PY'
import json
prompt = 'Return only JSON: {"scores":[{"index":0,"score":0.0},{"index":1,"score":0.0}]}. Score passage relevance to the query from 0 to 1. Query: leave policy. Passage 0: annual leave policy. Passage 1: cafeteria menu.'
print(json.dumps({"model": "qwen2.5:3b", "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 128}}))
PY
)"
generate_json="$(curl --fail-with-body --silent --show-error \
  -H 'Content-Type: application/json' --data-binary "$generate_body" \
  "$ollama_url/api/generate")"
python3 -c 'import json,sys; envelope=json.load(sys.stdin); scores=json.loads(envelope["response"])["scores"]; assert len(scores) == 2; print(json.dumps(scores))' <<<"$generate_json"

tags_json="$(curl --fail-with-body --silent --show-error "$ollama_url/api/tags")"
for model in bge-large-zh-v1.5:latest qwen2.5:3b; do
  digest="$(python3 -c 'import json,re,sys; model=sys.argv[1]; body=json.load(sys.stdin); matches=[item for item in body["models"] if item.get("name") == model or item.get("model") == model]; len(matches) == 1 or sys.exit(f"expected exactly one model match: {model}"); digest=str(matches[0]["digest"]).removeprefix("sha256:"); re.fullmatch(r"[0-9a-f]{64}", digest) or sys.exit(f"invalid digest: {model}"); print(digest)' "$model" <<<"$tags_json")"
  printf '%s %s\n' "$model" "$digest"
done
```

把 Bash 探针输出的 `EMBEDDING_MODEL_DIMENSIONS` 填入实际环境文件。老 Ollama 只提供 legacy endpoint 时，显式
设置 `OLLAMA_EMBEDDING_PATH=/api/embeddings`，并把
`EMBEDDING_ENCODING_PROFILE_SHA256` 改为 legacy/BGE profile hash
`b8e7252a57feef349f02d6b2624ef3f9e8bc9e989d9073e37aa5df424cf26de4`；不得在任意错误后自动切换路径。
上面的 Bash 命令还必须确认 3B 模型能返回 JSON score shape，并从 `/api/tags` 提取目标模型的真实 digest；去掉可选
`sha256:` 前缀后再写入配置，不要复制示例占位符。

关键配置必须绑定实测值和固定 profile：

```env
EMBEDDING_MODEL_NAME=bge-large-zh-v1.5:latest
EMBEDDING_MODEL_VERSION=ollama-bge-large-zh-v15-v1
EMBEDDING_MODEL_DIMENSIONS=<len(embeddings[0])>
EMBEDDING_MODEL_SHA256=<真实 embedding digest，无 sha256: 前缀>
EMBEDDING_ENCODING_PROFILE_SHA256=3d5db261732d456b51fa4f9aa89cb15054c21772c0809a50a31f0911eb960170
OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
OLLAMA_EMBEDDING_PATH=/api/embed
OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5
RERANKER_MODEL_NAME=qwen2.5:3b
RERANKER_MODEL_SHA256=<真实 reranker digest，无 sha256: 前缀>
RERANKER_PROMPT_PROFILE_SHA256=e474bae5997a24385e95ae8fb3bef00ac066a9afe3999aa6e89ceae6d1c72bbd
```

升级后，缺少完整 embedding fingerprint 的旧 retrieval publication 会保持不可用；请用当前
model digest、实测 dimensions、protocol、endpoint、query profile 和前缀指纹重新构建。必须选择一个
从未使用过的 `knowledge_chunks_qwen3_vN` collection 名称，完成全量验证后才能激活 Alias。

目标防火墙只允许 DC-Agent 主机访问批准的 Ollama IP/端口；Ollama 侧代理/ACL 只开放
`/api/tags`、`/api/embed`（或 `/api/embeddings`）和 `/api/generate`。Compose 中只有
`embedding-service`、`reranker-service` 接入 `ollama-egress`，API、worker、数据库和 Qdrant
不得借此获得通用出口。

所有已有 Word/PDF/TXT/Excel 内容必须用新 BGE embedding 全量重建，不能复用 Qwen3 或其他模型旧
向量。保持 `knowledge_chunks_qwen3_vN` 和 `RETRIEVAL_MODE=qwen3` 兼容命名，先构建新的不可变
collection 并验证模型元数据、实测 dimensions、归一化、profile/digest、点数和检索质量：

```bash
./tools/invoke_offline_compose.sh exec -T api \
  python -m app.retrieval_index_worker --collection knowledge_chunks_qwen3_v1
```

通过所有人工 gate 后才允许使用新版本执行 `--activate`，由现有 publisher 原子切换
`knowledge_chunks_current` Alias 和 PostgreSQL publication 状态。保留旧 collection 与旧 env；
回滚时先把 `RETRIEVAL_MODE=legacy`，恢复上一组模型 digest/profile/dimensions，再以新的版本号
全量重建已知可用组合并执行受 fence 保护的 `--activate`，禁止直接修改 Qdrant Alias。

```bash
# 仅在目标 acceptance 全部通过后人工执行
./tools/invoke_offline_compose.sh exec -T api \
  python -m app.retrieval_index_worker --collection knowledge_chunks_qwen3_v2 --activate
```

`qwen2.5:3b` 生成式 rerank 只是兼容模式。8/4/4/20 配置分别为
`RETRIEVAL_RERANK_TOP_K=8`、`RETRIEVAL_DEGRADED_RERANK_TOP_K=4`、
`RETRIEVAL_FINAL_TOP_K=4`、`RETRIEVAL_TOTAL_TIMEOUT_SECONDS=20`；必须在目标服务器跑 15 并发
容量测试并观测 controlled fallback。上线 acceptance 至少要求：向量维度一致；adapter 无 5xx；
每个候选得到有限 `[0,1]` score 或记录脱敏 fallback；失败时不能只返回原始 chunks；Excel 聚合
继续走结构化 ClickHouse 路径；15-user gate 达到批准的延迟/错误/fallback 阈值。全部通过后才能
激活 Alias。

本机没有目标 wheelhouse。必须在真实离线构建环境中，用实际 artifacts/wheels 按 Dockerfile 的
相同 `uv sync` flags 构建并验证镜像，以弥补本机限制：

```bash
set -Eeuo pipefail
export UV_PYTHON_DOWNLOADS=never
uv sync --project backend --frozen --offline --no-install-project --no-dev --group offline --no-index --find-links artifacts/wheels
uv sync --project backend --frozen --offline --no-install-project --no-dev --no-index --find-links artifacts/wheels
./tools/invoke_offline_compose.sh build schema-migration embedding-service api ingestion-worker
```

真实 Ollama probe、镜像构建、全量索引、15 用户容量、目标服务器 acceptance 和 Alias activation
都是上线前人工 gate，本次仓库修改不会执行这些动作。

详细的 artifact 校验、构建、索引发布、监控和回滚命令见
[`deploy/offline/README.md`](deploy/offline/README.md)。

当前开发机没有执行真实目标服务器门禁，因此仓库具备内网部署路线，但不能据此声明任意内网
服务器均可“复制后立即使用”。

需要明确区分三个状态：本节所述功能已有代码实现；仓库中的相关自动化契约可在本地运行并
验证；真实目标服务器上的镜像、模型、网络、权限、数据和并发仍必须单独完成部署验收。前两项
不能替代第三项。

### 当前规模限制与升级路线

混合检索已经移除该路线中的 PostgreSQL 全表逐片评分瓶颈，但“代码已接入”不等于
“千万级容量已证明”。必须用批准数据集验证 Qdrant 点数、过滤选择率、模型队列、CPU/内存、
P95、错误率和 fallback rate；强制命令见离线部署手册。

后续重点是统一 Docling/PaddleOCR 解析、跨文档分层 Map-Reduce 汇总、真实用户权限、
Redis/Celery 异步任务、ClamAV 上传扫描、细粒度引用和千万级整体验收。详细阶段和退出门禁见
[`企业知识库升级路线`](docs/superpowers/plans/2026-07-24-enterprise-knowledge-base-qa-rollout.md)。

## 本地开发补充

管理端提供概览、知识库维护、指定资料详情和 Agent 执行审计等独立路由。UI smoke 使用
Playwright/Pillow 和独立 QA Python 环境，不由 backend UV dependency groups 管理；测试使用临时
SQLite 数据和临时端口，结束后临时数据会销毁。

安装后端开发依赖并启动本地 API：

```bash
set -Eeuo pipefail
uv sync --project backend --group dev
(
  cd backend
  uv run --project . --group dev python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
)
```

运行后端测试和 Ruff：

```bash
set -Eeuo pipefail
(
  cd backend
  uv run --project . --group dev python -m unittest discover -s tests -p "test_*.py" -v
)
uv run --project backend --group dev ruff check backend
uv run --project backend --group dev ruff format backend
```

## 离线平台运行手册

离线单机部署、依赖锁定、模型与解析器 artifact 审核、Compose profile、容量门禁以及 32GB/64GB 结果记录，统一见 [`docs/offline-platform-runbook.md`](docs/offline-platform-runbook.md)。当前开发机没有 Docker，不能在本机宣称 Compose smoke 或容量门禁通过；请在满足手册前置条件的目标 Linux 主机上执行这些门禁。

## 当前能力

- 用户侧一次性知识检索问答，不展示历史会话侧栏。
- 首次从小输入框进入大聊天时带启动动画和 loading。
- DCAgent 回答支持等待态和逐步显现。
- 用户侧不暴露管理员资料原文，只在回答文本中保留必要引用。
- LangGraph 只读 Agent 最多执行两轮检索、深入检查三个资料来源，并将最多五条证据交给模型。
- Agent 会在证据不足时自动扩展检索词，命中多个来源时执行证据对比；没有证据时拒绝调用模型编造答案。
- 管理端采用路由级模块拆分，支持管理概览、知识库维护、资料解析详情和 Agent 执行审计。
- 知识库模块支持多文件上传、解析状态轮询、失败原因展示、重新解析、单条删除、批量删除和列表筛选。
- 后端支持 PostgreSQL 持久化、上传文件解析、知识片段索引、基础语义扩展检索、Agent 审计持久化和 OpenAI-compatible LLM 接入。
