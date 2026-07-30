# DC-Agent

DC-Agent 是一个公司内部只读知识 Agent。管理员把制度、合同、会议纪要、经营数据等资料上传到知识库后，DCAgent 会围绕用户问题执行有界的多步检索、资料深挖和证据对比，再基于已索引资料生成回答。

## 项目结构

- `frontend`：用户检索端，面向普通用户提问和查看 DCAgent 生成的答案。
- `admin-frontend`：知识库管理端，面向管理员上传、筛选、重新解析、删除资料源，查看解析片段和 Agent 执行审计。
- `backend`：Python + FastAPI + LangGraph 服务，提供只读 Agent、问答、资料上传、解析、知识库索引和审计接口。

## 本地环境

建议环境：

- Python 3.12.x（不支持 3.13）
- Node.js 20 或更高版本
- PostgreSQL，本地默认库名为 `dc_agent`

后端默认数据库连接：

```text
postgresql+psycopg://postgres:123456@127.0.0.1:5432/dc_agent
```

手动创建数据库示例：

```powershell
$env:PGPASSWORD="123456"
D:\PostgreSQL\18\bin\createdb.exe -h 127.0.0.1 -p 5432 -U postgres dc_agent
Remove-Item Env:PGPASSWORD
```

也可以用 `DATABASE_URL` 覆盖默认连接串。管理员上传的文件默认保存在 `backend/uploads/knowledge`，该目录只用于本地运行数据，不进入版本管理。

## 环境变量

后端启动时会自动读取项目根目录 `.env` 和 `backend/.env`。读取顺序为根目录 `.env` 后读取 `backend/.env`，但系统环境变量优先级最高，不会被文件覆盖。

复制示例文件：

```powershell
Copy-Item .env.example .env
Copy-Item backend\.env.example backend\.env
```

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

```powershell
& tools/invoke_offline_compose.ps1 exec -T api `
  python -m app.physoc_probe --report /tmp/physoc-probe.json
if ($LASTEXITCODE -ne 0) { throw "Physoc probe failed; do not persist evidence." }
New-Item -ItemType Directory -Force artifacts/benchmarks | Out-Null
& tools/invoke_offline_compose.ps1 cp api:/tmp/physoc-probe.json artifacts/benchmarks/physoc-probe.json
```

探针成功报告只记录 provider、model、streamPath、elapsedMs、answerChars 和 citationCount 等运行元数据，不会输出提示词、证据正文或模型回答正文。只有容器内 probe exit 0 后才创建 host 目录并执行 `cp`；将 `artifacts/benchmarks/physoc-probe.json` 作为切换门禁证据保存。探针失败时不得复制旧报告或启用该生产路由。

## Structured spreadsheet aggregation

Exact Excel/CSV aggregation is an opt-in local feature. The shipped default is
`STRUCTURED_QUERY_ENABLED=false`, which preserves the existing template/legacy document RAG path.
Enabling it does not add an external API: published rows stay in local Parquet staging and the
private ClickHouse service.

The query API uses the `CLICKHOUSE_QUERY_USER` account and its password file; the indexing worker
uses the separate `CLICKHOUSE_INGEST_USER` account and password file. Password values must not be
placed in `.env` or an example file. The 4-second timeout applies only to API aggregate connection
and query execution. Structured publication keeps the storage gateway's independent 30-second
execution default, and the default bounded ingestion batch is 50,000 rows.

The feature is usable only after an administrator has approved a confirmed schema for the XLSX/CSV
dataset and the offline `--profile indexing` worker has published that schema version. See
[`deploy/offline/README.md`](deploy/offline/README.md) for migration, worker startup, smoke aggregate,
rollback, and ClickHouse failure handling.

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
    O -->|"shadow / qwen3"| P1["Qwen2.5 Embedding adapter + BM25"]
    P1 --> P2["Qdrant Dense + Sparse"]
    P2 --> P3["RRF 融合 + Qwen2.5 生成式 Reranker adapter"]
    P3 --> P["Top 8 授权证据"]
    P0 --> P
    P -.->|"无可靠证据"| Y["拒绝回答，不调用模型"]
    P --> Q["Physoc DeepSeek POST/SSE"]
    Q --> R["基于证据归纳的最终答案"]
    Q -.->|"超时、非 2xx 或异常 SSE"| Z["HTTP 502"]
```

### Ollama Qwen2.5 混合检索链路（保留 qwen3 兼容命名）

普通文档问答采用 `Qdrant Dense + Sparse/BM25 + RRF`：

1. PostgreSQL 保存权威文档、切片、权限标签和发布状态；Qdrant 保存版本化检索点。
2. 轻量 Embedding adapter 通过公司内网 Ollama 的 `qwen2.5:0.5b` 生成 Dense vector，并按
   固定 profile 做归一化；维度必须以目标 `/api/embed` 的 `len(embeddings[0])` 实测值为准。
   本地 BM25 生成 Sparse query vector。
3. 两路检索都先应用 knowledge-base、permission-tag 和 publication filters，各取 Top 50。
4. RRF（`k=60`）融合候选，Reranker adapter 通过 `qwen2.5:3b` 的 `/api/generate` 生成并严格
   校验 `[0,1]` JSON scores。初始 8/4/4/20 配置是 `RETRIEVAL_RERANK_TOP_K=8`、
   `RETRIEVAL_DEGRADED_RERANK_TOP_K=4`、`RETRIEVAL_FINAL_TOP_K=4`、
   `RETRIEVAL_TOTAL_TIMEOUT_SECONDS=20`，最终只给 Agent 提供有界授权证据。
5. Agent 把授权证据、调查摘要和近期会话组成完整 RAG 提示词，再通过
   `POST /api/physoc/deepseeks/stream` 交给私有 Physoc DeepSeek 归纳。

`RETRIEVAL_MODE=legacy|shadow|qwen3` 的语义如下：

- `legacy`：只运行 PostgreSQL Legacy 检索，不构造混合检索/Qdrant 查询资源。
- `shadow`：前台仍返回 Legacy 结果；按 `RETRIEVAL_SHADOW_PERCENT` 在后台运行混合检索对比，
  只存 case/chunk ID、排名指标、耗时和脱敏失败码，不保存原始问题或证据正文。
- `qwen3`：为保持 collection、数据库记录和环境变量兼容而保留的路由名；按稳定会话哈希和
  `RETRIEVAL_CANARY_PERCENT` 选择 Ollama-backed 混合检索；未命中 canary 或
  Embedding、Reranker、Qdrant、Alias、超时/熔断异常时安全回退 Legacy。

DC-Agent 本身不再运行 Embedding/Reranker 权重；只有两个 adapter 服务调用公司内网可达的
Ollama，不依赖外部 API。`qwen2.5:3b` 生成式 rerank 是兼容模式，不等价于专用 cross-encoder，
必须在目标服务器完成 15 并发容量测试并观测 429/503、延迟和 controlled fallback。
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

- `STRUCTURED_QUERY_ENABLED=false`（默认）：XLSX/CSV 被展开为普通文本并进入切片 RAG。
  这种模式适合查找某行或某项说明，不适合计算整列平均值、总和等全量统计。
- `STRUCTURED_QUERY_ENABLED=true`：管理员必须确认字段类型、别名和统计权限，随后由
  `--profile indexing` worker 分批写入 Parquet 并发布到 ClickHouse。对于已成功发布且已通过
  行数和内容校验的 ClickHouse publication，`avg`、`sum`、`count`、`min`、`max` 由
  ClickHouse 对该 publication 的数据确定性计算，不调用 Physoc，也不会从文档切片估算结果。

这条路线称为 `ClickHouse complete-data aggregation`。Spreadsheet averages must not be
calculated from RAG chunks；Qdrant 中只保存表结构、字段和安全摘要，绝不把局部切片平均值当作
完整数据结果。

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
- Qdrant、Ollama Qwen2.5 Embedding adapter、BM25、RRF 和生成式 Reranker adapter 已接入
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
   两个前端还需要预构建静态产物，或可用的 npm 离线缓存/公司内部 npm 源；仓库本身不包含
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

### Ollama Qwen2.5 部署准备与上线门禁

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
set -euo pipefail
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b
ollama_url='http://127.0.0.1:11434'

embed_json="$(curl --fail-with-body --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"qwen2.5:0.5b","input":["dimension-probe"],"truncate":true,"keep_alive":"30m"}' \
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
for model in qwen2.5:0.5b qwen2.5:3b; do
  digest="$(python3 -c 'import json,re,sys; model=sys.argv[1]; body=json.load(sys.stdin); matches=[item for item in body["models"] if item.get("name") == model or item.get("model") == model]; len(matches) == 1 or sys.exit(f"expected exactly one model match: {model}"); digest=str(matches[0]["digest"]).removeprefix("sha256:"); re.fullmatch(r"[0-9a-f]{64}", digest) or sys.exit(f"invalid digest: {model}"); print(digest)' "$model" <<<"$tags_json")"
  printf '%s %s\n' "$model" "$digest"
done
```

#### Windows (PowerShell)

```powershell
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b
$ollama = 'http://127.0.0.1:11434'
$embedBody = @{ model = 'qwen2.5:0.5b'; input = @('dimension-probe'); truncate = $true; keep_alive = '30m' } | ConvertTo-Json -Compress
$embedJson = curl.exe --fail-with-body --silent --show-error -H 'Content-Type: application/json' --data-binary $embedBody "$ollama/api/embed"
$embedProbe = $embedJson | ConvertFrom-Json
$dimensions = $embedProbe.embeddings[0].Count
if ($dimensions -le 0) { throw 'Ollama returned no embedding dimensions' }
```

把 `$dimensions` 填入 `EMBEDDING_MODEL_DIMENSIONS`。老 Ollama 只提供 legacy endpoint 时，显式
设置 `OLLAMA_EMBEDDING_PATH=/api/embeddings`，并把
`EMBEDDING_ENCODING_PROFILE_SHA256` 改为 legacy profile hash
`23e5b954b6099dcc4427a33745ad03b9ce7dc6fbf2d8fd4728f1d7e1ce7db34c`；不得在任意错误后自动切换路径。用原生生成接口
确认 3B 模型能返回 JSON score shape：

```powershell
$prompt = 'Return only JSON: {"scores":[{"index":0,"score":0.0},{"index":1,"score":0.0}]}. Score relevance from 0 to 1. Query: leave policy. Passage 0: annual leave policy. Passage 1: cafeteria menu.'
$generateBody = @{ model = 'qwen2.5:3b'; prompt = $prompt; stream = $false; format = 'json'; options = @{ temperature = 0; num_predict = 128 } } | ConvertTo-Json -Depth 5 -Compress
$generateJson = curl.exe --fail-with-body --silent --show-error -H 'Content-Type: application/json' --data-binary $generateBody "$ollama/api/generate"
$scoreProbe = (($generateJson | ConvertFrom-Json).response | ConvertFrom-Json)
if ($scoreProbe.scores.Count -ne 2) { throw 'Ollama JSON score probe returned the wrong count' }
```

从 `/api/tags` 提取目标模型的真实 digest，去掉可选 `sha256:` 前缀后再写入配置。不要复制示例
占位符：

```powershell
$tags = (curl.exe --fail-with-body --silent --show-error "$ollama/api/tags") | ConvertFrom-Json
function Get-OllamaDigest([string]$model) {
  $modelMatches = @($tags.models | Where-Object { $_.name -ceq $model -or $_.model -ceq $model })
  if ($modelMatches.Count -ne 1) { throw "Expected exactly one Ollama model match: $model" }
  $digest = ([string]$modelMatches[0].digest) -replace '^sha256:', ''
  if ($digest -cnotmatch '^[0-9a-f]{64}$') { throw "Invalid Ollama digest for $model" }
  $digest
}
$embeddingDigest = Get-OllamaDigest 'qwen2.5:0.5b'
$rerankerDigest = Get-OllamaDigest 'qwen2.5:3b'
```

关键配置必须绑定实测值和固定 profile：

```env
EMBEDDING_MODEL_NAME=qwen2.5:0.5b
EMBEDDING_MODEL_DIMENSIONS=<len(embeddings[0])>
EMBEDDING_MODEL_SHA256=<真实 embedding digest，无 sha256: 前缀>
EMBEDDING_ENCODING_PROFILE_SHA256=fc5141eb8e304cacf598a7ad39ba75dbed3f22fa144c81f918ec58cd1efa3d10
RERANKER_MODEL_NAME=qwen2.5:3b
RERANKER_MODEL_SHA256=<真实 reranker digest，无 sha256: 前缀>
RERANKER_PROMPT_PROFILE_SHA256=e474bae5997a24385e95ae8fb3bef00ac066a9afe3999aa6e89ceae6d1c72bbd
```

升级后，缺少完整 embedding fingerprint 的旧 retrieval publication 会保持不可用；请用当前
model digest、dimensions、protocol 和所选 endpoint profile 重新构建并激活 collection。

目标防火墙只允许 DC-Agent 主机访问批准的 Ollama IP/端口；Ollama 侧代理/ACL 只开放
`/api/tags`、`/api/embed`（或 `/api/embeddings`）和 `/api/generate`。Compose 中只有
`embedding-service`、`reranker-service` 接入 `ollama-egress`，API、worker、数据库和 Qdrant
不得借此获得通用出口。

所有已有 Word/PDF/TXT/Excel 内容必须用新 embedding 全量重建，不能复用 Qwen3 或其他模型旧
向量。保持 `knowledge_chunks_qwen3_vN` 和 `RETRIEVAL_MODE=qwen3` 兼容命名，先构建新的不可变
collection 并验证模型元数据、实测 dimensions、归一化、profile/digest、点数和检索质量：

```powershell
& tools/invoke_offline_compose.ps1 exec -T api `
  python -m app.retrieval_index_worker --collection knowledge_chunks_qwen3_v1
```

通过所有人工 gate 后才允许使用新版本执行 `--activate`，由现有 publisher 原子切换
`knowledge_chunks_current` Alias 和 PostgreSQL publication 状态。保留旧 collection 与旧 env；
回滚时先把 `RETRIEVAL_MODE=legacy`，恢复上一组模型 digest/profile/dimensions，再以新的版本号
全量重建已知可用组合并执行受 fence 保护的 `--activate`，禁止直接修改 Qdrant Alias。

```powershell
# 仅在目标 acceptance 全部通过后人工执行
& tools/invoke_offline_compose.ps1 exec -T api `
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

```powershell
$env:UV_PYTHON_DOWNLOADS = 'never'
uv sync --project backend --frozen --offline --no-install-project --no-dev --group offline --no-index --find-links artifacts/wheels
uv sync --project backend --frozen --offline --no-install-project --no-dev --no-index --find-links artifacts/wheels
& tools/invoke_offline_compose.ps1 build schema-migration embedding-service reranker-service api ingestion-worker
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

## 启动

后端：

```powershell
uv sync --project backend --group dev
Set-Location backend
uv run --project . --group dev python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Set-Location ..
```

用户检索端：

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

默认访问地址：`http://127.0.0.1:5173`

知识库管理端：

```powershell
cd admin-frontend
npm.cmd install
npm.cmd run dev
```

默认访问地址：`http://127.0.0.1:5174`

管理端按功能拆分为独立路由：

- `/overview`：管理概览与最近活动。
- `/knowledge`：资料上传、筛选、重建索引和删除。
- `/knowledge/{sourceId}`：指定资料的解析详情与片段预览。
- `/agent-runs`：DCAgent 只读执行审计。

## 本地验证

后端测试：

```powershell
Set-Location backend
uv run --project . --group dev python -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..
```

后端代码质量：

> 历史文档曾写 `uv run --project backend --group dev ruff check backend` 和
> `uv run --project backend --group dev ruff format backend`，但当前 `dev` group 不包含 Ruff；
> 不要使用这些旧命令。以下 `uvx ruff` 调用已在本仓库实际验证。

```powershell
uvx ruff check backend tools
uvx ruff format --check backend tools
uvx ruff format backend tools
```

用户检索端：

```powershell
cd frontend
npm.cmd run test:run
npm.cmd run build
```

知识库管理端：

```powershell
cd admin-frontend
npm.cmd run test:run
npm.cmd run build
```

页面级冒烟（可选，需先确保已安装 Playwright Chromium）：

UI smoke 使用独立、已安装 Playwright/Pillow 的 QA Python 环境，不由 backend UV dependency groups 管理。

```powershell
# 终端 1
tools\start_smoke_backend.cmd

# 终端 2
tools\start_smoke_frontend.cmd

# 终端 3
tools\start_smoke_admin.cmd

# 终端 4
py tools\ui_smoke.py
```

冒烟脚本会使用临时后端 `8015`、用户端 `5177`、管理端 `5178`，截图输出到 `qa-screenshots`。

冒烟流程还会在临时 SQLite 环境中验证质量评测工作台：导入预览不会提前落库，确认后可创建评测案例，两个不同阈值的批次能够完成，并可查看报告详情、批次比较以及桌面端和 390×844 移动端布局。临时服务退出后测试数据自动销毁。

完整冒烟建议：

1. 启动后端。
2. 启动知识库管理端，上传一份 `.txt`、`.md`、`.docx`、`.xlsx`、`.csv`，或文本型、未加密且在已验收大小内的 `.pdf` 文档。
3. 等待资料源状态从 `解析中` 变为 `已索引`。
4. 启动用户检索端，询问文档中的制度、合同或业务问题。
5. 确认 DCAgent 回答只基于知识库内容，不在用户侧暴露资料原文管理入口。
6. 回到知识库管理端，确认“Agent 执行审计”中出现本次检索、资料检查、证据对比和回答生成步骤。

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
