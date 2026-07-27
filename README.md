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

> 本节描述 `main` 分支当前已经接入的运行链路，不代表路线图中的后续目标架构已经完成。

```mermaid
flowchart TD
    A["管理端上传文件"] --> B["FastAPI API"]
    B --> C["本地原始文件目录"]
    B --> D{"是否启用结构化 Excel/CSV"}

    D -->|"否或普通文档"| E["基础文件解析器"]
    E --> F["600 字符切片 / 120 字符重叠"]
    F --> G["PostgreSQL 文档、切片和轻量哈希向量"]

    D -->|"是"| H["推断并人工确认表结构"]
    H --> I["结构化 indexing worker"]
    I --> J["Parquet 暂存"]
    I --> K["ClickHouse 不可变发布表"]

    L["用户提问"] --> M{"是否为已发布数据集的统计问题"}
    M -->|"avg / sum / count / min / max"| K
    K --> N["确定性统计答案，不调用大模型"]
    K -.->|"不可用或超时"| X["显式失败，不回退到切片计算"]

    M -->|"普通文档问题"| O["LangGraph 只读检索 Agent"]
    O --> G
    G --> P["最多 5 个证据切片"]
    P -.->|"无可靠证据"| Y["拒绝回答，不调用模型"]
    P --> Q["Physoc DeepSeek POST/SSE"]
    Q --> R["基于证据归纳的最终答案"]
    Q -.->|"超时、非 2xx 或异常 SSE"| Z["HTTP 502"]
```

### 普通文档问答链路

当前普通文档问答属于检索增强生成（RAG）：

1. 上传文件保存在 API 主机的受控原始文件目录中。
2. 解析器提取文本，并按 600 个字符切片、保留 120 个字符重叠。
3. 切片和 48 维轻量哈希向量保存在 PostgreSQL。
4. 查询时从 PostgreSQL 读取已索引切片，在 Python 中计算关键词和向量分数。
5. LangGraph Agent 最多保留 5 个证据切片、深入检查 3 个资料来源，并把证据、调查摘要和最近 6 条消息组成完整 RAG 提示词。
6. `physoc_deepseek` Provider 将完整提示词通过 `POST /api/physoc/deepseeks/stream` 发送给私有 Physoc 服务，收集 SSE `message` 事件中的 `response`，再返回归纳后的答案。

没有可靠证据时不会调用模型。Physoc 超时、返回非 2xx、SSE 格式异常或答案为空时，API
返回 HTTP 502；系统不得把检索切片或模板文本伪装成模型答案。

这条链路是“检索少量相关证据后回答”，不是“读取整个知识库后进行全库总结”。跨章节、
跨文档和全库 Map-Reduce 汇总仍属于后续阶段。

### Excel/CSV 的两条处理路线

- `STRUCTURED_QUERY_ENABLED=false`（默认）：XLSX/CSV 被展开为普通文本并进入切片 RAG。
  这种模式适合查找某行或某项说明，不适合计算整列平均值、总和等全量统计。
- `STRUCTURED_QUERY_ENABLED=true`：管理员必须确认字段类型、别名和统计权限，随后由
  `--profile indexing` worker 分批写入 Parquet 并发布到 ClickHouse。对于已成功发布且已通过
  行数和内容校验的 ClickHouse publication，`avg`、`sum`、`count`、`min`、`max` 由
  ClickHouse 对该 publication 的数据确定性计算，不调用 Physoc，也不会从文档切片估算结果。

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

离线 Compose 声明 PostgreSQL、ClickHouse、Qdrant、Redis、ClamAV、Embedding Service、API，
以及可选的 indexing worker 和 llama.cpp profile。当前实际状态如下：

- PostgreSQL 已用于会话、文档、切片、Agent 审计和结构化元数据。
- ClickHouse 已用于启用后的 Excel/CSV 精确统计。
- Physoc 是独立部署的公司内网模型服务，不包含在本 Compose 项目中。
- Qdrant、Redis 和独立 Embedding Service 当前主要完成了离线部署与健康检查契约，普通文档
  问答仍未迁移到 Qdrant/BGE 检索。
- ClamAV 服务当前进入了部署健康检查，但文件上传路由尚未调用病毒扫描。

Compose 会把 Qdrant、Redis、ClamAV 和 Embedding Service 的健康状态作为部署启动门禁；这只
表示这些容器必须成功启动并通过健康检查。它们尚未成为普通问答业务主链路，因此不能据此
认为混合向量检索、异步任务队列和上传安全扫描已经启用。

### 安全与开放范围

当前 FastAPI 路由没有接入身份认证、用户主体、角色或知识库级权限校验，CORS 目前也配置为
`allow_origins=["*"]`。管理端和用户端之间尚无管理员隔离、部门隔离或文档授权边界。

因此，当前版本只能部署在网络和人员范围均受控的隔离验收环境，不得直接面向公司全员或其他
不受信任客户端开放。身份认证、管理员隔离、知识库授权、审计闭环和安全加固属于升级路线的
Phase 6；在这些门禁完成前，反向代理和网络 ACL 不能替代应用层权限控制。

### 搬入公司内网前的条件

项目运行时可以不调用公共 API，但不能只复制仓库目录后直接开放使用。目标环境至少需要：

1. 一台符合运行手册要求的 Linux 主机、rootful Docker 和 Compose v2。
2. 已审核并预置的内部基础镜像、Python wheelhouse、Embedding/解析模型文件和镜像 digest。
   两个前端还需要预构建静态产物，或可用的 npm 离线缓存/公司内部 npm 源；仓库本身不包含
   所有运行所需 artifact。
3. PostgreSQL、ClickHouse、Qdrant、Redis、ClamAV 和 Embedding Service 所需的数据目录、权限和 Secret 文件。
4. API 容器可以访问的私有 Physoc 地址，以及正确的 `LLM_PROVIDER`、`LLM_API_BASE`、
   `LLM_STREAM_PATH` 和 `LLM_MODEL`。
5. 同机反向代理。离线 Compose 只将 API 发布到 `127.0.0.1:8000`，内网客户端不应直接访问
   容器端口。
6. 在目标服务器完成 Compose smoke、Physoc probe、真实文档问答、模型故障 HTTP 502、
   配置恢复和结构化统计验收。

当前开发机没有执行真实目标服务器门禁，因此仓库具备内网部署路线，但不能据此声明任意内网
服务器均可“复制后立即使用”。

需要明确区分三个状态：本节所述功能已有代码实现；仓库中的相关自动化契约可在本地运行并
验证；真实目标服务器上的镜像、模型、网络、权限、数据和并发仍必须单独完成部署验收。前两项
不能替代第三项。

### 当前规模限制与升级路线

普通文档检索目前会从 PostgreSQL 读取已索引切片，并在 Python 中逐条计算分数。该实现适合
功能验证和中小规模知识库，不适合数百万或数千万普通文档切片。

面向大规模企业知识库的后续顺序是：统一 Docling/PaddleOCR 解析、BGE-M3 Dense/Sparse
Embedding、Qdrant 混合检索、BGE Reranker、相邻上下文扩展、分层 Map-Reduce 汇总，最后补齐
Redis/Celery 异步任务、ClamAV 上传扫描、权限、引用和千万级验收。详细阶段和退出门禁见
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

```powershell
uv run --project backend --group dev ruff check backend
uv run --project backend --group dev ruff format --check backend
uv run --project backend --group dev ruff format backend
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
