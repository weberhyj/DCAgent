# DC-Agent 公司内网部署配置清单

本文用于把 DC-Agent 部署到公司内网。当前生产路线为：

- 普通 Word、文本型 PDF、TXT、Markdown 等文档：解析与切片 → BGE 中文 Embedding → Qdrant Dense/Sparse 混合检索 → Qwen2.5 生成式 Reranker → Physoc DeepSeek 汇总回答。
- Excel/CSV 统计问题：管理员确认表结构 → structured worker → Parquet/ClickHouse → 精确执行求和、平均、计数、去重计数、最值、中位数、分位数、方差、标准差、分组和 Top/Bottom N。
- 不调用公网大模型 API。Embedding 和 Reranker 通过公司内网 Ollama 提供，大模型回答通过公司内网 Physoc 接口提供。

除非内网拓扑与当前部署方式不兼容，一般不需要修改源代码。需要修改的是服务器上的实际环境文件、密码文件、镜像地址、服务地址和反向代理配置。

## Ubuntu 20.04 事务部署与恢复

生产主路径仅为 Ubuntu 20.04 Bash：`prepare_offline_env.sh`、`invoke_offline_compose.sh` 与
`recover_offline_deployment.sh`。PowerShell 仅用于 Windows 开发机。`DEPLOYMENT_STATE_ROOT` 固定为
`DATA_ROOT/.dcagent-deployment-state` 并与 data/model/secret roots 绑定；普通 prepare/Compose
不隐式创建 identity，更换 `DATA_ROOT` 视为新部署。

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

旧部署必须先接管，再普通 prepare：

```bash
set -Eeuo pipefail
export HOST_DATA_ROOT=/absolute/data/root
export HOST_MODEL_ROOT=/absolute/model/root
./tools/recover_offline_deployment.sh adopt-existing --state-root /absolute/data/root/.dcagent-deployment-state
./tools/prepare_offline_env.sh
```

部署锁超时为 30 秒。六个 Compose verb：config/build/up/down/exec/cp；
`./tools/invoke_offline_compose.sh up`、`./tools/invoke_offline_compose.sh exec`、
`./tools/invoke_offline_compose.sh cp` 在执行前 durable 写入 `deployment-started.json`，失败保留；
`./tools/invoke_offline_compose.sh config`、`./tools/invoke_offline_compose.sh build`、
`./tools/invoke_offline_compose.sh down` 不写 marker。marker 存在时普通 `--rotate-secrets` 拒绝；只有经
`recover_offline_deployment.sh clear-start-marker` 且确认无 `PG_VERSION`、无未完成事务后，才可能恢复
pre-init rotation。任意形态 `PG_VERSION` 存在后永久拒绝；不提供在线 PostgreSQL role 密码修改或单行删除 marker 命令。

先用 `./tools/recover_offline_deployment.sh inspect --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>` 分类。
自动回滚完成后可继续；`rollback_failed` 用
`./tools/recover_offline_deployment.sh resume-rollback --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>`；
`committed_cleanup_required` 用
`./tools/recover_offline_deployment.sh finalize-cleanup --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>`；
损坏 journal/quarantine 人工修复后用
`./tools/recover_offline_deployment.sh acknowledge-repaired --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id> --evidence /absolute/path/sanitized-repair-evidence.json`。
人工运行 `./tools/recover_offline_deployment.sh clear-start-marker --state-root /absolute/data/root/.dcagent-deployment-state` 前，确认无 DC-Agent 容器、无 `PG_VERSION`、PostgreSQL 目录不存在或未初始化、无未完成事务。日志和 evidence receipt 不含 secret、数据库 URL、模型正文或原始 SSE。

开发机本地测试不是Ubuntu live gate通过；缺少真实 Docker、Physoc、Ollama 拓扑时只能记录未运行。

## 1. 部署前检查

1. 从 GitHub `main` 分支拉取最新代码，并记录部署的 Commit SHA。
2. 准备 Ubuntu 20.04、Bash、Python 3.12、uv、Docker Engine 和 Docker Compose v2。
3. 确认后端服务器或 API 容器可以访问 PostgreSQL、ClickHouse、Qdrant、Redis、Ollama 和 Physoc。
4. 确认内网 DNS、防火墙、端口和容器网络已经放行。
5. 不要把真实密码、Token、私有 IP 清单或证书提交到 Git。

推荐使用仓库提供的 Compose 部署。Compose 目录虽然名为 `offline`，这里表示不依赖公网运行，并不要求使用本地离线大模型。当前部署应使用 Physoc 和 Ollama，不要启用可选的 `generation` profile。

## 2. 创建实际环境文件

Compose 部署首次准备时直接运行 `./tools/prepare_offline_env.sh --initialize-state`；不要手工复制模板。首次运行
该命令会自动创建 `deploy/offline/.env`，读取当前非 root 部署账号的
`id -u` 和 `id -g`，并写入 `DCAGENT_UID` 和 `DCAGENT_GID`。如果已有配置中的
`DCAGENT_UID` 或 `DCAGENT_GID` 与当前账号不匹配，脚本会 fail closed，不匹配时拒绝继续。

```bash
set -Eeuo pipefail
./tools/prepare_offline_env.sh --initialize-state
```

随后修改：

```text
deploy/offline/.env
```

如果不使用 Compose、直接运行后端，则创建并修改：

```bash
cp backend/.env.example backend/.env
```

`.env.example` 只是模板。生产配置必须写入实际 `.env` 或由部署平台注入系统环境变量。

## 3. 配置 Physoc DeepSeek

在实际环境文件中配置：

```env
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://<Physoc内网IP>:<端口>
LLM_STREAM_PATH=/api/physoc/deepseeks/stream
LLM_MODEL=my_deepseek_r1_7b
```

示例：

```env
LLM_API_BASE=http://172.16.0.10:8090
```

注意事项：

- Physoc 路线不需要 `LLM_API_KEY`。
- 后端会向 `LLM_API_BASE + LLM_STREAM_PATH` 发送 `POST` 请求，请求体包含 `query` 和 `model`。
- 响应必须是 `text/event-stream`，SSE `message` 的 JSON 数据中使用 `response` 和 `done` 字段。
- 容器内的 `127.0.0.1` 指向容器自身。如果 Physoc 不在 API 容器内，必须填写 API 容器可以访问的内网 IP 或内网域名。
- 前端不直接访问 Physoc，由后端统一调用。

部署后可以在 API 容器中运行探针：

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
```

探针返回非零状态时，不要切换生产流量。

## 当前生产模型链路

Ubuntu + Supervisor 生产默认使用 `RETRIEVAL_MODE=qwen3`、`RERANKER_ENABLED=true`，
Embedding 为 `bge-m3-Q4_K_M.gguf`，Reranker 为 `bge-reranker-v2-m3-Q4_K_M.gguf`，两者均通过 llama.cpp
的 `/v1/embeddings` 与 `/v1/rerank` 提供服务。下方 Ollama/RRF-only 章节仅用于旧环境兼容和应急回滚。

## 4. 配置 Ollama Embedding（兼容回滚）

> 当前 Ubuntu + Supervisor 生产方案已切换为 llama.cpp BGE-M3 Q4 Embedding 和
> BGE-Reranker-v2-M3 Q4。请优先执行
> `deploy/ubuntu/LLAMA_CPP_EMBEDDING.md`；本节 Ollama 内容仅保留给旧环境兼容与回滚。

默认检索链路为 `Dense + BM25 + RRF + BGE-Reranker-v2-M3 → Top 8 → 邻接扩展 → Physoc DeepSeek 总结`：

```env
RETRIEVAL_MODE=qwen3
RERANKER_ENABLED=true
RETRIEVAL_FINAL_TOP_K=8
OLLAMA_BASE_URL=http://ollama.inner:11434
OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
```

默认部署构建、启动并探测 llama.cpp Reranker；只有明确设置 `RERANKER_ENABLED=false` 才进入 RRF-only 应急回滚。

在 Ollama 服务器上只准备 Embedding 模型：

```bash
set -Eeuo pipefail
ollama pull bge-large-zh-v1.5:latest

ollama_url='http://127.0.0.1:11434'
embed_json="$(curl --fail-with-body --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"bge-large-zh-v1.5:latest","input":["dimension-probe"],"truncate":true,"keep_alive":"30m"}' \
  "$ollama_url/api/embed")"
dimensions="$(python3 -c 'import json,sys; body=json.load(sys.stdin); value=len(body["embeddings"][0]); assert value > 0; print(value)' <<<"$embed_json")"
printf 'EMBEDDING_MODEL_DIMENSIONS=%s\n' "$dimensions"

tags_json="$(curl --fail-with-body --silent --show-error "$ollama_url/api/tags")"
for model in bge-large-zh-v1.5:latest; do
  digest="$(python3 -c 'import json,re,sys; model=sys.argv[1]; body=json.load(sys.stdin); matches=[item for item in body["models"] if item.get("name") == model or item.get("model") == model]; len(matches) == 1 or sys.exit(f"expected exactly one model match: {model}"); digest=str(matches[0]["digest"]).removeprefix("sha256:"); re.fullmatch(r"[0-9a-f]{64}", digest) or sys.exit(f"invalid digest: {model}"); print(digest)' "$model" <<<"$tags_json")"
  printf '%s %s\n' "$model" "$digest"
done
```

在 DC-Agent 实际环境文件中配置：

```env
OLLAMA_BASE_URL=http://<Ollama内网IP>:11434
OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
OLLAMA_EMBEDDING_PATH=/api/embed
OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5
OLLAMA_KEEP_ALIVE=30m
OLLAMA_REQUEST_TIMEOUT_SECONDS=15
```

必须从 API 容器所在网络验证以下接口：

```text
GET  http://<Ollama内网IP>:11434/api/tags
POST http://<Ollama内网IP>:11434/api/embed
```

### 4.1 替换模型指纹和向量维度

模板中的以下值是占位符，不能直接用于生产：

```env
EMBEDDING_MODEL_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EMBEDDING_MODEL_DIMENSIONS=1024
```

需要执行：

1. 从 Ollama `/api/tags` 获取 `bge-large-zh-v1.5:latest` 的真实 digest。
2. 去掉可选的 `sha256:` 前缀，保存为 64 位小写十六进制字符。
3. 调用 `/api/embed`，将 `len(embeddings[0])` 的实测结果写入 `EMBEDDING_MODEL_DIMENSIONS`。

最终配置形态：

```env
EMBEDDING_MODEL_NAME=bge-large-zh-v1.5:latest
EMBEDDING_MODEL_VERSION=ollama-bge-large-zh-v15-v1
EMBEDDING_MODEL_SHA256=<bge-large-zh-v1.5:latest真实digest>
EMBEDDING_MODEL_DIMENSIONS=<实测向量维度，BGE示例为1024>
EMBEDDING_MODEL_NORMALIZED=true
EMBEDDING_ENCODING_PROFILE_SHA256=3d5db261732d456b51fa4f9aa89cb15054c21772c0809a50a31f0911eb960170
OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
OLLAMA_EMBEDDING_PATH=/api/embed
OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5

```

只有在 endpoint、query profile 和前缀契约均未修改时，才可以保留项目提供的 profile SHA-256。
BGE query 会自动添加 `为这个句子生成表示以用于检索相关文章：`，document 文本不添加前缀；如果只提供 legacy
`/api/embeddings`，请使用 BGE legacy profile hash `b8e7252a57feef349f02d6b2624ef3f9e8bc9e989d9073e37aa5df424cf26de4`。

### 4.2 可选：重新启用 Reranker

`kopens/bge-reranker-large:latest` 在当前 Ollama 上通过 `/api/embed` 返回 1024 维向量，
并不返回 query/passage 的单一相关性分数，因此不能直接替换 `/v1/rerank`。只有准备好经过验证的
打分服务后，才设置：

```env
RERANKER_ENABLED=true
RERANKER_SERVICE_URL=http://reranker-service:8082
RERANKER_MODEL_NAME=<已验证模型>
RERANKER_MODEL_VERSION=<固定版本>
RERANKER_MODEL_SHA256=<真实digest>
RERANKER_PROMPT_PROFILE_SHA256=<固定prompt profile hash>
RERANKER_PROTOCOL_VERSION=v1
```

旧的 `qwen2.5:3b` `/api/generate` 生成式打分仅为兼容模式，不等价于专用 cross-encoder。启用时使用：

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh build reranker-service api
./tools/invoke_offline_compose.sh up -d reranker-service api
```

## 5. 配置数据库和内部服务

如果使用仓库 Compose 启动全部基础设施，服务之间通常使用 Compose 服务名。如果基础设施独立部署，则改成实际内网地址：

```env
DATABASE_URL=postgresql+psycopg://<用户>:<密码>@<PostgreSQL内网IP>:5432/dc_agent
CLICKHOUSE_URL=http://<ClickHouse内网IP>:8123
QDRANT_URL=http://<Qdrant内网IP>:6333
REDIS_URL=redis://<Redis内网IP>:6379/0
CLAMAV_HOST=<ClamAV内网IP或服务名>
EMBEDDING_SERVICE_URL=http://<Embedding适配器内网IP>:8081
PARQUET_ROOT=<持久化Parquet目录>
```

Compose 内部配置示例：

```env
CLICKHOUSE_URL=http://clickhouse:8123
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379/0
EMBEDDING_SERVICE_URL=http://embedding-service:8081
PARQUET_ROOT=/data/parquet
```

不要把宿主机浏览器使用的地址直接复制为容器内部地址。必须从 API 容器内部验证连通性。

## 6. 配置密码文件

Compose 模板使用文件传递数据库密码。至少准备：

```text
artifacts/secrets/postgres-password
artifacts/secrets/database-url
artifacts/secrets/clickhouse-query-password
artifacts/secrets/clickhouse-ingest-password
```

对应环境变量：

```env
POSTGRES_PASSWORD_FILE=../../artifacts/secrets/postgres-password
DATABASE_URL_SECRET_FILE=../../artifacts/secrets/database-url
CLICKHOUSE_QUERY_PASSWORD_FILE=../../artifacts/secrets/clickhouse-query-password
CLICKHOUSE_INGEST_PASSWORD_FILE=../../artifacts/secrets/clickhouse-ingest-password
CLICKHOUSE_QUERY_USER=dc_agent_query
CLICKHOUSE_INGEST_USER=dc_agent_ingest
```

要求：

- 查询账号和写入账号分离。
- 密码文件不得提交到 Git。
- Linux 上应限制文件所有者和读取权限。
- 修改密码后，同时更新服务端账号和对应 secret 文件。

## 7. 替换内部镜像和其他占位符

`deploy/offline/.env.example` 中的 `registry.internal/...@sha256:000...`、`111...` 等镜像 digest 是模板值。需要替换为公司内部镜像仓库中的真实不可变 digest。

同时搜索并替换所有重复字符形式的占位 hash，例如：

```text
aaaaaaaa...
cccccccc...
eeeeeeee...
00000000...
11111111...
22222222...
```

渲染 Compose 前执行：

```bash
./tools/invoke_offline_compose.sh config
```

配置渲染失败时不要直接绕过 wrapper，也不要临时改用公网镜像。

## 8. 启用结构化 Excel/CSV 查询

实际环境文件必须包含：

```env
STRUCTURED_QUERY_ENABLED=true
CLICKHOUSE_URL=http://clickhouse:8123
CLICKHOUSE_QUERY_USER=dc_agent_query
CLICKHOUSE_INGEST_USER=dc_agent_ingest
PARQUET_ROOT=/data/parquet
STRUCTURED_QUERY_TIMEOUT_SECONDS=4
STRUCTURED_INGEST_BATCH_ROWS=50000
```

仅设置 `STRUCTURED_QUERY_ENABLED=true` 还不够。要回答整列平均值、总和、数量、最大值或最小值，还必须：

1. 完成 PostgreSQL migration。
2. 启动 ClickHouse。
3. 启动 indexing worker。
4. 上传 Excel/CSV。
5. 在管理端确认字段类型、字段别名和统计权限。
6. 等待 publication 状态变为 `published`。

启动包含结构化 worker 的服务：

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh --profile indexing up -d
```

在 publication 成为 `published` 之前，不要验收平均值等全量统计问题。如果 ClickHouse 不可用或查询超时，系统应明确失败，不应退回文档切片估算平均值。

## 9. 构建并激活 Qdrant 检索索引

Ubuntu 原生部署首次启动时，建议在 Embedding、Qdrant 和 PostgreSQL 可用后执行一次幂等引导：

```bash
cd /opt/DCAgent
sudo bash tools/start_ubuntu_supervisor_chain.sh
```

普通文档上传后的解析、切片和检索索引由 API 的 FastAPI `BackgroundTasks` 执行，Supervisor
不再启动独立的 `dcagent-ingestion-worker`。Excel/CSV 的正式结构化发布仍由
`dcagent-structured-worker` 常驻处理。

该命令会检查 `retrieval_publications` 与 Qdrant alias：已有一致的 active publication 时直接跳过；
已有知识片段但没有 active publication 时自动创建下一个 `knowledge_chunks_qwen3_vN` collection、
完成 BGE-M3 全量构建并激活 alias；知识库为空时安全跳过。数据库审计记录与 Qdrant alias 不一致时会
失败并要求人工处理，不会覆盖旧 collection。

启动时知识库为空也不会留下永久的“无索引”状态：首次导入普通文档并生成片段后，导入流程会
自动重复同一套幂等引导，再进行源级增量写入。这样新部署不需要手工预先创建一个空的 Qdrant
collection。

模板默认使用影子模式：

```env
RETRIEVAL_MODE=shadow
RETRIEVAL_SHADOW_PERCENT=10
RETRIEVAL_CANARY_PERCENT=0
RETRIEVAL_PERMISSION_TAGS=公开
```

后续未显式选择分类的新上传文档默认使用“公开”，该值必须与检索权限标签保持一致。显式分类不会被覆盖。升级不会批量修改数据库，现有文档不会自动迁移；管理员可以删除旧文档后重新导入。

影子模式仍把 Legacy 结果返回给用户，混合检索主要用于后台对比。首次上线应先保持影子模式，重建全部文档向量：

```bash
./tools/invoke_offline_compose.sh exec -T api \
  python -m app.retrieval_index_worker \
  --collection knowledge_chunks_qwen3_v1
```

确认 publication、向量维度、点数量、权限过滤和检索样本均正确后，再激活 collection：

```bash
./tools/invoke_offline_compose.sh exec -T api \
  python -m app.retrieval_index_worker \
  --collection knowledge_chunks_qwen3_v1 \
  --activate
```

正式启用混合检索：

```env
RETRIEVAL_MODE=qwen3
RETRIEVAL_SHADOW_PERCENT=0
RETRIEVAL_CANARY_PERCENT=100
QDRANT_COLLECTION_ALIAS=knowledge_chunks_current
```

`qwen3` 是为兼容既有 collection、数据库记录和环境变量保留的路由名称，当前默认实际链路是 BGE-M3 + BM25 + RRF + BGE-Reranker-v2-M3。

更换 Embedding 模型、digest、向量维度、endpoint 或 query profile 后，必须使用从未使用过的新 collection 名称并重新构建全部 Word/PDF/TXT/Excel 向量，不能复用旧向量。

## 10. 配置前端和反向代理

开发模式可以设置：

```env
VITE_API_PROXY_TARGET=http://<后端内网IP>:8000
```

生产环境建议由 Nginx 或公司网关提供同源入口：

```text
/api/* -> DC-Agent 后端 8000 端口
```

仓库 Compose 默认只把 API 暴露到宿主机 `127.0.0.1:8000`。同一宿主机上的 Nginx 可以访问该地址；如果 Nginx 在另一个容器或另一台服务器，需要通过受控网络重新配置可达地址，不应直接把数据库、Qdrant、Ollama 或 Physoc 暴露给浏览器。

### 10.1 无公网内网需要处理 CDN

当前用户端运行时会访问：

```text
https://cdn.jsdelivr.net/npm/three@0.185.1/build/three.module.min.js
https://cdn.jsdelivr.net/npm/cn-fontsource-ding-talk-jin-bu-ti-regular@1.0.3/font.css
```

如果公司内网不能访问 `cdn.jsdelivr.net`，需要选择一种方案：

1. 由网络管理员只放行经过批准的 jsDelivr 资源；或
2. 把 Three.js 和字体放到公司内部静态资源服务器；或
3. 将资源改为随前端一起打包部署。

否则页面主体可能仍能打开，但 Three.js 背景或字体资源会加载失败。

## 11. 推荐启动顺序

从仓库根目录执行：

```bash
set -Eeuo pipefail
install -d -m 0700 /srv/dcagent/data /srv/dcagent/models
./tools/prepare_offline_env.sh --initialize-state
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh build schema-migration embedding-service api ingestion-worker
./tools/invoke_offline_compose.sh up -d
./tools/invoke_offline_compose.sh --profile indexing up -d
```

然后依次完成：

1. PostgreSQL migration 成功。
2. `/api/readyz` 返回正常。
3. Ollama `/api/tags`、`/api/embed` 可用；只有启用 Reranker 时才验收 `/api/generate` 或专用打分端点。
4. Physoc 探针通过。
5. 构建并激活 Qdrant collection。
6. 上传并发布测试文档。
7. 上传并发布测试 Excel。
8. 最后再接入 Nginx 和正式用户流量。

## 12. 上线验收清单

### 12.1 基础功能

- [ ] `GET /api/readyz` 正常。
- [ ] 输入“你好”直接返回固定欢迎词。
- [ ] 前端和管理端均能访问后端 `/api`。
- [ ] 浏览器不直接访问 Ollama、Physoc、数据库或 Qdrant。

### 12.2 普通文档问答

- [ ] 上传 TXT、Word 或文本型 PDF 后成功解析和发布。
- [ ] Qdrant collection 已激活。
- [ ] 提问能够检索到相关证据。
- [ ] 最终答案由 Physoc 根据检索证据总结，而不是直接返回切片。
- [ ] 没有可靠证据时明确提示依据不足。

### 12.3 Excel/CSV 精确统计

- [ ] `STRUCTURED_QUERY_ENABLED=true`。
- [ ] indexing worker 正常运行。
- [ ] 管理员已确认字段结构和统计权限。
- [ ] publication 状态为 `published`。
- [ ] 求和、平均、计数、去重计数、最值、中位数、P90、方差、标准差与原表人工计算结果一致。
- [ ] 按字段分组、升降序、Top/Bottom N 的分组数、顺序和结果与原表一致。
- [ ] ClickHouse 故障时没有退回切片估算。

### 12.4 网络和性能

- [ ] API 容器可以访问 Physoc 和 Ollama 的内网地址。
- [ ] jsDelivr 已放行或相关资源已经内网化。
- [ ] 完成至少 15 个并发用户的内网容量测试。
- [ ] 记录 502、429、503、超时、P95 延迟和资源使用情况。

## 13. 常见问题

### 13.1 提问后返回 HTTP 502

优先检查：

1. `LLM_API_BASE` 是否填写成了容器内不可达的 `127.0.0.1`。
2. 路径是否为 `/api/physoc/deepseeks/stream`，注意 `deepseeks` 的复数形式。
3. Physoc 是否接受 `POST`，并正确处理 `query`、`model` JSON 字段。
4. 响应 Content-Type 是否为 `text/event-stream`。
5. SSE 数据是否包含 `response` 和最终的 `done: true`。
6. 防火墙、代理或网关是否提前关闭长连接。
7. 查看后端 Loguru 的 `logger.exception(error)` 完整异常栈。

### 13.2 Excel 平均值仍然返回切片内容

优先检查：

1. 服务器实际 `.env` 是否明确设置 `STRUCTURED_QUERY_ENABLED=true`。
2. indexing profile 是否已启动。
3. 字段类型和统计权限是否已由管理员确认。
4. publication 是否达到 `published`。
5. ClickHouse 查询账号、密码文件和网络连接是否正确。
6. 问题中的字段名称是否与确认后的字段名或别名匹配。

### 13.3 Word/PDF 回答只是原始切片

优先检查：

1. `LLM_PROVIDER` 是否错误设置为 `template` 或 `mock`。
2. Physoc 探针是否通过。
3. 是否仍处于不符合预期的 Legacy/Shadow 检索配置。
4. Qdrant collection 是否已经构建并激活。
5. Embedding adapter 是否能访问 Ollama；若显式启用 Reranker，再检查其 adapter 和打分端点。

## 14. 相关文档

- [项目 README](../README.md)
- [Compose 内网部署说明](../deploy/offline/README.md)
- [平台运行手册](offline-platform-runbook.md)
- [API 契约](api-contract.md)
