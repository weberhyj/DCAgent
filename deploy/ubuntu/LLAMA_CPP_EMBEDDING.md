# Ubuntu + Supervisor：llama.cpp BGE-M3 Embedding

生产内网使用两个独立的 llama.cpp 进程：Embedding 使用 `bge-m3-Q4_K_M.gguf`，Reranker 使用
`bge-reranker-v2-m3-Q4_K_M.gguf`。Embedding 进程只提供 OpenAI-compatible
`/v1/embeddings`，不与 DC-Agent API 共用端口或模型上下文。

## 1. 启动 llama.cpp 进程

以下命令以非 root 账号 `dcagent` 运行；把模型目录、CPU 线程数和端口替换为内网实际值。

```bash
set -Eeuo pipefail
install -d -o dcagent -g dcagent -m 0750 /srv/dcagent/models /var/log/dcagent
sha256sum /srv/dcagent/models/bge-m3-Q4_K_M.gguf
sha256sum /srv/dcagent/models/bge-reranker-v2-m3-Q4_K_M.gguf
```

Embedding 服务：

```bash
/usr/local/bin/llama-server \
  -m /srv/dcagent/models/bge-m3-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8083 \
  --embedding \
  -c 4096 -np 2
```

Reranker 服务：

```bash
/usr/local/bin/llama-server \
  -m /srv/dcagent/models/bge-reranker-v2-m3-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  --reranking \
  -c 4096 -np 2
```

不要把两个模型加载到同一个进程。启动前确认 `llama-server --help` 包含对应的
`--embedding` 和 `--reranking` 选项；如果构建版本不支持，先替换为公司批准的 llama.cpp
构建，不要在应用层绕过探针。

## 2. Supervisor 配置

创建 `/etc/supervisor/conf.d/dcagent-llama-cpp.conf`：

```ini
[program:dcagent-llama-embedding]
command=/usr/local/bin/llama-server -m /srv/dcagent/models/bge-m3-Q4_K_M.gguf --host 127.0.0.1 --port 8083 --embedding -c 4096 -np 2
directory=/srv/dcagent
user=dcagent
autostart=true
autorestart=true
startsecs=5
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/dcagent/llama-embedding.log
stderr_logfile=/var/log/dcagent/llama-embedding-error.log
environment=HOME="/nonexistent",PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"

[program:dcagent-llama-reranker]
command=/usr/local/bin/llama-server -m /srv/dcagent/models/bge-reranker-v2-m3-Q4_K_M.gguf --host 127.0.0.1 --port 8080 --reranking -c 4096 -np 2
directory=/srv/dcagent
user=dcagent
autostart=true
autorestart=true
startsecs=5
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/dcagent/llama-reranker.log
stderr_logfile=/var/log/dcagent/llama-reranker-error.log
environment=HOME="/nonexistent",PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
```

加载配置并确认进程：

```bash
set -Eeuo pipefail
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart dcagent-llama-embedding dcagent-llama-reranker
sudo supervisorctl status dcagent-llama-embedding dcagent-llama-reranker
```

## 3. 探针与维度校验

先从 API/适配器所在主机执行：

```bash
set -Eeuo pipefail
curl --fail-with-body --silent --show-error \
  -X POST http://127.0.0.1:8083/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3-Q4_K_M.gguf","input":["dimension-probe"]}' \
  | tee /tmp/bge-m3-embedding.json

python3 - <<'PY'
import json
from pathlib import Path
body = json.loads(Path('/tmp/bge-m3-embedding.json').read_text())
items = body.get('data')
assert isinstance(items, list) and len(items) == 1
vector = items[0].get('embedding')
assert isinstance(vector, list) and len(vector) > 0
print(f'EMBEDDING_MODEL_DIMENSIONS={len(vector)}')
PY

curl --fail-with-body --silent --show-error \
  -X POST http://127.0.0.1:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-v2-m3-Q4_K_M.gguf","query":"probe","documents":["relevant","unrelated"]}'
```

把真实 GGUF `sha256sum` 和探测到的维度写入 Supervisor/API 的环境文件：

```dotenv
EMBEDDING_RUNTIME=llama_cpp
EMBEDDING_MODEL_NAME=bge-m3-Q4_K_M.gguf
EMBEDDING_MODEL_VERSION=llama-cpp-bge-m3-q4km-v1
EMBEDDING_MODEL_SHA256=<bge-m3 GGUF 的 64 位小写 sha256>
EMBEDDING_MODEL_DIMENSIONS=<探针返回的维度>
EMBEDDING_MODEL_NORMALIZED=true
EMBEDDING_ENCODING_PROFILE_SHA256=52fd367c0a46ecfffc3fcf7f688d0a9e6c80f26cacf772eeeee78cc7f6c16254
LLAMA_CPP_EMBEDDING_URL=http://127.0.0.1:8083
LLAMA_CPP_EMBEDDING_PATH=/v1/embeddings
LLAMA_CPP_EMBEDDING_MODEL=bge-m3-Q4_K_M.gguf
LLAMA_CPP_EMBEDDING_TIMEOUT_SECONDS=15
LLAMA_CPP_EMBEDDING_BATCH_MAX_ITEMS=32

RERANKER_ENABLED=true
RERANKER_RUNTIME=llama_cpp
RERANKER_MODEL_NAME=bge-reranker-v2-m3-Q4_K_M.gguf
RERANKER_MODEL_VERSION=llama-cpp-bge-reranker-v2-m3-q4km-v1
RERANKER_MODEL_SHA256=<reranker GGUF 的 64 位小写 sha256>
RERANKER_PROMPT_PROFILE_SHA256=6f7fb308e56ddbdb5e2cf8536141b9d038e5fe69e12791c9a5142e6e68ef0cc9
LLAMA_CPP_RERANKER_URL=http://127.0.0.1:8080
LLAMA_CPP_RERANKER_PATH=/v1/rerank
LLAMA_CPP_RERANKER_MODEL=bge-reranker-v2-m3-Q4_K_M.gguf
LLAMA_CPP_RERANK_TIMEOUT_SECONDS=10
LLAMA_CPP_RERANK_BATCH_MAX_ITEMS=32
```

## 4. 重建向量与切换 alias

BGE-M3 与旧 Ollama/BGE-large 向量不可混用。先创建新 collection，使用新的 Embedding 服务
全量重建所有 Word/普通文档 chunk，完成计数、维度和抽样检索校验后，再原子切换
`QDRANT_COLLECTION_ALIAS`。旧 collection 保留，回滚时只切回旧 alias 和旧 Embedding 指纹。

Excel 行级查询和 Word 年龄/性别/职务事实查询不依赖 Embedding、Reranker 或 LLM；只有开放式
文档问题进入 Embedding → Qdrant/BM25 → Reranker → LLM 链路。

## 5. 重启 DC-Agent 与回滚

```bash
set -Eeuo pipefail
sudo supervisorctl restart dcagent-api dcagent-structured-worker
sudo supervisorctl status dcagent-api dcagent-structured-worker dcagent-llama-embedding dcagent-llama-reranker
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/readyz
```

如果任一探针、维度、指纹或检索抽样失败，先停止切换并恢复旧 alias；不要删除旧 collection。
Supervisor 回滚命令：

```bash
set -Eeuo pipefail
sudo supervisorctl stop dcagent-api dcagent-structured-worker
# 恢复旧环境文件和 Qdrant alias 后：
sudo supervisorctl start dcagent-llama-embedding dcagent-llama-reranker
sudo supervisorctl start dcagent-api dcagent-structured-worker
sudo supervisorctl status dcagent-api dcagent-structured-worker
```
