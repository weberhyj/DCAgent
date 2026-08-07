# llama.cpp BGE GGUF Reranker

当目标服务器只能运行 `bge-reranker-v2-m3-Q4_K_M.gguf` 时，使用 llama.cpp 原生重排接口。

```bash
llama-server \
  -m /models/bge-reranker-v2-m3-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --reranking \
  -c 4096 \
  -np 4
```

先确认安装的 llama.cpp 支持 `--reranking`，并测试：

```bash
curl --fail-with-body -sS \
  -X POST http://127.0.0.1:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-v2-m3-Q4_K_M.gguf","query":"XX的地理位置","documents":["XX位于北京市。","这是无关内容。"]}'
```

DC-Agent 的 `reranker-service` 会把内部 `query/passages` 请求转换成 llama.cpp 的
`query/documents`，再把 `results[].relevance_score` 按 index 转换为内部 `scores`。

配置：

```env
RERANKER_ENABLED=true
RERANKER_RUNTIME=llama_cpp
RERANKER_MODEL_NAME=bge-reranker-v2-m3-Q4_K_M.gguf
RERANKER_MODEL_VERSION=llama-cpp-bge-reranker-v2-m3-q4km-v1
RERANKER_MODEL_SHA256=<sha256sum of the GGUF file>
RERANKER_PROMPT_PROFILE_SHA256=6f7fb308e56ddbdb5e2cf8536141b9d038e5fe69e12791c9a5142e6e68ef0cc9
RERANKER_PROTOCOL_VERSION=v1
LLAMA_CPP_RERANKER_URL=http://172.16.0.11:8080
LLAMA_CPP_RERANKER_PATH=/v1/rerank
LLAMA_CPP_RERANKER_MODEL=bge-reranker-v2-m3-Q4_K_M.gguf
LLAMA_CPP_RERANK_TIMEOUT_SECONDS=10
LLAMA_CPP_RERANK_BATCH_MAX_ITEMS=32
```

容器内不要把 `LLAMA_CPP_RERANKER_URL` 配置为 `127.0.0.1`，除非 llama.cpp 与适配器在同一容器。
`RERANKER_MODEL_SHA256` 必须填写目标 GGUF 文件真实的 `sha256sum`。
