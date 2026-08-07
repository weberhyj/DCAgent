# llama.cpp BGE GGUF Reranker 设计

## 目标

支持服务器使用 `bge-reranker-v2-m3-Q4_K_M.gguf` 与 llama.cpp 原生 `/v1/rerank` 接口，
同时保持 DC-Agent 现有的私有 `/v1/rerank` 响应协议和 API 检索代码不变。

## 方案

`reranker-service` 继续作为内部适配层。它接收现有的 `query/passages` 请求，将 passages
转换为 llama.cpp 的 `query/documents` 请求，再把 `results[].relevance_score` 按原始 index
恢复成 `scores` 数组。通过 `RERANKER_RUNTIME=llama_cpp` 选择该路线，默认 Ollama 路线保持兼容。

llama.cpp 服务地址通过 `LLAMA_CPP_RERANKER_URL` 配置，模型名称通过
`LLAMA_CPP_RERANKER_MODEL` 配置。启动自检仍然执行两条 query/passage，确保分数数量、顺序和
有限值满足内部协议。

## 错误处理

- llama.cpp 连接、超时或非 2xx 响应转换为受控的 Reranker backend failure（HTTP 503）。
- 缺失、重复、越界 index 或非有限 relevance score 转换为 malformed scores（HTTP 500）。
- Reranker 元数据仍由 DC-Agent 内部协议返回，API 侧无需改动。

## 验证

- 单元测试覆盖 documents/results 转换、乱序 index、重复/越界 index、HTTP 错误和 runtime 选择。
- 使用 llama.cpp 主机上的 curl 验证原生 `/v1/rerank`，再使用适配器 `/v1/rerank` 验证内部协议。
- 完整运行后端 pytest、Ruff 和 `fast lint --ty`。
