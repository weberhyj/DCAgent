# llama.cpp BGE-M3 与知识路由隔离设计

## 目标

将生产检索模型固定为：

- Embedding：BGE-M3 Q4，通过独立 llama.cpp embedding 服务提供 OpenAI-compatible `/v1/embeddings`；
- Reranker：BGE-Reranker-v2-M3 Q4，通过独立 llama.cpp `/v1/rerank` 服务提供 cross-encoder 分数；
- Excel 行级问题只访问 ClickHouse；
- Word 实体字段问题只访问 `knowledge_facts`；
- 只有真正的开放式文档问题才进入 Embedding → Qdrant/BM25 → Reranker → LLM。

## 架构

```text
API
 ├─ Embedding adapter :8081
 │    └─ llama.cpp BGE-M3-Q4 :8083 /v1/embeddings
 ├─ Reranker adapter :8082
 │    └─ llama.cpp BGE-Reranker-v2-M3-Q4 :8080 /v1/rerank
 └─ Knowledge router
      ├─ excel_row_lookup → ClickHouse
      ├─ excel_aggregate → ClickHouse
      ├─ word_factual → PostgreSQL knowledge_facts
      └─ document_qa → hybrid retrieval → Reranker → LLM
```

适配层保留 DC-Agent 内部稳定协议：

- `POST /v1/embeddings`：输入 `texts` 与 `purpose`，返回带模型指纹的向量；
- `POST /v1/rerank`：输入 `query` 与 `passages`，返回有序分数；
- `GET /readyz` 与 `GET /v1/metadata`：用于启动自检和运维检查。

两个 llama.cpp 服务使用不同进程和端口，避免 Embedding 与 Reranker 的模型上下文、并发和运行参数互相影响。

## Embedding 适配器

新增 `EMBEDDING_RUNTIME=llama_cpp`。服务启动时：

1. 校验私有 URL、模型名、接口路径、超时和批量上限；
2. 分别发送 query/document 启动探针；
3. 解析 OpenAI-compatible `data[].embedding` 响应；
4. 校验向量数量、维度、有限数值和非零范数；
5. 按 `EMBEDDING_MODEL_NORMALIZED` 执行统一 L2 归一化；
6. 将模型名、版本、维度、归一化、运行时和编码 profile 纳入 Embedding 指纹。

BGE-M3 不使用 BGE-large-v1.5 的中文查询前缀，query/document 均发送原始文本。远程 llama.cpp 无法从标准接口证明 GGUF 文件摘要，因此 `EMBEDDING_MODEL_SHA256` 作为部署锁定值保存并展示，实际文件摘要由部署脚本和模型主机验收负责。

## Excel 行级查询

新增 `excel_row_lookup` 意图和 ClickHouse 参数化查询计划，支持：

- 一个等值条件：确认列 = 用户提供的值；
- 返回一个或多个确认列；
- 限制返回行数并报告是否截断；
- 返回 dataset、schema version、publication version 和来源引用。

结构化候选问题解析失败时返回澄清或不支持结果，不能回退到 Word factual 或普通 RAG。结构化查询不调用 Embedding、Reranker 和 LLM。

## Word factual 查询

扩展“实体 + 字段”语法，覆盖自然语言前缀、字段后缀和常见问句。命中字段意图后只查询目标实体和目标字段：

- 禁止 adjacency expansion；
- 禁止历史消息污染；
- 禁止调用 Embedding、Reranker 和 LLM；
- 多来源冲突返回明确冲突提示；
- 多字段问题要求拆分或进入显式澄清，不让 LLM 自由扩写。

无法解析但明显属于实体属性问题时返回澄清，不落入通用 RAG。

## 普通 RAG 上下文策略

独立问题默认不携带最近历史消息。只有检测到追问、省略主语或代词依赖时才使用历史，并限制在当前检索范围内。相邻 chunk 扩展只对开放式文档问题启用，精确字段路由永不启用。

## 索引迁移

切换到 BGE-M3 后必须创建新的 Qdrant collection 并全量重建文档向量，验证后切换 alias。禁止混用旧 Embedding 向量。旧 collection 保留用于回滚。

## 验收

- llama.cpp Embedding query/document 探针和维度校验通过；
- llama.cpp Reranker 返回分数数量、索引和范围校验通过；
- Excel 行查询只产生 ClickHouse 调用；
- Excel 解析失败不会调用 Word/RAG；
- Word 问年龄、性别、职务只返回目标字段；
- 精确字段问题不调用 Reranker/LLM、不扩展相邻 chunk；
- 普通独立问题不带旧历史，开放式问题仍可使用相邻 chunk；
- 新旧 Embedding collection 指纹和 alias 切换流程可回滚。
