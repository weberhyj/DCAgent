# BGE Large 中文 Ollama 查询配置档设计

## 目标

将 Dense Embedding 从通用生成模型 `qwen2.5:0.5b` 切换为专业中文向量模型
`bge-large-zh-v1.5:latest`，同时保留 DC-Agent 现有的私有
`POST /v1/embeddings` 协议、Ollama `/api/embed` 与旧版 `/api/embeddings`
双入口，以及严格的模型元数据和 Qdrant 版本化发布机制。

本次变更只处理 Embedding 查询配置档，不修改 Reranker、Physoc/DeepSeek 回答接口、
稀疏检索或混合检索排序参数。

## 问题背景

`qwen2.5:0.5b` 可以在部分 Ollama 版本中返回向量，但它不是面向检索训练的
Embedding 模型。现有发布门禁已经观察到
`dense validation query did not recover expected target`，说明向量已经写入 Qdrant，
但 Dense 查询无法稳定回查预期切片。

`bge-large-zh-v1.5` 是中文检索模型。它区分查询和文档编码：文档保持原文，查询需要
使用固定检索指令。仅替换模型名称而继续对 query 使用原文，不能完整采用该模型的推荐
检索协议。

## 选定方案

新增显式枚举环境变量：

```env
OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5
```

允许值只有：

- `raw`：query 和 document 都保持原文，用于现有兼容路线；
- `bge-large-zh-v1.5`：只为 query 添加固定中文检索指令，document 保持原文。

不根据 Ollama 模型名称自动判断配置档，因为模型可能带命名空间、标签或企业内部别名。
不允许通过环境变量直接传入任意前缀，避免不可审计的空格、标点和文本差异产生不同的
Embedding 空间。

## 查询转换

`bge-large-zh-v1.5` 配置档使用以下固定 UTF-8 前缀：

```text
为这个句子生成表示以用于检索相关文章：
```

转换规则：

```text
purpose=query     -> 固定前缀 + 原始查询
purpose=document  -> 原始文档文本
```

前缀和查询之间不额外插入空格或换行。前缀字节的 SHA-256 固定为：

```text
2bb658b7e092d6b4b1dbde4c3fc5f281f9ed9f1ace5b49566fb8b10f57836e48
```

批量请求中的每条 query 独立转换，输入顺序和输出向量顺序保持不变。转换只发生在
Ollama Embedding adapter 内部，API、worker 和 Qdrant publisher 继续提交原始查询。

## 编码配置档与指纹

Embedding 编码配置档必须同时绑定：

- Ollama endpoint：`/api/embed` 或 `/api/embeddings`；
- query profile：`raw` 或 `bge-large-zh-v1.5`；
- query 前缀 SHA-256；
- document 原文策略；
- 批量/单条请求形态；
- 向量校验和 L2 归一化规则。

配置档文本继续使用规范化 ASCII 行，并通过
`query_prefix_sha256=2bb658...` 绑定中文前缀的准确 UTF-8 字节，避免不同平台的文本编码
影响指纹。四种 endpoint/profile 组合必须产生四个不同的
`EMBEDDING_ENCODING_PROFILE_SHA256`。

`raw` 配置档的前缀是空字节串，其 SHA-256 固定为
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`；BGE 配置档使用上文
固定的中文前缀 SHA-256。实现不得省略这一行或用模型名称替代前缀指纹。

服务启动时根据 `OLLAMA_EMBEDDING_PATH` 和
`OLLAMA_EMBEDDING_QUERY_PROFILE` 重新计算期望指纹。配置的
`EMBEDDING_ENCODING_PROFILE_SHA256` 不匹配时必须拒绝启动，不能回退到 `raw`。

## 模型身份和维度

部署环境使用 Ollama `/api/tags` 返回的准确模型名称和 digest：

```env
EMBEDDING_MODEL_NAME=bge-large-zh-v1.5:latest
OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
EMBEDDING_MODEL_VERSION=ollama-bge-large-zh-v15-v1
EMBEDDING_MODEL_SHA256=<真实64位digest>
EMBEDDING_MODEL_DIMENSIONS=<目标Ollama实测维度>
EMBEDDING_MODEL_NORMALIZED=true
OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5
```

不能仅根据模型资料假定维度为 1024。部署人员必须调用目标 Ollama endpoint，并将实际
`len(embeddings[0])` 写入环境文件。模型名称、digest、维度、查询配置档或 endpoint
任意一项变化，都视为新的 Embedding 指纹。

## 启动与错误处理

Embedding 服务启动探针继续分别执行 `query` 和 `document`：

- query 探针必须经过选定配置档转换；
- document 探针必须保持原文；
- 两条路径都必须返回正确数量、正确维度、有限且非零的向量；
- 无效 query profile、模型 digest 不匹配、配置档指纹不匹配或 Ollama 调用失败时，
  服务不得报告 ready。

对外错误继续脱敏，不返回原始查询、文档、向量或 Ollama 响应正文。现有的超时、队列
上限和响应大小限制保持不变。

## Qdrant 迁移

切换模型和查询配置档后，所有旧 Dense 向量都不可复用。上线顺序固定为：

1. 在目标 Ollama 验证模型名称、digest、endpoint 和实测维度；
2. 配置新的模型元数据、query profile 和编码配置档 SHA-256；
3. 重启 Embedding adapter，确认 `/readyz`、`/v1/metadata` 和单条 query/document
   `/v1/embeddings` 均正常；
4. 使用从未使用过的 `knowledge_chunks_qwen3_vN` 全量构建；
5. 通过结构、数量、权限、Dense/Sparse 回查和验收问题集；
6. 仅在全部通过后激活 `knowledge_chunks_current` Alias；
7. 保留前一个 active collection 作为短期回滚目标。

数据库中已经使用过的 collection 名称不得复用，即使对应 Qdrant collection 已被清理。

## 兼容性和回滚

- `/v1/embeddings` 请求和响应结构不变；
- `/api/embed` 和 `/api/embeddings` 都支持两个 query profile；
- `raw` 配置档保留现有行为，但必须显式配置；
- Reranker 继续使用当前 `qwen2.5:3b` 兼容路线；
- 回滚代码时必须同时回滚环境中的 query profile、编码配置档 SHA-256、模型元数据和
  Qdrant Alias，不能让旧代码查询新向量 collection。

## 测试

单元测试必须覆盖：

1. `raw` profile 对 query/document 均保持原文；
2. BGE profile 只转换 query，不转换 document；
3. 现代 endpoint 对批量 query 的每一项添加一次前缀；
4. 旧版 endpoint 对每个 query prompt 添加一次前缀；
5. 四种 endpoint/profile 配置档规范、互异且哈希可复算；
6. 未知、空白或大小写错误的 profile fail closed；
7. 启动元数据同时校验 endpoint 和 query profile 指纹；
8. 服务启动的 query/document 探针经过正确转换；
9. 现有响应校验、归一化、超时和错误脱敏测试保持通过；
10. 环境示例、Compose 和内网文档不再把 `qwen2.5:0.5b` 作为默认 Embedding。

## 验收标准

- `bge-large-zh-v1.5` query 向量包含固定检索指令，document 向量不包含；
- 对外 `/v1/embeddings` 协议无变化；
- query profile 与 endpoint 被严格纳入 Embedding 指纹；
- 旧配置缺少 query profile 时服务拒绝启动并给出明确配置错误；
- 新 collection 通过 Dense/Sparse 发布验证和真实中文知识库验收集；
- 完整后端测试、Ruff 检查和部署文档一致性检查通过。
