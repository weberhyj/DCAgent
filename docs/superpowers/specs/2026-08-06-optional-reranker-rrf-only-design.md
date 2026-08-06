# 可选 Reranker 与 RRF-only 模式设计

## 背景

当前 Qwen3 兼容命名的混合检索链路依次执行 BGE Dense 检索、BM25 Sparse 检索、RRF 融合和 Reranker 精排。项目已经能够在 Reranker 返回 429、503、超时或非法响应时保留 RRF 候选，但 Reranker 仍然是启动配置、运行时资源、API 就绪检查、Compose 服务和部署验收中的必需依赖。

目标内网 Ollama 只有 `kopens/bge-reranker-large:latest`。实测 `/api/embed` 为每个输入返回 1024 维向量，而不是每个 query-document pair 的单一相关性分数，因此它不能替代当前 `/api/generate` JSON 打分协议。为了在不部署额外推理运行时的条件下稳定提供知识库问答，系统需要一个正式的 RRF-only 运行模式。

## 目标

- 新增显式布尔配置 `RERANKER_ENABLED`。
- `RERANKER_ENABLED=false` 时，混合检索使用 Dense + BM25 + RRF，不创建、不调用也不检查 Reranker。
- Reranker 关闭时，Qdrant、Embedding、索引 Alias 和权限过滤仍然保持 fail-closed。
- API `/api/readyz` 不因有意关闭 Reranker 而返回 503 或 degraded。
- 默认部署示例使用 RRF-only，并保留未来重新启用 Reranker 的兼容路径。
- 不改变现有结构化 Excel/ClickHouse 查询链路。

## 方案比较

### 方案一：继续依赖异常降级

保持当前配置，每次请求先调用 Reranker，失败后返回 RRF 候选。修改量最少，但每次请求仍要承担超时或 503，API 就绪检查仍失败，Compose 仍启动无效服务，不满足稳定部署要求。

### 方案二：新增独立 Retrieval Mode

增加 `RETRIEVAL_MODE=rrf`。语义直观，但会把检索路由选择和排序策略混在同一枚举里，同时扩大 Shadow、Canary、日志和迁移兼容面的改动。

### 方案三：显式能力开关

保留 `RETRIEVAL_MODE=legacy|shadow|qwen3`，增加 `RERANKER_ENABLED=true|false`。路由模式继续决定使用 Legacy 还是 Hybrid，能力开关只决定 Hybrid 内是否执行二次精排。该方案边界清晰，能够保留现有索引和路由命名，作为本次实现方案。

## 配置语义

`RERANKER_ENABLED` 在代码中默认 `true`，避免已有部署升级后静默改变排序行为。新的内网部署示例显式设置：

```env
RERANKER_ENABLED=false
RETRIEVAL_FINAL_TOP_K=8
```

当 `RETRIEVAL_MODE=legacy` 时，该开关不产生运行时影响。当 `RETRIEVAL_MODE=shadow|qwen3` 且开关为 `false` 时：

- `RERANKER_SERVICE_URL` 和全部 `RERANKER_MODEL_*` 配置不再是 API、Worker 或索引生命周期的必填项；
- `RetrievalSettings.reranker_service_url` 和 `RetrievalSettings.reranker` 为 `None`；
- 非法布尔值必须使启动失败，不能模糊回退；
- Embedding 指纹、Sparse profile、Qdrant Alias 和权限标签仍然必须完整配置。

当开关为 `true` 或未设置时，保持当前严格配置和元数据校验。

## 运行时数据流

RRF-only 请求执行以下步骤：

1. 对 query 并行生成 BGE Dense vector 和本地 BM25 Sparse vector。
2. 使用相同权限、知识库和 publication filters 并行查询 Qdrant Dense/Sparse Top K。
3. 使用现有确定性 RRF 算法融合候选。
4. 跳过 Reranker 网络请求，保留 RRF 顺序和 `rerank_score=None`。
5. 使用 `RETRIEVAL_FINAL_TOP_K` 限制证据数量；内网示例从 4 调整为 8。
6. 继续执行相邻片段扩展、字符预算、去重、来源引用和大模型总结。

Reranker 开启时，现有精排、忙碌重试和失败降级行为保持不变。Reranker 无法找回 Dense/Sparse 初次召回之外的片段，因此关闭它不会改变召回集合，只会取消二次排序。

## 组件改动

### 检索配置

`backend/app/retrieval_settings.py` 增加 `reranker_enabled: bool`。只有开启时才解析私有 Reranker URL 和 pinned metadata。现有 Top K 数值约束保持兼容。

### Hybrid Retriever

`backend/app/hybrid_retriever.py` 允许 `reranker` 和 `reranker_metadata` 同时为 `None`。两者必须同时存在或同时缺失，避免部分配置。缺失时不提交线程池任务、不消耗超时预算，直接使用 RRF 顺序。`stage_ms` 继续包含 `reranker`，跳过时记录 `0.0`，保持观测字段稳定。

### 应用启动

`backend/app/main.py` 只在开关开启时创建和托管 `SyncHttpRerankerClient`。Router 日志中的 Reranker 版本在关闭时为 `None`。索引生命周期仍然只依赖 Embedding 和 Sparse artifact，不因关闭 Reranker 要求重建向量索引。

### 健康检查

`backend/app/infra/health.py` 只在 Reranker 开启时校验其 URL、元数据和 readiness。关闭时报告中不出现伪造的 `reranker=ready`，而是完全省略该依赖；Qdrant、Embedding 和 Alias 校验仍然决定 Hybrid 路由是否 Ready。

### Compose 和部署工具

- `deploy/offline/.env.example` 显式设置 `RERANKER_ENABLED=false` 和 `RETRIEVAL_FINAL_TOP_K=8`。
- API 与 ingestion worker 接收该开关，Reranker 环境变量在关闭时允许为空。
- `reranker-service` 放入显式 Compose profile，默认 `up -d` 不启动；未来启用时由 profile 启动并恢复严格启动校验。
- `tools/compose_smoke.py` 和 `tools/intranet_deployment_gate.py` 根据开关决定是否启动、探测和验收 Reranker。关闭时审计报告必须明确记录 `disabled`，不能将未执行写成通过。
- Bash 部署文档只要求 Ollama `/api/embed` 和 BGE Embedding；`/api/generate` Reranker probe 移入可选启用章节。

## 错误处理与安全

- 关闭 Reranker 不是异常，也不产生 `reranker_unavailable` fallback code。
- Embedding、Qdrant、Alias、权限范围或 publication fingerprint 异常仍按现有规则失败或回退 Legacy。
- RRF-only 仍在检索前应用权限过滤，不能通过增加 Top K 绕过授权边界。
- 日志和验收报告不得记录 query、片段正文、向量坐标或模型原始响应。
- 将来重新开启 Reranker 时，必须重新提供私有 URL、模型 digest、prompt profile 和协议版本，并通过目标机容量与质量 gate。

## 测试策略

- 配置测试：关闭时无需 Reranker URL/metadata；默认开启保持现有严格行为；非法布尔值失败。
- Retriever 测试：关闭时客户端零调用、候选保持 RRF 顺序、Top 8 与相邻片段行为正常、`rerank_score` 均为 `None`。
- 启动测试：关闭时工厂不创建 Reranker client，其他资源仍按顺序创建和关闭。
- 健康测试：关闭时不访问 `8082` 且 Qdrant/Embedding 正常即可 Ready；开启时继续检查 metadata mismatch。
- Compose 合约测试：默认服务集合不包含 Reranker，显式 profile 可以启动它。
- Smoke/Gate 测试：关闭时记录 disabled 并跳过探针；开启时保持原有三项探针。
- 回归测试：现有 Reranker 开启、失败降级、Shadow 和 structured query 测试继续通过。

## 发布和兼容性

- 不需要数据库迁移。
- 不需要重新导入文档或重建 BGE/Qdrant 索引，因为 Embedding fingerprint 不变。
- 只提升后端版本；前端代码和版本不变。
- 现有部署不设置 `RERANKER_ENABLED` 时继续使用 Reranker。
- 目标内网部署升级后必须显式设置 `RERANKER_ENABLED=false`，并重新执行 Compose config、API readiness、Word 文档命中测试和 15 并发验收。

## 验收标准

1. Reranker 容器未运行时，Qwen3 Hybrid 路由能够启动并通过与 Reranker 无关的就绪检查。
2. 一次知识库查询不会向 `RERANKER_SERVICE_URL` 发起任何请求。
3. Dense/Sparse/RRF 命中的 Word 片段能够作为证据进入大模型总结，`evidenceCount` 不因 Reranker 缺失变为 0。
4. 返回证据最多为配置的 Top 8，并保持权限、去重、邻接和字符预算约束。
5. Reranker 重新开启后，现有模型元数据校验、精排和降级测试全部保持兼容。
6. 后端定向测试、完整测试和 Ruff 检查通过；部署脚本与文档不再把 Reranker 描述为默认必需服务。
