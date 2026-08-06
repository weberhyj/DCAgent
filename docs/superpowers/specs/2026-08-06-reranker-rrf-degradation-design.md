# Reranker 失败时的 RRF 降级设计

## 背景

内网检索使用 BGE Embedding、Qdrant Dense/Sparse 检索和基于 Ollama `qwen2.5:3b` 的生成式 Reranker。真实 Word 片段进入 Reranker 后，模型可能返回 `{"system_prompt": ...}`，而不是约定的 `{"scores": [...]}`。Reranker 服务会将该协议错误包装为 HTTP 503，API 再将其映射为 `reranker_unavailable`。当前 Hybrid 检索因此整体失败并回退 Legacy；Legacy 也未命中时，最终 `evidenceCount=0`。

## 目标

Reranker 仍然保持严格输出校验，但它作为可选的排序增强组件，发生忙碌、服务不可用或响应协议异常时，不再清空已经由 Dense、Sparse 和 RRF 找到的候选证据。

## 方案

在 `HybridRetriever._rerank` 内处理可降级的 Reranker 客户端异常：

- 首次遇到 `RerankerBusy` 时维持现有行为，缩小到 `degraded_rerank_top_k` 后重试一次。
- 重试仍忙碌，或者任意一次遇到 `RerankerServiceError`、`RerankerResponseError` 时，直接返回当前 RRF 顺序的前 `degraded_rerank_top_k` 个候选。
- 不修改 `ollama_reranker_backend._parse_scores` 的严格协议校验，不接受 `system_prompt` 等异常结构。
- 降级候选没有 `rerank_score`，后续继续使用现有 RRF 分数作为知识命中分数。
- Embedding、Qdrant、总检索期限以及其他未知编程错误仍按原方式失败，避免掩盖非 Reranker 故障。

## 数据流

```text
Embedding + Sparse
  -> Qdrant Dense/Sparse
  -> RRF 候选
  -> Reranker
       -> 成功：按 rerank_score 排序
       -> 可降级失败：保留 RRF 顺序
  -> 邻接片段扩展
  -> knowledge_hits
  -> 大模型总结
```

## 测试

- 将现有“非 busy Reranker 失败会抛异常”测试改为“服务失败时保留降级数量的 RRF 候选”。
- 新增 Reranker 返回协议异常时同样降级的测试。
- 保留现有 busy 后缩小批次重试成功的测试。
- 验证后端版本从 `0.1.4` 升级到 `0.1.5`，前端版本不变。
