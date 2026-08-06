# Reranker RRF Degradation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reranker 返回 503、忙碌或协议异常时保留 RRF 检索证据，避免 `evidenceCount=0`。

**Architecture:** 严格 Reranker 协议保持不变；仅在 `HybridRetriever._rerank` 边界捕获明确的可降级客户端异常，并返回已经按 RRF 排序的受限候选。Embedding、Qdrant 和未知异常继续失败，避免扩大容错边界。

**Tech Stack:** Python 3.12、unittest、HybridRetriever、Reranker HTTP client、Ruff、uv

---

### Task 1: 用回归测试定义 RRF 降级行为

**Files:**
- Modify: `backend/tests/test_hybrid_retriever.py`

- [ ] **Step 1: 增加协议异常测试替身和失败测试**

```python
class InvalidResponseReranker(RecordingReranker):
    def rerank(self, query, passages, *, expected, timeout_seconds=None):
        self.batch_sizes.append(len(passages))
        self.timeouts.append(timeout_seconds)
        raise RerankerResponseError("invalid response")

def test_degrades_to_rrf_candidates_when_reranker_service_is_unavailable(self) -> None:
    reranker = FailingReranker()
    retriever = self.addCleanupFor(build_retriever(reranker=reranker))

    outcome = retriever.retrieve(request())

    self.assertEqual(reranker.batch_sizes, [24])
    self.assertEqual(len(outcome.candidates), 8)
    self.assertTrue(all(item.rerank_score is None for item in outcome.candidates))

def test_degrades_to_rrf_candidates_when_reranker_response_is_invalid(self) -> None:
    reranker = InvalidResponseReranker()
    retriever = self.addCleanupFor(build_retriever(reranker=reranker))

    outcome = retriever.retrieve(request())

    self.assertEqual(reranker.batch_sizes, [24])
    self.assertEqual(len(outcome.candidates), 8)
```

- [ ] **Step 2: 运行测试并确认因异常仍向外抛出而失败**

Run: `uv run --project backend --group dev python -m pytest backend/tests/test_hybrid_retriever.py -q`

Expected: FAIL，`RerankerServiceError` 或 `RerankerResponseError` 从 `HybridRetriever._rerank` 向外传播。

### Task 2: 实现受限 RRF 降级

**Files:**
- Modify: `backend/app/hybrid_retriever.py`
- Test: `backend/tests/test_hybrid_retriever.py`

- [ ] **Step 1: 导入可降级异常并实现最小处理**

```python
from .reranker_client import (
    RerankerBusy,
    RerankerResponseError,
    RerankerServiceError,
)

try:
    scores = self._run_one(...)
except RerankerBusy:
    candidates = candidates[: self._degraded_rerank_top_k]
    try:
        scores = self._run_one(...)
    except (RerankerBusy, RerankerResponseError, RerankerServiceError):
        return candidates
except (RerankerResponseError, RerankerServiceError):
    return candidates[: self._degraded_rerank_top_k]
```

- [ ] **Step 2: 运行定向测试并确认通过**

Run: `uv run --project backend --group dev python -m pytest backend/tests/test_hybrid_retriever.py -q`

Expected: PASS。

### Task 3: 独立升级后端版本并验证

**Files:**
- Modify: `backend/app/__init__.py`
- Test: `tools/tests/test_version_contract.py`

- [ ] **Step 1: 将后端补丁版本升级到 0.1.5**

```python
__version__ = "0.1.5"
```

- [ ] **Step 2: 运行版本契约测试**

Run: `uv run --project backend --group dev python -m pytest tools/tests/test_version_contract.py -q`

Expected: PASS，用户端与管理端版本保持不变。

### Task 4: 完整验证

**Files:**
- Verify: `backend/app/hybrid_retriever.py`
- Verify: `backend/tests/test_hybrid_retriever.py`
- Verify: `backend/app/__init__.py`

- [ ] **Step 1: 运行 Ruff**

Run: `uv run --project backend --group dev ruff check backend/app backend/tests`

Expected: exit 0。

- [ ] **Step 2: 运行后端完整测试**

Run: `uv run --project backend --group dev python -m pytest backend/tests -q`

Expected: exit 0，无失败用例。

- [ ] **Step 3: 检查差异仅覆盖已批准范围**

Run: `git diff --check && git status --short`

Expected: 无空白错误；仅包含设计、计划、Reranker 降级、测试和后端版本文件。
