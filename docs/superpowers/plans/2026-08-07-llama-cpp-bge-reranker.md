# llama.cpp BGE GGUF Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 reranker-service 通过 llama.cpp 原生 `/v1/rerank` 使用 BGE GGUF 模型，同时保持 DC-Agent 现有内部协议。

**Architecture:** 增加独立的 llama.cpp HTTP backend，使用 `RERANKER_RUNTIME=llama_cpp` 选择；请求适配为 `query/documents`，响应按 index 转换为内部 `scores`。Ollama 路线保持默认兼容。

**Tech Stack:** Python 3.12、FastAPI、httpx、pytest、uv、Ruff、ty。

---

### Task 1: llama.cpp 响应转换测试

**Files:**
- Create: `backend/tests/test_llama_cpp_reranker_backend.py`

- [ ] 写入 fake HTTP transport 和成功、乱序、重复 index、HTTP 错误测试。
- [ ] 运行 `uv run --project backend python -m pytest backend/tests/test_llama_cpp_reranker_backend.py -q`，确认新增模块不存在导致失败。

### Task 2: 实现 llama.cpp backend

**Files:**
- Create: `backend/app/llama_cpp_reranker_backend.py`
- Modify: `backend/app/reranker_service.py`

- [ ] 实现 URL 校验、请求转换、响应 index 校验和分数 materialize。
- [ ] 在 production lifespan 中按 `RERANKER_RUNTIME` 选择 Ollama 或 llama.cpp loader。

### Task 3: 配置和部署文档

**Files:**
- Modify: `deploy/offline/compose.yaml`
- Modify: `deploy/offline/.env.example`
- Modify: `deploy/offline/README.md`

- [ ] 增加 llama.cpp URL、模型名、路径和运行时环境变量。
- [ ] 添加 llama-server `--reranking` 启动和 curl 探针。

### Task 4: 版本和验收

- [ ] 升级后端版本号。
- [ ] 运行相关 pytest、完整 pytest、Ruff、`fast lint --ty` 和 `git diff --check`。
- [ ] 提交并推送到 `main`。
