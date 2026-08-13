# llama.cpp BGE-M3 与知识路由隔离实施计划

> **For agentic workers:** Implement this plan task-by-task with a test-first cycle. Keep each task independently testable and commit after verification.

**Goal:** 接入 llama.cpp BGE-M3 Q4 Embedding 与 BGE-Reranker-v2-M3 Q4，并阻断 Excel/Word 精确问题误入通用 RAG。

**Architecture:** 在现有稳定的 `/v1/embeddings` 与 `/v1/rerank` 内部协议外增加 llama.cpp Embedding adapter；Reranker 复用现有 llama.cpp adapter。知识路由使用显式终态：结构化问题解析失败时澄清，不再回退到其他知识链路；Excel 行查询和 Word factual 查询均使用确定性数据源。

**Tech Stack:** Python 3.12, FastAPI, httpx, ClickHouse, PostgreSQL, Qdrant, llama.cpp, pytest/unittest.

## Global Constraints

- BGE-M3 query/document 不添加 BGE-large-v1.5 查询前缀，使用 raw profile。
- Embedding 与 Reranker 使用两个独立 llama.cpp 服务和独立端口。
- 精确 Excel/Word 查询不调用 Embedding、Reranker 或 LLM。
- 新 Embedding 必须使用新的 Qdrant collection 并全量重建向量后才能切 alias。
- 所有外部模型响应必须校验数量、维度、索引、有限数值和范围。
- 不提交生产 `.env`、密码文件、模型文件或真实 checksum。

### Task 1: llama.cpp Embedding adapter

**Files:**
- Create: `backend/app/llama_cpp_embedding_backend.py`
- Modify: `backend/app/embedding_service.py`
- Modify: `backend/app/embedding_contracts.py` only if metadata needs additive runtime field
- Modify: `backend/tests/test_llama_cpp_embedding_backend.py`
- Modify: `backend/tests/test_embedding_service.py`
- Modify: `deploy/offline/.env.example`
- Modify: `.env.example`
- Modify: `deploy/offline/README.md`
- Modify: `deploy/offline/compose.yaml` only for adapter environment names

**Interfaces:**
- `LlamaCppEmbeddingBackend.embed(texts, purpose) -> list[list[float]]`
- `SyncLlamaCppEmbeddingClient.post_json(path, payload) -> object`
- `EMBEDDING_RUNTIME=ollama|llama_cpp`, with existing Ollama default preserved for compatibility
- llama.cpp response accepts OpenAI-compatible `data[].embedding` and validates optional index ordering

- [ ] Write failing tests for query/document payloads, response parsing, normalization, vector count mismatch, malformed data, URL/model/path validation, and runtime selection.
- [ ] Run the focused tests and confirm they fail because the backend/runtime branch is absent.
- [ ] Implement the smallest HTTP client and backend adapter; keep the existing `/v1/embeddings` API unchanged.
- [ ] Run focused tests and then existing embedding service tests.
- [ ] Add deployment configuration examples for BGE-M3 Q4 and llama.cpp endpoint variables.
- [ ] Commit as `feat: add llama.cpp bge-m3 embedding adapter`.

### Task 2: Excel row lookup

**Files:**
- Modify: `backend/app/structured_models.py`
- Modify: `backend/app/structured_query.py`
- Modify: `backend/app/structured_answer.py`
- Modify: `backend/app/knowledge_route_models.py`
- Modify: `backend/tests/test_structured_query.py`
- Modify: `backend/tests/test_structured_answer.py`

**Interfaces:**
- `StructuredRowLookupIntent(dataset_id, filter, selected_columns, limit)`
- `StructuredRowLookupPlan(publication_id, dataset_id, sql, parameters, filter, selected_columns, limit)`
- `StructuredRowLookupResult(dataset_id, schema_version, columns, rows, total_count, truncated, source metadata)`
- `KnowledgeRouteType.EXCEL_ROW_LOOKUP`

- [ ] Write failing tests for “条件列=值返回其他列”、多行结果、未知字段、未发布数据集和超限截断。
- [ ] Run focused tests and confirm the new intent/result types are missing.
- [ ] Implement parameterized ClickHouse SELECT with whitelist-only columns and bounded LIMIT.
- [ ] Render a table artifact without LLM and attach dataset/publication metadata.
- [ ] Run structured query and answer regression tests.
- [ ] Commit as `feat: add deterministic excel row lookup`.

### Task 3: Hard route isolation

**Files:**
- Modify: `backend/app/knowledge_router.py`
- Modify: `backend/app/structured_answer.py`
- Modify: `backend/app/structured_query.py`
- Modify: `backend/tests/test_knowledge_router.py`
- Modify: `backend/tests/test_structured_answer.py`

- [ ] Add failing tests proving Excel-looking questions that cannot parse never call Word or document Agent.
- [ ] Add failing tests proving row lookup questions are recognized even without aggregate words.
- [ ] Implement explicit structured candidate decisions and terminal clarification/unavailable results.
- [ ] Preserve existing non-structured document routing behavior.
- [ ] Run routing and structured test suites.
- [ ] Commit as `fix: isolate structured questions from document rag`.

### Task 4: Word factual precision and context policy

**Files:**
- Modify: `backend/app/word_facts.py`
- Modify: `backend/app/word_fact_answer.py`
- Modify: `backend/app/knowledge_router.py`
- Modify: `backend/app/agent.py`
- Modify: `backend/app/llm.py`
- Modify: `backend/app/retrieval_models.py` or request construction only if needed for expansion policy
- Modify: `backend/tests/test_word_facts.py`
- Modify: `backend/tests/test_word_fact_answer.py`
- Modify: `backend/tests/test_knowledge_router.py`
- Modify: `backend/tests/test_agent.py`
- Modify: `backend/tests/test_llm_provider.py`

- [ ] Add failing tests for natural-language age/gender/job queries and unknown entity-field questions.
- [ ] Add failing tests proving factual routes omit history and adjacency expansion.
- [ ] Add failing tests proving independent RAG questions omit history while follow-ups retain bounded history.
- [ ] Implement robust factual candidate detection and terminal clarification.
- [ ] Pass `EvidenceExpansionPolicy.NONE` for factual/structured routes and only enable adjacency for open document QA.
- [ ] Add a conservative follow-up detector before including history in `build_prompt`.
- [ ] Run focused Word, Agent, LLM, and routing tests.
- [ ] Commit as `fix: keep factual answers scoped to requested fields`.

### Task 5: Deployment and migration documentation

**Files:**
- Modify: `deploy/offline/LLAMA_CPP_RERANKER.md`
- Modify: `deploy/offline/README.md`
- Modify: `deploy/ubuntu/KNOWLEDGE_ROUTING_ROLLOUT.md`
- Modify: `docs/intranet-deployment-configuration.md`
- Create: `deploy/ubuntu/LLAMA_CPP_EMBEDDING.md`

- [ ] Document two independent llama.cpp services, BGE-M3 embedding startup probe, BGE reranker probe, Supervisor commands, and rollback.
- [ ] Document new Qdrant collection/reindex requirements and exact metadata checks.
- [ ] Run documentation/config contract tests.
- [ ] Commit as `docs: document llama.cpp bge retrieval rollout`.

### Task 6: Full verification

- [ ] Run backend unit tests with the project virtualenv.
- [ ] Run ruff/lint and type-oriented checks available in the repository.
- [ ] Run frontend build only if frontend files changed; otherwise record it as not applicable.
- [ ] Inspect `git diff --check`, staged diff, and secret patterns.
- [ ] Run code review checklist across correctness, architecture, security and performance.
- [ ] Do not claim completion until all commands have fresh passing output.
