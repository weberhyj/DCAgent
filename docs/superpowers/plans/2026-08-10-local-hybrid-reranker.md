# Local Hybrid Retrieval and Remote LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run BGE Embedding and BGE Reranker locally while keeping answer generation on the approved remote LLM service.

**Architecture:** Docker Compose runs PostgreSQL, Qdrant, Redis, ClamAV, the API, the Ollama-backed embedding adapter, and the optional llama.cpp reranker adapter. The API uses `RETRIEVAL_MODE=qwen3`, Dense + BM25 + RRF, and `RERANKER_ENABLED=true`; Physoc/OpenAI-compatible generation remains an external endpoint configured through `LLM_API_BASE`.

**Tech Stack:** Docker Compose v2, FastAPI, Qdrant, Ollama BGE embedding adapter, llama.cpp `bge-reranker-v2-m3-Q4_K_M.gguf`, PostgreSQL, Redis, and remote Physoc/OpenAI-compatible LLM.

## Global Constraints

- Do not commit `deploy/offline/.env`, model files, secrets, or generated data directories.
- Keep the remote LLM endpoint and credentials outside tracked files.
- The reranker must expose `/v1/rerank` and return per-passage relevance scores.
- The API must fail readiness when qwen3 dependencies or reranker metadata are invalid.
- Structured Excel/CSV aggregation remains disabled unless ClickHouse is separately provisioned and approved.
- Windows development requires WSL2/Docker Desktop; the repository production runbook targets Ubuntu 20.04.

---

### Task 1: Prepare the local host and model artifacts

**Files:**
- Read: `docs/offline-platform-runbook.md`
- Read: `deploy/offline/README.md`
- Read: `deploy/offline/LLAMA_CPP_RERANKER.md`
- External inputs: Docker Desktop + WSL2 or Ubuntu 20.04, BGE embedding model available through Ollama, `bge-reranker-v2-m3-Q4_K_M.gguf`, approved remote LLM URL/model

**Interfaces:**
- Produces fixed writable `DATA_ROOT`, `MODEL_ROOT`, and secret paths consumed by the deployment scripts.

- [ ] Install Docker Engine/Compose v2 in Ubuntu or WSL2 and verify `docker version` and `docker compose version`.
- [ ] Place the approved reranker GGUF at `${MODEL_ROOT}/bge-reranker-v2-m3-Q4_K_M.gguf` and calculate its SHA-256.
- [ ] Pull or pre-seed the approved Ollama embedding model and verify `POST /api/embed` returns a vector with the configured dimension.
- [ ] Confirm the remote LLM endpoint, stream path, model name, and network reachability from the API container.

### Task 2: Initialize deployment state and secrets

**Files:**
- Modify runtime-only: `deploy/offline/.env`
- Create runtime-only: `artifacts/secrets/database-url`, `artifacts/secrets/clickhouse-query-password`, `artifacts/secrets/clickhouse-ingest-password`
- Run: `tools/prepare_offline_env.sh --initialize-state`

**Interfaces:**
- Consumes: fixed `HOST_DATA_ROOT` and `HOST_MODEL_ROOT` from Task 1.
- Produces: validated Compose environment, data/model directories, and secret files.

- [ ] Run the initialization command from the repository root with explicit absolute data/model roots.
- [ ] Set `RETRIEVAL_MODE=qwen3`, `RERANKER_ENABLED=true`, `RERANKER_RUNTIME=llama_cpp`, and the approved embedding/reranker metadata in `deploy/offline/.env`.
- [ ] Set `LLAMA_CPP_RERANKER_URL`, `LLAMA_CPP_RERANKER_MODEL`, `LLAMA_CPP_RERANKER_PATH=/v1/rerank`, timeout, and batch limits.
- [ ] Set `LLM_PROVIDER=physoc_deepseek`, the approved `LLM_API_BASE`, `LLM_STREAM_PATH`, and `LLM_MODEL`; keep API keys in runtime-only environment/secret handling.
- [ ] Run the repository’s environment validation and confirm no secret values are printed.

### Task 3: Start infrastructure and model adapters

**Files:**
- Use: `deploy/offline/compose.yaml`
- Use: `tools/invoke_offline_compose.sh`

**Interfaces:**
- Consumes: Task 2 environment and secrets.
- Produces: healthy PostgreSQL, Qdrant, Redis, ClamAV, embedding-service, reranker-service, and API containers.

- [ ] Build/start the base services using the documented transactional deployment order.
- [ ] Start the reranker profile with `./tools/invoke_offline_compose.sh --profile reranker up -d reranker-service api`.
- [ ] Verify service health checks and API `/api/readyz`; readiness must be 200 with embedding, Qdrant, alias, and reranker checks healthy.
- [ ] Call the reranker adapter contract with one query and two passages and verify stable scores/order.

### Task 4: Publish an index and verify hybrid retrieval

**Files:**
- Use: admin frontend knowledge management flow
- Inspect: `backend/app/retrieval_router.py`, `backend/app/hybrid_retriever.py`, `backend/app/retrieval_publication.py`

**Interfaces:**
- Consumes: a test document with the configured permission tag and knowledge-base ID.
- Produces: a published Qdrant alias and an auditable qwen3 retrieval result.

- [ ] Upload a UTF-8 test document through `/api/knowledge/uploads`.
- [ ] Wait for indexing and confirm the source is published in PostgreSQL and Qdrant alias state.
- [ ] Ask a question that requires the test document and verify the response contains authorized evidence.
- [ ] Inspect retrieval audit data to confirm dense/sparse candidates, RRF fusion, reranker invocation, final top-k, latency, and no fallback code.

### Task 5: Verify remote generation and failure behavior

**Files:**
- Inspect: `backend/app/physoc_sse.py`, `backend/app/llm.py`, `backend/app/retrieval_router.py`
- Verify: `backend/tests/test_physoc_sse.py`, `backend/tests/test_reranker_service.py`, `backend/tests/test_retrieval_router.py`

**Interfaces:**
- Consumes: Task 4 evidence and remote LLM stream.
- Produces: a complete answer or an explicit HTTP 502 on generation failure.

- [ ] Run a normal question and confirm the answer is generated by the remote LLM from bounded evidence.
- [ ] Temporarily make the remote LLM unavailable and confirm the API returns HTTP 502 without exposing raw chunks/template text.
- [ ] Temporarily make the reranker unavailable and confirm the configured degraded/fallback behavior is recorded and bounded.
- [ ] Record readiness, P95 latency, error rate, reranker fallback rate, and model metadata for the acceptance report.

### Task 6: Final acceptance

**Files:**
- Verify: `tools/tests/test_compose_smoke.py`, `tools/tests/test_intranet_deployment_gate.py`, `tools/ui_smoke.py`
- Read: `docs/offline-platform-runbook.md`

- [ ] Run Compose smoke and model probes on the Docker/Ubuntu host.
- [ ] Run frontend tests/build with the pnpm workspace.
- [ ] Run backend contract tests in the synced uv environment.
- [ ] Complete the approved concurrency and real-document acceptance gate before treating the setup as production-ready.
