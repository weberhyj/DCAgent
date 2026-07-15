# Offline Mixed-Data Architecture Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each phase task-by-task. Each phase has its own plan and must keep the existing legacy path working until its cutover gate passes.

**Goal:** Deliver an offline, mixed-data DC-Agent that separates exact Excel analytics from document semantic retrieval while preserving a reversible migration path.

**Architecture:** PostgreSQL owns control-plane metadata, permissions, job state, and publication manifests. ClickHouse owns structured Excel data. Qdrant owns document and selected free-text vectors. A local llama.cpp service provides optional streamed generation with bounded concurrency and deterministic fallbacks.

**Tech Stack:** Python 3.12 container, FastAPI, SQLAlchemy/Alembic, PostgreSQL, ClickHouse, Qdrant, Redis, Docling, PaddleOCR, openpyxl/pyxlsb/LibreOffice, Polars/PyArrow, private BGE/FlagEmbedding or ONNX Runtime service, optional local BGE Reranker, Jieba, SQLGlot, llama.cpp, Qwen GGUF, Locust.

**Approved Design:** [`../specs/2026-07-15-offline-mixed-data-architecture-design.md`](../specs/2026-07-15-offline-mixed-data-architecture-design.md)

---

## Phase decomposition

The approved architecture spans independent subsystems. Implement them in this order:

1. [Offline platform foundation](2026-07-15-offline-platform-foundation.md)
   - Environment contract, dependency lock, Alembic baseline, Compose services, private shared Embedding service, health checks, and benchmark harness.
   - Does not change the existing answer or ingestion path.
2. [Batch ingestion and indexing](2026-07-15-offline-ingestion-and-indexing.md)
   - Durable PostgreSQL jobs with lease tokens/heartbeats/checkpoints, streaming files and redaction, ClickHouse staging, document parsing, changed-row detection, shared Embedding, BM25 artifacts, dense-vector cache, Qdrant collections, and one canonical publication manifest.
   - Keeps legacy PostgreSQL JSON vectors behind a feature flag.
3. [Online query and local answer](2026-07-15-offline-query-and-answer.md)
   - PostgreSQL-backed permission scopes, query routing, safe ClickHouse plans, shared-Embedding Qdrant hybrid retrieval, optional bounded reranking, namespaced caches, mixed result fusion, local generation, exactly-once degradation/SSE persistence, and frontend streaming.
   - Enables the new path only after Phase 2 publication and integration gates pass.
4. [Acceptance and cutover](2026-07-15-offline-acceptance-and-cutover.md)
   - Timed off-host backup/restore drills, separate 32GB/64GB and batch-under-load gates, security/failure testing, machine-readable shadow comparison, principal/department rollout, rollback, and guarded legacy cleanup.

## Global invariants

- No runtime external API calls. Model files, wheels, and container images are supplied from the internal offline artifact mirror.
- A request pins one PostgreSQL publication manifest and uses the exact ClickHouse physical table and Qdrant collection named by that manifest.
- PostgreSQL is the durable source of truth for ingestion jobs. Redis is only a cache and worker-dispatch layer.
- Ingestion and query use one checksum-pinned private Embedding service; no API or worker loads a second model copy.
- Semantic ingestion is sink-based and bounded; changed rows reuse versioned fingerprints and dense-vector cache entries.
- Numeric Excel facts come from ClickHouse; Qdrant evidence can explain them but cannot overwrite them.
- Existing SQLite unit tests and the legacy `InMemoryChatRepository` remain available until the final cutover gate.
- Every phase ends with focused tests, the existing backend regression suite, and a small, reversible commit.

## Cross-phase verification

Run from the repository root after each phase:

```powershell
Set-Location backend
py -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..
```

The full 30-million-row/5-million-point benchmark is never committed to Git. It runs only on the target server or dedicated CI host using the manifest and artifact versions recorded by Phase 1.

## Phase handoff gates

- Phase 1 gate: core services (PostgreSQL, migration, ClickHouse, Qdrant, Redis, ClamAV, Embedding, and API) have contract-checked configuration and the target host can run the core smoke profile; worker and llama.cpp profile smoke is executed after their Phase 2/3 entrypoints exist.
- Phase 2 gate: a test batch publishes a complete canonical ClickHouse/Qdrant manifest, survives Redis loss and lease expiry, reuses unchanged vectors, and leaves the legacy path usable.
- Phase 3 gate: structured, document, and mixed queries pass PostgreSQL-backed permission, provenance, timeout, reranker/cache, exactly-once persistence, and degradation tests; streaming starts promptly when a model slot is available.
- Phase 4 gate: both memory profiles and all batch windows have separate measured results, a timed off-host restore meets RPO/RTO, cohort rollout and rollback are exercised, and legacy cleanup guards pass.
