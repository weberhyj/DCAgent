# Enterprise Knowledge Base QA Rollout Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this roadmap phase-by-phase. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved enterprise knowledge-base architecture through six independently testable and reversible phases, beginning with a production-safe Physoc route.

**Architecture:** Keep FastAPI, LangGraph, PostgreSQL, Qdrant, ClickHouse, Redis, and the existing conversation API, then replace the legacy parsing and retrieval path behind explicit phase gates. Each phase produces working software, retains a rollback path, and freezes its public contracts before the next detailed plan is written.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, PostgreSQL, Qdrant, ClickHouse, Redis/Celery, Docling, PaddleOCR, Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Reranker-0.6B, FastEmbed BM25, SQLGlot, Polars/PyArrow/openpyxl, Physoc DeepSeek SSE, Docker Compose, unittest, Ruff, uv.

**Approved Design:** [`../specs/2026-07-24-enterprise-knowledge-base-qa-design.md`](../specs/2026-07-24-enterprise-knowledge-base-qa-design.md)

---

## Phase order

### Phase 1: Physoc production routing and deployment gate

Detailed plan: [`2026-07-24-physoc-production-routing.md`](2026-07-24-physoc-production-routing.md)

- [ ] Production startup rejects `template` and `mock` providers.
- [ ] Offline Compose passes `LLM_STREAM_PATH` and does not require an API key for Physoc.
- [ ] The offline deployment example selects `physoc_deepseek` with a container-reachable private address.
- [ ] A target-host probe verifies POST body, SSE completion, model identity, timeout handling, and safe reporting.
- [ ] A model failure returns an explicit gateway error and never returns retrieved slices as the answer.

Exit gate: a real target server produces a passing Physoc probe report and a model-outage request returns HTTP 502 without any evidence text in the response body.

### Phase 2: Unified document parsing

- [ ] Introduce versioned `DocumentSource`, `DocumentNode`, `DocumentChunk`, and `CitationLocator` persistence contracts.
- [ ] Route PDF/DOCX/PPTX through Docling and TXT/Markdown through native parsers.
- [ ] Route scanned PDF pages and meaningful slide images through offline PaddleOCR.
- [ ] Preserve page, section, slide, worksheet, table, cell-range, OCR confidence, parent, and adjacency metadata.
- [ ] Reject `.doc`, `.ppt`, and `.xls` rather than reading binary data as text.
- [ ] Publish parser output by content hash and parser version while the old version remains queryable.

Exit gate: the fixed parser regression corpus reproduces expected locations and text for every supported format, and failed parsing leaves the previous version active.

### Phase 3: Qwen3 Qdrant hybrid retrieval

Detailed implementation plan: [`2026-07-27-qwen3-hybrid-retrieval-gray-migration.md`](2026-07-27-qwen3-hybrid-retrieval-gray-migration.md)

- [x] Add versioned Qwen3 Dense and local BM25 Sparse indexes in Qdrant.
- [x] Apply knowledge-base, permission-tag, and publication filters before both retrieval branches.
- [x] Fuse Dense/Sparse Top 50 with RRF, rerank Top 24 with Qwen/Qwen3-Reranker-0.6B, and return Top 8 authorized evidence with adjacent context.
- [x] Move the Qwen3 route away from PostgreSQL full-table chunk scans and Python per-row scoring while preserving Legacy fallback.
- [x] Add `RETRIEVAL_MODE=legacy|shadow|qwen3`, Shadow comparison, stable canary routing, versioned collection publication, and quality/capacity gates.
- [ ] Complete the approved target-host Shadow 10 -> 50 -> 100 and canary 5 -> 25 -> 50 -> 100 rollout.

Exit gate: permission leakage is zero, Recall@50 is at least 90%, NDCG@8 does not regress, critical Top-8 regressions are zero, and Qdrant retrieval meets the five-second p95 target in the mandatory 15-user acceptance run. Immediate rollback is `RETRIEVAL_MODE=legacy rollback`; index recovery uses the governed Alias rollback procedure.

### Phase 4: Query routing and hierarchical synthesis

- [ ] Implement `factual_qa`, `topic_summary`, `document_summary`, `document_compare`, `structured_aggregate`, `follow_up`, and `clarification` routes.
- [ ] Implement chapter-to-document-to-corpus Map-Reduce summaries with coverage and conflict checks.
- [ ] Build authorized evidence packages instead of concatenating raw slices.
- [ ] Validate every generated claim against citations before persistence.
- [ ] Keep hidden search-mode controls hidden; routing remains automatic and full-knowledge-base by default.

Exit gate: factual, full-document, cross-document, comparison, no-evidence, and model-outage acceptance cases return synthesized answers or explicit errors, never slice lists.

### Phase 5: Excel/CSV exact analytics

- [ ] Complete the existing schema confirmation, Parquet staging, ClickHouse publication, and governed query path.
- [ ] Enforce the approved null, invalid-value, formula-cache, duplicate-row, hidden-row, Decimal precision, and `ROUND_HALF_UP` rules.
- [ ] Route `avg`, `sum`, `count`, `min`, `max`, grouping, and filters to ClickHouse over the complete published dataset.
- [ ] Keep numeric facts immutable when a mixed question is explained by Physoc.
- [ ] Return file, worksheet, range, row counts, valid/null counts, schema version, publication ID, and audit ID.

Exit gate: every supported aggregate matches the Polars reference result exactly, including boundary-value rounding cases, and ClickHouse failure never falls back to slice arithmetic.

### Phase 6: Citations, asynchronous operations, security, and acceptance

- [ ] Add Redis/Celery parser, OCR, embedding, and structured worker queues with PostgreSQL durable state.
- [ ] Add retries, exponential backoff, dead-letter handling, resource isolation, version publication, and rollback drills.
- [ ] Add citation UI contracts while leaving the chat attachment entry hidden.
- [ ] Add ClamAV upload checks, permission tests, prompt-injection tests, audit logging, backup/restore, and observability.
- [ ] Run 15-in-flight and 15-user closed-loop capacity tests against at least 30 million ClickHouse rows and 5 million Qdrant points.

Exit gate: security, recovery, citation, ingestion, load, and end-to-end business acceptance reports all pass on the approved internal deployment profile.

## Cross-phase invariants

- [ ] Runtime services never download packages, containers, OCR assets, Embedding models, Reranker models, or LLM weights from the public network.
- [ ] Every request pins one PostgreSQL publication manifest and uses the exact Qdrant collection and ClickHouse table named by it.
- [ ] Authorization filters are applied before retrieval, statistics, context construction, and model invocation.
- [ ] Document synthesis requires Physoc; model failure is explicit and cannot reveal raw slices.
- [ ] Pure structured statistics may return deterministic results without Physoc, but never use slice estimates.
- [ ] Excel/CSV averages, sums, counts, minima, and maxima use ClickHouse complete-data aggregation; averages are never calculated from RAG chunks.
- [ ] User-side attachment and manual search-mode controls remain hidden.
- [ ] Each phase uses TDD, focused commits, full backend regression tests, Ruff checks, and a documented rollback.

## Planning rule

Phase 1 and the Phase 3 Qwen3 implementation now have detailed plans. A phase is not operationally complete merely because its deterministic tests pass: the target-host model checksums, Alias state, permission corpus, Physoc route, live dependencies, and capacity report must also pass. Before changing the remaining parser, synthesis, security, or queue phases, inspect the resulting code and write the next detailed plan with exact post-migration paths and signatures.
