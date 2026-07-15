# Offline Query and Local Answer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route authorized questions to ClickHouse, Qdrant, or both; preserve provenance; use the shared private Embedding service; optionally rerank a bounded candidate set; and stream local answers with deterministic degradation.

**Architecture:** A request first resolves its principal from PostgreSQL and derives a non-empty `QueryAccessScope`, then pins one canonical `PublicationManifest`. Every engine receives the same scope and pinned physical versions. Redis caches are namespaced by manifest, scope, catalog, and model checksums. The asynchronous `AnswerEngine` adapter keeps legacy synchronous repositories working while providing exactly-once persistence for non-streaming and SSE requests.

**Tech Stack:** FastAPI, Pydantic/dataclasses, PostgreSQL, ClickHouse, Qdrant, Redis, SQLGlot, shared `HttpEmbeddingClient`, optional local BGE reranker, httpx streaming, llama.cpp, Vue 3 fetch streaming, unittest/Vitest.

---

## Canonical query contracts

`QueryAccessScope` is defined only in `backend/app/authorization.py`. `PinnedPublication` is imported from `backend/app/publication_models.py`; this plan never redeclares it. `PrincipalResolver` resolves scope from the PostgreSQL authorization repository rather than trusting a client-supplied scope header. Every `AnswerEngine` method receives `scope` and `publication`.

Query tests extend `backend/tests/support/offline_fakes.py` from Phase 2 with `sample_scope`, `sample_publication`, recording engine clients, and deterministic evidence/fact builders used in the snippets below. Test-specific failure runtimes remain in their own test module.

### Task 1: Resolve principals, permissions, and one publication

**Files:**
- Create: `backend/app/authorization.py`
- Create: `backend/app/sql_authorization_repository.py`
- Create: `backend/app/query_models.py`
- Create: `backend/alembic/versions/20260715_02_query_authorization.py`
- Create: `backend/tests/test_authorization_filters.py`
- Create: `backend/tests/test_principal_resolver.py`
- Create: `backend/tests/test_sql_authorization_repository.py`
- Modify: `backend/tests/support/offline_fakes.py`
- Modify: `backend/app/publication.py`
- Modify: `backend/app/offline_settings.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes.py`

- [ ] **Step 1: Write failing deny-by-default and equivalent-filter tests**

```python
class AuthorizationTest(unittest.TestCase):
    def test_scope_is_loaded_from_authorization_source_and_filters_are_nonempty(self) -> None:
        scope = QueryAccessScope("u1", ("tenant-a",), ("finance",), ("internal",))
        sql, parameters = build_clickhouse_permission_predicate(scope)
        qdrant = build_qdrant_permission_filter(scope)
        self.assertIn("tenant_id IN", sql)
        self.assertEqual(parameters["tenant_ids"], ["tenant-a"])
        self.assertEqual(qdrant["must"][0]["key"], "tenant_id")
        self.assertEqual(qdrant["must"][0]["match"]["any"], ["tenant-a"])

    def test_empty_scope_and_missing_identity_are_denied(self) -> None:
        with self.assertRaises(ValueError):
            QueryAccessScope("u1", (), (), ())
        with self.assertRaises(HTTPException) as error:
            PrincipalResolver(FakeAuthorizationRepository(scope=None)).resolve(fake_request())
        self.assertEqual(error.exception.status_code, 403)

    def test_admin_capability_is_database_backed_and_deny_by_default(self) -> None:
        allowed = PrincipalResolver(FakeAuthorizationRepository(permissions={"shadow_audit:read"}))
        allowed.require_permission(fake_request(principal="acceptance-runner"), "shadow_audit:read")
        denied = PrincipalResolver(FakeAuthorizationRepository(permissions=set()))
        with self.assertRaises(HTTPException) as error:
            denied.require_permission(fake_request(principal="ordinary-user"), "shadow_audit:read")
        self.assertEqual(error.exception.status_code, 403)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_authorization_filters tests.test_principal_resolver tests.test_sql_authorization_repository -v
```

Expected: FAIL because scope resolution and filter builders do not exist.

- [ ] **Step 3: Implement scope and principal resolution**

```python
@dataclass(frozen=True, slots=True)
class QueryAccessScope:
    user_id: str
    tenant_ids: tuple[str, ...]
    department_ids: tuple[str, ...]
    classifications: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.user_id or not (self.tenant_ids or self.department_ids or self.classifications):
            raise ValueError("query scope must be non-empty")

class AuthorizationRepository(Protocol):
    def resolve_scope(self, user_id: str) -> QueryAccessScope | None: ...
    def has_permission(self, user_id: str, permission: str) -> bool: ...
```

`20260715_02_query_authorization.py` has `down_revision = "20260715_01"` and creates `authorization_principals`, `authorization_roles`, `principal_roles`, `authorization_permissions`, `role_permissions`, and role/direct tenant, department, and classification mapping tables with foreign keys and unique mappings. Seed no permissions by default. `SqlAuthorizationRepository.resolve_scope()` unions direct mappings with effective role mappings in one transaction, sorts/deduplicates them, and returns `None` for disabled/missing principals. `has_permission(user_id, permission)` returns true only for an enabled principal whose assigned enabled role contains that exact permission; there is no wildcard or header fallback. Role assignments, effective mappings, and capabilities are the PostgreSQL source of truth. Add a PostgreSQL/SQLite contract test for allowed, disabled, role-derived, empty-scope, capability-granted, and capability-denied principals.

`PrincipalResolver` checks the peer address against `TRUSTED_PROXY_CIDRS`, reads only `IDENTITY_HEADER` (default and deployed value `X-Identity`), and calls the SQL repository. Missing identity is 401; missing/empty PostgreSQL query scope or missing capability is 403. Query-route `resolve()` requires a non-empty `QueryAccessScope`. Admin-route `require_permission(request, permission)` authenticates the same principal identity and calls `has_permission()` without requiring a query-data scope, so a least-privilege acceptance reader can have only `shadow_audit:read`; it never accepts a permission header. The scope builders always emit tenant, department, and classification constraints where present; neither query engine accepts an empty filter. Add `IDENTITY_HEADER=X-Identity`, `TRUSTED_PROXY_CIDRS`, and `REQUEST_ID_HEADER`; no client or proxy scope header is accepted as an authorization source. `create_app()` constructs the SQL repository/resolver lazily and the message routes require its dependency before calling any conversation or answer service.

Extend `PublicationService.pin(publication_scope="default")` to return the exact canonical `PinnedPublication` from `publication_models.py`. The route dependency calls `pin("default")` once and passes that object downstream; the `QueryAccessScope` is never overloaded as a publication scope string.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_authorization_filters tests.test_principal_resolver tests.test_sql_authorization_repository tests.test_publication -v
Set-Location ..
git add backend/app/authorization.py backend/app/sql_authorization_repository.py backend/app/query_models.py backend/app/publication.py backend/app/offline_settings.py backend/app/main.py backend/app/routes.py backend/alembic/versions/20260715_02_query_authorization.py backend/tests/support/offline_fakes.py backend/tests/test_authorization_filters.py backend/tests/test_principal_resolver.py backend/tests/test_sql_authorization_repository.py
git commit -m "feat: enforce database-backed query scopes"
```

### Task 2: Add a governed semantic catalog and cache keys

**Files:**
- Create: `backend/config/semantic_catalog.json`
- Create: `backend/app/semantic_catalog.py`
- Create: `backend/app/query_cache.py`
- Create: `backend/tests/test_semantic_catalog.py`
- Create: `backend/tests/test_query_cache.py`

- [ ] **Step 1: Write failing catalog/cache tests**

```python
class SemanticCatalogTest(unittest.TestCase):
    def test_rejects_unapproved_expression(self) -> None:
        with self.assertRaises(SemanticCatalogError):
            SemanticCatalog.from_mapping({"datasets": {"sales": {"metrics": {"bad": {"expression": "url(x)"}}}}})

class QueryCacheTest(unittest.TestCase):
    def test_key_contains_manifest_scope_catalog_and_model_versions(self) -> None:
        key = QueryCacheKey(
            kind=CacheKind.RETRIEVAL,
            manifest_id="m1",
            scope_hash="scope-hash",
            catalog_checksum="catalog-hash",
            embedding_sha256="embedding-sha",
            sparse_sha256="sparse-sha",
            reranker_sha256="reranker-sha",
            input_hash="question-hash",
            filter_hash="filters-hash",
            limit=10,
        )
        self.assertEqual(
            key.as_redis_key(),
            "dc:q:retrieval:m1:scope-hash:catalog-hash:embedding-sha:sparse-sha:reranker-sha:question-hash:filters-hash:10",
        )
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_semantic_catalog tests.test_query_cache -v
```

Expected: FAIL because catalog validation and namespaced cache keys do not exist.

- [ ] **Step 3: Implement approved expressions and cache policy**

The catalog permits only `sum`, `count`, `avg`, `min`, `max`, `uniqExact`, explicit dimensions, dates, partitions, and approved joins. It records a SHA-256 checksum. Define `class CacheKind(StrEnum)` with `PLAN`, `EMBEDDING`, `RETRIEVAL`, and `AGGREGATION`. `QueryCacheKey` contains kind, manifest ID, stable scope hash, catalog checksum, Embedding SHA, `publication.sparse.profile_sha256`, optional reranker SHA, normalized-input hash, filter hash, and result limit; it never stores raw permission lists or question text in Redis keys. Separate prefixes/TTLs prevent plan, embedding, retrieval, and aggregation collisions. Redis failure is a cache miss, not a query failure.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_semantic_catalog tests.test_query_cache -v
Set-Location ..
git add backend/config/semantic_catalog.json backend/app/semantic_catalog.py backend/app/query_cache.py backend/tests/test_semantic_catalog.py backend/tests/test_query_cache.py
git commit -m "feat: govern semantic plans and cache namespaces"
```

### Task 3: Build safe bounded ClickHouse plans

**Files:**
- Create: `backend/app/clickhouse_query.py`
- Create: `backend/tests/test_clickhouse_query.py`
- Create: `backend/tests/integration/test_clickhouse_query_integration.py`

- [ ] **Step 1: Write failing SQL safety and permission tests**

```python
class ClickHouseQueryTest(unittest.TestCase):
    def test_plan_uses_pinned_table_scope_partition_and_limit(self) -> None:
        plan = StructuredQueryPlanner(sample_catalog()).plan(
            StructuredIntent("sales", ("revenue",), ("region",), {"order_date": {"gte": "2026-01-01"}}, 100),
            sample_publication(),
            sample_scope(),
        )
        self.assertIn("dc_sales_batch_", plan.sql)
        self.assertIn("tenant_id", plan.sql)
        self.assertLessEqual(plan.limit, 1000)

    def test_unknown_metric_and_unfiltered_table_are_rejected(self) -> None:
        with self.assertRaises(UnsafeQueryError):
            StructuredQueryPlanner(sample_catalog()).plan(
                StructuredIntent("sales", ("shell",)), sample_publication(), sample_scope()
            )
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_clickhouse_query -v
```

Expected: FAIL because the planner and executor do not exist.

- [ ] **Step 3: Implement SQLGlot planning and read-only execution**

Resolve metrics/dimensions through the catalog, use the physical table in the pinned manifest, inject mandatory authorization and partition predicates, and build a parameterized SQLGlot AST. Enforce SELECT-only, whitelisted functions/columns, max limit 1000, read-only account, `max_execution_time=4`, `max_memory_usage=1_000_000_000`, `max_result_rows=1000`, and overflow mode `break`. Cache only successful common aggregations using the namespaced key. Timeout becomes a typed `StructuredUnavailable` result.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_clickhouse_query -v
$env:RUN_OFFLINE_INTEGRATION="1"
py -m unittest tests.integration.test_clickhouse_query_integration -v
Remove-Item Env:RUN_OFFLINE_INTEGRATION
Set-Location ..
git add backend/app/clickhouse_query.py backend/tests/test_clickhouse_query.py backend/tests/integration/test_clickhouse_query_integration.py
git commit -m "feat: add safe bounded ClickHouse queries"
```

### Task 4: Add dense/BM25 retrieval with shared Embedding and Redis cache

**Files:**
- Create: `backend/app/qdrant_retrieval.py`
- Create: `backend/app/bm25_query_encoder.py`
- Modify: `backend/app/embedding_client.py`
- Create: `backend/tests/test_qdrant_retrieval.py`
- Create: `backend/tests/test_bm25_query_encoder.py`
- Create: `backend/tests/integration/test_qdrant_retrieval_integration.py`

- [ ] **Step 1: Write failing filter, metadata, and cache tests**

```python
class QdrantHybridRetrieverTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_same_nonempty_scope_for_dense_and_sparse(self) -> None:
        client = RecordingQdrantSearchClient()
        retriever = QdrantHybridRetriever(client, RecordingEmbeddingClient(), FakeBm25Encoder(), FakeQueryCache())
        hits = await retriever.search("policy", sample_publication(), sample_scope(), limit=10)
        self.assertEqual(client.dense_filter, client.sparse_filter)
        self.assertEqual(len({hit.evidence_id for hit in hits}), len(hits))
        self.assertLessEqual(len(hits), 10)
        self.assertEqual(client.embedding_purpose, "query")
```

```python
class Bm25QueryEncoderTest(unittest.TestCase):
    def test_loads_pinned_vocabulary_and_rejects_checksum_mismatch(self) -> None:
        encoder = Bm25QueryEncoder.from_publication(sample_publication(), artifact_root=fixture_artifacts())
        vector = encoder.encode("alpha beta")
        self.assertEqual(vector.qdrant_idf_modifier, "none")
        self.assertAlmostEqual(vector.value_for("alpha"), math.log(1 + (10 - 2 + 0.5) / (2 + 0.5)))
        with self.assertRaises(ArtifactMismatch):
            Bm25QueryEncoder.from_publication(
                replace(sample_publication(), sparse=bad_sparse_checksum()),
                artifact_root=fixture_artifacts(),
            )
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_bm25_query_encoder tests.test_qdrant_retrieval -v
```

Expected: FAIL because production query encoding, pinned BM25 loading, and retrieval do not exist.

- [ ] **Step 3: Implement bounded hybrid retrieval**

`HttpEmbeddingClient` is the concrete implementation of the Phase 1 `EmbeddingClient` protocol (the protocol remains the canonical injected type). Call it with `purpose="query"` and the pinned `EmbeddingArtifactRef`; reject metadata mismatch. `Bm25QueryEncoder.from_publication()` loads the tokenizer, dictionary, stop words, vocabulary, DF, and `k1/b/k3` files named by the manifest, verifies every checksum, and applies the exact ingestion formula. Search at most 50 dense and 50 sparse candidates in the pinned collection with identical tenant/department/classification/date/source/business-key filters. Apply explicit RRF with `k=60`, deduplicate stable IDs and normalized text hashes, fetch only parent/neighbor IDs, and return at most ten chunks with source/page/slide/worksheet/section/business-key provenance. Cache query embeddings and retrieval results under keys from Task 2.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_bm25_query_encoder tests.test_qdrant_retrieval -v
$env:RUN_OFFLINE_INTEGRATION="1"
py -m unittest tests.integration.test_qdrant_retrieval_integration -v
Remove-Item Env:RUN_OFFLINE_INTEGRATION
Set-Location ..
git add backend/app/qdrant_retrieval.py backend/app/bm25_query_encoder.py backend/app/embedding_client.py backend/tests/test_bm25_query_encoder.py backend/tests/test_qdrant_retrieval.py backend/tests/integration/test_qdrant_retrieval_integration.py
git commit -m "feat: add permission-filtered hybrid retrieval"
```

### Task 5: Add bounded local reranking and authoritative result fusion

**Files:**
- Create: `backend/app/reranker.py`
- Create: `backend/app/result_fusion.py`
- Modify: `backend/app/offline_settings.py`
- Create: `backend/tests/test_reranker.py`
- Create: `backend/tests/test_result_fusion.py`

- [ ] **Step 1: Write failing reranker/fusion tests**

```python
class RerankerTest(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_keeps_rrf_order_and_records_skip(self) -> None:
        reranker = LocalReranker(FakeRerankerRuntime(delay=2.0), timeout_seconds=1.5)
        outcome = await reranker.rerank("q", sample_evidence(20), sample_publication().retrieval)
        self.assertEqual(outcome.reason, RerankReason.TIMEOUT)
        self.assertEqual([item.evidence_id for item in outcome.items], [f"e{i}" for i in range(20)])

    async def test_receives_only_authorized_redacted_candidates(self) -> None:
        runtime = RecordingRerankerRuntime()
        await LocalReranker(runtime, timeout_seconds=1.5).rerank(
            "q",
            [evidence("allowed", authorized=True, text="masked")],
            sample_publication().retrieval,
        )
        self.assertEqual(runtime.seen_ids, ["allowed"])
        self.assertEqual(runtime.seen_texts, ["masked"])

class ResultFusionTest(unittest.TestCase):
    def test_clickhouse_values_cannot_be_overwritten_by_evidence(self) -> None:
        outcome = fuse_results(structured={"revenue": 100}, evidence=[evidence("revenue is 90")], join_keys=("department_id",))
        self.assertEqual(outcome.facts["revenue"].value, 100)
        self.assertEqual(outcome.facts["revenue"].engine, "clickhouse")
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_reranker tests.test_result_fusion -v
```

Expected: FAIL because no reranker protocol or fusion implementation exists.

- [ ] **Step 3: Implement optional reranker and governed joins**

`Reranker` accepts only permission-filtered, redacted top-20 candidates and returns at most ten. `OfflineSettings` adds optional `RERANKER_MODEL_ROOT`, `RERANKER_SLOTS`, and `RERANKER_TIMEOUT_SECONDS`; the production factory loads the checksum-pinned local BGE reranker lazily from the existing read-only `/models` mount with network access disabled. The runtime validates artifact name/version/SHA/protocol, uses a semaphore, and on busy/timeout/checksum mismatch returns the original RRF order with typed `RERANKER_SKIPPED`. An absent configured artifact is not an error. Fusion joins only catalog-approved keys/time ranges, preserves ClickHouse table/manifest/query hash/row IDs, and never lets evidence replace numeric facts.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_reranker tests.test_result_fusion -v
Set-Location ..
git add backend/app/reranker.py backend/app/result_fusion.py backend/app/offline_settings.py backend/tests/test_reranker.py backend/tests/test_result_fusion.py
git commit -m "feat: add optional local reranking and fusion"
```

### Task 6: Build an async `AnswerEngine` and thread scope through repositories

**Files:**
- Create: `backend/app/answer_engine.py`
- Create: `backend/app/query_router.py`
- Create: `backend/app/query_service.py`
- Create: `backend/app/conversation_service.py`
- Create: `backend/alembic/versions/20260715_03_answer_requests.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Create: `backend/tests/test_query_service.py`
- Create: `backend/tests/test_answer_engine.py`
- Create: `backend/tests/test_conversation_service.py`

- [ ] **Step 1: Write failing scope/timeout/exactly-once tests**

```python
class QueryServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_branches_degrade_independently(self) -> None:
        service = build_query_service(clickhouse=SlowStructuredEngine(), qdrant=HealthyDocumentEngine())
        outcome = await service.execute("mixed question", sample_scope(), sample_publication())
        self.assertTrue(outcome.document_available)
        self.assertTrue(outcome.structured_unavailable)

class ConversationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_request_id_appends_once(self) -> None:
        service = build_conversation_service()
        first = await service.complete("c1", "question", sample_scope(), sample_publication(), request_id="r1")
        second = await service.complete("c1", "question", sample_scope(), sample_publication(), request_id="r1")
        self.assertEqual(first.message_id, second.message_id)
        self.assertEqual(service.repository.count_request("r1"), 1)

    async def test_concurrent_duplicate_reserves_one_owner_and_replays_terminal_result(self) -> None:
        service = build_conversation_service()
        first, second = await asyncio.gather(
            service.complete("c1", "question", sample_scope(), sample_publication(), request_id="r2"),
            service.complete("c1", "question", sample_scope(), sample_publication(), request_id="r2"),
        )
        self.assertEqual(first.message_id, second.message_id)
        self.assertEqual(service.repository.count_request("r2"), 1)

    async def test_conflicting_fingerprint_is_rejected_and_stale_owner_is_reclaimable(self) -> None:
        service = build_conversation_service()
        await service.reserve_only("c1", "question", "deep", sample_scope(), sample_publication(), "r3")
        with self.assertRaises(IdempotencyConflict):
            await service.complete("c1", "different", sample_scope(), sample_publication(), request_id="r3")
        service.repository.force_reservation_expiry("r3")
        recovered = await service.complete("c1", "question", sample_scope(), sample_publication(), request_id="r3")
        self.assertEqual(recovered.status, "completed")

    async def test_terminal_state_and_exchange_commit_or_rollback_together(self) -> None:
        service = build_conversation_service(repository=CrashPointRepository("before_commit"))
        with self.assertRaises(SimulatedCrash):
            await service.complete("c1", "question", sample_scope(), sample_publication(), request_id="r4")
        self.assertEqual(service.repository.count_request("r4"), 0)
        self.assertEqual(service.repository.count_messages_for_request("r4"), 0)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_query_service tests.test_answer_engine tests.test_conversation_service -v
```

Expected: FAIL because asynchronous engine and idempotent persistence do not exist.

- [ ] **Step 3: Implement explicit protocols and adapters**

```python
class AnswerEngine(Protocol):
    async def complete(self, request: AnswerRequest, scope: QueryAccessScope, publication: PinnedPublication) -> QueryOutcome: ...
    async def stream(self, request: AnswerRequest, scope: QueryAccessScope, publication: PinnedPublication) -> AsyncIterator[AnswerEvent]: ...
```

`20260715_03_answer_requests.py` has `down_revision = "20260715_02"`. The route dependency pins once with `PublicationService.pin("default")`; `QueryService.execute(question, scope, publication)` receives that object and never pins again. It classifies rules-first and wraps structured/document branches with independent timeouts so one exception becomes a typed unavailable result without cancelling the other. `LegacyAnswerEngine` wraps `ReadOnlyKnowledgeAgent.run()` with `asyncio.to_thread`; at this phase boundary, `OfflineAnswerEngine` calls the new service and returns its deterministic facts/evidence outcome without generation. Task 8 later modifies this adapter to invoke `LocalGenerationClient` when `publication.generation` is configured. Repository protocols add `reserve_answer_request`, `heartbeat_answer_request`, `finalize_answer_request_with_exchange`, and `cancel_answer_request` backed by a unique `answer_requests.request_id` row. The migration stores `reserved|streaming|completed|degraded|cancelled|failed`, owner token, lease expiry/heartbeat, request fingerprint, message ID, and timestamps. The fingerprint is conversation ID + content + mode + scope hash + manifest ID; same key with a different fingerprint returns 409, and stale reserved/streaming rows are reclaimable after the lease. One database transaction/CAS writes the user/assistant exchange and terminal state together; a crash-point test proves neither half can remain. `ConversationService` passes the same scope/publication to repository, engine, prompt builder, and audit; concurrent duplicates replay a terminal result or return an explicit in-progress response, and disconnects cancel the reservation without persisting partial text. Routes and `create_app()` inject this service for both legacy and offline modes.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_query_service tests.test_answer_engine tests.test_conversation_service tests.test_agent tests.test_sql_repository -v
Set-Location ..
git add backend/app/answer_engine.py backend/app/query_router.py backend/app/query_service.py backend/app/conversation_service.py backend/app/repository.py backend/app/sql_repository.py backend/app/routes.py backend/app/main.py backend/alembic/versions/20260715_03_answer_requests.py backend/tests/test_query_service.py backend/tests/test_answer_engine.py backend/tests/test_conversation_service.py
git commit -m "feat: add scoped asynchronous answer engine"
```

### Task 7: Add typed degradation and audit records

**Files:**
- Create: `backend/app/degradation.py`
- Create: `backend/tests/test_degradation.py`
- Modify: `backend/app/database.py`
- Modify: `backend/app/sql_repository.py`
- Create: `backend/alembic/versions/20260715_04_query_audit_fields.py`
- Create: `backend/tests/test_shadow_audit_repository.py`

- [ ] **Step 1: Write failing degradation/audit tests**

```python
class DegradationTest(unittest.TestCase):
    def test_model_busy_preserves_facts_and_evidence(self) -> None:
        answer = build_degraded_answer(DegradationReason.MODEL_BUSY, facts=[fact("revenue", 100)], evidence=[evidence("risk")])
        self.assertIn("100", answer.text)
        self.assertEqual(answer.reason, DegradationReason.MODEL_BUSY)

class ShadowAuditRepositoryTest(unittest.TestCase):
    def test_comparison_rows_are_idempotent_and_raw_text_free(self) -> None:
        repository = SqlChatRepository(test_database())
        row = ShadowComparisonRow(
            request_id="r1",
            principal_hash="p" * 64,
            scope_hash="s" * 64,
            manifest_id="m1",
            question_hash="q" * 64,
            legacy_answer_hash="l" * 64,
            offline_answer_hash="o" * 64,
            terminal_status="completed",
            legacy_status="completed",
            offline_status="completed",
            terminal_reason=None,
            offline_timeout=False,
            numeric_disagreement=False,
            permission_leak=False,
            publication_mismatch=False,
            unexpected_error=False,
            recall_at_5=1.0,
            citation_correctness=1.0,
            grounded_claim_rate=1.0,
            queue_feedback_ms=100.0,
            first_token_ms=500.0,
            created_at=fixed_time(),
        )
        repository.append_shadow_comparison(row)
        repository.append_shadow_comparison(row)
        self.assertEqual(repository.get_shadow_comparison("r1"), row)
        self.assertNotIn("secret question", repository.raw_shadow_storage())
        cancelled = replace(
            row,
            request_id="r2",
            legacy_answer_hash=None,
            offline_answer_hash=None,
            terminal_status="cancelled",
            legacy_status="cancelled",
            offline_status="cancelled",
            terminal_reason="client_disconnect",
        )
        repository.append_shadow_comparison(cancelled)
        self.assertEqual(repository.get_shadow_comparison("r2").terminal_reason, "client_disconnect")
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_degradation tests.test_shadow_audit_repository -v
```

Expected: FAIL because typed fallbacks, query-audit fields, and the shadow comparison repository do not exist.

- [ ] **Step 3: Implement failure taxonomy and privacy-safe audit**

`20260715_04_query_audit_fields.py` has `down_revision = "20260715_03"`. Cover ClickHouse timeout, Qdrant outage/no evidence, reranker skipped, model busy/failure, partial mixed result, disk pressure, and publication mismatch. Audit `query_kind`, manifest ID, scope hash, plan hash, candidate count, degradation reasons, queue wait, first-token time, and cache hits; never store raw scopes, prompts, or sensitive evidence. The migration is additive and leaves legacy rows readable. It also creates `shadow_comparisons`, keyed by `request_id`, with principal/scope/question hashes; optional legacy/offline answer hashes; manifest ID; typed `terminal_status`, `legacy_status`, `offline_status`, and `terminal_reason`; `offline_timeout`, numeric-disagreement, permission-leak, publication-mismatch, and unexpected-error flags; retrieval/grounding/citation metrics; queue/first-token timings; and created-at. Status values are `completed|degraded|cancelled|failed`; reason values come from the typed degradation/error taxonomy, not exception text. Cancelled/failed rows may have null answer hashes and metrics. The table contains no prompt, answer, evidence, raw permission list, stack trace, or exception message. Extend the existing `SqlChatRepository`—do not introduce a parallel `SqlRepository` class—with idempotent `append_shadow_comparison()` and typed `get_shadow_comparison()` methods used by Task 12 and the acceptance tool.

```python
ShadowStatus = Literal["completed", "degraded", "cancelled", "failed"]

@dataclass(frozen=True, slots=True)
class ShadowComparisonRow:
    request_id: str
    principal_hash: str
    scope_hash: str
    manifest_id: str
    question_hash: str
    legacy_answer_hash: str | None
    offline_answer_hash: str | None
    terminal_status: ShadowStatus
    legacy_status: ShadowStatus
    offline_status: ShadowStatus
    terminal_reason: str | None
    offline_timeout: bool
    numeric_disagreement: bool
    permission_leak: bool
    publication_mismatch: bool
    unexpected_error: bool
    recall_at_5: float | None
    citation_correctness: float | None
    grounded_claim_rate: float | None
    queue_feedback_ms: float | None
    first_token_ms: float | None
    created_at: datetime
```

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_degradation tests.test_shadow_audit_repository tests.test_sql_repository -v
Set-Location ..
git add backend/app/degradation.py backend/app/database.py backend/app/sql_repository.py backend/alembic/versions/20260715_04_query_audit_fields.py backend/tests/test_degradation.py backend/tests/test_shadow_audit_repository.py
git commit -m "feat: audit deterministic query degradation"
```

### Task 8: Add bounded local llama.cpp generation

**Files:**
- Create: `backend/app/local_generation.py`
- Create: `backend/tests/test_local_generation.py`
- Modify: `backend/app/answer_engine.py`
- Modify: `backend/app/query_service.py`
- Modify: `backend/app/conversation_service.py`

- [ ] **Step 1: Write failing slot/private-endpoint/stream tests**

```python
class LocalGenerationTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_public_endpoint_and_limits_slots(self) -> None:
        with self.assertRaises(ValueError):
            LocalGenerationClient("https://public.example/v1", slots=2)
        client = LocalGenerationClient("http://llama:8080/v1", slots=1, transport=FakeStreamingTransport(["a"]))
        async with client.slot():
            with self.assertRaises(GenerationBusy):
                async with client.slot(wait_timeout=0):
                    pass

    async def test_offline_answer_engine_uses_pinned_generation_artifact_after_retrieval(self) -> None:
        engine = build_offline_answer_engine(generation=RecordingGenerationClient())
        outcome = await engine.complete(sample_request(), sample_scope(), sample_publication_with_generation())
        self.assertTrue(outcome.generated)
        self.assertEqual(engine.generation_client.seen_artifact, sample_publication_with_generation().generation)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_local_generation tests.test_answer_engine tests.test_query_service tests.test_conversation_service -v
```

Expected: FAIL because slot leasing and streaming client do not exist.

- [ ] **Step 3: Implement bounded streaming**

Validate private/loopback URL and the `publication.generation` artifact name/version/SHA/protocol, use an async semaphore plus Redis lease keys, cap context/output at the manifest values (baseline 2,048/256), set temperature 0.1, and expire crashed leases. Parse llama.cpp SSE deltas and record queue/first-token latency. `OfflineAnswerEngine` calls this client only after retrieval and passes an authorized, redacted context; it never passes raw scopes or unrestricted SQL. Busy, timeout, or failure returns a typed degradation while preserving facts/evidence.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_local_generation tests.test_llm_provider tests.test_answer_engine tests.test_query_service tests.test_conversation_service -v
Set-Location ..
git add backend/app/local_generation.py backend/app/answer_engine.py backend/app/query_service.py backend/app/conversation_service.py backend/tests/test_local_generation.py backend/tests/test_answer_engine.py backend/tests/test_query_service.py backend/tests/test_conversation_service.py
git commit -m "feat: stream bounded local generation"
```

### Task 9: Expose exactly-once SSE events

**Files:**
- Modify: `backend/app/routes.py`
- Modify: `backend/app/schemas.py`
- Create: `backend/tests/test_streaming_api.py`

- [ ] **Step 1: Write failing event, disconnect, and idempotency tests**

```python
class StreamingApiTest(unittest.TestCase):
    def test_emits_ordered_events_and_persists_once(self) -> None:
        app, repository = build_streaming_test_app()
        client = TestClient(app)
        with client.stream(
            "POST",
            "/api/conversations/c1/messages/stream",
            headers={"X-Identity": "u1", "Idempotency-Key": "r1"},
            json={"content": "question", "mode": "deep"},
        ) as response:
            body = "".join(response.iter_text())
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertLess(body.index("event: accepted"), body.index("event: retrieval"))
        self.assertLess(body.index("event: retrieval"), body.index("event: completed"))
        self.assertEqual(repository.count_request("r1"), 1)

    def test_accepted_event_flushes_within_two_seconds(self) -> None:
        app, _ = build_streaming_test_app(generation_blocked=True)
        started = time.monotonic()
        with TestClient(app).stream(
            "POST",
            "/api/conversations/c1/messages/stream",
            headers={"X-Identity": "u1", "Idempotency-Key": "r2"},
            json={"content": "question", "mode": "deep"},
        ) as response:
            first = next(response.iter_text())
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("event: accepted", first)

    def test_disconnect_releases_generation_slot_without_persisting_partial_answer(self) -> None:
        app, repository, engine = build_disconnect_test_app()
        with TestClient(app).stream(
            "POST",
            "/api/conversations/c1/messages/stream",
            headers={"X-Identity": "u1", "Idempotency-Key": "r3"},
            json={"content": "question", "mode": "deep"},
        ) as response:
            next(response.iter_text())
        self.assertTrue(engine.cancelled)
        self.assertEqual(repository.count_request("r3"), 0)

    def test_terminal_replay_and_conflicting_reuse(self) -> None:
        app, repository = build_streaming_test_app()
        client = TestClient(app)
        headers = {"X-Identity": "u1", "Idempotency-Key": "r4"}
        first = client.post("/api/conversations/c1/messages", headers=headers, json={"content": "q", "mode": "deep"})
        replay = client.post("/api/conversations/c1/messages", headers=headers, json={"content": "q", "mode": "deep"})
        conflict = client.post("/api/conversations/c1/messages", headers=headers, json={"content": "different", "mode": "deep"})
        self.assertEqual(first.json()["messageId"], replay.json()["messageId"])
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(repository.count_request("r4"), 1)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_streaming_api -v
```

Expected: FAIL because the streaming route and exactly-once persistence do not exist.

- [ ] **Step 3: Implement typed SSE and cleanup**

Events are `accepted`, `retrieval`, `queued`, `delta`, `degraded`, `completed`, and `error`; every payload includes request ID and safe manifest ID. Send `accepted` within two seconds, flush UTF-8-safe chunks, emit heartbeat comments, set `Cache-Control: no-cache` and `X-Accel-Buffering: no`, and cancel/release model slots on client disconnect. The route calls only `ConversationService.finalize_answer_request_with_exchange()` for completed/degraded terminal events; that method atomically writes both messages and the terminal state. Repeated idempotency keys replay the stored message ID, concurrent duplicates cannot reserve a second owner, and cancelled requests do not persist partial text. Keep the existing non-streaming endpoint backed by the same engine.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_streaming_api tests.test_api_contract -v
Set-Location ..
git add backend/app/routes.py backend/app/schemas.py backend/tests/test_streaming_api.py
git commit -m "feat: expose exactly-once streamed answers"
```

### Task 10: Consume SSE safely in the user frontend

**Files:**
- Modify: `frontend/src/services/api.ts`
- Create: `frontend/src/services/streaming.spec.ts`
- Modify: `frontend/src/composables/useChat.ts`
- Modify: `frontend/src/composables/useChat.spec.ts`
- Modify: `frontend/src/types/chat.ts`
- Modify: `frontend/src/components/chat/ChatTranscript.vue`

- [ ] **Step 1: Write a failing UTF-8/chunk-boundary parser test**

```typescript
it('keeps multibyte text across Uint8Array boundaries', async () => {
  const bytes = new TextEncoder().encode(
    'event: delta\\ndata: {"text":"答案"}\\n\\nevent: completed\\ndata: {}\\n\\n',
  )
  const chineseStart = bytes.findIndex(value => value >= 0xe0)
  const events = parseSseStream([
    bytes.slice(0, chineseStart + 1),
    bytes.slice(chineseStart + 1),
  ])
  expect(events.map((event) => event.type)).toEqual(['delta', 'completed'])
  expect(events[0].data.text).toBe('答案')
})
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location frontend
npm.cmd run test:run -- src/services/streaming.spec.ts
```

Expected: FAIL because the streaming parser/state machine does not exist.

- [ ] **Step 3: Implement fetch streaming and fallback**

Use `fetch()` with `ReadableStream` and `TextDecoder.decode(chunk, {stream: true})`; buffer partial lines, dispatch typed events, append deltas, show queued/degraded state, abort on unmount, and fall back to the existing POST endpoint on 404/feature flag off. Never use EventSource for the POST JSON request.

- [ ] **Step 4: Run tests/build and commit**

```powershell
npm.cmd run test:run
npm.cmd run build
Set-Location ..
git add frontend/src/services/api.ts frontend/src/services/streaming.spec.ts frontend/src/composables/useChat.ts frontend/src/composables/useChat.spec.ts frontend/src/types/chat.ts frontend/src/components/chat/ChatTranscript.vue
git commit -m "feat: consume offline answer streams"
```

### Task 11: Extend retrieval/answer evaluation

**Files:**
- Modify: `backend/app/evaluation.py`
- Modify: `backend/app/evaluation_batches.py`
- Create: `backend/tests/test_query_evaluation.py`

- [ ] **Step 1: Write failing metric tests**

```python
class QueryEvaluationTest(unittest.TestCase):
    def test_reports_recall_mrr_grounding_and_retrieval_variant(self) -> None:
        result = evaluate_query_outcome(
            expected_source_ids=["s2"],
            retrieved_source_ids=["s1", "s2"],
            answer_claims=[claim(["s2"], grounded=True)],
            retrieval_variant="reranked",
        )
        self.assertEqual(result.source_recall, 1.0)
        self.assertEqual(result.mrr, 0.5)
        self.assertEqual(result.grounded_claim_rate, 1.0)
        self.assertEqual(result.retrieval_variant, "reranked")
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_query_evaluation -v
```

Expected: FAIL because version-aware query metrics do not exist.

- [ ] **Step 3: Implement evaluation fields**

Store query kind, manifest ID, dense/sparse/fused/reranked ranks, Recall@K, MRR, no-answer accuracy, citation correctness, structured-fact correctness, grounding, degradation, queue wait, first-token latency, and cache hit/miss. Preserve existing API fields.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_query_evaluation tests.test_quality_evaluation tests.test_evaluation_batches -v
Set-Location ..
git add backend/app/evaluation.py backend/app/evaluation_batches.py backend/tests/test_query_evaluation.py
git commit -m "feat: evaluate hybrid query quality"
```

### Task 12: Add real shadow execution and document flags

**Files:**
- Create: `backend/app/shadow_answer_engine.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes.py`
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `README.md`
- Create: `docs/offline-query-runbook.md`
- Create: `backend/tests/test_shadow_answer_engine.py`
- Create: `backend/tests/test_query_engine_flags.py`

- [ ] **Step 1: Write flag, shadow, and protected audit-route tests**

```python
class QueryEngineFlagTest(unittest.TestCase):
    def test_default_and_cohort_selection(self) -> None:
        self.assertEqual(resolve_query_engine({}, principal="u1", department="d1"), "legacy")
        self.assertEqual(resolve_query_engine({"QUERY_ENGINE": "offline", "OFFLINE_QUERY_PRINCIPALS": "u1"}, "u1", "d9"), "offline")
        self.assertEqual(resolve_query_engine({"QUERY_ENGINE": "offline", "OFFLINE_QUERY_ALL_USERS": "true"}, "u9", "d9"), "offline")
        self.assertEqual(resolve_query_engine({"QUERY_ENGINE": "offline"}, "u9", "d9"), "legacy")
        with self.assertRaises(ValueError):
            resolve_query_engine({"QUERY_ENGINE": "unknown"}, "u1", "d1")

class ShadowAnswerEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_legacy_and_runs_offline_with_same_scope_and_manifest(self) -> None:
        legacy = RecordingAnswerEngine(result=legacy_outcome())
        offline = RecordingAnswerEngine(error=RuntimeError("offline failed"))
        audit = RecordingShadowAudit()
        result = await ShadowAnswerEngine(legacy, offline, audit).complete(
            sample_request(), sample_scope(), sample_publication()
        )
        self.assertEqual(result, legacy_outcome())
        self.assertEqual(legacy.seen_scope, offline.seen_scope)
        self.assertEqual(legacy.seen_publication, offline.seen_publication)
        self.assertEqual(len(audit.rows), 1)
        row = audit.rows[0]
        self.assertEqual(row.terminal_status, "degraded")
        self.assertEqual(row.legacy_status, "completed")
        self.assertEqual(row.offline_status, "failed")
        self.assertEqual(row.terminal_reason, "offline_runtime_error")
        self.assertTrue(row.unexpected_error)
        self.assertIsNotNone(row.legacy_answer_hash)
        self.assertIsNone(row.offline_answer_hash)

    async def test_stream_forwards_only_legacy_events_and_consumes_offline_for_audit(self) -> None:
        legacy = RecordingAnswerEngine(events=[accepted(), delta("legacy"), completed()])
        offline = RecordingAnswerEngine(events=[accepted(), delta("offline"), completed()])
        audit = RecordingShadowAudit()
        engine = ShadowAnswerEngine(legacy, offline, audit)
        events = [event async for event in engine.stream(sample_request(), sample_scope(), sample_publication())]
        self.assertEqual([event.text for event in events if event.type == "delta"], ["legacy"])
        self.assertEqual(offline.consumed_event_types, ["accepted", "delta", "completed"])
        self.assertEqual(len(audit.rows), 1)

    async def test_stream_disconnect_cancels_both_branches(self) -> None:
        legacy = BlockingRecordingAnswerEngine(first_event=accepted())
        offline = BlockingRecordingAnswerEngine(first_event=accepted())
        audit = RecordingShadowAudit()
        stream = ShadowAnswerEngine(legacy, offline, audit).stream(
            sample_request(), sample_scope(), sample_publication()
        )
        await anext(stream)
        await stream.aclose()
        self.assertTrue(legacy.cancelled)
        self.assertTrue(offline.cancelled)
        self.assertEqual(len(audit.rows), 1)
        row = audit.rows[0]
        self.assertEqual(row.terminal_status, "cancelled")
        self.assertEqual(row.legacy_status, "cancelled")
        self.assertEqual(row.offline_status, "cancelled")
        self.assertEqual(row.terminal_reason, "client_disconnect")
        self.assertIsNone(row.legacy_answer_hash)
        self.assertIsNone(row.offline_answer_hash)

    def test_trusted_shadow_audit_route_returns_only_hashed_metrics(self) -> None:
        app = build_shadow_test_app(
            shadow_reader_principal="acceptance-runner",
            principal_roles={"acceptance-runner": {"acceptance-reader"}},
            role_permissions={"acceptance-reader": {"shadow_audit:read"}},
        )
        response = TestClient(app).get(
            "/api/admin/shadow-comparisons/r1",
            headers={"X-Identity": "acceptance-runner"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["requestId"], "r1")
        self.assertEqual(response.json()["terminalStatus"], "completed")
        self.assertEqual(response.json()["legacyStatus"], "completed")
        self.assertEqual(response.json()["offlineStatus"], "completed")
        self.assertIsNone(response.json()["terminalReason"])
        self.assertFalse(response.json()["offlineTimeout"])
        self.assertFalse(response.json()["publicationMismatch"])
        self.assertFalse(response.json()["unexpectedError"])
        self.assertNotIn("question", response.json())
        self.assertNotIn("answer", response.json())
        self.assertEqual(
            TestClient(app).get("/api/admin/shadow-comparisons/r1").status_code,
            401,
        )
        self.assertEqual(
            TestClient(build_shadow_test_app(principal_roles={"ordinary-user": {"ordinary"}})).get(
                "/api/admin/shadow-comparisons/r1",
                headers={"X-Identity": "ordinary-user"},
            ).status_code,
            403,
        )
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_query_engine_flags tests.test_shadow_answer_engine -v
Set-Location ..
```

Expected: FAIL because cohort-aware flag resolution, SQL-backed shadow execution, and the protected audit route do not exist.

- [ ] **Step 3: Implement shadow engine, runbook, and flags**

`ShadowAnswerEngine.complete()` runs legacy and offline engines concurrently with the same `QueryAccessScope`, canonical publication, request fingerprint, and timeout policy; it returns the legacy answer while atomically appending one hashed `shadow_comparisons` row through the Task 7 repository. `stream()` starts both engine streams with the same inputs, forwards only legacy events in their original order, consumes the offline events into hashed comparison metrics, and writes one audit row after both reach a terminal state. Offline errors/timeouts become audit metrics and never replace or delay an available legacy event. A legacy terminal error is forwarded and cancels the offline branch. Client disconnect cancels both iterators, releases both engines' model slots, and records only a typed cancelled comparison without partial text. Neither branch persists messages directly; the outer `ConversationService` persists the one user-visible legacy terminal result exactly once. Add tests proving both engines receive identical scope/manifest, offline failure cannot replace the legacy response, streamed offline deltas never reach the client, and disconnect cancels both branches. `main.py` injects the SQL-backed shadow audit writer. `routes.py` adds `GET /api/admin/shadow-comparisons/{request_id}` on the loopback/internal admin listener and gates it with the Task 1 `PrincipalResolver.require_permission(request, "shadow_audit:read")`; no client-supplied scope or permission header is accepted. The response explicitly includes request/manifest IDs, allowed hashes, `terminalStatus`, `legacyStatus`, `offlineStatus`, `terminalReason`, `offlineTimeout`, `numericDisagreement`, `permissionLeak`, `publicationMismatch`, `unexpectedError`, retrieval/grounding/citation metrics, and queue/first-token timings; it never includes raw question/answer/evidence. Support `QUERY_ENGINE=legacy|shadow|offline`, `OFFLINE_QUERY_PRINCIPALS`, `OFFLINE_QUERY_DEPARTMENTS`, and explicit `OFFLINE_QUERY_ALL_USERS`; offline mode with empty cohorts and all-users false remains legacy. The runbook documents scope resolution, manifest pinning, cache namespaces, reranker skip behavior, queue/degradation events, the audit-reader credential, rollback, and the fact that shadow never changes the user-visible legacy answer.

- [ ] **Step 4: Run full Phase 3 verification and commit**

```powershell
Set-Location backend
py -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..\frontend
npm.cmd run test:run
npm.cmd run build
Set-Location ..
docker compose --env-file deploy/offline/.env -f deploy/offline/compose.yaml --profile generation up -d --wait llama
git diff --check
git add .env.example backend/.env.example README.md docs/offline-query-runbook.md backend/app/shadow_answer_engine.py backend/app/main.py backend/app/routes.py backend/tests/test_shadow_answer_engine.py backend/tests/test_query_engine_flags.py
git commit -m "docs: add offline query rollout runbook"
```

## Phase 3 completion gate

- Every query/answer route resolves a PostgreSQL-backed non-empty query scope; the isolated shadow-audit admin route instead resolves the exact PostgreSQL capability and has no data-query scope. Forbidden records cannot reach either engine or the model prompt.
- Every request pins one canonical publication containing all Embedding, optional reranker/generation, sparse, redaction, permission, and physical-resource metadata.
- Structured SQL is catalog-driven, SELECT-only, permission-filtered, partition-filtered, bounded, and read-only.
- Dense and sparse retrieval use the shared private Embedding client, identical filters, explicit RRF, bounded optional reranking, and namespaced caches.
- Mixed answers preserve ClickHouse numeric authority and source provenance.
- Local generation is private, slot-bounded, streamed, cancellable, and optional.
- Busy, timeout, outage, and no-evidence cases degrade deterministically.
- SSE and non-streaming paths persist each request exactly once and legacy behavior remains available.
