# Unified Knowledge Routing and Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize greeting, Excel structured queries, Word factual questions, and ordinary document RAG behind one auditable router that enforces terminal no-fallback behavior and can be rolled out safely on Ubuntu/Supervisor.

**Architecture:** A `KnowledgeAnswerRouter` owns route ordering and returns one `AgentRunResult` carrying typed route metadata. Excel and Word services are authoritative: once they recognize a question, clarification, not-found, timeout, or validation failure is terminal. Only ordinary document and summary/compare routes invoke hybrid retrieval; the Agent stops performing the current second full-document `inspect_document` pass, so all document evidence comes from the configured Dense/Sparse/RRF/BGE Reranker pipeline and bounded adjacency assembly.

**Tech Stack:** Python 3.12, dataclasses/`StrEnum`, LangGraph Agent, Qdrant hybrid retrieval, llama.cpp BGE Reranker, SQLAlchemy/Alembic, Loguru, FastAPI, Python `unittest`, uv, Ubuntu, Supervisor.

## Global Constraints

- Route order is fixed: `greeting -> Excel structured -> Word factual -> summary/compare document RAG -> ordinary document RAG`.
- A recognized Excel or Word route is terminal even when it returns clarification, not found, unavailable, conflict, or validation failure.
- Excel routes never invoke Qdrant, Word chunks, `inspect_document`, BGE Reranker, or the LLM.
- Word factual routes never invoke Embedding, Qdrant, adjacency expansion, `inspect_document`, BGE Reranker, or the LLM.
- Document routes continue to use the production BGE Dense/Sparse, RRF, llama.cpp `bge-reranker-v2-m3-Q4.gguf`, and LLM path.
- Remove the Agent's `DEFAULT_EMBEDDING_PROVIDER` full-document inspection path; follow-up evidence must use another routed hybrid search query, not local default embeddings.
- Bounded adjacency is allowed only for document routes. Exact routes do not create a `RetrievalRequest`.
- Every persisted Agent run records route type, dataset/entity, target fields, candidate sources, degradation reason, and validation outcome without storing secrets or full document text.
- `WORD_FACTUAL_QA_ENABLED=false` and `UNIFIED_KNOWLEDGE_ROUTING_ENABLED=false` are safe defaults. Enable factual routing only after Word fact reindex verification.
- Production deployment remains Ubuntu without Docker and uses Supervisor programs `dcagent-api` and `dcagent-structured-worker`.

---

## File map

- `backend/app/knowledge_route_models.py`: route enum and bounded metadata contract.
- `backend/app/knowledge_router.py`: centralized orchestration and document route classification.
- `backend/app/agent.py`: route-aware Agent result fields and removal of local full-document inspection.
- `backend/app/repository.py` and `backend/app/sql_repository.py`: delegate message answering to the unified router.
- `backend/app/structured_answer.py`: mark Excel single/multi/clarification/unavailable outcomes.
- `backend/app/word_fact_answer.py`: mark Word factual/clarification/not-found/conflict outcomes.
- `backend/app/retrieval_models.py`: explicit evidence expansion policy.
- `backend/app/hybrid_retriever.py`: honor bounded/no-adjacency policy.
- `backend/app/database.py`: route audit columns on `agent_runs`.
- `backend/alembic/versions/20260811_08_knowledge_route_audit.py`: route audit migration after Word facts.
- `backend/app/sql_repository.py`: route audit persistence.
- `backend/app/schemas.py`: route audit API fields.
- `backend/app/offline_settings.py` and `backend/app/main.py`: feature flags and service construction.
- `backend/tests/test_knowledge_router.py`: route order and terminal behavior.
- `backend/tests/test_agent.py`: routed repeated-search behavior without `inspect_document`.
- `backend/tests/test_hybrid_retriever.py`: adjacency policy.
- `backend/tests/test_structured_answer.py` and `backend/tests/test_word_fact_answer.py`: route metadata.
- `backend/tests/test_sql_repository.py` and `backend/tests/test_api_contract.py`: audit persistence/API.
- `backend/tests/test_alembic_baseline.py`, `backend/tests/test_migration_entrypoint.py`, `backend/tests/test_lazy_startup.py`: revision `20260811_08`.
- `backend/tests/test_rag_acceptance.py` and `backend/tests/test_quality_evaluation.py`: cross-source regression.
- `.env.example`, `backend/.env.example`, `deploy/offline/.env.example`, `README.md`, `deploy/offline/README.md`, `deploy/ubuntu/KNOWLEDGE_ROUTING_ROLLOUT.md`: rollout and rollback.

### Task 1: Define route types and centralized orchestration

**Files:**
- Create: `backend/app/knowledge_route_models.py`
- Create: `backend/app/knowledge_router.py`
- Modify: `backend/app/agent.py`
- Create: `backend/tests/test_knowledge_router.py`

**Interfaces:**
- Consumes: greeting Agent method, optional `StructuredAnswerService`, optional `WordFactAnswerService`, and normal `ReadOnlyKnowledgeAgent.run()`.
- Produces: `KnowledgeRouteType`, `KnowledgeRouteMetadata`, `LegacyKnowledgeAnswerRouter`, and `KnowledgeAnswerRouter.answer() -> AgentRunResult`.

- [ ] **Step 1: Write failing route-order and terminal tests**

```python
def test_route_order_is_greeting_then_excel_then_word_then_document(self) -> None:
    router = KnowledgeAnswerRouter(
        agent=RecordingAgent(self.calls),
        structured_service=RecordingStructured(self.calls),
        word_fact_service=RecordingWordFacts(self.calls),
    )
    result = router.answer(
        conversation_id="conv-1",
        content="地区为华东的销售额汇总",
        mode="deep",
        previous_messages=[],
    )
    self.assertEqual(self.calls, ["greeting", "excel"])
    self.assertEqual(result.route_type, KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE)

def test_excel_unavailable_is_terminal_and_never_calls_word_or_agent(self) -> None:
    router = self.router(structured_result=excel_unavailable_run())
    result = router.answer("conv-1", "地区为华东的销售额汇总", "deep", [])
    self.assertEqual(result.route_metadata.degradation_reason, "clickhouse_unavailable")
    self.assertEqual(self.calls, ["greeting", "excel"])

def test_word_not_found_is_terminal_and_never_calls_document_agent(self) -> None:
    router = self.router(word_result=word_not_found_run())
    result = router.answer("conv-1", "张三几岁", "deep", [])
    self.assertEqual(result.route_type, KnowledgeRouteType.WORD_FACTUAL)
    self.assertEqual(self.calls, ["greeting", "excel", "word"])

def test_open_introduction_routes_to_summary_compare(self) -> None:
    router = self.router()
    result = router.answer("conv-1", "介绍张三", "deep", [])
    self.assertEqual(result.route_type, KnowledgeRouteType.SUMMARY_COMPARE)
    self.assertEqual(self.calls, ["greeting", "excel", "word", "document"])

def test_policy_question_routes_to_document_qa(self) -> None:
    router = self.router()
    result = router.answer("conv-1", "报销流程是什么", "quick", [])
    self.assertEqual(result.route_type, KnowledgeRouteType.DOCUMENT_QA)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_knowledge_router -v
Pop-Location
```

Expected: FAIL because route contracts and the orchestrator do not exist.

- [ ] **Step 3: Implement typed routing**

Define:

```python
class KnowledgeRouteType(StrEnum):
    GREETING = "greeting"
    EXCEL_FILTERED_AGGREGATE = "excel_filtered_aggregate"
    EXCEL_MULTI_AGGREGATE = "excel_multi_aggregate"
    WORD_FACTUAL = "word_factual"
    DOCUMENT_QA = "document_qa"
    SUMMARY_COMPARE = "summary_compare"
    CLARIFICATION = "clarification"

@dataclass(frozen=True, slots=True)
class KnowledgeRouteMetadata:
    dataset_id: str | None = None
    entity: str | None = None
    target_fields: tuple[str, ...] = ()
    candidate_source_ids: tuple[str, ...] = ()
    origin_route: KnowledgeRouteType | None = None
    degradation_reason: str | None = None
    validation_passed: bool | None = None
    adjacency_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "entity": self.entity,
            "target_fields": list(self.target_fields),
            "candidate_source_ids": list(self.candidate_source_ids),
            "origin_route": None if self.origin_route is None else self.origin_route.value,
            "degradation_reason": self.degradation_reason,
            "validation_passed": self.validation_passed,
            "adjacency_allowed": self.adjacency_allowed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> KnowledgeRouteMetadata:
        target_fields = _string_tuple(payload.get("target_fields"), "target_fields")
        candidate_sources = _string_tuple(
            payload.get("candidate_source_ids"), "candidate_source_ids"
        )
        origin_value = payload.get("origin_route")
        origin = None if origin_value is None else KnowledgeRouteType(str(origin_value))
        return cls(
            dataset_id=_optional_string(payload.get("dataset_id")),
            entity=_optional_string(payload.get("entity")),
            target_fields=target_fields,
            candidate_source_ids=candidate_sources,
            origin_route=origin,
            degradation_reason=_optional_string(
                payload.get("degradation_reason")
            ),
            validation_passed=_optional_bool(
                payload.get("validation_passed")
            ),
            adjacency_allowed=bool(payload.get("adjacency_allowed", False)),
        )

def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("route metadata string field is invalid")
    return value

def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("route metadata boolean field is invalid")
    return value

def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"route metadata {field_name} must be a string list")
    return tuple(value)
```

Add defaulted fields to the end of `AgentRunResult` and `AgentRunAudit`:

```python
route_type: KnowledgeRouteType = KnowledgeRouteType.DOCUMENT_QA
route_metadata: KnowledgeRouteMetadata = field(
    default_factory=KnowledgeRouteMetadata
)
```

`to_audit()` copies both values.

Implement `KnowledgeAnswerRouter.answer()` with concrete arguments and no implicit fallthrough:

```python
def answer(
    self,
    conversation_id: str,
    content: str,
    mode: ComposerMode,
    previous_messages: Sequence[ChatMessageModel],
) -> AgentRunResult:
    greeting = self._agent.try_answer_greeting(
        conversation_id=conversation_id,
        content=content,
        mode=mode,
    )
    if greeting is not None:
        return replace(
            greeting,
            route_type=KnowledgeRouteType.GREETING,
            route_metadata=KnowledgeRouteMetadata(validation_passed=True),
        )
    if self._structured_service is not None:
        structured = self._structured_service.try_answer(
            conversation_id, content, mode, previous_messages
        )
        if structured is not None:
            return structured
    if self._word_fact_service is not None:
        factual = self._word_fact_service.try_answer(
            conversation_id, content, mode, previous_messages
        )
        if factual is not None:
            return factual
    document_route = classify_document_route(content, mode)
    return self._agent.run(
        conversation_id=conversation_id,
        content=content,
        mode=mode,
        previous_messages=list(previous_messages),
        route_type=document_route,
    )
```

`classify_document_route()` returns `SUMMARY_COMPARE` when mode is `source` or the normalized question contains `介绍/总结/概括/比较/对比/异同/分别`; otherwise it returns `DOCUMENT_QA`.

Also implement `LegacyKnowledgeAnswerRouter.answer()` with the current greeting → optional structured service → Agent order and without the Word factual service. It returns the same `AgentRunResult` contract, so `OfflineSettings` can switch routers without changing repository persistence.

- [ ] **Step 4: Run tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_knowledge_router tests.test_agent -v
Pop-Location
git add backend/app/knowledge_route_models.py backend/app/knowledge_router.py backend/app/agent.py backend/tests/test_knowledge_router.py
git commit -m "feat: centralize knowledge answer routing"
```

### Task 2: Mark Excel and Word routes and enforce terminal no-fallback behavior

**Files:**
- Modify: `backend/app/structured_answer.py`
- Modify: `backend/app/word_fact_answer.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_structured_answer.py`
- Modify: `backend/tests/test_word_fact_answer.py`
- Modify: `backend/tests/test_knowledge_router.py`

**Interfaces:**
- Consumes: `KnowledgeRouteType`, `KnowledgeRouteMetadata`, and existing optional services.
- Produces: route-tagged service runs and repository delegation to one `KnowledgeAnswerRouter`.

- [ ] **Step 1: Write failing route metadata tests**

```python
def test_excel_multi_result_records_dataset_fields_and_validation(self) -> None:
    result = self.ask_structured("地区为华东的销售额、成本汇总")
    self.assertEqual(result.route_type, KnowledgeRouteType.EXCEL_MULTI_AGGREGATE)
    self.assertEqual(result.route_metadata.dataset_id, "ds-sales")
    self.assertEqual(result.route_metadata.target_fields, ("销售额", "成本"))
    self.assertTrue(result.route_metadata.validation_passed)
    self.assertFalse(result.route_metadata.adjacency_allowed)

def test_word_fact_records_entity_field_and_candidate_sources(self) -> None:
    result = self.ask_word("张三几岁")
    self.assertEqual(result.route_type, KnowledgeRouteType.WORD_FACTUAL)
    self.assertEqual(result.route_metadata.entity, "张三")
    self.assertEqual(result.route_metadata.target_fields, ("年龄",))
    self.assertEqual(result.route_metadata.candidate_source_ids, ("kb-people",))
    self.assertTrue(result.route_metadata.validation_passed)

def test_structured_clarification_records_origin_and_stays_terminal(self) -> None:
    result = self.ask_structured("金额汇总")
    self.assertEqual(result.route_type, KnowledgeRouteType.CLARIFICATION)
    self.assertEqual(
        result.route_metadata.origin_route,
        KnowledgeRouteType.EXCEL_MULTI_AGGREGATE,
    )
    self.assertEqual(self.document_calls, 0)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_structured_answer tests.test_word_fact_answer tests.test_knowledge_router -v
Pop-Location
```

Expected: FAIL because service results do not carry route metadata and repositories still own manual branching.

- [ ] **Step 3: Populate metadata and delegate repository answering**

For successful structured single results, use `EXCEL_FILTERED_AGGREGATE` and record dataset ID, one target field, source ID, and `validation_passed=True`. For multi results, use `EXCEL_MULTI_AGGREGATE` and record all display fields in result order. Structured clarification uses `CLARIFICATION` with `origin_route` set to the intended Excel route. Structured unavailable retains the intended Excel route and records one bounded reason code from:

```python
STRUCTURED_DEGRADATION_REASONS = {
    "catalog unavailable": "catalog_unavailable",
    "structured intent unavailable": "intent_unavailable",
    "active publication unavailable": "publication_unavailable",
    "structured query planning failed": "plan_rejected",
    "structured query unavailable": "clickhouse_unavailable",
}
```

For Word, successful, not-found, and conflicting results use `WORD_FACTUAL`; ambiguity uses `CLARIFICATION` with `origin_route=WORD_FACTUAL`. Record entity, one target field, sorted unique source IDs, and validator status.

Construct one `KnowledgeAnswerRouter` in each repository constructor and replace the greeting/structured/word/Agent branching in `send_message()` with:

```python
agent_result = self._answer_router.answer(
    conversation_id=conversation_id,
    content=clean_content,
    mode=mode,
    previous_messages=previous_messages,
)
```

Keep message and audit persistence exactly once after the router returns. `main.py` passes optional services into repository construction; no route service opens another database or ClickHouse client.

- [ ] **Step 4: Run repository route tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_knowledge_router tests.test_structured_answer tests.test_word_fact_answer tests.test_sql_repository -v
Pop-Location
git add backend/app/structured_answer.py backend/app/word_fact_answer.py backend/app/repository.py backend/app/sql_repository.py backend/app/main.py backend/tests/test_structured_answer.py backend/tests/test_word_fact_answer.py backend/tests/test_knowledge_router.py
git commit -m "feat: enforce terminal knowledge routes"
```

### Task 3: Remove local document inspection and keep all document evidence reranked

**Files:**
- Modify: `backend/app/agent.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/retrieval_models.py`
- Modify: `backend/app/hybrid_retriever.py`
- Modify: `backend/tests/test_agent.py`
- Modify: `backend/tests/test_hybrid_retriever.py`
- Modify: `backend/tests/test_rag_acceptance.py`

**Interfaces:**
- Consumes: production routed search, `RetrievalRequest`, and the current two-query deep mode.
- Produces: `EvidenceExpansionPolicy`, `AgentSearchResult`, no local `DEFAULT_EMBEDDING_PROVIDER` inspection, and document-only adjacency.

- [ ] **Step 1: Write failing evidence-policy tests**

```python
def test_agent_never_calls_inspect_document_after_reranked_search(self) -> None:
    agent = ReadOnlyKnowledgeAgent(
        tools=KnowledgeAgentTools(search_knowledge=self.search),
        llm_provider=self.provider,
    )
    result = agent.run(
        "conv-1",
        "介绍张三",
        "deep",
        [],
        route_type=KnowledgeRouteType.SUMMARY_COMPARE,
    )
    self.assertNotIn("inspect_document", [item.tool_name for item in result.steps])
    self.assertGreaterEqual(self.search_calls, 1)

def test_factual_route_creates_no_retrieval_request(self) -> None:
    self.repository.send_message("conv-1", "张三几岁", "deep")
    self.assertEqual(self.recorded_retrieval_requests, [])

def test_hybrid_retriever_skips_adjacency_when_policy_is_none(self) -> None:
    request = sample_request(
        expansion_policy=EvidenceExpansionPolicy.NONE
    )
    outcome = self.retriever.retrieve(request)
    self.assertEqual(self.gateway.retrieve_point_calls, [])
    self.assertEqual(outcome.stage_ms["adjacency"], 0.0)

def test_document_route_uses_bounded_adjacency_after_reranking(self) -> None:
    request = sample_request(
        expansion_policy=EvidenceExpansionPolicy.BOUNDED_ADJACENCY
    )
    outcome = self.retriever.retrieve(request)
    self.assertGreaterEqual(len(outcome.candidates), 1)
    self.assertEqual(len(self.reranker.calls), 1)

def test_document_route_records_retrieval_degradation_reason(self) -> None:
    result = self.agent.run(
        "conv-1",
        "报销流程是什么",
        "deep",
        [],
        route_type=KnowledgeRouteType.DOCUMENT_QA,
    )
    self.assertEqual(
        result.route_metadata.degradation_reason,
        "reranker_service_error",
    )
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_agent tests.test_hybrid_retriever tests.test_rag_acceptance -v
Pop-Location
```

Expected: FAIL because `inspect_document` is part of the Agent graph and retrieval expansion is implicit.

- [ ] **Step 3: Remove the bypass and add explicit expansion policy**

Define:

```python
class EvidenceExpansionPolicy(StrEnum):
    NONE = "none"
    BOUNDED_ADJACENCY = "bounded_adjacency"
```

Add `expansion_policy: EvidenceExpansionPolicy = EvidenceExpansionPolicy.BOUNDED_ADJACENCY` to `RetrievalRequest`. In `HybridRetriever._retrieve()`:

```python
if request.expansion_policy is EvidenceExpansionPolicy.NONE:
    evidence = tuple(reranked[:evidence_limit])
    stage_ms["adjacency"] = 0.0
else:
    stage_started = self._monotonic()
    evidence = self._expand_adjacency(
        reranked[:evidence_limit],
        scope=request.scope,
        limit=evidence_limit,
        deadline=deadline,
        generation=generation,
    )
    stage_ms["adjacency"] = _elapsed_ms(stage_started, self._monotonic())
```

Change `KnowledgeAgentTools.search_knowledge` to return `AgentSearchResult`, remove `inspect_document`, remove `rank_inspected_chunks()` and its `DEFAULT_EMBEDDING_PROVIDER` imports, and remove the `inspect` node/edge from the Agent graph. The `_search` node consumes `search_result.hits` and appends `search_result.fallback_reason` to bounded state. After search rounds, go directly to `compare` then `answer` when hits exist. Deep/source mode gets additional evidence only through additional `search_knowledge` calls, which pass through `RetrievalRouter` and the configured BGE Reranker.

Replace the search tool's raw list return with:

```python
@dataclass(frozen=True, slots=True)
class AgentSearchResult:
    hits: tuple[KnowledgeSearchHitModel, ...]
    fallback_reason: str | None = None
```

Repository routed searches convert `RoutedRetrievalOutcome` into `AgentSearchResult`, preserve `outcome.fallback_reason`, and set `expansion_policy=EvidenceExpansionPolicy.BOUNDED_ADJACENCY` on every document `RetrievalRequest`. Agent state accumulates bounded fallback reason codes and final document route metadata records the first reason plus sorted candidate source IDs. Exact Excel/Word routes never call the routed search method and therefore never create a request.

- [ ] **Step 4: Run retrieval regressions and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_agent tests.test_hybrid_retriever tests.test_retrieval_router tests.test_rag_acceptance -v
Pop-Location
git add backend/app/agent.py backend/app/repository.py backend/app/sql_repository.py backend/app/retrieval_models.py backend/app/hybrid_retriever.py backend/tests/test_agent.py backend/tests/test_hybrid_retriever.py backend/tests/test_rag_acceptance.py
git commit -m "fix: keep document evidence on the reranked path"
```

### Task 4: Persist and expose route audit metadata

**Files:**
- Modify: `backend/app/database.py`
- Create: `backend/alembic/versions/20260811_08_knowledge_route_audit.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/tests/test_sql_repository.py`
- Modify: `backend/tests/test_api_contract.py`
- Modify: `backend/tests/test_alembic_baseline.py`
- Modify: `backend/tests/test_migration_entrypoint.py`
- Modify: `backend/tests/test_lazy_startup.py`

**Interfaces:**
- Consumes: `AgentRunResult.route_type` and `route_metadata`.
- Produces: non-null `agent_runs.route_type`, non-null `agent_runs.route_metadata`, and API aliases `routeType`/`routeMetadata`.

- [ ] **Step 1: Write failing audit persistence tests**

```python
def test_sql_agent_run_round_trips_route_audit(self) -> None:
    self.repository.send_message("conv-1", "张三几岁", "quick")
    run = self.repository.list_agent_runs(1)[0]
    self.assertEqual(run.route_type, KnowledgeRouteType.WORD_FACTUAL)
    self.assertEqual(run.route_metadata.entity, "张三")
    self.assertEqual(run.route_metadata.target_fields, ("年龄",))
    self.assertTrue(run.route_metadata.validation_passed)

def test_agent_audit_api_exposes_bounded_route_fields(self) -> None:
    response = self.client.get("/api/agent/runs")
    payload = response.json()[0]
    self.assertEqual(payload["routeType"], "word_factual")
    self.assertEqual(payload["routeMetadata"]["entity"], "张三")
    self.assertNotIn("chunkText", payload["routeMetadata"])
    self.assertNotIn("password", str(payload).casefold())

def test_route_audit_migration_backfills_legacy_runs(self) -> None:
    command.upgrade(config, "20260811_07")
    self.insert_legacy_agent_run()
    command.upgrade(config, "20260811_08")
    row = self.read_agent_route_columns()
    self.assertEqual(row.route_type, "document_qa")
    self.assertEqual(row.route_metadata, {})
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_sql_repository tests.test_api_contract tests.test_alembic_baseline -v
Pop-Location
```

Expected: FAIL because route audit columns and serializers do not exist.

- [ ] **Step 3: Add the audit migration and serializers**

Add to `AgentRunRecord`:

```python
route_type: Mapped[str] = mapped_column(
    String(40), nullable=False, default="document_qa"
)
route_metadata: Mapped[dict[str, object]] = mapped_column(
    JSON, nullable=False, default=dict, server_default="{}"
)
```

Migration `20260811_08` uses `down_revision = "20260811_07"`, adds nullable columns, backfills existing rows to `document_qa` and `{}`, then uses `batch_alter_table` to make them non-null. Downgrade drops both columns.

Persist `run.route_type.value` and `run.route_metadata.to_dict()`. Reconstruct the enum and tuple fields with a strict `KnowledgeRouteMetadata.from_dict()` that ignores unknown keys and rejects wrong scalar/list types.

Extend `AgentRunAudit` Pydantic output:

```python
route_type: KnowledgeRouteType = Field(alias="routeType")
route_metadata: dict[str, object] = Field(alias="routeMetadata")
```

Update expected head/revision constants to `20260811_08`.

- [ ] **Step 4: Run audit tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_sql_repository tests.test_api_contract tests.test_alembic_baseline tests.test_migration_entrypoint tests.test_lazy_startup -v
Pop-Location
git add backend/app/database.py backend/alembic/versions/20260811_08_knowledge_route_audit.py backend/app/sql_repository.py backend/app/schemas.py backend/tests/test_sql_repository.py backend/tests/test_api_contract.py backend/tests/test_alembic_baseline.py backend/tests/test_migration_entrypoint.py backend/tests/test_lazy_startup.py
git commit -m "feat: audit knowledge answer routes"
```

### Task 5: Add feature flags, cross-source acceptance, and Ubuntu rollout

**Files:**
- Modify: `backend/app/offline_settings.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_offline_settings.py`
- Modify: `backend/tests/test_rag_acceptance.py`
- Modify: `backend/tests/test_quality_evaluation.py`
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `deploy/offline/.env.example`
- Modify: `README.md`
- Modify: `deploy/offline/README.md`
- Create: `deploy/ubuntu/KNOWLEDGE_ROUTING_ROLLOUT.md`

**Interfaces:**
- Consumes: existing `STRUCTURED_QUERY_ENABLED`, new Word fact index, Supervisor programs, and production BGE Reranker topology.
- Produces: feature-gated rollout, smoke tests, rollback, and acceptance evidence.

- [ ] **Step 1: Write failing flags and cross-source acceptance tests**

```python
def test_routing_flags_default_off_and_parse_explicit_true(self) -> None:
    defaults = OfflineSettings.from_environ({})
    self.assertFalse(defaults.unified_knowledge_routing_enabled)
    self.assertFalse(defaults.word_factual_qa_enabled)
    enabled = OfflineSettings.from_environ(
        {
            "UNIFIED_KNOWLEDGE_ROUTING_ENABLED": "true",
            "WORD_FACTUAL_QA_ENABLED": "true",
        }
    )
    self.assertTrue(enabled.unified_knowledge_routing_enabled)
    self.assertTrue(enabled.word_factual_qa_enabled)

def test_excel_question_never_returns_seeded_word_answer(self) -> None:
    self.seed_word_source("irrelevant.docx", "报销制度与本问题无关。")
    response = self.ask("地区为华东的销售额、成本汇总")
    self.assertEqual(response.route_type, KnowledgeRouteType.EXCEL_MULTI_AGGREGATE)
    self.assertEqual(self.rag_search_calls, 0)
    self.assertNotIn("报销制度", response.reply.paragraphs[0].text)

def test_age_question_returns_only_age_even_when_chunk_has_all_fields(self) -> None:
    self.seed_word_record("姓名：张三，年龄：28岁，性别：女，职务：工程师")
    response = self.ask("张三几岁")
    text = response.reply.paragraphs[0].text
    self.assertEqual(text, "张三的年龄是28岁。")
    self.assertNotIn("女", text)
    self.assertNotIn("工程师", text)
    self.assertEqual(self.reranker_calls, 0)

def test_open_introduction_uses_production_reranker_and_llm(self) -> None:
    response = self.ask("介绍张三")
    self.assertEqual(response.route_type, KnowledgeRouteType.SUMMARY_COMPARE)
    self.assertGreater(self.reranker_calls, 0)
    self.assertEqual(self.llm_calls, 1)

def test_exact_routes_still_work_when_llm_is_unavailable(self) -> None:
    self.llm.fail_with = RuntimeError("llm unavailable")
    excel = self.ask("地区为华东的销售额汇总")
    word = self.ask("张三几岁")
    self.assertEqual(
        excel.route_type,
        KnowledgeRouteType.EXCEL_FILTERED_AGGREGATE,
    )
    self.assertEqual(word.route_type, KnowledgeRouteType.WORD_FACTUAL)
```

- [ ] **Step 2: Run acceptance tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest tests.test_offline_settings tests.test_rag_acceptance tests.test_quality_evaluation -v
Pop-Location
```

Expected: FAIL because flags, cross-route isolation, and rollout documentation do not exist.

- [ ] **Step 3: Implement flags and the exact Ubuntu/Supervisor runbook**

Add booleans `unified_knowledge_routing_enabled` and `word_factual_qa_enabled` to `OfflineSettings` using `parse_bool(environ.get("UNIFIED_KNOWLEDGE_ROUTING_ENABLED"), default=False)` and `parse_bool(environ.get("WORD_FACTUAL_QA_ENABLED"), default=False)`. When unified routing is disabled, construct a `LegacyKnowledgeAnswerRouter` that preserves the current greeting → structured → Agent order for emergency rollback. When unified routing is enabled but Word factual QA is disabled, construct `KnowledgeAnswerRouter` with `word_fact_service=None`. Refuse startup with `WORD_FACTUAL_QA_ENABLED=true` and `UNIFIED_KNOWLEDGE_ROUTING_ENABLED=false`.

Add to all environment examples:

```dotenv
UNIFIED_KNOWLEDGE_ROUTING_ENABLED=false
WORD_FACTUAL_QA_ENABLED=false
STRUCTURED_IMPLICIT_SUMMARY_MAX_METRICS=12
```

Create `deploy/ubuntu/KNOWLEDGE_ROUTING_ROLLOUT.md` with this exact deployment sequence for `/opt/DCAgent`:

```bash
cd /opt/DCAgent
git pull --ff-only origin main
uv sync --project backend --frozen --offline --no-install-project --no-dev --group offline --no-index --find-links artifacts/wheels
cd /opt/DCAgent/backend
./.venv/bin/python -m app.migration_entrypoint
cd /opt/DCAgent
sudo supervisorctl restart dcagent-api
sudo supervisorctl restart dcagent-structured-worker
sudo supervisorctl status dcagent-api dcagent-structured-worker
```

Then keep `WORD_FACTUAL_QA_ENABLED=false`, reindex every Word source through `POST /api/knowledge/sources/{source_id}/reindex`, verify fact counts and retrieval publication health, set:

```dotenv
UNIFIED_KNOWLEDGE_ROUTING_ENABLED=true
WORD_FACTUAL_QA_ENABLED=true
```

and restart `dcagent-api` only. The structured worker remains running for Excel imports.

Document smoke questions and expected route audit:

```text
地区为华东的销售额、成本汇总 -> excel_multi_aggregate; no citations; no reranker/LLM
张三几岁 -> word_factual; only age; no reranker/LLM
介绍张三 -> summary_compare; BGE reranker and LLM used
报销流程是什么 -> document_qa; BGE reranker and LLM used
```

Rollback changes only the two flags to `false` and restarts `dcagent-api`; do not drop `knowledge_facts`, remove ClickHouse publications, or stop `dcagent-structured-worker`.

- [ ] **Step 4: Run the complete gate and commit**

```powershell
Push-Location backend
uv run --project . --group dev --group offline python -m unittest discover -s tests -p "test_*.py" -v
uv run --project . --group dev ruff format --check app tests
uv run --project . --group dev ruff check app tests
Pop-Location
git diff --check
git add backend/app/offline_settings.py backend/app/main.py backend/tests/test_offline_settings.py backend/tests/test_rag_acceptance.py backend/tests/test_quality_evaluation.py .env.example backend/.env.example deploy/offline/.env.example README.md deploy/offline/README.md deploy/ubuntu/KNOWLEDGE_ROUTING_ROLLOUT.md
git commit -m "docs: add unified knowledge routing rollout"
```

Expected: all backend tests pass; exact Excel/Word routes make zero retrieval/reranker/LLM calls; open document routes use the configured production reranker; Ubuntu runbook has a reversible two-flag cutover.
