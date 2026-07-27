# Qwen3 Hybrid Retrieval Gray Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PostgreSQL full-scan retrieval with a CPU-only, private Qwen3 Dense + Qdrant BM25 hybrid path that can run in Shadow, canary, and production modes while preserving deterministic ClickHouse aggregation and automatic Legacy rollback.

**Architecture:** Keep the current synchronous conversation repository and Physoc answer generation. Add synchronous private model clients, independent Embedding and Reranker services with bounded dynamic batching, a versioned Qdrant index, a `HybridRetriever`, and a `RetrievalRouter` injected into `SqlChatRepository`. PostgreSQL remains authoritative, ClickHouse remains the complete-data spreadsheet engine, and the current Hashing/PostgreSQL search remains the Legacy fallback.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLAlchemy/Alembic, PostgreSQL, Qdrant, ClickHouse, Redis, httpx, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, FastEmbed BM25, OpenVINO/ONNX Runtime, Loguru, unittest, uv, Ruff, Docker Compose.

---

## Locked file structure

The implementation uses these boundaries so model inference, indexing, retrieval, and rollout can be tested independently:

- `backend/app/retrieval_settings.py`: parse and validate retrieval mode, model identities, limits, timeouts, scope, Shadow, and canary settings.
- `backend/app/retrieval_models.py`: internal request scope, candidate, outcome, stage timing, publication, and comparison dataclasses.
- `backend/app/reranker_contracts.py`: versioned private Reranker wire models and metadata validation.
- `backend/app/reranker_client.py`: synchronous private HTTP client with pinned metadata checks and busy/unavailable errors.
- `backend/app/inference_batching.py`: reusable bounded asynchronous dynamic batch worker used only inside model services.
- `backend/app/qwen3_embedding_runtime.py`: Qwen3 last-token pooling and OpenVINO/ONNX/PyTorch embedding adapters.
- `backend/app/qwen3_reranker_runtime.py`: Qwen3 yes/no scoring and OpenVINO/ONNX/PyTorch reranker adapters.
- `backend/app/reranker_service.py`: private checksum-pinned Reranker FastAPI service.
- `backend/app/qdrant_retrieval.py`: collection schema, payloads, Dense/Sparse search, Alias operations, and point lifecycle.
- `backend/app/sparse_embedding.py`: pinned local FastEmbed/Qdrant BM25 encoder.
- `backend/app/retrieval_publication.py`: full collection build, validation, activation, incremental source updates, and structured metadata updates.
- `backend/app/hybrid_retriever.py`: parallel query encoding/search, RRF, reranking, adjacency expansion, and final evidence conversion.
- `backend/app/retrieval_router.py`: Legacy/Shadow/Qwen3 selection, stable canary hashing, circuit breakers, background comparison, and fallback.
- `backend/app/retrieval_audit.py`: PostgreSQL publication and Shadow comparison persistence without raw query/document text.
- `backend/app/retrieval_index_worker.py`: command-line full build and Alias activation entrypoint.
- `backend/alembic/versions/20260727_04_qwen3_retrieval.py`: chunk metadata, publication, source-index state, and Shadow audit schema.
- `deploy/docker/reranker.Dockerfile`: independent private Reranker image.
- `tools/hybrid_retrieval_benchmark.py`: 15-user latency and fallback acceptance benchmark.

Existing files are modified only where their current responsibility requires integration:

- `backend/app/embedding_contracts.py`, `embedding_client.py`, and `embedding_service.py`: preserve the current v1 protocol while adding synchronous query use, Qwen3 runtime loading, and dynamic batching.
- `backend/app/models.py`, `database.py`, `sql_repository.py`, `repository.py`, `agent.py`, `ingestion.py`, `structured_worker.py`, `main.py`, and `infra/health.py`: add metadata and inject retrieval/index lifecycle services without changing the public conversation response schema.
- `backend/pyproject.toml`, `backend/uv.lock`, Compose, environment examples, artifact schema, README, and deployment README: package and document the internal-only services.

## Implementation tasks

### Task 1: Add retrieval settings and internal contracts

**Files:**
- Create: `backend/app/retrieval_settings.py`
- Create: `backend/app/retrieval_models.py`
- Modify: `backend/app/offline_settings.py`
- Test: `backend/tests/test_retrieval_settings.py`
- Modify test: `backend/tests/test_offline_settings.py`

- [ ] **Step 1: Write failing configuration and contract tests**

Create tests that cover all three modes, private URLs, fixed model identities, limits, total timeout, stable scope, and fail-closed permission tags:

```python
class RetrievalSettingsTest(unittest.TestCase):
    def test_qwen3_defaults_match_approved_design(self) -> None:
        settings = RetrievalSettings.from_environ(private_qwen_environment())
        self.assertEqual(settings.mode, RetrievalMode.QWEN3)
        self.assertEqual(settings.dense_top_k, 50)
        self.assertEqual(settings.sparse_top_k, 50)
        self.assertEqual(settings.rerank_top_k, 24)
        self.assertEqual(settings.degraded_rerank_top_k, 12)
        self.assertEqual(settings.final_top_k, 8)
        self.assertEqual(settings.rrf_k, 60)
        self.assertEqual(settings.total_timeout_seconds, 5.0)
        self.assertEqual(settings.embedding.name, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(settings.embedding.dimensions, 1024)
        self.assertEqual(settings.reranker.name, "Qwen/Qwen3-Reranker-0.6B")

    def test_legacy_does_not_require_qwen_services(self) -> None:
        settings = RetrievalSettings.from_environ({"RETRIEVAL_MODE": "legacy"})
        self.assertEqual(settings.mode, RetrievalMode.LEGACY)

    def test_shadow_and_qwen3_require_private_services_and_permission_tags(self) -> None:
        for mode in ("shadow", "qwen3"):
            with self.subTest(mode=mode):
                environ = private_qwen_environment()
                environ["RETRIEVAL_MODE"] = mode
                environ["RETRIEVAL_PERMISSION_TAGS"] = ""
                with self.assertRaisesRegex(ValueError, "RETRIEVAL_PERMISSION_TAGS"):
                    RetrievalSettings.from_environ(environ)

    def test_rejects_public_reranker_url_and_invalid_percentages(self) -> None:
        environ = private_qwen_environment()
        environ["RERANKER_SERVICE_URL"] = "https://public.example"
        with self.assertRaises(ValueError):
            RetrievalSettings.from_environ(environ)
        for value in ("-1", "101", "nan"):
            with self.subTest(value=value):
                environ = private_qwen_environment()
                environ["RETRIEVAL_CANARY_PERCENT"] = value
                with self.assertRaises(ValueError):
                    RetrievalSettings.from_environ(environ)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
uv run --directory backend --group dev python -m unittest tests.test_retrieval_settings tests.test_offline_settings -v
```

Expected: FAIL because `retrieval_settings` and the Reranker URL setting do not exist.

- [ ] **Step 3: Implement exact settings and dataclasses**

Define these public types in `retrieval_models.py`:

```python
class RetrievalMode(StrEnum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    QWEN3 = "qwen3"

@dataclass(frozen=True, slots=True)
class RetrievalScope:
    knowledge_base_id: str
    permission_tags: tuple[str, ...]
    publication_version: str

    def __post_init__(self) -> None:
        if not self.knowledge_base_id.strip():
            raise ValueError("knowledge_base_id must not be empty")
        if not self.permission_tags:
            raise ValueError("permission_tags must not be empty")

@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    limit: int
    routing_key: str
    scope: RetrievalScope

@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    source_id: str
    source_name: str
    source_type: str
    classification: str
    chunk_id: str
    chunk_index: int
    text: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    parent_chunk_id: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None

@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    mode: RetrievalMode
    candidates: tuple[RetrievalCandidate, ...]
    stage_ms: Mapping[str, float]
    fallback_reason: str | None = None
```

Implement `RetrievalSettings.from_environ()` with these defaults and rules:

```text
RETRIEVAL_MODE=legacy
RETRIEVAL_KNOWLEDGE_BASE_ID=default
RETRIEVAL_DENSE_TOP_K=50
RETRIEVAL_SPARSE_TOP_K=50
RETRIEVAL_RERANK_TOP_K=24
RETRIEVAL_DEGRADED_RERANK_TOP_K=12
RETRIEVAL_FINAL_TOP_K=8
RETRIEVAL_RRF_K=60
RETRIEVAL_TOTAL_TIMEOUT_SECONDS=5
RETRIEVAL_SHADOW_PERCENT=0
RETRIEVAL_CANARY_PERCENT=100
QDRANT_COLLECTION_ALIAS=knowledge_chunks_current
RERANKER_SERVICE_URL=http://127.0.0.1:8082
```

When the mode is `shadow` or `qwen3`, require comma-separated `RETRIEVAL_PERMISSION_TAGS`, all Embedding metadata fields, all Reranker metadata fields, and private service URLs. Extend `require_private_url()` to allow the `reranker-service` host and add `reranker_service_url` to `OfflineSettings`. The router reads the current active publication ID from `RetrievalAuditRepository` and places that exact ID in `RetrievalScope`; it is not supplied by an untrusted request or a manually drifting environment value.

- [ ] **Step 4: Run the tests and verify they pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Commit the settings boundary**

```powershell
git add backend/app/retrieval_settings.py backend/app/retrieval_models.py backend/app/offline_settings.py backend/tests/test_retrieval_settings.py backend/tests/test_offline_settings.py
git commit -m "feat: define qwen3 retrieval settings"
```

### Task 2: Add pinned Reranker protocol and synchronous model clients

**Files:**
- Create: `backend/app/reranker_contracts.py`
- Create: `backend/app/reranker_client.py`
- Modify: `backend/app/embedding_client.py`
- Test: `backend/tests/test_reranker_contracts.py`
- Test: `backend/tests/test_reranker_client.py`
- Modify test: `backend/tests/test_embedding_client.py`

- [ ] **Step 1: Write failing wire-contract and client tests**

Use strict Pydantic models and test metadata pinning, batch splitting, private URL validation, timeout mapping, queue-busy mapping, and client reuse:

```python
class RerankerClientTest(unittest.TestCase):
    def test_scores_pairs_and_preserves_order(self) -> None:
        transport = FakeSyncTransport(scores=[0.8, 0.2])
        client = SyncHttpRerankerClient("http://reranker-service:8082", transport=transport)
        scores = client.rerank(
            "policy",
            ["required policy", "unrelated"],
            expected=reranker_metadata(),
        )
        self.assertEqual(scores, [0.8, 0.2])
        self.assertEqual(transport.calls[0][0], "http://reranker-service:8082/v1/rerank")

    def test_maps_429_to_busy_and_503_to_unavailable(self) -> None:
        with self.assertRaises(RerankerBusy):
            SyncHttpRerankerClient(
                "http://reranker-service:8082", transport=StatusTransport(429)
            ).rerank("q", ["p"], expected=reranker_metadata())
        with self.assertRaises(RerankerServiceError):
            SyncHttpRerankerClient(
                "http://reranker-service:8082", transport=StatusTransport(503)
            ).rerank("q", ["p"], expected=reranker_metadata())

class SyncEmbeddingClientTest(unittest.TestCase):
    def test_sync_client_reuses_one_httpx_client_and_checks_metadata(self) -> None:
        client = SyncHttpEmbeddingClient(
            "http://embedding-service:8081", transport=FakeSyncEmbeddingTransport()
        )
        vectors = client.embed(["one", "two"], purpose="query", expected=metadata())
        self.assertEqual(len(vectors), 2)
        client.close()
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
uv run --directory backend --group dev python -m unittest tests.test_reranker_contracts tests.test_reranker_client tests.test_embedding_client -v
```

Expected: FAIL because the Reranker protocol and synchronous clients do not exist.

- [ ] **Step 3: Implement the private Reranker contract**

Use these wire fields and bounds:

```python
MAX_RERANK_PASSAGES = 32
MAX_RERANK_TEXT_BYTES = 16 * 1024
MAX_RERANK_REQUEST_BYTES = 384 * 1024

class RerankerRequest(_WireModel):
    query: str
    passages: list[str] = Field(min_length=1, max_length=MAX_RERANK_PASSAGES)

class RerankerResponse(RerankerMetadataResponse):
    passage_count: int = Field(alias="passageCount", ge=1, le=MAX_RERANK_PASSAGES)
    scores: list[float] = Field(min_length=1, max_length=MAX_RERANK_PASSAGES)

    @model_validator(mode="after")
    def validate_scores(self) -> RerankerResponse:
        if len(self.scores) != self.passage_count:
            raise ValueError("reranker score count mismatch")
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in self.scores):
            raise ValueError("reranker scores must be finite values in [0, 1]")
        return self
```

`RerankerModelMetadata` contains `name`, `version`, `sha256`, `prompt_profile_sha256`, and `protocol_version`. Reject extra fields and non-lowercase SHA-256 values exactly as the Embedding protocol does.

- [ ] **Step 4: Implement synchronous clients**

Add `SyncHttpEmbeddingClient` beside the existing asynchronous client and implement `SyncHttpRerankerClient` with a persistent `httpx.Client` configured as follows:

```python
self._client = httpx.Client(
    timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
    follow_redirects=False,
    trust_env=False,
)
```

Both clients validate every returned metadata field before accepting vectors/scores, split requests at protocol limits, expose idempotent `close()`, and never retry model mismatches. Reranker HTTP 429 maps to `RerankerBusy`; timeout, connection, and 5xx errors map to `RerankerServiceError` without including upstream response bodies.

- [ ] **Step 5: Run the tests and verify they pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit the service clients**

```powershell
git add backend/app/reranker_contracts.py backend/app/reranker_client.py backend/app/embedding_client.py backend/tests/test_reranker_contracts.py backend/tests/test_reranker_client.py backend/tests/test_embedding_client.py
git commit -m "feat: add private reranker and sync model clients"
```

### Task 3: Add bounded batching and Qwen3 CPU runtimes

**Files:**
- Create: `backend/app/inference_batching.py`
- Create: `backend/app/qwen3_embedding_runtime.py`
- Create: `backend/app/qwen3_reranker_runtime.py`
- Create: `backend/app/reranker_service.py`
- Modify: `backend/app/embedding_service.py`
- Test: `backend/tests/test_inference_batching.py`
- Test: `backend/tests/test_qwen3_embedding_runtime.py`
- Test: `backend/tests/test_qwen3_reranker_runtime.py`
- Test: `backend/tests/test_reranker_service.py`
- Modify test: `backend/tests/test_embedding_service.py`

- [ ] **Step 1: Write failing batching tests**

```python
class DynamicBatcherTest(unittest.IsolatedAsyncioTestCase):
    async def test_merges_requests_and_splits_results_in_original_order(self) -> None:
        observed: list[list[int]] = []

        def process(items: list[int]) -> list[int]:
            observed.append(items)
            return [item * 10 for item in items]

        batcher = DynamicBatcher(process, max_items=8, max_queue_items=16, wait_ms=10)
        await batcher.start()
        left, right = await asyncio.gather(batcher.submit([1, 2]), batcher.submit([3]))
        await batcher.close()
        self.assertEqual(left, [10, 20])
        self.assertEqual(right, [30])
        self.assertEqual(observed, [[1, 2, 3]])

    async def test_rejects_immediately_when_queue_is_full(self) -> None:
        batcher = DynamicBatcher(lambda items: items, max_items=1, max_queue_items=1, wait_ms=50)
        await batcher.start()
        first = asyncio.create_task(batcher.submit([1]))
        with self.assertRaises(InferenceQueueFull):
            await batcher.submit([2])
        await first
        await batcher.close()
```

- [ ] **Step 2: Write failing Qwen3 pooling and scoring tests**

Test the algorithms independently from model libraries:

```python
def test_last_token_pool_handles_left_and_right_padding() -> None:
    hidden = numpy.array([[[1, 0], [2, 0], [9, 0]], [[3, 0], [4, 0], [0, 0]]], dtype=float)
    mask = numpy.array([[1, 1, 1], [1, 1, 0]])
    pooled = last_token_pool(hidden, mask)
    numpy.testing.assert_allclose(pooled, [[9, 0], [4, 0]])

def test_yes_probability_uses_only_yes_and_no_logits() -> None:
    scores = yes_probability(numpy.array([[1.0, 3.0], [4.0, 2.0]]))
    self.assertGreater(scores[0], 0.8)
    self.assertLess(scores[1], 0.2)
```

Also test that `runtime=openvino`, `onnxruntime`, and `torch` load only local paths, use `trust_remote_code=False`, and set `local_files_only=True` before importing model libraries.

- [ ] **Step 3: Run the new tests and verify they fail**

```powershell
uv run --directory backend --group dev --group offline python -m unittest tests.test_inference_batching tests.test_qwen3_embedding_runtime tests.test_qwen3_reranker_runtime tests.test_reranker_service tests.test_embedding_service -v
```

Expected: FAIL because the batcher, Qwen3 runtimes, and Reranker service do not exist.

- [ ] **Step 4: Implement the dynamic batch worker**

`DynamicBatcher.submit()` reserves capacity by item count, waits at most `wait_ms` for compatible work, calls the blocking processor through `asyncio.to_thread`, and resolves each request future with its original result slice. Cancellation removes no already-reserved capacity until the worker completes. `close()` rejects new work, drains accepted work, and joins the worker task.

Use one Embedding batcher per purpose so query and document instructions are never mixed. Use one Reranker batcher for `(query, passage)` pairs.

- [ ] **Step 5: Implement Qwen3 Embedding runtimes**

`qwen3_embedding_runtime.py` exposes:

```python
class Qwen3EmbeddingBackend:
    def embed(self, texts: Sequence[str], *, purpose: EmbeddingPurpose) -> list[list[float]]: ...

def load_qwen3_embedding_backend(
    model_root: Path,
    metadata: EmbeddingModelMetadata,
    *,
    runtime: Literal["openvino", "onnxruntime", "torch"],
) -> Qwen3EmbeddingBackend: ...
```

Tokenize query and document texts with the pinned local tokenizer, request the last hidden state,
select the last non-padding token, L2-normalize, and truncate only when `metadata.dimensions` is
smaller than the native 1024 dimensions. Reject output with the wrong native size before truncation.
OpenVINO uses `OVModelForFeatureExtraction`; ONNX Runtime uses
`ORTModelForFeatureExtraction`; PyTorch uses `AutoModel`. All use the local artifact only.

For `purpose="query"`, format every input with the exact pinned profile below; document inputs remain
unchanged. Include the profile SHA-256 in `embedding-metadata.json`:

```python
DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

def format_embedding_query(query: str) -> str:
    return f"Instruct: {DEFAULT_RETRIEVAL_INSTRUCTION}\nQuery:{query}"
```

- [ ] **Step 6: Implement Qwen3 Reranker runtimes and service**

Build each pair with the pinned prompt profile, tokenize to the configured maximum length, run the
causal language model, select final-token logits for the tokenizer IDs of `"yes"` and `"no"`, and
return `softmax([no, yes])[1]`. OpenVINO uses `OVModelForCausalLM`, ONNX Runtime uses
`ORTModelForCausalLM`, and the compatibility path uses `AutoModelForCausalLM`.

Pin this exact prompt profile and include its SHA-256 in `reranker-metadata.json`:

```python
DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def format_rerank_pair(query: str, passage: str) -> str:
    body = (
        f"<Instruct>: {DEFAULT_RETRIEVAL_INSTRUCTION}\n"
        f"<Query>: {query}\n<Document>: {passage}"
    )
    return f"{RERANK_PREFIX}{body}{RERANK_SUFFIX}"
```

`reranker_service.py` mirrors the existing Embedding service contract:

```text
GET  /readyz
GET  /v1/metadata
POST /v1/rerank
```

The service verifies the complete model directory checksum and `reranker-metadata.json` at startup.
Queue saturation returns HTTP 429, backend failure returns sanitized HTTP 503, malformed model output
returns HTTP 500, and readiness stays false until the startup self-test scores one positive and one
negative passage with finite values.

- [ ] **Step 7: Route Embedding requests through bounded batching**

Modify `embedding_service.py` so production startup loads `load_qwen3_embedding_backend` according
to `EMBEDDING_RUNTIME`, creates query/document batchers, and closes them during lifespan shutdown.
Preserve `create_embedding_app()` for existing direct-backend tests. Add a batcher-aware app factory
used by production and tests that returns HTTP 429 on `InferenceQueueFull`.

- [ ] **Step 8: Run tests and verify they pass**

Run the Step 3 command.

Expected: PASS without downloading model files.

- [ ] **Step 9: Commit the CPU model services**

```powershell
git add backend/app/inference_batching.py backend/app/qwen3_embedding_runtime.py backend/app/qwen3_reranker_runtime.py backend/app/reranker_service.py backend/app/embedding_service.py backend/tests/test_inference_batching.py backend/tests/test_qwen3_embedding_runtime.py backend/tests/test_qwen3_reranker_runtime.py backend/tests/test_reranker_service.py backend/tests/test_embedding_service.py
git commit -m "feat: add batched qwen3 cpu model services"
```

### Task 4: Add retrieval persistence and chunk metadata

**Files:**
- Modify: `backend/app/models.py`
- Modify: `backend/app/database.py`
- Create: `backend/app/retrieval_audit.py`
- Create: `backend/alembic/versions/20260727_04_qwen3_retrieval.py`
- Test: `backend/tests/test_retrieval_audit.py`
- Modify test: `backend/tests/test_alembic_baseline.py`
- Modify test: `backend/tests/test_sql_repository.py`

- [ ] **Step 1: Write failing migration and repository tests**

```python
class RetrievalAuditRepositoryTest(unittest.TestCase):
    def test_records_publication_and_redacted_shadow_comparison(self) -> None:
        publication = repository.create_publication(
            collection_name="knowledge_chunks_qwen3_v1",
            alias_name="knowledge_chunks_current",
            embedding_model_version="qwen3-0.6b-1",
            sparse_profile_sha256="a" * 64,
            dimensions=1024,
        )
        repository.mark_publication_validated(publication.id, point_count=12)
        repository.mark_publication_active(publication.id, point_count=12)
        repository.record_shadow(
            request_id="request-1",
            routing_key_hash="b" * 64,
            query_hash="c" * 64,
            legacy_chunk_ids=("legacy-1",),
            qwen_chunk_ids=("qwen-1",),
            legacy_ms=40.0,
            qwen_ms=220.0,
            status="completed",
        )
        stored = repository.list_shadow(limit=1)[0]
        self.assertEqual(stored.query_hash, "c" * 64)
        self.assertFalse(hasattr(stored, "query"))

    def test_chunk_metadata_round_trips(self) -> None:
        chunk = KnowledgeChunkModel(
            id="chunk-1", source_id="source-1", chunk_index=0,
            text="body", token_count=1,
            metadata={"section_title": "Policy", "page_number": 3},
        )
        repository.complete_knowledge_source_indexing("source-1", [chunk])
        self.assertEqual(repository.list_knowledge_chunks("source-1")[0].metadata["page_number"], 3)
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
uv run --directory backend --group dev python -m unittest tests.test_retrieval_audit tests.test_alembic_baseline tests.test_sql_repository -v
```

Expected: FAIL because the new records, migration, and chunk metadata do not exist.

- [ ] **Step 3: Add the database schema**

The migration adds `knowledge_chunks.metadata JSON NOT NULL DEFAULT '{}'` and these tables:

```text
retrieval_publications(
  id PK, collection_name UNIQUE, alias_name, status,
  embedding_model_version, sparse_profile_sha256, dimensions,
  point_count, error_message, created_at, completed_at
)

retrieval_source_indexes(
  source_id PK/FK knowledge_sources ON DELETE CASCADE,
  publication_id FK retrieval_publications ON DELETE SET NULL,
  status, indexed_chunk_count, error_message, updated_at
)

retrieval_shadow_comparisons(
  id PK, request_id UNIQUE, routing_key_hash, query_hash,
  legacy_chunk_ids JSON, qwen_chunk_ids JSON,
  legacy_ms, qwen_ms, status, fallback_reason, created_at
)
```

Use indexes on publication status, source-index status, Shadow creation time, and Shadow status.
The downgrade removes the three tables before removing `knowledge_chunks.metadata`.

- [ ] **Step 4: Implement retrieval persistence**

`RetrievalAuditRepository` validates legal transitions:

```text
building -> validated -> active -> retired
building|validated -> failed
```

Activating one publication retires the previous active publication in the same transaction. Shadow
records accept only hashes, identifiers, numeric timings, status, and sanitized fallback codes.
`KnowledgeChunkModel.metadata` defaults to an empty dictionary, and SQL/in-memory repositories
round-trip it without changing the current API schema.

- [ ] **Step 5: Run tests and verify they pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit persistence**

```powershell
git add backend/app/models.py backend/app/database.py backend/app/retrieval_audit.py backend/alembic/versions/20260727_04_qwen3_retrieval.py backend/tests/test_retrieval_audit.py backend/tests/test_alembic_baseline.py backend/tests/test_sql_repository.py
git commit -m "feat: persist retrieval publications and shadow audits"
```

### Task 5: Add Qdrant hybrid index and local BM25 encoding

**Files:**
- Create: `backend/app/sparse_embedding.py`
- Create: `backend/app/qdrant_retrieval.py`
- Test: `backend/tests/test_sparse_embedding.py`
- Test: `backend/tests/test_qdrant_retrieval.py`
- Test: `backend/tests/integration/test_qdrant_retrieval.py`

- [ ] **Step 1: Write failing sparse and Qdrant gateway tests**

Use injected fake FastEmbed and Qdrant clients so unit tests do not start services:

```python
class SparseEmbeddingTest(unittest.TestCase):
    def test_emits_sorted_finite_sparse_vectors(self) -> None:
        encoder = LocalBm25Encoder(model=FakeSparseModel(indices=[7, 2], values=[0.4, 0.8]))
        vector = encoder.embed_query("leave policy")
        self.assertEqual(vector.indices, (2, 7))
        self.assertEqual(vector.values, (0.8, 0.4))

class QdrantRetrievalTest(unittest.TestCase):
    def test_creates_named_dense_and_sparse_vectors(self) -> None:
        gateway = QdrantRetrievalGateway(FakeQdrantClient(), alias_name="knowledge_chunks_current")
        gateway.create_collection("knowledge_chunks_qwen3_v1", dense_dimensions=1024)
        config = gateway.client.created[0]
        self.assertEqual(config["dense"].size, 1024)
        self.assertEqual(config["dense"].distance, models.Distance.COSINE)
        self.assertTrue(config["sparse"].index.on_disk)

    def test_search_filter_is_applied_before_scoring(self) -> None:
        scope = RetrievalScope("default", ("internal",), "v1")
        gateway.search_dense([0.0] * 1024, scope=scope, limit=50)
        observed = gateway.client.query_calls[0].query_filter
        self.assertIn("knowledge_base_id", serialized_filter_keys(observed))
        self.assertIn("permission_tags", serialized_filter_keys(observed))
        self.assertIn("publication_version", serialized_filter_keys(observed))

    def test_rejects_empty_scope_instead_of_unfiltered_search(self) -> None:
        with self.assertRaises(ValueError):
            gateway.search_dense([0.0] * 1024, scope=None, limit=50)
```

- [ ] **Step 2: Run unit tests and verify they fail**

```powershell
uv run --directory backend --group dev --group offline python -m unittest tests.test_sparse_embedding tests.test_qdrant_retrieval -v
```

Expected: FAIL because the sparse encoder and gateway do not exist.

- [ ] **Step 3: Implement the local BM25 encoder**

Load the pinned `Qdrant/bm25` FastEmbed artifact from `SPARSE_MODEL_ROOT` with runtime networking
disabled. Convert FastEmbed outputs into this immutable internal value:

```python
@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]
```

Sort by index, combine duplicate indices, reject negative indices, zero vectors, and non-finite
values, and expose `embed_documents(texts)` plus `embed_query(text)`.

- [ ] **Step 4: Implement the Qdrant gateway**

Create collections with named vectors and conservative CPU/storage defaults:

```python
vectors_config = {
    "dense": models.VectorParams(
        size=dense_dimensions,
        distance=models.Distance.COSINE,
        on_disk=True,
    )
}
sparse_vectors_config = {
    "sparse": models.SparseVectorParams(
        index=models.SparseIndexParams(on_disk=True),
        modifier=models.Modifier.IDF,
    )
}
quantization_config = models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,
        quantile=0.99,
        always_ram=True,
    )
)
```

Expose `create_collection`, `delete_collection`, `upsert_points`, `delete_source`, `search_dense`,
`search_sparse`, `retrieve_points`, `validate_collection`, `activate_alias`, and `resolve_alias`.
Every search requires a `RetrievalScope` and creates `must` conditions for knowledge base,
publication version, and `MatchAny` permission tags. Convert Qdrant payloads into
`RetrievalCandidate`; reject any hit missing source, chunk, text, or permission metadata.

- [ ] **Step 5: Run unit tests and verify they pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Add an opt-in real-Qdrant integration test**

The integration test uses `QDRANT_INTEGRATION_URL`, creates a uniquely named collection, upserts
three points with disjoint permissions, verifies Dense and Sparse filters return only the allowed
point, atomically activates a temporary Alias, and deletes the collection in `addCleanup()`.

Run:

```powershell
$env:QDRANT_INTEGRATION_URL='http://127.0.0.1:6333'
uv run --directory backend --group dev --group offline python -m unittest tests.integration.test_qdrant_retrieval -v
```

Expected with Qdrant running: PASS. Expected without the environment variable: SKIP.

- [ ] **Step 7: Commit the Qdrant index boundary**

```powershell
git add backend/app/sparse_embedding.py backend/app/qdrant_retrieval.py backend/tests/test_sparse_embedding.py backend/tests/test_qdrant_retrieval.py backend/tests/integration/test_qdrant_retrieval.py
git commit -m "feat: add qdrant dense sparse index gateway"
```

### Task 6: Add versioned publication, backfill, and incremental indexing

**Files:**
- Create: `backend/app/retrieval_publication.py`
- Create: `backend/app/retrieval_index_worker.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/ingestion.py`
- Modify: `backend/app/structured_worker.py`
- Modify: `backend/app/structured_repository.py`
- Test: `backend/tests/test_retrieval_publication.py`
- Test: `backend/tests/test_retrieval_index_worker.py`
- Modify test: `backend/tests/test_knowledge_ingestion_pipeline.py`
- Modify test: `backend/tests/test_structured_worker.py`

- [ ] **Step 1: Write failing full-build tests**

```python
class RetrievalPublicationTest(unittest.TestCase):
    def test_build_validates_before_alias_switch(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(3))
        result = publisher.build_and_activate("knowledge_chunks_qwen3_v1")
        self.assertEqual(result.point_count, 3)
        self.assertEqual(
            publisher.gateway.events,
            ["create", "upsert:3", "validate:3", "activate_alias"],
        )

    def test_failed_build_never_moves_alias(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(3), fail_validation=True)
        with self.assertRaises(IndexValidationError):
            publisher.build_and_activate("knowledge_chunks_qwen3_v1")
        self.assertNotIn("activate_alias", publisher.gateway.events)
        self.assertEqual(publisher.audit.active_publication(), None)

    def test_document_batches_never_exceed_embedding_protocol_limit(self) -> None:
        publisher = build_publisher(chunks=sample_chunks(130))
        publisher.build_and_activate("knowledge_chunks_qwen3_v1")
        self.assertEqual([len(batch) for batch in publisher.embedding.batches], [64, 64, 2])
```

- [ ] **Step 2: Write failing incremental lifecycle tests**

```python
def test_new_document_updates_postgres_then_active_qdrant_collection() -> None:
    queue = KnowledgeIngestionQueue(repository, index_lifecycle=index_lifecycle)
    queue.enqueue("source-1", sample_txt, "TXT")
    queue.drain()
    self.assertEqual(repository.get_source_index_status("source-1"), "indexed")
    self.assertEqual(index_lifecycle.upserted_source_ids, ["source-1"])

def test_qdrant_failure_does_not_delete_postgres_chunks() -> None:
    queue = KnowledgeIngestionQueue(repository, index_lifecycle=FailingIndexLifecycle())
    queue.enqueue("source-1", sample_txt, "TXT")
    queue.drain()
    self.assertEqual(len(repository.list_knowledge_chunks("source-1")), 1)
    self.assertEqual(repository.get_source_index_status("source-1"), "failed")

def test_structured_publication_indexes_only_catalog_metadata() -> None:
    worker = build_structured_worker(metadata_indexer=RecordingMetadataIndexer())
    worker.run_once()
    point = worker.metadata_indexer.points[0]
    self.assertIn("worksheet_name", point.payload)
    self.assertNotIn("complete_rows", point.payload)
```

- [ ] **Step 3: Run tests and verify they fail**

```powershell
uv run --directory backend --group dev --group offline python -m unittest tests.test_retrieval_publication tests.test_retrieval_index_worker tests.test_knowledge_ingestion_pipeline tests.test_structured_worker -v
```

Expected: FAIL because publication and lifecycle hooks do not exist.

- [ ] **Step 4: Implement the point builder and full publisher**

For each narrative chunk, derive missing adjacency from ordered source chunks and build a payload
containing the approved fields. Use deterministic point UUIDs from `source_id + chunk_id +
publication_version`. For structured publications, create one point per dataset/worksheet containing
only schema, aliases, units, types, row count, safe sample values, and publication identifiers.

`build_and_activate()` performs this exact sequence:

```text
create PostgreSQL publication(building)
create Qdrant collection
stream PostgreSQL chunks/catalogs in bounded batches
embed Dense and Sparse vectors
upsert with deterministic IDs
validate vector size, point count, payload fields, permission probes, sample queries
mark publication validated
atomically switch Alias
mark publication active and retire the previous publication
```

On any error, mark the publication failed, leave the Alias unchanged, and delete the incomplete
collection only when it is not referenced by an Alias.

- [ ] **Step 5: Implement the full-build entrypoint**

`python -m app.retrieval_index_worker` accepts only environment configuration and these flags:

```text
--collection knowledge_chunks_qwen3_v1
--activate
--batch-size 64
--validation-sample-size 50
```

`--activate` is required to move the Alias. Without it, a successful build remains `validated` for
manual review. Reject collection names that do not match `^knowledge_chunks_qwen3_v[0-9]+$`.

- [ ] **Step 6: Implement incremental source and structured metadata hooks**

Add an optional `KnowledgeIndexLifecycle` to `KnowledgeIngestionQueue`. PostgreSQL chunk completion
happens first; Qdrant upsert updates `retrieval_source_indexes` independently. Qdrant failure marks
only retrieval indexing failed and preserves chunks for Legacy fallback. Source deletion must delete
Qdrant points before committing the PostgreSQL deletion; a failed Qdrant delete returns an explicit
503 instead of leaving searchable stale points.

Inject `StructuredMetadataIndexer` into `StructuredIngestionWorker`. Call it only after
`complete_publication()` succeeds. Its failure records a retrieval source-index error without
rolling back the valid ClickHouse publication.

- [ ] **Step 7: Run tests and verify they pass**

Run the Step 3 command.

Expected: PASS.

- [ ] **Step 8: Commit publication and lifecycle support**

```powershell
git add backend/app/retrieval_publication.py backend/app/retrieval_index_worker.py backend/app/sql_repository.py backend/app/repository.py backend/app/ingestion.py backend/app/structured_worker.py backend/app/structured_repository.py backend/tests/test_retrieval_publication.py backend/tests/test_retrieval_index_worker.py backend/tests/test_knowledge_ingestion_pipeline.py backend/tests/test_structured_worker.py
git commit -m "feat: publish versioned hybrid retrieval indexes"
```

### Task 7: Implement hybrid retrieval, RRF, reranking, and adjacency

**Files:**
- Create: `backend/app/hybrid_retriever.py`
- Modify: `backend/app/retrieval.py`
- Modify: `backend/app/models.py`
- Modify: `backend/app/sql_repository.py`
- Test: `backend/tests/test_hybrid_retriever.py`
- Modify test: `backend/tests/test_retrieval_threshold.py`
- Modify test: `backend/tests/test_sql_repository.py`

- [ ] **Step 1: Write failing RRF and retrieval tests**

```python
class ReciprocalRankFusionTest(unittest.TestCase):
    def test_fuses_by_chunk_id_and_records_both_ranks(self) -> None:
        fused = reciprocal_rank_fusion(
            dense=[candidate("a"), candidate("b")],
            sparse=[candidate("b"), candidate("c")],
            k=60,
        )
        self.assertEqual([item.chunk_id for item in fused], ["b", "a", "c"])
        self.assertEqual(fused[0].dense_rank, 2)
        self.assertEqual(fused[0].sparse_rank, 1)

class HybridRetrieverTest(unittest.TestCase):
    def test_runs_dense_and_sparse_search_then_reranks_top_24(self) -> None:
        retriever = build_retriever(dense_count=50, sparse_count=50)
        outcome = retriever.retrieve(request("policy"))
        self.assertEqual(retriever.reranker.passage_count, 24)
        self.assertLessEqual(len(outcome.candidates), 8)
        self.assertIn("embedding", outcome.stage_ms)
        self.assertIn("qdrant", outcome.stage_ms)
        self.assertIn("reranker", outcome.stage_ms)

    def test_retries_busy_reranker_once_with_top_12(self) -> None:
        retriever = build_retriever(reranker=BusyOnceReranker())
        outcome = retriever.retrieve(request("policy"))
        self.assertEqual(retriever.reranker.batch_sizes, [24, 12])
        self.assertTrue(outcome.candidates)

    def test_adds_only_bounded_adjacent_evidence(self) -> None:
        retriever = build_retriever(top_hit=chunk_with_neighbors("c2", "c1", "c3"))
        outcome = retriever.retrieve(request("policy"))
        self.assertEqual([item.chunk_id for item in outcome.candidates[:3]], ["c2", "c1", "c3"])
        self.assertLessEqual(sum(len(item.text) for item in outcome.candidates), 24_000)
```

- [ ] **Step 2: Run tests and verify they fail**

```powershell
uv run --directory backend --group dev --group offline python -m unittest tests.test_hybrid_retriever tests.test_retrieval_threshold tests.test_sql_repository -v
```

Expected: FAIL because hybrid retrieval is not implemented.

- [ ] **Step 3: Implement deterministic RRF**

For each list, assign one-based ranks and sum `1 / (k + rank)` by chunk ID. Preserve the candidate
payload from the better individual rank, record both ranks, reject duplicate IDs inside one input,
and sort ties by source name, chunk index, then chunk ID. This makes Shadow comparisons reproducible.

- [ ] **Step 4: Implement `HybridRetriever.retrieve()`**

Use one persistent `ThreadPoolExecutor(max_workers=4)` and one absolute deadline based on
`time.monotonic()`. Execute Dense query embedding and Sparse query encoding in parallel, then Dense
and Sparse Qdrant searches in parallel. Fuse Top 50 + Top 50, rerank Top 24, retry once with Top 12
only for `RerankerBusy`, and select Top 8.

Fetch referenced parent/previous/next points in one Qdrant call. Add them only when absent and stop at
both the configured evidence count and character budget. Convert the final candidates to existing
`KnowledgeSearchHitModel` values so Physoc prompt and citation code remain unchanged. Set `score` to
the reranker score when present and keep Dense/Sparse/RRF diagnostics only in internal outcomes.

- [ ] **Step 5: Keep Legacy search explicit and scoped**

Rename the current SQL full scan implementation to `_search_legacy_knowledge_chunks()` and preserve
its scoring. Add the configured classification/permission filter to its SQL statement so automatic
fallback cannot search a broader scope than Qdrant. Do not remove `HashingEmbeddingProvider`.

- [ ] **Step 6: Run tests and verify they pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 7: Commit hybrid retrieval**

```powershell
git add backend/app/hybrid_retriever.py backend/app/retrieval.py backend/app/models.py backend/app/sql_repository.py backend/tests/test_hybrid_retriever.py backend/tests/test_retrieval_threshold.py backend/tests/test_sql_repository.py
git commit -m "feat: add qwen3 hybrid retrieval and reranking"
```

### Task 8: Add Legacy, Shadow, canary, circuit-breaker, and fallback routing

**Files:**
- Create: `backend/app/retrieval_router.py`
- Modify: `backend/app/agent.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/repository.py`
- Test: `backend/tests/test_retrieval_router.py`
- Modify test: `backend/tests/test_agent.py`
- Modify test: `backend/tests/test_sql_repository.py`

- [ ] **Step 1: Write failing routing tests**

```python
class RetrievalRouterTest(unittest.TestCase):
    def test_legacy_never_calls_qwen(self) -> None:
        router = build_router(mode="legacy")
        result = router.search(request("policy"))
        self.assertEqual(result.mode, "legacy")
        self.assertEqual(router.hybrid.calls, 0)

    def test_shadow_returns_legacy_and_records_qwen_off_thread(self) -> None:
        router = build_router(mode="shadow", shadow_percent=100)
        result = router.search(request("policy", routing_key="conv-1"))
        self.assertEqual(result.mode, "legacy")
        router.shadow_queue.drain_for_test()
        self.assertEqual(router.audit.shadow_count(), 1)
        self.assertEqual(router.audit.list_shadow(1)[0].status, "completed")

    def test_canary_assignment_is_stable(self) -> None:
        router = build_router(mode="qwen3", canary_percent=25)
        assignments = [router.uses_qwen("conv-7") for _ in range(20)]
        self.assertEqual(len(set(assignments)), 1)

    def test_qwen_failure_falls_back_once_and_opens_circuit(self) -> None:
        router = build_router(mode="qwen3", hybrid=AlwaysFailingHybrid(), failure_threshold=2)
        first = router.search(request("one"))
        second = router.search(request("two"))
        third = router.search(request("three"))
        self.assertEqual(first.fallback_reason, "hybrid_unavailable")
        self.assertEqual(second.fallback_reason, "hybrid_unavailable")
        self.assertEqual(third.fallback_reason, "circuit_open")
        self.assertEqual(router.hybrid.calls, 2)

    def test_empty_qwen_results_use_nonempty_legacy_results(self) -> None:
        router = build_router(mode="qwen3", hybrid=EmptyHybrid())
        result = router.search(request("known term"))
        self.assertEqual(result.mode, "legacy")
        self.assertEqual(result.fallback_reason, "qwen_empty_legacy_nonempty")
```

- [ ] **Step 2: Run tests and verify they fail**

```powershell
uv run --directory backend --group dev --group offline python -m unittest tests.test_retrieval_router tests.test_agent tests.test_sql_repository -v
```

Expected: FAIL because mode routing and routing keys do not exist.

- [ ] **Step 3: Implement stable percentage selection and circuit breaker**

Compute the bucket as `int.from_bytes(sha256(routing_key.encode()).digest()[:8], "big") % 100`.
The circuit opens after the configured consecutive failure count, remains open for the configured
reset interval, permits one half-open probe, and closes only after that probe succeeds. Use a lock so
sync FastAPI worker threads cannot corrupt state.

- [ ] **Step 4: Implement the bounded Shadow queue**

Use a fixed-size `queue.Queue` and one daemon worker per API process. `submit()` is non-blocking; a
full queue increments a metric and skips Shadow work. The worker records SHA-256 hashes of query and
routing key, ordered chunk IDs, timings, status, and sanitized error code. It never stores query text,
passage text, internal URLs, or exception messages. `close()` stops and joins the worker.

- [ ] **Step 5: Implement `RetrievalRouter.search()`**

Apply this order exactly:

```text
legacy -> return Legacy
shadow not selected -> return Legacy
shadow selected -> return Legacy and enqueue Qwen comparison
qwen3 canary not selected -> return Legacy
qwen3 selected + circuit open -> return Legacy(circuit_open)
qwen3 selected + Qwen success/nonempty -> return Qwen
qwen3 selected + Qwen empty -> call Legacy; fallback only when Legacy is nonempty
qwen3 selected + Qwen failure/timeout/busy -> return Legacy with sanitized reason
```

Modify `KnowledgeAgentTools.search_knowledge` to accept the current conversation ID as its routing
key. The structured service remains before Agent execution in `SqlChatRepository.send_message()`, so
ClickHouse aggregate questions never enter this router.

- [ ] **Step 6: Run tests and verify they pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 7: Commit the migration router**

```powershell
git add backend/app/retrieval_router.py backend/app/agent.py backend/app/sql_repository.py backend/app/repository.py backend/tests/test_retrieval_router.py backend/tests/test_agent.py backend/tests/test_sql_repository.py
git commit -m "feat: add shadow canary and retrieval fallback"
```

### Task 9: Wire production startup, health checks, ownership, and sanitized logging

**Files:**
- Modify: `backend/app/main.py`
- Modify: `backend/app/infra/health.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/app/sql_repository.py`
- Test: `backend/tests/test_app_configuration.py`
- Modify test: `backend/tests/test_lazy_startup.py`
- Modify test: `backend/tests/test_infra_health.py`
- Modify test: `backend/tests/test_api_contract.py`

- [ ] **Step 1: Write failing startup ownership and health tests**

```python
class RetrievalProductionStartupTest(unittest.IsolatedAsyncioTestCase):
    async def test_qwen3_startup_builds_and_closes_owned_resources(self) -> None:
        resources = RecordingResourceFactory()
        app = create_production_app(
            environment_override=private_qwen_environment(),
            retrieval_resource_factory=resources,
        )
        async with app.router.lifespan_context(app):
            self.assertIs(app.state.repository.retrieval_router, resources.router)
        self.assertEqual(resources.closed, ["shadow_queue", "hybrid", "reranker", "embedding", "qdrant"])

    async def test_legacy_startup_does_not_construct_qwen_clients(self) -> None:
        resources = RecordingResourceFactory(fail_if_called=True)
        environ = private_environment()
        environ["RETRIEVAL_MODE"] = "legacy"
        app = create_production_app(
            environment_override=environ,
            retrieval_resource_factory=resources,
        )
        async with app.router.lifespan_context(app):
            self.assertEqual(resources.calls, [])

class RetrievalHealthTest(unittest.TestCase):
    def test_shadow_reports_qwen_degraded_without_failing_api_readiness(self) -> None:
        report = build_registry(mode="shadow", reranker_ok=False).report()
        self.assertTrue(report["retrieval_shadow"]["ok"])
        self.assertEqual(report["reranker"]["detail"], "degraded")

    def test_qwen3_requires_qdrant_embedding_and_reranker_readiness(self) -> None:
        report = build_registry(mode="qwen3", qdrant_ok=True, embedding_ok=True, reranker_ok=False).report()
        self.assertFalse(report["reranker"]["ok"])
```

- [ ] **Step 2: Write failing sanitized API logging test**

Patch the hybrid retriever to raise `RuntimeError("secret passage http://reranker-service:8082")`,
send a message, and assert the client receives a normal Legacy answer while the captured Loguru
record contains request ID and `hybrid_unavailable` but not `secret passage` or the internal URL.

- [ ] **Step 3: Run tests and verify they fail**

```powershell
uv run --directory backend --group dev --group offline python -m unittest tests.test_app_configuration tests.test_lazy_startup tests.test_infra_health tests.test_api_contract -v
```

Expected: FAIL because production does not construct or health-check retrieval resources.

- [ ] **Step 4: Implement production resource construction**

In `main.py`, parse `RetrievalSettings` after `OfflineSettings`. For Shadow/Qwen3 modes, construct
one persistent sync Embedding client, Reranker client, Qdrant client/gateway, BM25 encoder,
`HybridRetriever`, `RetrievalAuditRepository`, and `RetrievalRouter`, then inject the router and index
lifecycle into `SqlChatRepository` and `KnowledgeIngestionQueue`. Legacy mode constructs none of
those model/retrieval resources.

Register every owned resource once and close in reverse dependency order. Preserve the existing
structured service, Physoc provider, database, storage, and evaluation resource ownership.

- [ ] **Step 5: Implement mode-aware dependency health**

Add private readiness checks for:

```text
GET embedding-service/v1/metadata
GET reranker-service/v1/metadata
GET qdrant/readyz
Qdrant Alias resolution and expected vector dimension
```

Metadata checks compare every pinned field. In `qwen3`, a failed required retrieval dependency makes
`/api/readyz` return 503. In `shadow`, the API remains ready but reports the Qwen dependency as
`degraded`; PostgreSQL, configured ClickHouse, and production LLM rules remain unchanged.

- [ ] **Step 6: Add stage metrics and sanitized Loguru events**

Emit one structured completion event with request ID, mode, model versions, Alias, candidate counts,
stage timings, fallback code, and result count. Log exceptions through `logger.bind(...).exception()`
only at the internal boundary and replace raw exception strings with an enumerated fallback code in
stored audit data and client-visible output.

- [ ] **Step 7: Run tests and verify they pass**

Run the Step 3 command.

Expected: PASS.

- [ ] **Step 8: Commit application integration**

```powershell
git add backend/app/main.py backend/app/infra/health.py backend/app/routes.py backend/app/sql_repository.py backend/tests/test_app_configuration.py backend/tests/test_lazy_startup.py backend/tests/test_infra_health.py backend/tests/test_api_contract.py
git commit -m "feat: wire production hybrid retrieval services"
```

### Task 10: Package the internal model services and deployment contract

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`
- Create: `deploy/docker/reranker.Dockerfile`
- Modify: `deploy/docker/embedding.Dockerfile`
- Modify: `deploy/offline/compose.yaml`
- Modify: `deploy/offline/artifacts.schema.json`
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `deploy/offline/.env.example`
- Modify: `tools/tests/test_backend_uv_contract.py`
- Modify: `tools/tests/test_compose_contract.py`
- Modify: `tools/tests/test_compose_smoke.py`
- Modify: `tools/tests/test_structured_deployment_contract.py`
- Modify: `backend/tests/test_project_dependencies.py`
- Modify: `backend/tests/test_offline_artifacts.py`

- [ ] **Step 1: Write failing dependency and Compose contract tests**

```python
def test_offline_group_contains_qwen3_cpu_runtime_packages(self) -> None:
    dependencies = offline_dependencies()
    for package in (
        "fastembed", "openvino", "optimum", "optimum-intel", "transformers", "torch"
    ):
        self.assertTrue(any(item.lower().startswith(package) for item in dependencies), package)

def test_compose_has_independent_reranker_service(self) -> None:
    compose = load_compose()
    service = compose["services"]["reranker-service"]
    self.assertEqual(service["networks"], ["offline"])
    self.assertEqual(service["environment"]["HF_HUB_OFFLINE"], "1")
    self.assertIn("RERANKER_MODEL_ROOT", service["environment"])
    self.assertNotIn("ports", service)

def test_api_receives_retrieval_mode_and_pinned_metadata(self) -> None:
    environment = load_compose()["services"]["api"]["environment"]
    for key in (
        "RETRIEVAL_MODE", "QDRANT_COLLECTION_ALIAS", "RERANKER_SERVICE_URL",
        "EMBEDDING_MODEL_SHA256", "RERANKER_MODEL_SHA256",
    ):
        self.assertIn(key, environment)
```

- [ ] **Step 2: Run contract tests and verify they fail**

```powershell
uv run --directory backend --group dev python -m unittest tests.test_project_dependencies tests.test_offline_artifacts -v
uv run --project backend --group dev python -m unittest tools.tests.test_backend_uv_contract tools.tests.test_compose_contract tools.tests.test_compose_smoke tools.tests.test_structured_deployment_contract -v
```

Expected: FAIL because the Reranker service and runtime dependencies are absent.

- [ ] **Step 3: Add CPU runtime dependencies and refresh the lock**

Add these packages without version upper bounds to the `offline` dependency group:

```toml
"fastembed>=0.7",
"numpy>=2",
"openvino>=2025",
"optimum[onnxruntime]>=1.27",
"optimum-intel[openvino]>=1.24",
"torch>=2.7",
"transformers>=4.53",
```

Keep `qdrant-client`, `FlagEmbedding`, and `onnxruntime`. Run:

```powershell
uv lock --project backend --upgrade
uv sync --project backend --group dev --group offline --frozen
```

Expected: lock succeeds and the environment installs from the configured package source. Before
offline image creation, mirror every locked wheel into `artifacts/wheels`; the Docker build remains
`--offline --find-links=/wheels` and must fail if any wheel is missing.

- [ ] **Step 4: Add independent model service images**

Keep `embedding.Dockerfile` on port 8081 and switch its command only to the updated Qwen3-capable
factory. Create `reranker.Dockerfile` from the same non-root/offline pattern with this command:

```dockerfile
CMD ["python", "-m", "uvicorn", "app.reranker_service:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8082"]
```

Neither image exposes a host port. Both mount `/models` read-only and use one Uvicorn worker so one
container loads one model copy.

- [ ] **Step 5: Update Compose and environment contracts**

Add `reranker-service` on the internal network with checksum-pinned model root, runtime, batch, queue,
thread, and offline environment variables. Add CPU and memory limits for both model services. The
standard internal Compose stack starts both model services, while `RETRIEVAL_MODE=legacy` prevents the
API from constructing their clients or requiring their readiness.

Add all retrieval variables from Task 1, with these production examples:

```text
RETRIEVAL_MODE=shadow
RETRIEVAL_SHADOW_PERCENT=10
RETRIEVAL_CANARY_PERCENT=0
RETRIEVAL_PERMISSION_TAGS=internal
QDRANT_COLLECTION_ALIAS=knowledge_chunks_current
EMBEDDING_MODEL_DIR=qwen3-embedding-0.6b
RERANKER_MODEL_DIR=qwen3-reranker-0.6b
EMBEDDING_RUNTIME=openvino
RERANKER_RUNTIME=openvino
```

Extend the artifact schema description to include `embedding-model`, `reranker-model`, and
`sparse-embedding-model` directories with local paths and SHA-256 checksums. Do not add download URLs.

- [ ] **Step 6: Run deployment contract tests and offline render**

Run the Step 2 command, then:

```powershell
docker compose --env-file deploy/offline/.env.example -f deploy/offline/compose.yaml config --quiet
```

Expected: tests PASS and Compose renders successfully when required secret/file placeholders are
provided by the existing test harness.

- [ ] **Step 7: Commit deployment packaging**

```powershell
git add backend/pyproject.toml backend/uv.lock deploy/docker/embedding.Dockerfile deploy/docker/reranker.Dockerfile deploy/offline/compose.yaml deploy/offline/artifacts.schema.json .env.example backend/.env.example deploy/offline/.env.example backend/tests/test_project_dependencies.py backend/tests/test_offline_artifacts.py tools/tests/test_backend_uv_contract.py tools/tests/test_compose_contract.py tools/tests/test_compose_smoke.py tools/tests/test_structured_deployment_contract.py
git commit -m "build: package private qwen3 retrieval services"
```

### Task 11: Add quality metrics, Shadow reporting, performance gates, and failure injection

**Files:**
- Modify: `backend/app/evaluation.py`
- Modify: `backend/app/evaluation_batches.py`
- Modify: `backend/app/sql_repository.py`
- Create: `tools/hybrid_retrieval_benchmark.py`
- Create: `tools/tests/test_hybrid_retrieval_benchmark.py`
- Create: `backend/tests/integration/test_hybrid_retrieval_e2e.py`
- Modify: `backend/tests/test_quality_evaluation.py`
- Modify: `backend/tests/test_evaluation_batches.py`
- Modify: `backend/tests/test_rag_acceptance.py`
- Modify: `tools/tests/test_benchmark_report.py`

- [ ] **Step 1: Write failing ranking metric tests**

```python
class RankingMetricTest(unittest.TestCase):
    def test_recall_mrr_and_ndcg(self) -> None:
        metrics = calculate_ranking_metrics(
            ranked_chunk_ids=["wrong", "relevant-a", "relevant-b"],
            relevant_chunk_ids={"relevant-a", "relevant-b"},
            k=3,
        )
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.mrr, 0.5)
        self.assertAlmostEqual(metrics.ndcg, 0.6934, places=4)

    def test_shadow_report_contains_no_raw_text(self) -> None:
        report = build_shadow_report(sample_shadow_records())
        serialized = json.dumps(report)
        self.assertNotIn("query", report["records"][0])
        self.assertNotIn("passage", serialized.lower())
```

- [ ] **Step 2: Write failing benchmark threshold tests**

```python
class HybridRetrievalBenchmarkTest(unittest.TestCase):
    def test_report_fails_p95_error_and_fallback_thresholds(self) -> None:
        report = summarize_results(
            latencies=[1.0] * 14 + [5.2],
            errors=1,
            fallbacks=2,
            requests=100,
            p95_limit=5.0,
            error_rate_limit=0.01,
            fallback_rate_limit=0.01,
        )
        self.assertFalse(report.passed)
        self.assertIn("p95_seconds", report.failed_gates)
        self.assertIn("fallback_rate", report.failed_gates)
```

- [ ] **Step 3: Run tests and verify they fail**

```powershell
uv run --directory backend --group dev --group benchmark python -m unittest tests.test_quality_evaluation tests.test_evaluation_batches tests.test_rag_acceptance -v
uv run --project backend --group dev --group benchmark python -m unittest tools.tests.test_hybrid_retrieval_benchmark tools.tests.test_benchmark_report -v
```

Expected: FAIL because NDCG/MRR, Shadow reporting, and the hybrid benchmark do not exist.

- [ ] **Step 4: Add retrieval quality metrics and comparison reports**

Extend evaluation results with `recall_at_k`, `mrr`, and `ndcg_at_k` while preserving existing source
and term recall fields. Batch comparison reports include Legacy vs Qwen3 deltas, critical-case Top 8
regressions, latency percentiles, errors, and fallback reasons. Reports use case IDs and chunk IDs,
never full source text.

The acceptance gate is:

```text
Recall@50 >= 0.90
NDCG@8 >= Legacy NDCG@8
target NDCG@8 improvement >= 0.05
critical Top-8 regressions = 0
permission leaks = 0
structured aggregate mismatches = 0
```

- [ ] **Step 5: Implement the concurrency benchmark**

`tools/hybrid_retrieval_benchmark.py` accepts:

```text
--concurrency 15
--requests 150
--p95-seconds 5
--max-error-rate 0.01
--max-fallback-rate 0.01
--questions-jsonl
--output-json
```

Construct one production `HybridRetriever` from the internal environment, use 15 closed-loop worker
threads, and record timings directly from `RetrievalOutcome`. This measures retrieval without calling
Physoc. Exit nonzero when any threshold fails. The unit test injects a fake retriever and never requires
live services. At process startup, resolve the repository root from `Path(__file__)`, prepend the
absolute `backend` directory to `sys.path`, and only then import `app.*`; this keeps the documented
root-level command reproducible without relying on ambient `PYTHONPATH`.

- [ ] **Step 6: Add opt-in end-to-end and failure-injection tests**

The integration test requires PostgreSQL, Qdrant, ClickHouse, Embedding, Reranker, and a fake Physoc
SSE server. It publishes narrative and spreadsheet fixtures, verifies cited document QA and exact
ClickHouse averages, then runs these failures one at a time:

```text
Embedding unavailable -> Legacy fallback
Reranker unavailable -> Legacy fallback
Qdrant timeout -> Legacy fallback
Qdrant Alias dimension mismatch -> readiness failure and Legacy fallback
ClickHouse unavailable for aggregate -> explicit structured unavailable answer
Physoc 502/interrupted SSE -> explicit model unavailable response, no raw chunks
Shadow queue full -> foreground Legacy response remains successful
```

Run only when `HYBRID_E2E=1`; otherwise SKIP.

- [ ] **Step 7: Run tests and verify they pass**

Run the Step 3 command.

Expected: PASS.

- [ ] **Step 8: Run the live acceptance commands**

```powershell
$env:HYBRID_E2E='1'
uv run --directory backend --group dev --group benchmark python -m unittest tests.integration.test_hybrid_retrieval_e2e -v
uv run --project backend --group benchmark python tools/hybrid_retrieval_benchmark.py --concurrency 15 --requests 150 --p95-seconds 5 --max-error-rate 0.01 --max-fallback-rate 0.01 --questions-jsonl artifacts/benchmarks/hybrid-questions.jsonl --output-json artifacts/benchmarks/hybrid-retrieval-report.json
```

Expected: E2E PASS and benchmark exit code 0. Preserve the JSON report as deployment evidence; do not
commit sensitive question text or generated reports.

- [ ] **Step 9: Commit evaluation and capacity gates**

```powershell
git add backend/app/evaluation.py backend/app/evaluation_batches.py backend/app/sql_repository.py backend/tests/test_quality_evaluation.py backend/tests/test_evaluation_batches.py backend/tests/test_rag_acceptance.py backend/tests/integration/test_hybrid_retrieval_e2e.py tools/hybrid_retrieval_benchmark.py tools/tests/test_hybrid_retrieval_benchmark.py tools/tests/test_benchmark_report.py
git commit -m "test: gate qwen3 retrieval quality and capacity"
```

### Task 12: Document operation, run full verification, and prepare Shadow rollout

**Files:**
- Modify: `README.md`
- Modify: `deploy/offline/README.md`
- Modify: `docs/superpowers/plans/2026-07-24-enterprise-knowledge-base-qa-rollout.md`
- Modify: `tools/tests/test_physoc_llm_contract.py`
- Modify: `tools/tests/test_structured_deployment_contract.py`

- [ ] **Step 1: Write failing documentation contract assertions**

Add assertions that active documentation contains:

```text
Qwen/Qwen3-Embedding-0.6B
Qwen/Qwen3-Reranker-0.6B
RETRIEVAL_MODE=legacy|shadow|qwen3
Qdrant Dense + Sparse/BM25 + RRF
ClickHouse complete-data aggregation
/api/physoc/deepseeks/stream
no raw-chunk answer on Physoc failure
Shadow 10 -> 50 -> 100 and canary 5 -> 25 -> 50 -> 100
Alias rollback and RETRIEVAL_MODE=legacy rollback
```

The contract also rejects statements that ordinary retrieval still uses BGE-M3 as the selected
production model or that spreadsheet averages are calculated from RAG chunks.

- [ ] **Step 2: Run documentation tests and verify they fail**

```powershell
uv run --project backend --group dev python -m unittest tools.tests.test_physoc_llm_contract tools.tests.test_structured_deployment_contract -v
```

Expected: FAIL because the documentation still describes the previous retrieval roadmap/model.

- [ ] **Step 3: Update operational documentation**

Document:

1. model artifact directory layout, metadata manifests, SHA-256 generation, and offline wheel bundle;
2. first full build without `--activate`, validation review, and explicit Alias activation;
3. Legacy, Shadow, and Qwen3 environment examples;
4. dashboard/log fields for latency, queue saturation, circuit state, and fallback;
5. exact rollback commands for mode, Alias, and model image;
6. spreadsheet routing and the rule forbidding chunk estimates;
7. Physoc endpoint configuration and the rule forbidding raw-chunk fallback;
8. CPU starting profile and the mandatory 15-user acceptance command.

Update the enterprise rollout document so its hybrid retrieval phase names Qwen3 Embedding/Reranker
and links to this implementation plan instead of retaining the earlier BGE choice.

- [ ] **Step 4: Run the complete deterministic test suite**

```powershell
uv run --directory backend --group dev --group offline --group benchmark python -m unittest discover -s tests -p "test_*.py" -v
uv run --project backend --group dev python -m unittest discover -s tools/tests -p "test_*.py" -v
```

Expected: PASS, with live-service integration tests skipped unless their explicit environment flags
are enabled.

- [ ] **Step 5: Run formatting, dependency, migration, and diff checks**

```powershell
uv lock --project backend --check
uv sync --project backend --no-dev --frozen --group offline
uv run --directory backend --no-dev --group offline python -c "from app.main import create_production_app; print('import-ok')"
ruff check backend/app backend/tests tools
ruff format --check backend/app backend/tests tools
git diff --check
```

Expected: all commands exit 0 and print `import-ok` for the import smoke test.

- [ ] **Step 6: Run the production-like model and retrieval probes**

With approved local artifacts and services running:

```powershell
Invoke-RestMethod http://127.0.0.1:8081/readyz
Invoke-RestMethod http://127.0.0.1:8082/readyz
Invoke-RestMethod http://127.0.0.1:8000/api/readyz
uv run --project backend --group benchmark python tools/hybrid_retrieval_benchmark.py --concurrency 15 --requests 150 --p95-seconds 5 --max-error-rate 0.01 --max-fallback-rate 0.01 --questions-jsonl artifacts/benchmarks/hybrid-questions.jsonl --output-json artifacts/benchmarks/hybrid-retrieval-report.json
```

Expected: model metadata matches pinned checksums, API readiness is 200, and the benchmark passes.

- [ ] **Step 7: Commit documentation and final verification changes**

```powershell
git add README.md deploy/offline/README.md docs/superpowers/plans/2026-07-24-enterprise-knowledge-base-qa-rollout.md tools/tests/test_physoc_llm_contract.py tools/tests/test_structured_deployment_contract.py
git commit -m "docs: add qwen3 hybrid retrieval operations"
```

## Rollout checklist after implementation

- [ ] Build `knowledge_chunks_qwen3_v1` without `--activate`.
- [ ] Verify point count, dimension, permission filters, sample queries, and model checksums.
- [ ] Activate the Alias while `RETRIEVAL_MODE=legacy`.
- [ ] Run Shadow at 10%, 50%, and 100%; review quality and capacity reports at every stage.
- [ ] Run Qwen3 canary at 5%, 25%, 50%, and 100% using stable conversation hashing.
- [ ] Stop promotion immediately if Recall@50, NDCG@8, critical Top-8, permission, error, fallback, or P95 gates fail.
- [ ] Roll back with `RETRIEVAL_MODE=legacy`; switch the Qdrant Alias or model image independently only when that layer is the cause.

## Completion definition

The feature is complete only when all deterministic tests pass, the internal deployment starts with
outbound model access disabled, a versioned Qdrant Alias is active, structured aggregates remain exact,
Physoc failures never return raw chunks, Shadow comparisons contain no raw sensitive text, permission
leakage is zero, and the approved dataset passes 15-user retrieval P95 <= 5 seconds.
