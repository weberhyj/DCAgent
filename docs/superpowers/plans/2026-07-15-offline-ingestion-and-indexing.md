# Offline Ingestion and Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, bounded-memory ingestion pipeline that publishes matching ClickHouse and Qdrant versions through one PostgreSQL manifest while preserving the legacy path until cutover.

**Architecture:** PostgreSQL is the only durable job state and publication pointer; Redis only wakes workers. Parsers write bounded artifacts through `SemanticSink`, unchanged text reuses a versioned dense-vector cache, and ingestion plus query call the same private Embedding service. ClickHouse tables and Qdrant collections remain immutable until one complete manifest validates and publishes.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/Alembic, PostgreSQL, Redis, openpyxl, pyxlsb, LibreOffice, Polars/PyArrow, ClickHouse, Docling, PaddleOCR, Jieba, Qdrant, private Embedding service, unittest.

---

## Canonical file ownership

- `backend/app/publication_models.py`: the only `PublicationManifest` and `PinnedPublication` definition.
- `backend/app/ingestion_models.py`: batches, jobs, claims, checkpoints, failures, and summaries.
- `backend/app/semantic_records.py`: `SemanticRecord`, `SemanticSink`, and `SemanticSource`.
- `backend/app/embedding_client.py`: the only dense encoder used by workers and queries.
- `backend/tests/support/offline_fakes.py`: all reusable fakes named below, so no test depends on an undefined helper.

### Task 1: Add canonical control-plane and publication models

**Files:**
- Create: `backend/app/publication_models.py`
- Create: `backend/app/ingestion_models.py`
- Create: `backend/alembic/versions/20260715_01_ingestion_control_plane.py`
- Create: `backend/tests/support/offline_fakes.py`
- Create: `backend/tests/test_ingestion_schema.py`
- Modify: `backend/app/database.py`

- [ ] **Step 1: Write failing schema and manifest tests**

```python
class IngestionSchemaTest(unittest.TestCase):
    def test_creates_jobs_dependencies_sources_and_publications(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        tables = set(inspect(database.engine).get_table_names())
        self.assertTrue({
            "ingestion_batches",
            "ingestion_jobs",
            "ingestion_job_dependencies",
            "source_versions",
            "publication_manifests",
            "publication_pointers",
        }.issubset(tables))

    def test_manifest_has_every_reproducibility_field(self) -> None:
        manifest = sample_manifest()
        self.assertEqual(manifest.embedding.dimensions, 768)
        self.assertEqual(manifest.sparse.bm25_k1, 1.2)
        self.assertEqual(manifest.retrieval.dense_limit, 50)
        self.assertEqual(manifest.permission_schema_sha256, "p" * 64)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_ingestion_schema -v
```

Expected: FAIL because the tables and canonical manifest do not exist.

- [ ] **Step 3: Implement the exact shared model**

```python
@dataclass(frozen=True, slots=True)
class ArtifactFileRef:
    key: str
    sha256: str

@dataclass(frozen=True, slots=True)
class EmbeddingArtifactRef:
    name: str
    version: str
    sha256: str
    dimensions: int
    normalized: bool
    encoding_profile_sha256: str
    protocol_version: str

@dataclass(frozen=True, slots=True)
class SparseArtifactRef:
    profile_sha256: str
    tokenizer_checksum: str
    dictionary_checksum: str
    stop_words_checksum: str
    vocabulary: ArtifactFileRef
    document_frequency: ArtifactFileRef
    token_counts: ArtifactFileRef
    coordinates: ArtifactFileRef
    corpus_documents: int
    average_document_length: float
    min_document_frequency: int
    max_document_frequency_ratio: float
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_k3: float = 0.0
    qdrant_idf_modifier: Literal["none"] = "none"

@dataclass(frozen=True, slots=True)
class RedactionArtifactRef:
    version: str
    policy_sha256: str
    normalization_sha256: str

@dataclass(frozen=True, slots=True)
class RerankerArtifactRef:
    name: str
    version: str
    sha256: str
    protocol_version: str
    scoring_profile_sha256: str
    candidate_limit: int = 20
    result_limit: int = 10

@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    dense_limit: int = 50
    sparse_limit: int = 50
    rrf_k: int = 60
    reranker: RerankerArtifactRef | None = None

@dataclass(frozen=True, slots=True)
class GenerationArtifactRef:
    name: str
    version: str
    sha256: str
    protocol_version: str
    context_tokens: int
    output_tokens: int

@dataclass(frozen=True, slots=True)
class PublicationManifest:
    manifest_id: str
    batch_id: str
    clickhouse_tables: Mapping[str, str]
    clickhouse_row_counts: Mapping[str, int]
    qdrant_collection: str
    qdrant_point_count: int
    embedding: EmbeddingArtifactRef
    sparse: SparseArtifactRef
    redaction: RedactionArtifactRef
    retrieval: RetrievalProfile
    generation: GenerationArtifactRef | None
    dense_vector_cache: ArtifactFileRef
    semantic_fingerprints: ArtifactFileRef
    permission_schema_sha256: str
    pipeline_profile_sha256: str

PinnedPublication = PublicationManifest
```

`EmbeddingArtifactRef` deliberately exposes the seven fields required by the Phase 1 `EmbeddingMetadataExpectation` protocol. Do not change the Phase 1 client signature or add parallel `expected_*` arguments; ingestion and query both call `embed(..., expected=manifest.embedding, purpose=...)`.

`20260715_01_ingestion_control_plane.py` has `down_revision = "20260715_00"`. `ingestion_models.py` adds `BatchStatus`, `JobStatus`, `ClaimedJob`, `JobCheckpoint`, `JobFailure`, and `IngestionSummary`. The migration starts from Phase 1 revision `20260715_00`, adds dependency edges, `lease_token`, `lease_expires_at`, `heartbeat_at`, `checkpoint_json`, `checkpoint_seq`, and `next_attempt_at`, and does not remove legacy vectors.

```python
@dataclass(frozen=True, slots=True)
class IngestionJob:
    id: str
    batch_id: str
    source_id: str
    job_type: str
    dedupe_key: str
    status: JobStatus
    attempt: int = 0
```

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_ingestion_schema -v
Set-Location ..
git add backend/app/publication_models.py backend/app/ingestion_models.py backend/app/database.py backend/alembic/versions/20260715_01_ingestion_control_plane.py backend/tests/support/offline_fakes.py backend/tests/test_ingestion_schema.py
git commit -m "feat: add canonical ingestion control plane"
```

### Task 2: Implement dependency-aware PostgreSQL job leasing

**Files:**
- Create: `backend/app/ingestion_repository.py`
- Create: `backend/tests/test_ingestion_repository.py`

- [ ] **Step 1: Write failing dependency, dedupe, and stale-token tests**

```python
class IngestionRepositoryTest(unittest.TestCase):
    def test_child_is_claimable_only_after_parent_completes(self) -> None:
        repository = SqlIngestionRepository.for_test()
        parent = repository.enqueue("b1", "s1", "parse", "parse:s1", ())
        child = repository.enqueue("b1", "s1", "index", "index:s1", (parent.id,))
        duplicate = repository.enqueue("b1", "s1", "index", "index:s1", (parent.id,))
        self.assertEqual(duplicate.id, child.id)
        claimed_parent = repository.claim_next("w1", 60)
        self.assertEqual(claimed_parent.id, parent.id)
        self.assertIsNone(repository.claim_next("w2", 60))
        repository.complete(parent.id, claimed_parent.lease_token, {})
        self.assertEqual(repository.claim_next("w2", 60).id, child.id)

    def test_expired_owner_cannot_checkpoint_after_reclaim(self) -> None:
        repository = SqlIngestionRepository.for_test()
        repository.enqueue("b1", "s1", "parse", "parse:s1", ())
        first = repository.claim_next("w1", 1)
        repository.force_lease_expiry(first.id)
        second = repository.claim_next("w2", 60)
        with self.assertRaises(LeaseLostError):
            repository.checkpoint(first.id, first.lease_token, JobCheckpoint(1, {"rows": 1}))
        with self.assertRaises(LeaseLostError):
            repository.complete(first.id, first.lease_token, {})
        with self.assertRaises(LeaseLostError):
            repository.fail(first.id, first.lease_token, JobFailure("stale", retryable=True))
        repository.checkpoint(second.id, second.lease_token, JobCheckpoint(1, {"rows": 1}))
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_ingestion_repository -v
```

Expected: FAIL because dependency rows and guarded lease operations do not exist.

- [ ] **Step 3: Implement the repository protocol**

```python
class IngestionRepository(Protocol):
    def enqueue(self, batch_id: str, source_id: str, job_type: str, dedupe_key: str, depends_on: tuple[str, ...]) -> IngestionJob: ...
    def claim_next(self, worker_id: str, lease_seconds: int) -> ClaimedJob | None: ...
    def heartbeat(self, job_id: str, lease_token: str, lease_seconds: int) -> None: ...
    def checkpoint(self, job_id: str, lease_token: str, checkpoint: JobCheckpoint) -> None: ...
    def complete(self, job_id: str, lease_token: str, artifact: Mapping[str, object]) -> None: ...
    def fail(self, job_id: str, lease_token: str, failure: JobFailure) -> None: ...
```

PostgreSQL claims with `FOR UPDATE SKIP LOCKED`, requires every dependency to be completed, creates a new lease token per claim, and increments attempt count. Every mutation filters by running status, unexpired lease, and exact token; affected rows other than one raise `LeaseLostError`. Expired jobs are reclaimable and excess attempts become quarantined. SQLite tests serialize the same transitions.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_ingestion_repository -v
Set-Location ..
git add backend/app/ingestion_repository.py backend/tests/test_ingestion_repository.py
git commit -m "feat: add durable dependency-aware jobs"
```

### Task 3: Add heartbeat/checkpoint execution and the production worker loop

**Files:**
- Create: `backend/app/job_execution.py`
- Create: `backend/app/ingestion_worker.py`
- Create: `backend/tests/test_ingestion_worker.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing heartbeat and Redis-loss tests**

```python
class IngestionWorkerTest(unittest.TestCase):
    def test_long_handler_renews_lease(self) -> None:
        repository = BlockingFakeRepository()
        worker = IngestionWorker(repository, {"parse": blocking_handler}, "w1", lease_seconds=1, heartbeat_interval=0.05)
        thread = Thread(target=worker.run_once)
        thread.start()
        repository.started.wait(1)
        time.sleep(1.4)
        self.assertIsNone(repository.claim_next("w2", 1))
        repository.release.set()
        thread.join(1)
        self.assertGreaterEqual(repository.heartbeat_count, 2)

    def test_second_worker_resumes_from_persisted_checkpoint(self) -> None:
        repository = CheckpointingFakeRepository.with_queued_job("parse")
        first = IngestionWorker(repository, {"parse": fail_after_checkpoint_one}, "w1", lease_seconds=1, heartbeat_interval=0.05)
        self.assertTrue(first.run_once())
        repository.make_retry_due()
        second = IngestionWorker(repository, {"parse": resume_from_checkpoint}, "w2", lease_seconds=1, heartbeat_interval=0.05)
        self.assertTrue(second.run_once())
        self.assertEqual(repository.handler_start_sequences, [0, 1])
        self.assertEqual(repository.checkpoints, [1, 2])
        self.assertFalse(repository.heartbeat_thread_alive)

    def test_postgres_polling_survives_redis_failure(self) -> None:
        repository = FakeIngestionRepository.with_queued_job("parse")
        worker = IngestionWorker(repository, {"parse": complete_handler}, "w1", wakeup=UnavailableRedisWakeup())
        self.assertTrue(worker.run_once())
        self.assertEqual(repository.job.status, "completed")
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_ingestion_worker -v
```

Expected: FAIL because the execution context and runnable module do not exist.

- [ ] **Step 3: Implement heartbeat and safe shutdown**

```python
class JobExecutionContext:
    def __enter__(self) -> "JobExecutionContext":
        self._thread = Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        return self

    def checkpoint(self, checkpoint: JobCheckpoint) -> None:
        self.raise_if_lease_lost()
        self.repository.checkpoint(self.job.id, self.job.lease_token, checkpoint)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join(timeout=self.heartbeat_interval * 2)

    def raise_if_lease_lost(self) -> None:
        if self._lease_lost.is_set():
            raise LeaseLostError(self.job.id)
```

The heartbeat runs at one-third of the lease. A handler is `Callable[[ClaimedJob, JobExecutionContext], Mapping[str, object] | Awaitable[Mapping[str, object]]]`, receives the last persisted checkpoint after a reclaim/restart, and checkpoints each bounded batch. The dedicated worker process runs awaitable handlers with one event loop per claimed job, so async Embedding/Qdrant operations do not block API workers. `__exit__` always stops and joins the heartbeat thread. `run_forever(stop_event)` polls PostgreSQL even when Redis is empty/unavailable, stops claiming on SIGTERM, and finishes or checkpoints the active job. `build_production_worker()` lazily creates repository, dispatchers, and clients; `python -m app.ingestion_worker` invokes it.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_ingestion_worker tests.test_ingestion_repository -v
Set-Location ..
git add backend/app/job_execution.py backend/app/ingestion_worker.py backend/app/main.py backend/tests/test_ingestion_worker.py
git commit -m "feat: add heartbeat-aware ingestion worker"
```

### Task 4: Stream uploads to quarantine before registering durable scan work

**Files:**
- Modify: `backend/app/storage.py`
- Modify: `backend/app/routes.py`
- Create: `backend/app/file_scanner.py`
- Modify: `backend/app/ingestion_repository.py`
- Create: `backend/tests/test_streaming_knowledge_upload.py`

- [ ] **Step 1: Write failing ordering tests**

```python
class StreamingUploadTest(unittest.IsolatedAsyncioTestCase):
    async def test_closed_staged_file_precedes_source_and_job_registration(self) -> None:
        storage = RecordingStorage()
        repository = RecordingIngestionRepository()
        receipt = await register_upload(FakeUploadFile("a.xlsx", [b"abc", b"def"]), storage, repository)
        self.assertEqual(storage.events, ["write", "close", "rename"])
        self.assertEqual(repository.events, ["source_version", "scan_upload"])
        self.assertEqual(receipt.size, 6)
        self.assertEqual(storage.raw_files, [])

    async def test_partial_write_creates_no_database_state(self) -> None:
        repository = RecordingIngestionRepository()
        with self.assertRaises(OSError):
            await register_upload(FailingUploadFile(), FailingStorage(), repository)
        self.assertEqual(repository.events, [])

    async def test_rejects_extension_mime_size_and_archive_expansion_before_parse(self) -> None:
        for upload in (
            FakeUploadFile("bad.exe", [b"MZ"], mime="application/octet-stream"),
            FakeUploadFile("fake.pdf", [b"MZ"], mime="application/x-dosexec"),
            FakeUploadFile("huge.pdf", [b"x" * 11], mime="application/pdf"),
            FakeArchiveUpload(expanded_bytes=1_000_000_000),
        ):
            with self.assertRaises(UploadRejected):
                await register_upload(upload, LimitedStorage(max_bytes=10, max_expanded_bytes=100), RecordingIngestionRepository())
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_streaming_knowledge_upload -v
```

Expected: FAIL because bounded quarantine and registration do not exist.

- [ ] **Step 3: Implement the upload transaction**

Read exactly 1 MiB per `UploadFile.read(size)`, hash while writing, enforce the configured byte ceiling, fsync, close, and atomically rename the staging file. Before durable registration, validate the extension allowlist, detected MIME versus declared MIME, archive member count/path traversal/compression ratio/expanded-byte ceiling, and other checks that do not require ClamAV. After the closed quarantine file exists, create a source version in `pending_scan` state and a durable `scan_upload` job in one PostgreSQL transaction. The dedupe key contains source ID, content hash, and pipeline-profile hash. The scan job performs ClamAV: clean moves the file to raw storage and creates parse children; timeout records a retryable checkpoint/failure on the existing scan job; infected or policy-rejected content marks the source quarantined and never creates parse/index jobs. Failed writes remove the partial file and create no database state.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_streaming_knowledge_upload tests.test_knowledge_upload tests.test_api_contract -v
Set-Location ..
git add backend/app/storage.py backend/app/routes.py backend/app/file_scanner.py backend/app/ingestion_repository.py backend/tests/test_streaming_knowledge_upload.py
git commit -m "feat: stage and scan durable uploads"
```

### Task 5: Define bounded semantic records and redaction

**Files:**
- Create: `backend/app/semantic_records.py`
- Create: `backend/app/redaction.py`
- Create: `backend/config/redaction_policy.json`
- Modify: `backend/app/job_execution.py`
- Modify: `backend/app/ingestion_repository.py`
- Create: `backend/tests/test_semantic_records.py`

- [ ] **Step 1: Write failing bounded/redaction tests**

```python
class SemanticRecordTest(unittest.TestCase):
    def test_summary_does_not_materialize_records(self) -> None:
        sink = RecordingSemanticSink(batch_rows=2)
        summary = write_semantic_batches(
            [raw_record("s1", "phone 13800138000")],
            sink,
            RedactionPolicy.from_mapping({"phone": "mask"}),
        )
        self.assertFalse(hasattr(summary, "semantic_records"))
        self.assertNotIn("13800138000", "".join(sink.texts))
        self.assertLessEqual(sink.max_batch_size, 2)

    def test_redaction_failure_is_fail_closed_and_payload_is_allowlisted(self) -> None:
        with self.assertRaises(RedactionError):
            build_semantic_record(
                raw_record("s2", "secret"),
                RedactionPolicy.from_mapping({"required": ["missing"]}),
            )
        record = build_semantic_record(
            raw_record("s3", "phone 13800138000", payload={"department_id": "d1", "secret": "x"}),
            RedactionPolicy.from_mapping({"phone": "mask", "payloadAllowlist": ["department_id"]}),
        )
        self.assertNotIn("13800138000", record.text)
        self.assertEqual(record.payload, {"department_id": "d1"})

    def test_job_handler_persists_non_retryable_quarantine(self) -> None:
        repository = RecordingIngestionRepository()
        result = SemanticJobHandler(
            repository,
            builder=FailingSemanticRecordBuilder(RedactionError("policy mismatch")),
        ).run(sample_claimed_job(), RecordingJobExecutionContext())
        self.assertEqual(result.failure.code, "redaction_failed")
        self.assertFalse(result.failure.retryable)
        self.assertEqual(repository.source_status("source-1"), "quarantined")
        self.assertEqual(repository.job_status("job-1"), "failed")
        self.assertEqual(repository.created_child_jobs, [])
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_semantic_records -v
```

Expected: FAIL because semantic sink/source and redaction do not exist.

- [ ] **Step 3: Implement the contracts**

```python
@dataclass(frozen=True, slots=True)
class SemanticRecord:
    semantic_id: str
    source_id: str
    source_version_id: str
    dataset: str
    business_key: str | None
    text: str
    text_sha256: str
    payload: Mapping[str, object]
    payload_sha256: str

class SemanticSink(Protocol):
    def write_batch(self, records: Sequence[SemanticRecord]) -> None: ...
    def close(self) -> ArtifactFileRef: ...

class SemanticSource(Protocol):
    def iter_batches(self, max_rows: int) -> Iterator[Sequence[SemanticRecord]]: ...
```

The order is compose text → `SemanticRecordBuilder(policy)` deterministic redaction → normalization → hashes → sink. Excel and document parsers receive the same builder instance; they cannot construct a `SemanticRecord` directly from raw text. At the job-handler boundary, `RedactionError` atomically marks the source version `quarantined` and the current job `failed` with `JobFailure(code="redaction_failed", retryable=False)`; no child dependency becomes claimable and the handler never falls back to raw text. Excel IDs use dataset/business key; document IDs use source/logical location/chunk ordinal. Only allowlisted payload fields reach semantic Parquet, checkpoints, Embedding, cache, logs, or Qdrant.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_semantic_records -v
Set-Location ..
git add backend/app/semantic_records.py backend/app/redaction.py backend/config/redaction_policy.json backend/app/job_execution.py backend/app/ingestion_repository.py backend/tests/test_semantic_records.py
git commit -m "feat: add bounded redacted semantic artifacts"
```

### Task 6: Stream Excel into Parquet and `SemanticSink`

**Files:**
- Create: `backend/app/excel_schema.py`
- Create: `backend/app/excel_ingestion.py`
- Create: `backend/tests/test_excel_ingestion.py`

- [ ] **Step 1: Write the failing single-pass test**

```python
class ExcelIngestionTest(unittest.TestCase):
    def test_writes_bounded_batches_and_selected_semantic_text(self) -> None:
        schema = DatasetSchema("sales", ("order_id",), ("comment",), ("department_id", "classification"), "order_date")
        parquet_sink = RecordingParquetSink()
        semantic_sink = RecordingSemanticSink(batch_rows=2)
        execution = RecordingJobExecutionContext()
        result = ExcelIngestionService(
            batch_rows=2,
            semantic_builder=SemanticRecordBuilder(sample_redaction_policy()),
        ).ingest_rows(
            SinglePassRows(sample_sales_rows()),
            schema,
            parquet_sink,
            semantic_sink,
            execution=execution,
        )
        self.assertEqual(parquet_sink.batch_sizes, [2, 1])
        self.assertEqual(semantic_sink.business_keys, ["1", "3"])
        self.assertEqual(result.semantic_records_written, 2)
        self.assertFalse(hasattr(result, "semantic_records"))
        self.assertEqual(execution.checkpoint_rows, [2, 3])
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_excel_ingestion -v
```

Expected: FAIL because the sink-based Excel API does not exist.

- [ ] **Step 3: Implement bounded readers**

Use openpyxl read-only/data-only for XLSX/XLSM, pyxlsb iteration for XLSB, and bounded LibreOffice conversion for XLS. Validate schema/business keys/formula cache before publication. `ingest_rows(..., execution: JobExecutionContext)` writes Arrow batches through one `ParquetWriter`, passes semantic batches directly to `SemanticSink.write_batch()`, and returns counts/artifact references only. Every bounded batch calls `execution.checkpoint()` with source offset, worksheet, emitted-row count, and current artifact part so a reclaimed worker resumes without replaying the whole workbook.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_excel_ingestion -v
Set-Location ..
git add backend/app/excel_schema.py backend/app/excel_ingestion.py backend/tests/test_excel_ingestion.py
git commit -m "feat: stream Excel into bounded artifacts"
```

### Task 7: Add immutable ClickHouse staging

**Files:**
- Create: `backend/app/clickhouse_gateway.py`
- Create: `backend/tests/test_clickhouse_gateway.py`
- Create: `backend/tests/integration/test_clickhouse_integration.py`

- [ ] **Step 1: Write failing identifier/count tests**

```python
class ClickHouseGatewayTest(unittest.TestCase):
    def test_generates_server_owned_name_and_validates_counts(self) -> None:
        gateway = ClickHouseGateway(FakeClickHouseClient())
        execution = RecordingJobExecutionContext()
        table = gateway.create_staging_table("sales", "batch-20260715", execution=execution)
        self.assertEqual(table, "dc_sales_batch_20260715")
        gateway.validate(table, expected_rows=3, aggregate_checks={"amount": 60}, execution=execution)
        self.assertGreater(len(execution.checkpoints), 0)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_clickhouse_gateway -v
```

Expected: FAIL because the gateway does not exist.

- [ ] **Step 3: Implement guarded staging**

Generate identifiers only from `[a-z0-9_]`, parameterize values, import bounded Arrow/Parquet batches, and include batch ID, business keys, permission columns, source version, and partition keys. `create_staging_table(..., execution: JobExecutionContext)` checkpoints each imported batch; `validate(..., execution=...)` checkpoints each count/aggregate partition. Retries rebuild only the immutable attempt table. Validate rows, schema, duplicate keys, aggregates, and permission fields.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_clickhouse_gateway -v
$env:RUN_OFFLINE_INTEGRATION="1"
py -m unittest tests.integration.test_clickhouse_integration -v
Remove-Item Env:RUN_OFFLINE_INTEGRATION
Set-Location ..
git add backend/app/clickhouse_gateway.py backend/tests/test_clickhouse_gateway.py backend/tests/integration/test_clickhouse_integration.py
git commit -m "feat: add immutable ClickHouse staging"
```

### Task 8: Parse and structurally chunk documents

**Files:**
- Create: `backend/app/document_models.py`
- Create: `backend/app/document_parser.py`
- Create: `backend/app/office_conversion.py`
- Create: `backend/app/document_chunking.py`
- Create: `backend/tests/test_document_parser.py`
- Create: `backend/tests/test_document_chunking.py`
- Create: `backend/tests/test_document_parser_offline.py`

- [ ] **Step 1: Write failing provenance/chunk tests**

```python
class DocumentChunkingTest(unittest.TestCase):
    def test_preserves_page_and_parent_section(self) -> None:
        chunks = list(chunk_blocks([DocumentBlock("heading", "Policy", page=2, parent_id="p1")], 300, 800, 0.1))
        self.assertEqual(chunks[0].page, 2)
        self.assertEqual(chunks[0].parent_id, "p1")
        self.assertLessEqual(chunks[0].token_count, 800)

class OfflineDocumentParserTest(unittest.TestCase):
    def test_pdf_ppt_word_ocr_and_legacy_conversion_use_only_local_artifacts(self) -> None:
        with deny_all_network():
            execution = RecordingJobExecutionContext()
            results = parse_fixture_bundle(local_parser_runtime(), sample_redaction_policy(), execution=execution)
        self.assertEqual({item.kind for item in results}, {"pdf", "pptx", "docx", "scanned_pdf", "legacy_ppt"})
        self.assertGreater(len(execution.checkpoints), 0)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_document_parser tests.test_document_chunking tests.test_document_parser_offline -v
```

Expected: FAIL because the parser/chunker do not exist.

- [ ] **Step 3: Implement adapters and sink output**

Docling preserves PDF page/section/table, PPT slide/title/body/notes, Word hierarchy/list/table, and Markdown blocks. `parse_fixture_bundle(..., execution: JobExecutionContext)` checkpoints each file/page/slide batch before releasing it. PaddleOCR runs only below the text threshold and receives the Phase 1 local parser runtime; a socket/network guard test fails if any parser attempts a download. Legacy DOC/PPT convert locally. Chunk 300–800 tokens with 10–15% overlap, retain parent IDs, and write bounded records through the injected `SemanticRecordBuilder`; never return a whole document list.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_document_parser tests.test_document_chunking tests.test_document_parser_offline -v
Set-Location ..
git add backend/app/document_models.py backend/app/document_parser.py backend/app/office_conversion.py backend/app/document_chunking.py backend/tests/test_document_parser.py backend/tests/test_document_chunking.py backend/tests/test_document_parser_offline.py
git commit -m "feat: add structured document ingestion"
```

### Task 9: Use the shared private Embedding client

**Files:**
- Modify: `backend/app/embedding_client.py`
- Modify: `backend/app/ingestion_worker.py`
- Create: `backend/tests/test_ingestion_embedding.py`

- [ ] **Step 1: Write failing purpose/metadata tests**

```python
class IngestionEmbeddingTest(unittest.IsolatedAsyncioTestCase):
    async def test_batches_document_requests_and_checks_metadata(self) -> None:
        client = RecordingEmbeddingClient()
        execution = RecordingJobExecutionContext()
        vectors = await embed_records(
            client,
            [record("a"), record("b"), record("c")],
            sample_embedding_artifact(),
            2,
            execution=execution,
        )
        self.assertEqual(client.batch_sizes, [2, 1])
        self.assertEqual(client.purposes, ["document", "document"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(execution.checkpoint_rows, [2, 3])
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_ingestion_embedding -v
```

Expected: FAIL because ingestion is not wired to `EmbeddingClient`.

- [ ] **Step 3: Implement the shared call path**

The Phase 1 `EmbeddingClient` is an async protocol and `HttpEmbeddingClient` is its concrete production implementation. `embed_records(..., execution: JobExecutionContext)` calls `await client.embed(texts, purpose="document", expected=manifest.embedding)` in bounded batches and checkpoints each response batch. Reject name/version/SHA/dimensions/normalization/encoding-profile/protocol mismatch without retry; retry only connection/503 errors. Workers and API imports must not load FlagEmbedding or ONNX weights.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_ingestion_embedding -v
Set-Location ..
git add backend/app/embedding_client.py backend/app/ingestion_worker.py backend/tests/test_ingestion_embedding.py
git commit -m "feat: share the offline embedding service"
```

### Task 10: Detect changes and reuse dense vectors

**Files:**
- Create: `backend/app/row_changes.py`
- Create: `backend/app/dense_vector_cache.py`
- Create: `backend/app/semantic_fingerprint_index.py`
- Create: `backend/tests/test_row_changes.py`
- Create: `backend/tests/test_dense_vector_cache.py`
- Create: `backend/tests/test_incremental_dense_builder.py`

- [ ] **Step 1: Write failing classification/cache tests**

```python
class ChangeAndCacheTest(unittest.IsolatedAsyncioTestCase):
    def test_only_new_or_text_changed_rows_need_embedding(self) -> None:
        changes = RowChangeDetector().classify_batch(sample_current_records(), sample_previous_index())
        self.assertEqual([item.kind for item in changes], [
            ChangeKind.UNCHANGED,
            ChangeKind.PAYLOAD_CHANGED,
            ChangeKind.TEXT_CHANGED,
            ChangeKind.NEW,
        ])
        cache = SqliteDenseVectorCache.for_test()
        key = DenseCacheKey("model-a", "profile-a", "text-a")
        cache.put_many([DenseVectorEntry(key, (1.0, 0.0), "now")])
        self.assertIn(key, cache.get_many([key]))
        self.assertEqual(cache.get_many([DenseCacheKey("model-b", "profile-a", "text-a")]), {})

    async def test_previous_manifest_reuses_vectors_and_updates_payload(self) -> None:
        client = RecordingEmbeddingClient()
        builder = IncrementalDenseBuilder.open_previous(
            previous_manifest=sample_previous_manifest(),
            artifact_root=fixture_artifacts(),
            embedding_client=client,
        )
        execution = RecordingJobExecutionContext()
        result = await builder.build(sample_changes(), execution=execution)
        self.assertEqual(client.seen_semantic_ids, ["text-changed", "new"])
        self.assertEqual(result.reused_ids, ["unchanged", "payload-only"])
        self.assertIn("payload-only", result.payload_updates)
        self.assertTrue(result.fingerprints.sealed)
        self.assertTrue(result.dense_cache.sealed)
        self.assertGreater(len(execution.checkpoints), 0)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_row_changes tests.test_dense_vector_cache tests.test_incremental_dense_builder -v
```

Expected: FAIL because change classification and cache do not exist.

- [ ] **Step 3: Implement bounded hash indexes**

`IncrementalDenseBuilder.open_previous()` opens the exact fingerprint and dense-cache artifacts named by the pinned previous manifest and verifies their checksums. `build(..., execution: JobExecutionContext)` checkpoints every bounded lookup/Embedding/write batch. The new fingerprint artifact is a complete snapshot: it copy-forwards unchanged IDs, replaces changed/new IDs, updates payload hashes, omits deleted IDs, then seals a new checksum. The fingerprint index stores semantic ID, text hash, payload hash, and redaction-policy hash only.

`DenseCacheKey` is model SHA + encoding-profile SHA + text SHA. The production cache is partitioned into 256 SQLite shards by the first two hex characters of the key; bounded `get_many`/`put_many` touch only relevant shards, append changed vectors, and copy-forward referenced immutable entries without loading a full shard. `seal()` records a manifest over shard checksums. Unchanged and payload-only records reuse vectors; payload-only records still update the new Qdrant collection's permission payload. Model/profile changes produce cache misses by construction.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_row_changes tests.test_dense_vector_cache tests.test_incremental_dense_builder -v
Set-Location ..
git add backend/app/row_changes.py backend/app/dense_vector_cache.py backend/app/semantic_fingerprint_index.py backend/tests/test_row_changes.py backend/tests/test_dense_vector_cache.py backend/tests/test_incremental_dense_builder.py
git commit -m "feat: reuse unchanged dense vectors"
```

### Task 11: Build BM25 artifacts in two streaming passes

**Files:**
- Create: `backend/app/sparse_index.py`
- Create: `backend/app/token_count_cache.py`
- Create: `backend/tests/test_sparse_index.py`

- [ ] **Step 1: Write failing two-pass test**

```python
class SparseIndexTest(unittest.TestCase):
    def test_build_is_reproducible_without_fit_list(self) -> None:
        builder = SparseIndexBuilder(Bm25Config(1.2, 0.75, 0.0, min_df=1, max_df_ratio=1.0))
        first_execution = RecordingJobExecutionContext()
        second_execution = RecordingJobExecutionContext()
        first = builder.build(FakeSemanticSource(["alpha beta", "alpha"]), DiskTokenCountCache.for_test(), temp_dir("a"), 1, execution=first_execution)
        second = builder.build(FakeSemanticSource(["alpha beta", "alpha"]), DiskTokenCountCache.for_test(), temp_dir("b"), 1, execution=second_execution)
        self.assertEqual(first, second)
        self.assertEqual(first.corpus_documents, 2)
        self.assertEqual(first.qdrant_idf_modifier, "none")
        self.assertEqual(read_vocabulary(first.vocabulary), {"alpha": 0, "beta": 1})
        self.assertTrue(verify_artifact(first.coordinates))
        self.assertGreater(len(first_execution.checkpoints), 0)

    def test_omits_terms_outside_document_frequency_limits(self) -> None:
        artifact = SparseIndexBuilder(Bm25Config(1.2, 0.75, 0.0, min_df=2, max_df_ratio=0.8)).build(
            FakeSemanticSource(["common rare", "common", "common"]),
            DiskTokenCountCache.for_test(),
            temp_dir("filtered"),
            1,
            execution=RecordingJobExecutionContext(),
        )
        self.assertEqual(read_vocabulary(artifact.vocabulary), {})
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_sparse_index -v
```

Expected: FAIL because the streaming sparse builder does not exist.

- [ ] **Step 3: Implement exact reproducible scoring**

`SparseIndexBuilder.build(..., execution: JobExecutionContext)` checkpoints every bounded source/token-cache/coordinate batch. Pass one streams/reuses a partitioned on-disk SQLite token-count cache keyed by tokenizer checksum + text hash and writes `N`, `df`, lengths, and `avgdl`. Terms below `min_df` or above `max_df_ratio` are removed. Remaining terms receive deterministic integer IDs by normalized UTF-8 byte sort, independent of input order. Pass two writes a versioned coordinates artifact keyed by semantic ID plus final indices/values; Qdrant indexing streams this artifact rather than recomputing scores.

Use versioned Jieba dictionary/stop words, document weight `tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))`, query IDF `ln(1+(N-df+0.5)/(df+0.5))`, `k1=1.2`, `b=0.75`, `k3=0`, and Qdrant IDF `none`. Persist vocabulary, DF, token counts, coordinates, thresholds, and every checksum in `SparseArtifactRef`. Compute `profile_sha256` from the canonical JSON serialization of all sparse fields except `profile_sha256` itself; verification reconstructs that same projection. It is the only sparse cache-key identity used by Phase 3.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_sparse_index -v
Set-Location ..
git add backend/app/sparse_index.py backend/app/token_count_cache.py backend/tests/test_sparse_index.py
git commit -m "feat: build streaming BM25 artifacts"
```

### Task 12: Stream versioned points into Qdrant

**Files:**
- Create: `backend/app/qdrant_gateway.py`
- Create: `backend/app/qdrant_index_builder.py`
- Create: `backend/tests/test_qdrant_gateway.py`
- Create: `backend/tests/test_qdrant_index_builder.py`
- Create: `backend/tests/integration/test_qdrant_integration.py`

- [ ] **Step 1: Write failing bounded-upsert test**

```python
class QdrantIndexBuilderTest(unittest.IsolatedAsyncioTestCase):
    async def test_batches_points_and_preserves_metadata(self) -> None:
        client = RecordingQdrantClient()
        execution = RecordingJobExecutionContext()
        await QdrantIndexBuilder(
            client=client,
            dense_artifact=sample_sealed_dense_vector_artifact(),
            sparse_artifact=sample_manifest().sparse,
            batch_points=2,
        ).build(
            FakeSemanticSource(["a", "b", "c"]),
            sample_manifest(),
            execution=execution,
        )
        self.assertLessEqual(max(client.upsert_batch_sizes), 2)
        self.assertTrue(all(point.payload["redaction_version"] == "r1" for point in client.points))
        self.assertTrue(all(point.payload["embedding_model_name"] == "bge-test" for point in client.points))
        self.assertTrue(all(point.payload["embedding_model_version"] == "1" for point in client.points))
        self.assertTrue(all(point.payload["embedding_normalized"] is True for point in client.points))
        self.assertTrue(all(point.payload["embedding_generated_at"] for point in client.points))
        self.assertEqual(client.idf_modifier, "none")
        self.assertGreater(len(execution.checkpoints), 0)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_qdrant_gateway tests.test_qdrant_index_builder -v
```

Expected: FAIL because the gateway/index builder do not exist.

- [ ] **Step 3: Implement collection and builder**

Create immutable collections with named `dense` and `bm25` vectors, on-disk vectors, deterministic point IDs, and payload indexes for permissions/provenance. The async builder receives a sealed dense-vector artifact produced by `build_dense_vectors` and the exact `SparseArtifactRef`; it rejects an unsealed artifact, verifies dense and coordinates checksums, and never calls Embedding or mutates the dense cache. It streams semantic batches, joins dense vectors and sparse coordinates by semantic ID, and upserts bounded batches. Every point stores the pinned Embedding model name, version, checksum, normalization flag, encoding-profile checksum, and the vector entry's `generated_at`, plus redaction version and allowlisted provenance/permission payload. `build(..., execution: JobExecutionContext)` checkpoints after every bounded upsert. Validate count, dimensions, model/redaction/sparse checksums, filter behavior, payload allowlist, absence of raw sensitive text, and sample retrieval.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_qdrant_gateway tests.test_qdrant_index_builder -v
$env:RUN_OFFLINE_INTEGRATION="1"
py -m unittest tests.integration.test_qdrant_integration -v
Remove-Item Env:RUN_OFFLINE_INTEGRATION
Set-Location ..
git add backend/app/qdrant_gateway.py backend/app/qdrant_index_builder.py backend/tests/test_qdrant_gateway.py backend/tests/test_qdrant_index_builder.py backend/tests/integration/test_qdrant_integration.py
git commit -m "feat: stream immutable Qdrant indexes"
```

### Task 13: Validate and publish one complete manifest

**Files:**
- Create: `backend/app/publication.py`
- Create: `backend/tests/test_publication.py`

- [ ] **Step 1: Write failing mismatch/round-trip tests**

```python
class PublicationServiceTest(unittest.TestCase):
    def test_mismatch_preserves_old_pointer(self) -> None:
        repository = FakePublicationRepository(active_manifest_id="old")
        with self.assertRaises(PublicationError):
            PublicationService(repository).publish(mixed_batch_candidate())
        self.assertEqual(repository.active_manifest_id, "old")

    def test_pin_returns_exact_persisted_manifest(self) -> None:
        repository = FakePublicationRepository.with_manifest(sample_manifest())
        self.assertEqual(PublicationService(repository).pin("default"), sample_manifest())
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_publication -v
```

Expected: FAIL because complete candidate validation/persistence do not exist.

- [ ] **Step 3: Implement atomic pointer publication**

Validate matching batch IDs, row/point counts, Embedding identity/dimensions/normalization, tokenizer/dictionary/stop-word/vocabulary/document-frequency/token-count/coordinates checksums, min/max document-frequency thresholds, BM25 constants, reconstructed sparse `profile_sha256`, redaction and permission checksums, sealed vector/fingerprint artifacts, optional reranker metadata, and optional generation model name/version/SHA/protocol/context/output limits from the artifact lock. Insert the immutable manifest and update the PostgreSQL pointer in one transaction. `pin(publication_scope="default")` returns the canonical type without field loss.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_publication -v
Set-Location ..
git add backend/app/publication.py backend/tests/test_publication.py
git commit -m "feat: publish complete offline manifests"
```

### Task 14: Wire the durable DAG behind migration flags

**Files:**
- Modify: `deploy/docker/worker.Dockerfile`
- Modify: `backend/app/ingestion.py`
- Modify: `backend/app/ingestion_worker.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/tests/test_knowledge_ingestion_pipeline.py`
- Create: `backend/tests/integration/test_offline_ingestion_pipeline.py`

- [ ] **Step 1: Write failing end-to-end recovery test**

```python
class OfflinePipelineTest(unittest.TestCase):
    def test_redis_loss_still_publishes_one_matching_manifest(self) -> None:
        app = build_offline_test_app()
        upload_fixture_batch(app)
        run_worker_until_idle(app, clear_redis_after_first_job=True)
        manifest = app.state.publication_service.pin("default")
        self.assertEqual(manifest.batch_id, app.state.clickhouse.last_batch_id)
        self.assertEqual(manifest.batch_id, app.state.qdrant.last_batch_id)
        self.assertEqual(app.state.ingestion_repository.pending_job_count(), 0)

    def test_document_only_batch_copy_forwards_structured_snapshot(self) -> None:
        app = build_offline_test_app(previous_manifest=sample_manifest())
        upload_document_only_batch(app)
        run_worker_until_idle(app)
        manifest = app.state.publication_service.pin("default")
        self.assertEqual(manifest.batch_id, app.state.clickhouse.last_batch_id)
        self.assertEqual(
            manifest.clickhouse_row_counts,
            sample_manifest().clickhouse_row_counts,
        )
        self.assertNotEqual(manifest.clickhouse_tables, sample_manifest().clickhouse_tables)

    def test_structured_only_batch_copy_forwards_semantic_snapshot(self) -> None:
        app = build_offline_test_app(previous_manifest=sample_manifest())
        upload_structured_only_batch(app)
        run_worker_until_idle(app)
        manifest = app.state.publication_service.pin("default")
        self.assertEqual(manifest.batch_id, app.state.qdrant.last_batch_id)
        self.assertEqual(manifest.qdrant_point_count, sample_manifest().qdrant_point_count)
        self.assertNotEqual(manifest.qdrant_collection, sample_manifest().qdrant_collection)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.integration.test_offline_ingestion_pipeline -v
Set-Location ..
```

Expected: FAIL because the dependency graph is not wired.

- [ ] **Step 3: Persist this exact graph**

```text
scan_upload -> parse_excel | parse_document
all parse_excel jobs -> finalize_excel_artifacts
finalize_excel_artifacts -> import_clickhouse [when structured snapshot changed]
finalize_excel_artifacts -> copy_forward_clickhouse [when structured snapshot unchanged]
all parse_excel + parse_document jobs -> finalize_semantic_artifact
finalize_semantic_artifact -> detect_changes [when semantic snapshot changed]
detect_changes -> build_dense_vectors
finalize_semantic_artifact -> build_sparse_artifacts [when semantic snapshot changed]
build_dense_vectors + build_sparse_artifacts -> index_qdrant
finalize_semantic_artifact -> copy_forward_qdrant [when semantic snapshot unchanged]
import_clickhouse | copy_forward_clickhouse -> validate_clickhouse
index_qdrant -> validate_qdrant
copy_forward_qdrant -> validate_qdrant
validate_clickhouse + validate_qdrant -> publish_manifest
```

The batch row stores the complete expected source/version set. Fan-in jobs are claimable only when every expected parse/import dependency has a terminal success; one failed/quarantined source blocks validation/publication. `KnowledgeIngestionQueue` becomes a durable repository facade plus optional Redis wake-up. `INGESTION_MODE=legacy|durable|dual` and `OFFLINE_INDEXING_ENABLED` preserve rollback. Failed dependencies never make publication claimable, and every handler checkpoints through `JobExecutionContext`.

Every publication is a full logical snapshot. Incremental batches copy-forward unchanged rows/records and apply changed/new/deleted IDs. Exactly one ClickHouse branch (`import_clickhouse` or `copy_forward_clickhouse`) and exactly one Qdrant branch (`index_qdrant` or `copy_forward_qdrant`) is created for a batch; the unselected branch has no job row. If a batch changes documents only, `copy_forward_clickhouse` creates a new immutable ClickHouse physical version with the new batch ID and identical validated counts; if it changes structured data only, `copy_forward_qdrant` creates a new immutable collection from the previous semantic snapshot/vector cache. The system never substitutes an empty artifact for an unchanged engine. `build_dense_vectors` consumes the finalized semantic snapshot plus prior fingerprints, embeds only new/text-changed records, and seals the dense cache. `build_sparse_artifacts` consumes the same full semantic snapshot. `index_qdrant` depends on both sealed artifacts. Publication is claimable only after both new-batch physical versions validate. Update the worker image command to `python -m app.ingestion_worker` in this task before starting the `indexing` profile.

- [ ] **Step 4: Run regressions and commit**

```powershell
Set-Location backend
py -m unittest discover -s tests -p "test_*.py" -v
py -m unittest tests.integration.test_offline_ingestion_pipeline -v
Set-Location ..
docker compose --env-file deploy/offline/.env -f deploy/offline/compose.yaml --profile indexing up -d --wait ingestion-worker
git diff --check
git add deploy/docker/worker.Dockerfile backend/app/ingestion.py backend/app/ingestion_worker.py backend/app/main.py backend/app/routes.py backend/app/schemas.py backend/tests/test_knowledge_ingestion_pipeline.py backend/tests/integration/test_offline_ingestion_pipeline.py
git commit -m "feat: orchestrate durable offline ingestion"
```

### Task 15: Expose safe ingestion progress

**Files:**
- Modify: `backend/app/routes.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/ingestion_repository.py`
- Create: `backend/tests/test_ingestion_api.py`
- Modify: `admin-frontend/src/types/chat.ts`
- Modify: `admin-frontend/src/services/api.ts`
- Modify: `admin-frontend/src/composables/useChatKnowledgeManagement.ts`
- Modify: `admin-frontend/src/composables/useChatKnowledgeManagement.spec.ts`
- Modify: `admin-frontend/src/views/KnowledgeManagementPage.vue`
- Modify: `admin-frontend/src/views/__tests__/KnowledgeManagementPage.spec.ts`

- [ ] **Step 1: Write failing API/UI tests**

```python
class IngestionApiTest(unittest.TestCase):
    def test_progress_hides_paths_and_lease_tokens(self) -> None:
        payload = TestClient(build_ingestion_test_app()).get("/api/admin/ingestion/batches").json()[0]
        self.assertEqual(payload["status"], "running")
        self.assertNotIn("rawPath", payload)
        self.assertNotIn("leaseToken", payload)
```

```typescript
it('stops polling after a terminal batch state', async () => {
  const state = createKnowledgeManagementState(fakeApi(['running', 'published']))
  await state.startPolling()
  expect(state.batches[0].status).toBe('published')
  expect(state.polling).toBe(false)
})
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_ingestion_api -v
Set-Location ..\admin-frontend
npm.cmd run test:run -- src/composables/useChatKnowledgeManagement.spec.ts src/views/__tests__/KnowledgeManagementPage.spec.ts
Set-Location ..
```

Expected: FAIL because progress projection and client state do not exist.

- [ ] **Step 3: Implement bounded projection**

List at most 100 newest batches with status, counts, phase, timestamps, manifest ID, and sanitized errors. Poll only queued/running/validating states and stop on unmount. Never expose paths, lease data, prompts, raw evidence, or credentials.

- [ ] **Step 4: Run tests/build and commit**

```powershell
Set-Location backend
py -m unittest tests.test_ingestion_api -v
Set-Location ..\admin-frontend
npm.cmd run test:run
npm.cmd run build
Set-Location ..
git add backend/app/routes.py backend/app/schemas.py backend/app/ingestion_repository.py backend/tests/test_ingestion_api.py admin-frontend/src/types/chat.ts admin-frontend/src/services/api.ts admin-frontend/src/composables/useChatKnowledgeManagement.ts admin-frontend/src/composables/useChatKnowledgeManagement.spec.ts admin-frontend/src/views/KnowledgeManagementPage.vue admin-frontend/src/views/__tests__/KnowledgeManagementPage.spec.ts
git commit -m "feat: show durable ingestion progress"
```

## Phase 2 completion gate

- Alembic reaches `20260715_01`; API/worker start only after migrations.
- Uploads register a scan job only after a closed staged file exists.
- Lease tokens, heartbeat, checkpoint, dependency, retry, quarantine, and Redis-loss recovery tests pass.
- Excel and documents use bounded sinks; no `result.semantic_records` list exists.
- Redaction precedes semantic storage, Embedding, cache, logging, and Qdrant.
- Unchanged/payload-only records reuse vectors; changed/new records call the shared Embedding service.
- BM25 artifacts contain complete tokenizer/vocabulary/DF/parameter checksums.
- ClickHouse and Qdrant validate the same batch/permission schema.
- One canonical manifest round-trips without field loss and pins exact physical versions plus Embedding, optional reranker, and optional generation artifacts.
- Legacy ingestion and PostgreSQL JSON vectors remain available for rollback.
