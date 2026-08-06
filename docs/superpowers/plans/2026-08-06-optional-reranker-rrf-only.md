# Optional Reranker RRF-only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-supported `RERANKER_ENABLED=false` path that runs Dense + BM25 + RRF without creating, probing, or starting a Reranker service.

**Architecture:** Keep `RETRIEVAL_MODE` responsible for Legacy/Shadow/Qwen3 routing and add one orthogonal capability flag for secondary ranking. When disabled, settings omit Reranker URL/metadata, startup passes `None` into `HybridRetriever`, health checks omit the dependency, and deployment tools record the service as disabled. Existing enabled behavior, failure fallback, index fingerprints, permissions, adjacency, and structured ClickHouse queries remain unchanged.

**Tech Stack:** Python 3.12, FastAPI, dataclasses, httpx, Qdrant, Docker Compose, pytest, uv, Ruff.

---

## File map

- `backend/app/retrieval_settings.py`: parse and validate `RERANKER_ENABLED`; make Reranker configuration conditional.
- `backend/app/hybrid_retriever.py`: skip the Reranker stage while preserving deterministic RRF order.
- `backend/app/main.py`: create and own a Reranker client only when enabled.
- `backend/app/infra/health.py`: omit Reranker URL and metadata checks when disabled.
- `deploy/offline/compose.yaml`: propagate the flag and place `reranker-service` behind a profile.
- `deploy/offline/.env.example`: make RRF-only the documented deployment default and return eight final candidates.
- `tools/compose_smoke.py`: conditionally start/probe Reranker and report `disabled` when omitted.
- `tools/intranet_deployment_gate.py`: conditionally build/probe Reranker while retaining an auditable disabled step.
- `README.md`, `deploy/offline/README.md`, `docs/intranet-deployment-configuration.md`: document the Ollama-only RRF route and optional Reranker re-enable procedure.
- `backend/app/__init__.py`: bump backend version from `0.1.5` to `0.1.6`.

### Task 1: Make Reranker configuration optional

**Files:**
- Modify: `backend/tests/test_retrieval_settings.py`
- Modify: `backend/app/retrieval_settings.py`

- [ ] **Step 1: Write failing settings tests**

Add tests proving the disabled path does not require Reranker configuration and invalid booleans fail closed:

```python
def test_rrf_only_does_not_require_reranker_service_or_metadata(self) -> None:
    environ = private_hybrid_environment()
    environ["RERANKER_ENABLED"] = "false"
    for key in tuple(environ):
        if key == "RERANKER_SERVICE_URL" or key.startswith("RERANKER_MODEL_"):
            environ.pop(key)

    settings = RetrievalSettings.from_environ(environ)

    self.assertFalse(settings.reranker_enabled)
    self.assertIsNone(settings.reranker_service_url)
    self.assertIsNone(settings.reranker)


def test_reranker_remains_enabled_by_default(self) -> None:
    settings = RetrievalSettings.from_environ(private_hybrid_environment())

    self.assertTrue(settings.reranker_enabled)
    self.assertIsNotNone(settings.reranker_service_url)
    self.assertIsNotNone(settings.reranker)


def test_rejects_invalid_reranker_enabled_value(self) -> None:
    environ = private_hybrid_environment()
    environ["RERANKER_ENABLED"] = "sometimes"

    with self.assertRaisesRegex(ValueError, "RERANKER_ENABLED must be a boolean"):
        RetrievalSettings.from_environ(environ)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
uv run --project backend python -m pytest backend/tests/test_retrieval_settings.py -q
```

Expected: failures because `RetrievalSettings` has no `reranker_enabled` field and still requires the URL and metadata.

- [ ] **Step 3: Implement conditional settings parsing**

Add the field and parse it with the existing strict boolean helper:

```python
@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    mode: RetrievalMode
    reranker_enabled: bool
    # existing fields remain in their current order
```

In `from_environ`:

```python
try:
    reranker_enabled = parse_bool(environ.get("RERANKER_ENABLED"), default=True)
except ValueError as error:
    raise RetrievalSettingsError("RERANKER_ENABLED must be a boolean") from error
```

For non-Legacy modes, always parse Qdrant and Embedding URLs, then conditionally parse Reranker configuration:

```python
reranker_service_url: str | None = None
reranker: RerankerModelSettings | None = None
if reranker_enabled:
    reranker_service_url = require_private_url(
        environ.get("RERANKER_SERVICE_URL", "http://127.0.0.1:8082"),
        "reranker_service_url",
    )
    reranker = _reranker_metadata(environ)
```

Set `reranker_enabled` in both Legacy and Hybrid constructor returns. Legacy continues to expose `reranker=None` regardless of the flag because it does not build Hybrid resources.

- [ ] **Step 4: Run the settings tests and verify GREEN**

Run:

```powershell
uv run --project backend python -m pytest backend/tests/test_retrieval_settings.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit the settings change**

```powershell
git add backend/app/retrieval_settings.py backend/tests/test_retrieval_settings.py
git commit -m "feat: make reranker configuration optional"
```

### Task 2: Skip Reranker execution in HybridRetriever

**Files:**
- Modify: `backend/tests/test_hybrid_retriever.py`
- Modify: `backend/app/hybrid_retriever.py`

- [ ] **Step 1: Write a failing RRF-only retrieval test**

Extend the test helper so it can pass `None` for both Reranker dependencies, then add:

```python
def test_rrf_only_skips_reranker_and_preserves_fused_order(self) -> None:
    reranker = RecordingReranker()
    retriever = self.addCleanupFor(
        build_retriever(
            reranker=reranker,
            reranker_enabled=False,
            final_top_k=8,
        )
    )

    outcome = retriever.retrieve(request())

    self.assertEqual(reranker.batch_sizes, [])
    self.assertEqual(len(outcome.candidates), 8)
    self.assertTrue(all(item.rerank_score is None for item in outcome.candidates))
    self.assertEqual(outcome.stage_ms["reranker"], 0.0)
    self.assertEqual(
        [item.chunk_id for item in outcome.candidates],
        [item.chunk_id for item in sorted(outcome.candidates, key=lambda item: -item.rrf_score)],
    )
```

Add a constructor invariant test:

```python
def test_requires_reranker_and_metadata_to_be_enabled_together(self) -> None:
    with self.assertRaisesRegex(ValueError, "reranker and reranker_metadata"):
        HybridRetriever(
            embedding=RecordingEmbedding(),
            sparse=RecordingSparse(),
            gateway=RecordingGateway((), (), ()),
            reranker=None,
            embedding_metadata=EMBEDDING,
            reranker_metadata=RERANKER,
        )
```

- [ ] **Step 2: Run the test and verify RED**

```powershell
uv run --project backend python -m pytest backend/tests/test_hybrid_retriever.py -q
```

Expected: failure because the constructor currently requires a concrete Reranker and always executes `_rerank`.

- [ ] **Step 3: Implement optional Reranker execution**

Change the constructor types while keeping both keyword arguments required:

```python
reranker: PassageReranker | None,
reranker_metadata: RerankerMetadataExpectation | None,
```

Validate the pair and store it:

```python
if (reranker is None) != (reranker_metadata is None):
    raise ValueError("reranker and reranker_metadata must both be configured or both be None")
self.reranker = reranker
self._reranker_metadata = reranker_metadata
```

In `_retrieve`, bypass the stage without consuming the shared deadline:

```python
if self.reranker is None:
    reranked = fused
    stage_ms["reranker"] = 0.0
else:
    stage_started = self._monotonic()
    reranked = self._rerank(query, fused, deadline=deadline, generation=generation)
    stage_ms["reranker"] = _elapsed_ms(stage_started, self._monotonic())
```

At the start of `_rerank`, narrow the optional fields before invoking the client:

```python
if self.reranker is None or self._reranker_metadata is None:
    raise RuntimeError("reranker is disabled")
```

- [ ] **Step 4: Run HybridRetriever tests and verify GREEN**

```powershell
uv run --project backend python -m pytest backend/tests/test_hybrid_retriever.py -q
```

Expected: enabled, degraded, timeout, adjacency, lifecycle, and new RRF-only tests all pass.

- [ ] **Step 5: Commit the retriever change**

```powershell
git add backend/app/hybrid_retriever.py backend/tests/test_hybrid_retriever.py
git commit -m "feat: support rrf-only hybrid retrieval"
```

### Task 3: Avoid creating a Reranker client at application startup

**Files:**
- Modify: `backend/tests/test_lazy_startup.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write a failing startup resource test**

Add a test using the existing recording factory:

```python
def test_rrf_only_startup_does_not_create_reranker_client(self) -> None:
    environ = private_qwen_environment(RERANKER_ENABLED="false")
    for key in tuple(environ):
        if key == "RERANKER_SERVICE_URL" or key.startswith("RERANKER_MODEL_"):
            environ.pop(key)
    factory = RecordingRetrievalFactory()

    with TestClient(create_production_app(environment_override=environ, resource_factory=factory)):
        pass

    self.assertIn("embedding", factory.events)
    self.assertIn("gateway", factory.events)
    self.assertNotIn("reranker", factory.events)
```

Use the actual factory argument names already present in `test_lazy_startup.py`; preserve its existing database and repository fixtures.

- [ ] **Step 2: Run the startup test and verify RED**

```powershell
uv run --project backend python -m pytest backend/tests/test_lazy_startup.py -q
```

Expected: failure because `_configure_retrieval_runtime` always calls `create_reranker_client`.

- [ ] **Step 3: Build resources conditionally**

Update `_DefaultRetrievalResourceFactory.create_hybrid_retriever` to accept nullable dependencies and remove the unconditional Reranker assertion:

```python
assert settings.embedding is not None
if settings.reranker_enabled:
    assert settings.reranker is not None
return HybridRetriever(
    # existing dependencies
    reranker=dependencies.get("reranker"),  # type: ignore[arg-type]
    reranker_metadata=settings.reranker,
)
```

Update `_configure_retrieval_runtime`:

```python
reranker = None
if settings.reranker_enabled:
    reranker = own(factory.create_reranker_client(settings))  # type: ignore[attr-defined]
```

Pass `reranker` into the existing `create_hybrid_retriever` call. Keep router logging unchanged; it already reports `None` when metadata is absent.

- [ ] **Step 4: Run startup and retrieval tests**

```powershell
uv run --project backend python -m pytest backend/tests/test_lazy_startup.py backend/tests/test_hybrid_retriever.py -q
```

Expected: all selected tests pass and the enabled path still records the Reranker resource.

- [ ] **Step 5: Commit the startup change**

```powershell
git add backend/app/main.py backend/tests/test_lazy_startup.py
git commit -m "feat: skip disabled reranker startup"
```

### Task 4: Make API health checks Reranker-aware

**Files:**
- Modify: `backend/tests/test_infra_health.py`
- Modify: `backend/app/infra/health.py`

- [ ] **Step 1: Write failing health tests**

Add a test that removes every Reranker setting and asserts no probe is built:

```python
def test_qwen3_rrf_only_health_omits_reranker_dependency(self) -> None:
    environ = retrieval_health_environment("qwen3")
    environ["RERANKER_ENABLED"] = "false"
    for key in tuple(environ):
        if key == "RERANKER_SERVICE_URL" or key.startswith("RERANKER_MODEL_"):
            environ.pop(key)
    gateway = RetrievalHealthGateway()
    http_client = RetrievalHealthHttpClient(retrieval_health_responses())

    checks = build_dependency_checks(
        OfflineSettings.from_environ(environ),
        database=object(),
        environ=environ,
        http_client=http_client,
        retrieval_settings=RetrievalSettings.from_environ(environ),
        retrieval_gateway=gateway,
        retrieval_scope_provider=retrieval_scope_provider(gateway),
    )

    self.assertNotIn("reranker", {check.name for check in checks})
    self.assertNotIn("http://127.0.0.1:8082/v1/metadata", http_client.urls)
```

- [ ] **Step 2: Run the health tests and verify RED**

```powershell
uv run --project backend python -m pytest backend/tests/test_infra_health.py -q
```

Expected: failure because URL validation and metadata checks currently treat every Hybrid route as Reranker-enabled.

- [ ] **Step 3: Implement conditional health registration**

In both `validate_health_service_urls` and `build_dependency_checks`, derive:

```python
retrieval_enabled = (
    retrieval_settings is not None
    and retrieval_settings.mode is not RetrievalMode.LEGACY
)
reranker_enabled = retrieval_enabled and retrieval_settings.reranker_enabled
```

Only append `RERANKER_SERVICE_URL`, assert metadata, create `_reranker_metadata_check`, and include `reranker` in the Shadow degraded set when `reranker_enabled` is true. Qdrant and Embedding remain required whenever `retrieval_enabled` is true.

- [ ] **Step 4: Run health and startup tests**

```powershell
uv run --project backend python -m pytest backend/tests/test_infra_health.py backend/tests/test_lazy_startup.py -q
```

Expected: RRF-only omits Reranker, while enabled Qwen3 and Shadow metadata mismatch tests retain their existing behavior.

- [ ] **Step 5: Commit the health change**

```powershell
git add backend/app/infra/health.py backend/tests/test_infra_health.py
git commit -m "feat: omit disabled reranker health checks"
```

### Task 5: Make the Reranker container optional in Compose

**Files:**
- Modify: `deploy/offline/.env.example`
- Modify: `deploy/offline/compose.yaml`
- Modify: `tools/tests/test_compose_contract.py`

- [ ] **Step 1: Write failing Compose contract tests**

Add assertions equivalent to:

```python
def test_rrf_only_is_the_documented_default(self) -> None:
    env_text = ENV_EXAMPLE.read_text(encoding="utf-8")
    self.assertIn("RERANKER_ENABLED=false", env_text)
    self.assertIn("RETRIEVAL_FINAL_TOP_K=8", env_text)


def test_reranker_service_requires_explicit_profile(self) -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    self.assertEqual(compose["services"]["reranker-service"]["profiles"], ["reranker"])
    self.assertEqual(
        compose["services"]["api"]["environment"]["RERANKER_ENABLED"],
        "${RERANKER_ENABLED:-false}",
    )
```

- [ ] **Step 2: Run the Compose contract test and verify RED**

```powershell
uv run --project backend python -m pytest tools/tests/test_compose_contract.py -q
```

Expected: failure because the service has no profile and the flag is absent.

- [ ] **Step 3: Update Compose and the environment example**

Apply these values:

```env
RERANKER_ENABLED=false
RETRIEVAL_FINAL_TOP_K=8
```

Add to `reranker-service`:

```yaml
profiles: ["reranker"]
```

Pass this into both `api` and `ingestion-worker`:

```yaml
RERANKER_ENABLED: ${RERANKER_ENABLED:-false}
```

For every Reranker-only variable in Compose, including values passed into the optional `reranker-service`, replace interpolation that uses `:?` with empty defaults such as `${RERANKER_MODEL_NAME:-}`. Docker Compose may expand inactive-profile services during `config`, so `:?` would incorrectly make Reranker configuration mandatory in RRF-only mode. When the `reranker` profile is explicitly enabled, `backend/app/reranker_service.py` keeps strict runtime validation through `_required()` and its startup health checks, so incomplete configuration still fails before becoming healthy.

- [ ] **Step 4: Run Compose contract tests**

```powershell
uv run --project backend python -m pytest tools/tests/test_compose_contract.py tools/tests/test_structured_deployment_contract.py -q
```

Expected: tests pass and the worker still receives all indexing and structured-query settings.

- [ ] **Step 5: Commit Compose changes**

```powershell
git add deploy/offline/.env.example deploy/offline/compose.yaml tools/tests/test_compose_contract.py
git commit -m "feat: make reranker compose service optional"
```

### Task 6: Make smoke and deployment gates conditional

**Files:**
- Modify: `tools/compose_smoke.py`
- Modify: `tools/intranet_deployment_gate.py`
- Modify: `tools/tests/test_compose_smoke.py`
- Modify: `tools/tests/test_intranet_deployment_gate.py`

- [ ] **Step 1: Write failing smoke tests**

Add tests proving the disabled command does not start or exec into the Reranker:

```python
def test_rrf_only_up_omits_reranker_service(self) -> None:
    command = build_compose_command("up", reranker_enabled=False)

    self.assertIn("embedding-service", command)
    self.assertIn("api", command)
    self.assertNotIn("reranker-service", command)


def test_rrf_only_checks_report_reranker_disabled(self) -> None:
    checks = _checks(DEFAULT_WRAPPER_PATH, reranker_enabled=False)

    self.assertFalse(any(check.component == "reranker" for check in checks))
```

Add a gate test asserting `compose_build` omits `reranker-service`, `ollama_generate` is marked disabled, and the report can still pass.

- [ ] **Step 2: Run gate tests and verify RED**

```powershell
uv run --project backend python -m pytest tools/tests/test_compose_smoke.py tools/tests/test_intranet_deployment_gate.py -q
```

Expected: failures because current commands always include Reranker build, startup, adapter probes, and Ollama generation probe.

- [ ] **Step 3: Implement a shared strict boolean decision inside each tool**

In `compose_smoke.py`, add an internal reader that defaults to enabled for backward compatibility when the key is absent and rejects values outside `true/false`:

```python
def _read_env_bool(path: Path, key: str, *, default: bool) -> bool:
    value = _read_env_value(path, key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{key} must be true or false")
```

Thread `reranker_enabled` through `build_compose_command`, `_checks`, and `run_compose_smoke`. When disabled:

```python
ready_results["reranker"] = {"status": "disabled", "enabled": False}
```

Do not create fake adapter payloads and do not mark an unexecuted probe as passed.

In `intranet_deployment_gate.py`, derive the same boolean from `deploy/offline/.env`, build `_SERVICES` dynamically, omit the `/api/generate` command, and append this auditable step:

```python
{
    "category": "ollama_generate",
    "started_at": timestamp,
    "finished_at": timestamp,
    "exit_code": None,
    "duration_ms": 0,
    "sanitized_status": "disabled",
}
```

- [ ] **Step 4: Run smoke and gate tests**

```powershell
uv run --project backend python -m pytest tools/tests/test_compose_smoke.py tools/tests/test_intranet_deployment_gate.py tools/tests/test_offline_compose.py -q
```

Expected: disabled and enabled paths pass; reports remain sanitized and deterministic.

- [ ] **Step 5: Commit deployment tool changes**

```powershell
git add tools/compose_smoke.py tools/intranet_deployment_gate.py tools/tests/test_compose_smoke.py tools/tests/test_intranet_deployment_gate.py
git commit -m "feat: skip disabled reranker deployment gates"
```

### Task 7: Update documentation and backend version

**Files:**
- Modify: `README.md`
- Modify: `deploy/offline/README.md`
- Modify: `docs/intranet-deployment-configuration.md`
- Modify: `backend/app/__init__.py`
- Test: `tools/tests/test_backend_uv_contract.py`

- [ ] **Step 1: Add failing documentation/version assertions**

Extend the existing contract test to require the independent backend version:

```python
def test_backend_version_is_0_1_6(self) -> None:
    text = (REPO_ROOT / "backend" / "app" / "__init__.py").read_text(encoding="utf-8")
    self.assertIn('__version__ = "0.1.6"', text)
```

Where documentation contract assertions already exist, require the literal configuration and chain:

```python
self.assertIn("RERANKER_ENABLED=false", readme_text)
self.assertIn("Dense + BM25 + RRF", readme_text)
```

- [ ] **Step 2: Run the contract test and verify RED**

```powershell
uv run --project backend python -m pytest tools/tests/test_backend_uv_contract.py tools/tests/test_compose_contract.py -q
```

Expected: failure because the backend remains `0.1.5` and documentation still describes Qwen2.5 Reranker as mandatory.

- [ ] **Step 3: Update operator documentation**

Document the default route:

```env
RETRIEVAL_MODE=qwen3
RERANKER_ENABLED=false
RETRIEVAL_FINAL_TOP_K=8
OLLAMA_BASE_URL=http://ollama.inner:11434
OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
```

State explicitly that `kopens/bge-reranker-large:latest` returning 1024-dimensional `/api/embed` vectors is not a score-producing replacement. Move `qwen2.5:3b`, `/api/generate`, profile startup, digest, prompt profile, and capacity probes into an optional re-enable section. Update build/up examples so the default command does not include `reranker-service`.

Change only the backend version:

```python
__version__ = "0.1.6"
```

- [ ] **Step 4: Run documentation/version contracts**

```powershell
uv run --project backend python -m pytest tools/tests/test_backend_uv_contract.py tools/tests/test_compose_contract.py -q
```

Expected: all selected contracts pass.

- [ ] **Step 5: Commit documentation and version**

```powershell
git add README.md deploy/offline/README.md docs/intranet-deployment-configuration.md backend/app/__init__.py tools/tests/test_backend_uv_contract.py
git commit -m "docs: document ollama-only rrf deployment"
```

### Task 8: Run full verification

**Files:**
- Verify all modified files from Tasks 1-7.

- [ ] **Step 1: Run focused backend and deployment tests**

```powershell
uv run --project backend python -m pytest backend/tests/test_retrieval_settings.py backend/tests/test_hybrid_retriever.py backend/tests/test_lazy_startup.py backend/tests/test_infra_health.py tools/tests/test_compose_contract.py tools/tests/test_compose_smoke.py tools/tests/test_intranet_deployment_gate.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run Ruff**

```powershell
uv tool run ruff check backend/app backend/tests tools
uv tool run ruff format --check backend/app backend/tests tools
```

Expected: both commands exit 0 without modifying files.

- [ ] **Step 3: Run the complete backend and tools test suite**

```powershell
uv run --project backend python -m pytest backend/tests tools/tests -q
```

Expected: the suite passes, apart from any explicitly documented pre-existing failure that reproduces unchanged on commit `eb4fbbc`.

- [ ] **Step 4: Validate repository state and diff**

```powershell
git diff --check
git status --short
git log --oneline -8
```

Expected: no whitespace errors; only intended tracked changes or commits are present.

- [ ] **Step 5: Record target-server gates**

Do not claim the following as locally passed because this Windows machine has no Docker or target Ollama access:

```text
Ubuntu Compose config/build/up
API /api/readyz with RERANKER_ENABLED=false
Word document evidenceCount > 0
15-concurrent-user latency and error-rate acceptance
```

These remain mandatory target-server checks after the branch is deployed.
