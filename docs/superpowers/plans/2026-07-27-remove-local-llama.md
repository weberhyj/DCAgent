# Remove Local llama.cpp Generation Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the obsolete local llama.cpp/GGUF answer-generation route while preserving Physoc SSE, remote OpenAI-compatible providers, deterministic ClickHouse aggregation, indexing, embeddings, OCR, and the rest of the internal deployment.

**Architecture:** The production answer path remains the configured network provider, with Physoc `/api/physoc/deepseek/stream` as the deployment example. Local model service discovery, model-slot configuration, GGUF probing, and generation Compose wiring are deleted; generic domain fields named `generation` remain when they describe ClickHouse staging or answer-audit records rather than a local model.

**Tech Stack:** FastAPI/Pydantic, Python `unittest`, Docker Compose, PowerShell deployment wrapper, `uv`, Ruff, ClickHouse, Qdrant, Redis, Docling, PaddleOCR, Physoc SSE.

---

### Task 1: Add a failing contract test for the removal boundary

**Files:**
- Create: `tools/tests/test_no_local_llama_contract.py`
- Test: `tools/tests/test_no_local_llama_contract.py`

- [ ] **Step 1: Write the failing test**

Create a repository contract test that scans only active source/configuration/operational files, not historical specs or plans. Use exact forbidden patterns so normal domain uses of `generation` are allowed:

```python
from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ROOTS = (
    ROOT / "backend" / "app",
    ROOT / "backend" / "tests",
    ROOT / "deploy" / "offline",
    ROOT / "tools",
    ROOT / "README.md",
    ROOT / ".env.example",
    ROOT / "backend" / ".env.example",
    ROOT / "docs" / "offline-platform-runbook.md",
)
FORBIDDEN = (
    re.compile(r"\bllama(?:\.cpp)?\b", re.IGNORECASE),
    re.compile(r"\bGGUF\b", re.IGNORECASE),
    re.compile(r"\bgeneration-model\b", re.IGNORECASE),
    re.compile(r"\bLLAMA_[A-Z0-9_]+\b"),
    re.compile(r"\bMODEL_SLOTS\b"),
    re.compile(r"--profile\s+generation\b", re.IGNORECASE),
    re.compile(r"profiles:\s*\[[^\]]*['\"]generation['\"]", re.IGNORECASE),
)


def _active_files() -> list[Path]:
    files: list[Path] = []
    contract_test = Path(__file__).resolve()
    for root in ACTIVE_ROOTS:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.resolve() != contract_test
            )
    return sorted(set(files))


class NoLocalLlamaContractTests(unittest.TestCase):
    def test_active_tree_has_no_local_llama_contract(self) -> None:
        violations: list[str] = []
        for path in _active_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    violations.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
        self.assertEqual([], violations, "local-model contract remains:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```powershell
py -m unittest tools.tests.test_no_local_llama_contract -v
```

Expected: `FAIL`, listing the current `llama`, `GGUF`, `LLAMA_*`, `MODEL_SLOTS`, and generation-profile references.

- [ ] **Step 3: Commit the failing contract test**

```powershell
git add tools/tests/test_no_local_llama_contract.py
git commit -m "test: define local model removal contract"
```

### Task 2: Remove llama settings and health dependencies

**Files:**
- Modify: `backend/app/offline_settings.py`
- Modify: `backend/app/infra/health.py`
- Modify: `backend/tests/test_offline_settings.py`
- Modify: `backend/tests/test_lazy_startup.py`

- [ ] **Step 1: Update settings tests first**

Remove `LLAMA_SERVER_URL` and `MODEL_SLOTS` from private-environment fixtures. Change the expected settings assertion to verify the remaining fields (for example, `settings.embedding_service_url`) and delete tests for public model endpoints and slot-range validation. Keep tests for PostgreSQL, ClickHouse, Qdrant, Redis, Embedding, structured-query, and provider validation.

- [ ] **Step 2: Remove the settings fields and parsing**

In `OfflineSettings`, delete the `llama_server_url` and `model_slots` dataclass fields, remove `llama` from the private-host allowlist, remove the `llama_server_url` entry from `values`, delete the `MODEL_SLOTS` integer parsing/validation block, and remove `model_slots=model_slots` from the constructor. The resulting model still contains `model_root` because embedding/OCR artifacts use it:

```python
    embedding_service_url: str
    raw_data_root: Path
    parquet_root: Path
    structured_ingest_batch_rows: int
    model_root: Path
    dependency_timeout_seconds: float
```

- [ ] **Step 3: Remove generation health logic**

In `backend/app/infra/health.py`, stop importing `parse_bool`; delete `_generation_enabled`; remove the conditional `LLAMA_SERVER_URL` entry from `validate_health_service_urls`; and remove the conditional `DependencyCheck("llama", ...)` append from `build_dependency_checks`. Keep the production-provider checks and all existing non-model dependency checks unchanged.

- [ ] **Step 4: Update lazy-startup health tests**

Delete tests that assert template skips llama or that enabling generation adds a llama dependency. Remove `LLAMA_SERVER_URL` from environment fixtures and the invalid-public-model URL case. Add/retain an assertion that `build_dependency_checks(...)` contains only database, ClickHouse, Qdrant, Redis, ClamAV, and embedding checks for both Physoc and remote-provider environments.

- [ ] **Step 5: Run focused backend tests**

```powershell
uv run --project backend --frozen python -m unittest backend.tests.test_offline_settings backend.tests.test_lazy_startup backend.tests.test_infra_health -v
```

Expected: PASS.

- [ ] **Step 6: Commit the backend removal**

```powershell
git add backend/app/offline_settings.py backend/app/infra/health.py backend/tests/test_offline_settings.py backend/tests/test_lazy_startup.py
git commit -m "refactor: remove local model runtime settings"
```

### Task 3: Remove the llama Compose service and environment contract

**Files:**
- Modify: `deploy/offline/compose.yaml`
- Modify: `deploy/offline/.env.example`
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `tools/invoke_offline_compose.ps1`
- Modify: `tools/tests/test_compose_contract.py`
- Modify: `tools/tests/test_structured_deployment_contract.py`
- Modify: `tools/tests/test_compose_smoke.py`

- [ ] **Step 1: Delete local-model Compose wiring**

Remove `LLAMA_SERVER_URL` and `MODEL_SLOTS` from the API environment in `deploy/offline/compose.yaml`. Delete the complete `llama` service block, including its image, model command, `/models` bind, healthcheck, network, and `profiles: ["generation"]`. Keep `MODEL_ROOT` and the `physoc-egress` network unchanged.

- [ ] **Step 2: Remove local-model environment examples**

Delete `LLAMA_IMAGE`, `LLAMA_MODEL_FILE`, and `MODEL_SLOTS` from `deploy/offline/.env.example`; delete `LLAMA_SERVER_URL` and `MODEL_SLOTS` from the root and backend `.env.example` files. Keep `MODEL_ROOT`, embedding model variables, and all Physoc/OpenAI-compatible variables.

- [ ] **Step 3: Simplify the wrapper contract**

In `tools/invoke_offline_compose.ps1`, remove `"llama"` from the required rendered service allowlist and delete the `"llama" = ...` bind contract entry. Do not remove `/models` from the API/embedding bind validation because local embedding/OCR artifacts still use it. Keep rejection of unapproved services, profiles, host binds, and public Physoc egress.

- [ ] **Step 4: Update Compose and deployment tests**

Change service lists to end at `ingestion-worker` (or the existing default services), assert `generation` profile is absent, remove synthetic `llama` services from rendered-config fixtures, and delete the missing-generation-profile scenario. Update structured deployment assertions so only the `indexing` profile remains. In compose smoke tests, retain the existing assertion that worker services are not started by default but remove `llama` from the forbidden tuple if it is now covered by the new contract test.

- [ ] **Step 5: Run Compose/deployment tests**

```powershell
py -m unittest tools.tests.test_compose_contract tools.tests.test_structured_deployment_contract tools.tests.test_compose_smoke -v
```

Expected: PASS.

- [ ] **Step 6: Commit the Compose removal**

```powershell
git add deploy/offline/compose.yaml deploy/offline/.env.example .env.example backend/.env.example tools/invoke_offline_compose.ps1 tools/tests/test_compose_contract.py tools/tests/test_structured_deployment_contract.py tools/tests/test_compose_smoke.py
git commit -m "refactor: remove local model compose profile"
```

### Task 4: Reduce model-probe tooling to embedding and reranker candidates

**Files:**
- Modify: `tools/benchmarks/model_probe.py`
- Modify: `tools/tests/test_model_probe.py`

- [ ] **Step 1: Delete generation-only tests and fixtures**

Remove generation metrics fixtures, GGUF byte fixtures, generation loader-audit tests, llama service arguments, generation throughput tests, and tests that construct `generation-model` candidates. Keep embedding metadata, reranker latency, checksum stability, private-network, fixed-argv, cleanup, mutex, and report tests that exercise retained kinds.

- [ ] **Step 2: Remove generation candidate code**

Change the supported kinds declaration to:

```python
SUPPORTED_KINDS = ("embedding-model", "reranker-model")
```

Delete `GGUF_FILE_TYPES`, `_GgufBudget`, `_read_exact`, `_read_gguf_string`, `_read_gguf_value`, and `read_gguf_metadata`. In `load_candidate_artifact`, accept only embedding and reranker artifact metadata and reject any other kind with the existing unsupported-kind error. Delete `_generation_metrics` and remove the `generation-model` branch from `evaluate_candidate_gate` and the probe result assembly.

- [ ] **Step 3: Remove llama Compose injection and CLI coupling**

Delete `_build_candidate_override`'s `llama_service` parameter and local-model service payload; remove the generation-specific checks from `_validate_candidate_injection`; change `run_model_probe` to accept only the embedding service identifier; always render the default profile and start only the embedding service when needed. Remove `--llama-service` from `main()` and stop passing it into `run_model_probe`.

- [ ] **Step 4: Preserve retained probe behavior**

Keep the embedding and reranker branches, candidate checksum verification, private Docker context validation, service cleanup in `finally`, bounded metrics, atomic report writes, and fail-closed handling for missing/non-finite metrics. Ensure reranker probing still uses the embedding service's candidate injection contract exactly as before.

- [ ] **Step 5: Run model-probe tests**

```powershell
py -m unittest tools.tests.test_model_probe -v
```

Expected: PASS with no generation-model test names or fixtures remaining.

- [ ] **Step 6: Commit the model-probe removal**

```powershell
git add tools/benchmarks/model_probe.py tools/tests/test_model_probe.py
git commit -m "refactor: remove local generation model probes"
```

### Task 5: Remove llama-only capacity report data

**Files:**
- Modify: `tools/benchmarks/run_capacity_benchmark.py`
- Modify: `tools/tests/test_benchmark_report.py`
- Modify: `docs/offline-platform-runbook.md`

- [ ] **Step 1: Update benchmark report tests**

Remove `model_slots` from every `create_report(...)` call and delete the report assertion for `modelSlots`. Add an assertion that the serialized `profile` still contains `name` and `vectorDimensions` and does not contain `modelSlots`.

- [ ] **Step 2: Remove the field and CLI option**

Delete the `model_slots: int` parameter and its positive-integer validation from `create_report`. Build the report profile as:

```python
    profile = {
        "name": selected.name,
        "vectorDimensions": vector_dimension,
    }
```

Remove `--model-slots` from `_parser()` and stop passing it from `main()`. Leave concurrency, latency, error-rate, service-round-trip, and batch gates intact.

- [ ] **Step 3: Update capacity runbook commands**

Remove the `MODEL_SLOTS` table row and delete `--model-slots 2` from the benchmark command description. Explain that capacity reports now describe service round-trip and workload concurrency without local model slots.

- [ ] **Step 4: Run benchmark tests**

```powershell
py -m unittest tools.tests.test_benchmark_report -v
```

Expected: PASS.

- [ ] **Step 5: Commit the capacity change**

```powershell
git add tools/benchmarks/run_capacity_benchmark.py tools/tests/test_benchmark_report.py docs/offline-platform-runbook.md
git commit -m "refactor: remove local model capacity slots"
```

### Task 6: Update active documentation and deployment guidance

**Files:**
- Modify: `README.md`
- Modify: `deploy/offline/README.md`
- Modify: `docs/offline-platform-runbook.md`

- [ ] **Step 1: Replace Compose topology wording**

State that the private Compose topology contains PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, Embedding Service, API, and optional indexing worker. Remove references to a generation profile, local model artifact, llama service, GGUF, and model slots.

- [ ] **Step 2: Document the approved answer provider**

Add the production contract explicitly: ordinary questions retrieve bounded evidence and send a grounded prompt to the configured network provider; the Physoc deployment uses `LLM_PROVIDER=physoc_deepseek`, `LLM_API_BASE` set to the internal gateway, and `LLM_STREAM_PATH=/api/physoc/deepseek/stream`. Provider failure remains an explicit failure response; there is no local-model or retrieved-chunk fallback.

- [ ] **Step 3: Preserve local non-generation artifacts**

Keep `MODEL_ROOT` documentation for the Embedding Service, Docling, and PaddleOCR. Keep indexing and ClickHouse structured-publication instructions unchanged except where they mention the removed profile.

- [ ] **Step 4: Run documentation contract tests**

```powershell
py -m unittest tools.tests.test_structured_deployment_contract tools.tests.test_compose_smoke -v
```

Expected: PASS.

- [ ] **Step 5: Commit documentation updates**

```powershell
git add README.md deploy/offline/README.md docs/offline-platform-runbook.md
git commit -m "docs: document provider-only answer generation"
```

### Task 7: Verify the complete removal and preserved behavior

**Files:**
- Modify: none unless verification exposes a missed active reference.

- [ ] **Step 1: Run the complete Python regression suite**

```powershell
uv run --project backend --frozen python -m unittest discover -s backend/tests -v
py -m unittest discover -s tools/tests -v
```

Expected: PASS; Physoc SSE, structured query/worker, lazy startup, and deployment contract tests remain green.

- [ ] **Step 2: Validate runtime dependency installation and import**

In a clean temporary directory, run:

```powershell
uv sync --project backend --no-dev --frozen
uv run --project backend --no-dev --frozen python -c "import app.main; print('app.main import ok')"
```

Expected: the frozen runtime environment installs successfully and prints `app.main import ok` without requiring a local model package or service.

- [ ] **Step 3: Run static and repository checks**

```powershell
uv run --project backend --frozen ruff check backend
uv run --project backend --frozen ruff format --check backend
uv lock --check
git diff --check
py -m unittest tools.tests.test_no_local_llama_contract -v
```

Expected: all commands succeed; the contract test reports no active local-model references.

- [ ] **Step 4: Audit remaining `generation` references**

```powershell
rg -n -i 'generation' backend/app deploy/offline tools README.md docs/offline-platform-runbook.md
```

Review each match manually. Remaining matches must be ClickHouse staging generations, structured-publication attempts, answer-generation audit steps, or other provider-agnostic domain fields; they must not describe a local service, GGUF artifact, profile, environment variable, probe, or capacity slot.

- [ ] **Step 5: Commit the final verification-only adjustments**

If the audit finds an active missed reference, update the smallest responsible file, rerun the failing focused test and the contract test, then commit:

Run `git add` with the exact reviewed file paths, then run:

```powershell
git commit -m "chore: finish local model removal audit"
```

Do not add `docs/superpowers/specs/2026-07-21-dual-llm-runtime-routing-design.md`; it is an existing user file and must remain untracked.

---

## Self-review checklist

- Spec coverage: Tasks 1 through 3 remove active runtime/configuration contracts; Task 4 removes model probing; Task 5 removes slot-only capacity data; Task 6 updates active docs; Task 7 validates preserved Physoc, structured-query, indexing, and dependency behavior.
- Placeholder scan: every implementation step names exact files, symbols, commands, and expected outcomes; no unresolved markers or unspecified error-handling steps are used.
- Type consistency: `OfflineSettings` no longer exposes either local-model field; `create_report` no longer accepts `model_slots`; `run_model_probe` no longer accepts `llama_service`; all corresponding tests and callers are updated in the same tasks.
- Scope guard: `MODEL_ROOT`, Embedding Service, Docling, PaddleOCR, ClickHouse `generation` fields, and Physoc/OpenAI-compatible providers remain in scope and are explicitly preserved.
