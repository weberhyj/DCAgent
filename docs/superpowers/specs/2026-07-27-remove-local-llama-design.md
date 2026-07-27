# Remove Local llama.cpp Generation Route Design

## Context

DC-Agent is deployed inside the company network and sends grounded RAG prompts to the approved
Physoc endpoint at `/api/physoc/deepseek/stream`. The repository still contains an optional local
generation route built around llama.cpp, a GGUF model, a Compose `generation` profile, local-model
probe logic, and related configuration and documentation. That route is no longer required.

This change removes only the local llama.cpp/GGUF generation capability. It must not remove or
degrade the Physoc provider, the OpenAI-compatible remote provider, PostgreSQL persistence,
ClickHouse structured spreadsheet aggregation, the indexing worker, Qdrant, Redis, ClamAV, the
Embedding Service, Docling, or PaddleOCR.

## Decision

Physoc remains the configured production answer-generation route. The repository will no longer
ship, validate, benchmark, document, or expose a local llama.cpp generation service. There will be
no fallback from Physoc failure to a local model, template output, or retrieved chunks.

The word `generation` will not be removed indiscriminately. ClickHouse staging generations,
structured-publication attempts, answer-generation audit steps, and other domain concepts that are
unrelated to llama.cpp remain unchanged.

## Changes

### Compose and environment contract

- Remove the `llama` service and the `generation` profile from `deploy/offline/compose.yaml`.
- Remove `LLAMA_IMAGE`, `LLAMA_MODEL_FILE`, `LLAMA_SERVER_URL`, and `MODEL_SLOTS` from deployment
  environment examples and API container wiring.
- Remove llama-specific bind validation, service allowlists, and profile expectations from
  `tools/invoke_offline_compose.ps1` and its contract tests.
- Keep `MODEL_ROOT` because the local Embedding Service, Docling, and PaddleOCR still use it.
- Keep the private `physoc-egress` network and all Physoc environment variables unchanged.

### Backend runtime and health checks

- Remove `llama_server_url` and `model_slots` from `OfflineSettings` and update its tests.
- Remove `GENERATION_ENABLED` and `LLM_GENERATION_ENABLED` parsing from infrastructure health
  checks.
- Remove the conditional llama readiness check and llama URL validation.
- Keep the existing production-provider validation: production must use a real configured provider
  and must fail closed when Physoc is unavailable.

### Model-probe and capacity tooling

- Remove `generation-model` from the supported model-probe kinds.
- Remove GGUF parsing, Qwen-family/size/quantization validation, llama service overrides,
  tokenization checks, local completion throughput probes, and generation-model metrics.
- Remove the `--llama-service` command-line argument and all generation-model probe tests.
- Preserve embedding-model and reranker-model probing.
- Remove the local `modelSlots` capacity-report field and its CLI/config coupling because it only
  represented llama.cpp parallel slots. Preserve the concurrency, latency, error-rate, and
  service-round-trip capacity gates used by the remaining system.

### Documentation

- Remove llama.cpp, GGUF, `generation` profile, local-generation artifact, and model-slot
  instructions from README and deployment runbooks.
- State explicitly that answer generation uses a network-reachable approved provider, with Physoc
  as the production deployment example.
- Keep documentation for local embedding/OCR artifacts because those are outside this deletion
  scope.

## Preserved behavior

The following behavior must remain unchanged:

1. Ordinary document questions retrieve bounded evidence and send the complete grounded prompt to
   Physoc.
2. Missing evidence prevents a model call.
3. Physoc timeout, non-2xx response, malformed SSE, interrupted stream, or empty answer returns the
   existing explicit failure response instead of falling back.
4. Published Excel/CSV aggregate questions execute deterministically through ClickHouse and do not
   call any LLM.
5. `uv sync --no-dev --frozen` installs all API runtime dependencies.
6. The optional indexing profile and its structured-publication workflow remain available.

## Validation

Implementation is complete only when all of the following are true:

- Active code, Compose, environment examples, and operational documentation contain no llama.cpp,
  GGUF, `LLAMA_*`, `MODEL_SLOTS`, `generation-model`, or Compose `generation` profile contract.
- A repository search confirms that remaining uses of `generation` are unrelated to the removed
  local model route.
- Backend settings, lazy-startup, health, Physoc SSE, structured-query, and structured-worker tests
  pass.
- Compose, deployment, model-probe, benchmark-report, and UV dependency contract tests pass after
  removing obsolete expectations.
- A clean temporary `uv sync --no-dev --frozen` environment imports `app.main` successfully.
- Ruff checks, lock validation, and `git diff --check` pass.

## Rollback

Rollback is a Git revert of the implementation commits. Operational rollback must restore the last
known-good Physoc configuration; it must not reintroduce template output or retrieved chunks as a
production answer. Reintroducing a local generation service would require a new reviewed design,
artifact/license approval, resource sizing, and deployment acceptance.
