# Offline Platform Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a reproducible, fully offline runtime contract, single-server service topology, health model, and capacity benchmark harness without changing the current ingestion or answer behavior.

**Architecture:** Keep FastAPI and existing SQLite/PostgreSQL tests independent from external services through lazy dependency factories. Add a Python 3.12 container contract for PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, a private shared Embedding service, API, migration job, worker, and llama.cpp. Record all offline artifacts by checksum and provide small smoke profiles plus production-scale benchmark manifests.

**Tech Stack:** Python 3.12, FastAPI, dataclasses, httpx, PostgreSQL, ClickHouse, Qdrant, Redis, Docker Compose, unittest, JSON benchmark manifests, Locust, psutil.

---

### Task 1: Add the offline runtime settings contract

**Files:**
- Create: `backend/app/offline_settings.py`
- Create: `backend/tests/test_offline_settings.py`
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `backend/app/llm.py`
- Modify: `backend/tests/test_llm_provider.py`
- Modify: `backend/app/database.py`
- Modify: `backend/tests/test_database_config.py`

- [ ] **Step 1: Write failing settings tests**

```python
from __future__ import annotations

import unittest

from app.offline_settings import OfflineSettings, OfflineSettingsError


class OfflineSettingsTest(unittest.TestCase):
    def test_builds_private_offline_service_settings(self) -> None:
        settings = OfflineSettings.from_environ(
            {
                "OFFLINE_MODE": "true",
                "DATABASE_URL": "postgresql+psycopg://dc_agent@postgres/dc_agent",
                "CLICKHOUSE_URL": "http://clickhouse:8123",
                "QDRANT_URL": "http://qdrant:6333",
                "REDIS_URL": "redis://redis:6379/0",
                "CLAMAV_HOST": "clamav",
                "EMBEDDING_SERVICE_URL": "http://embedding-service:8081",
                "LLAMA_SERVER_URL": "http://llama:8080",
                "RAW_DATA_ROOT": "/data/raw",
                "PARQUET_ROOT": "/data/parquet",
                "MODEL_ROOT": "/models",
                "MODEL_SLOTS": "2",
            }
        )

        self.assertTrue(settings.offline_mode)
        self.assertEqual(settings.model_slots, 2)
        self.assertEqual(settings.clickhouse_url, "http://clickhouse:8123")
        self.assertEqual(settings.embedding_service_url, "http://embedding-service:8081")

    def test_rejects_public_model_endpoint_in_offline_mode(self) -> None:
        with self.assertRaisesRegex(OfflineSettingsError, "private or loopback"):
            OfflineSettings.from_environ(
                {
                    "OFFLINE_MODE": "true",
                    "LLAMA_SERVER_URL": "https://api.example.com/v1",
                }
            )

    def test_existing_llm_provider_rejects_public_api_in_offline_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "private or loopback"):
            create_llm_provider(
                {
                    "OFFLINE_MODE": "true",
                    "LLM_PROVIDER": "openai_compatible",
                    "LLM_API_BASE": "https://api.example.com/v1",
                    "LLM_API_KEY": "offline-test",
                    "LLM_MODEL": "remote-model",
                }
            )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
Set-Location backend
py -m unittest tests.test_offline_settings -v
```

Expected: FAIL because `app.offline_settings` does not exist.

- [ ] **Step 3: Implement the immutable settings object**

```python
from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse


class OfflineSettingsError(ValueError):
    pass


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def require_private_url(value: str, field: str) -> str:
    parsed = urlparse(value)
    host = parsed.hostname or ""
    if host in {"localhost", "postgres", "clickhouse", "qdrant", "redis", "embedding-service", "llama"}:
        return value.rstrip("/")
    try:
        address = ip_address(host)
    except ValueError as error:
        raise OfflineSettingsError(f"{field} must use a private or loopback host") from error
    if not (address.is_private or address.is_loopback):
        raise OfflineSettingsError(f"{field} must use a private or loopback host")
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class OfflineSettings:
    offline_mode: bool
    database_url: str
    clickhouse_url: str
    qdrant_url: str
    redis_url: str
    clamav_host: str
    embedding_service_url: str
    llama_server_url: str
    raw_data_root: Path
    parquet_root: Path
    model_root: Path
    model_slots: int
    dependency_timeout_seconds: float

    @classmethod
    def from_environ(cls, environ: dict[str, str]) -> "OfflineSettings":
        offline_mode = parse_bool(environ.get("OFFLINE_MODE"), default=True)
        values = {
            "database_url": resolve_database_url(environ),
            "clickhouse_url": environ.get("CLICKHOUSE_URL", "http://127.0.0.1:8123"),
            "qdrant_url": environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
            "redis_url": environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
            "embedding_service_url": environ.get("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8081"),
            "llama_server_url": environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080"),
        }
        if offline_mode:
            values = {key: require_private_url(value, key) for key, value in values.items()}
        slots = int(environ.get("MODEL_SLOTS", "2"))
        if slots not in {1, 2, 3, 4}:
            raise OfflineSettingsError("MODEL_SLOTS must be between 1 and 4")
        return cls(
            offline_mode=offline_mode,
            clamav_host=environ.get("CLAMAV_HOST", "127.0.0.1"),
            raw_data_root=Path(environ.get("RAW_DATA_ROOT", "./data/raw")),
            parquet_root=Path(environ.get("PARQUET_ROOT", "./data/parquet")),
            model_root=Path(environ.get("MODEL_ROOT", "./models")),
            model_slots=slots,
            dependency_timeout_seconds=float(environ.get("DEPENDENCY_TIMEOUT_SECONDS", "2.0")),
            **values,
        )
```

Add `resolve_database_url()` to accept exactly one of `DATABASE_URL` or `DATABASE_URL_FILE`; the latter reads a mounted secret file and rejects empty content. `backend/tests/test_database_config.py` covers direct URL, secret-file URL, both-set rejection, and missing/empty files.

- [ ] **Step 4: Add the same variables to both example environment files**

```dotenv
OFFLINE_MODE=true
DATABASE_URL=postgresql+psycopg://dc_agent@127.0.0.1/dc_agent
CLICKHOUSE_URL=http://127.0.0.1:8123
QDRANT_URL=http://127.0.0.1:6333
REDIS_URL=redis://127.0.0.1:6379/0
CLAMAV_HOST=127.0.0.1
EMBEDDING_SERVICE_URL=http://127.0.0.1:8081
LLAMA_SERVER_URL=http://127.0.0.1:8080
LLM_PROVIDER=template
LLM_API_BASE=http://127.0.0.1:8080/v1
LLM_API_KEY=local-offline
LLM_MODEL=local-model
RAW_DATA_ROOT=./data/raw
PARQUET_ROOT=./data/parquet
MODEL_ROOT=./models
MODEL_SLOTS=2
DEPENDENCY_TIMEOUT_SECONDS=2.0
```

- [ ] **Step 5: Run tests and commit**

Before running the test, change `create_llm_provider()` as follows:

```python
provider = source.get("LLM_PROVIDER", "template").strip().lower().replace("-", "_")
offline_mode = parse_bool(source.get("OFFLINE_MODE"), default=True)
if provider == "openai_compatible":
    api_base = source.get("LLM_API_BASE", "").strip()
    api_key = source.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise ValueError("LLM_API_KEY is required")
    if offline_mode:
        api_base = require_private_url(api_base, "LLM_API_BASE")
    return OpenAICompatibleLLMProvider(api_base=api_base, api_key=api_key, model=source["LLM_MODEL"])
```

Template mode remains available; no offline configuration may silently fall back to a public endpoint.

```powershell
py -m unittest tests.test_offline_settings tests.test_runtime_env tests.test_llm_provider tests.test_database_config -v
Set-Location ..
git add .env.example backend/.env.example backend/app/offline_settings.py backend/app/llm.py backend/app/database.py backend/tests/test_offline_settings.py backend/tests/test_llm_provider.py backend/tests/test_database_config.py
git commit -m "feat: define offline runtime settings"
```

Expected: all focused tests PASS.

### Task 2: Split and lock offline dependencies

**Files:**
- Create: `backend/requirements-offline.in`
- Create: `backend/requirements-benchmark.in`
- Create: `backend/requirements-offline.txt`
- Create: `backend/requirements-benchmark.txt`
- Create: `deploy/offline/artifacts.schema.json`
- Create: `backend/app/offline_artifacts.py`
- Create: `backend/app/parser_runtime.py`
- Create: `backend/tests/test_offline_artifacts.py`
- Create: `backend/tests/test_parser_runtime.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the artifact manifest validation test**

```python
import unittest

from app.offline_artifacts import validate_artifact_manifest


class OfflineArtifactManifestTest(unittest.TestCase):
    def test_requires_checksum_license_and_local_path(self) -> None:
        manifest = {
            "artifacts": [
                {
                    "name": "embedding-model",
                    "kind": "model",
                    "version": "1",
                    "sha256": "a" * 64,
                    "license": "MIT",
                    "localPath": "/models/embedding",
                }
            ]
        }

        validate_artifact_manifest(manifest)

        broken = {"artifacts": [{"name": "missing-checksum"}]}
        with self.assertRaises(ValueError):
            validate_artifact_manifest(broken)
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_offline_artifacts tests.test_parser_runtime -v
Set-Location ..
```

Expected: FAIL because the validator does not exist.

- [ ] **Step 3: Add dependency input files**

`backend/requirements-offline.in`:

```text
-r requirements.txt
alembic>=1.16,<2
clickhouse-connect>=0.8,<1
qdrant-client>=1.14,<2
redis>=5,<7
psycopg[binary]>=3.2,<4
polars>=1.30,<2
pyarrow>=19,<22
pyxlsb>=1.0,<2
docling>=2.40,<3
paddlepaddle>=3,<4
paddleocr>=3,<4
jieba>=0.42,<1
sqlglot>=27,<30
FlagEmbedding>=1.3,<2
onnxruntime>=1.22,<2
psutil>=7,<8
```

`backend/requirements-benchmark.in`:

```text
-r requirements-offline.in
locust>=2.37,<3
pip-tools>=7.4,<8
```

Do not add these heavy packages to `backend/requirements.txt`; the existing unit-test environment must remain lightweight.

Add `artifacts/wheels/`, `artifacts/models/`, and `artifacts/benchmarks/` to `.gitignore`. These directories are populated by the internal artifact mirror or benchmark runner and must never be committed.

- [ ] **Step 4: Implement manifest validation and schema**

```python
from __future__ import annotations

import re
from collections.abc import Mapping


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_artifact_manifest(payload: Mapping[str, object]) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact manifest must contain artifacts")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("artifact entries must be objects")
        for field in ("name", "kind", "version", "sha256", "license", "localPath"):
            if not str(artifact.get(field, "")).strip():
                raise ValueError(f"artifact is missing {field}")
        if not SHA256_PATTERN.fullmatch(str(artifact["sha256"]).lower()):
            raise ValueError("artifact sha256 must be 64 lowercase hexadecimal characters")
        if str(artifact["localPath"]).startswith(("http://", "https://")):
            raise ValueError("offline artifact paths must be local")
```

The JSON schema must require the same fields and reject additional network URL fields. The lock includes local Docling artifacts, PaddleOCR detection/recognition/classification models, PaddlePaddle CPU wheels, LibreOffice, Poppler/native libraries, and their licenses/checksums. `parser_runtime.py` requires `DOCLING_ARTIFACTS_PATH`, `PADDLEOCR_HOME`, and `LIBREOFFICE_BIN` to exist locally, sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, and rejects any HTTP model path. `test_parser_runtime.py` patches socket/network helpers to fail and proves runtime validation does not attempt a download. Add `deploy/offline/artifacts.lock.json` to `.gitignore`; the environment-specific lock is generated on the target host and is not committed.

- [ ] **Step 5: Produce hashed lock files on the Python 3.12 build host**

Run on the offline build host:

```powershell
py -3.12 -m pip install "pip-tools>=7.4,<8"
py -3.12 -m piptools compile backend/requirements-offline.in --generate-hashes --output-file backend/requirements-offline.txt
py -3.12 -m piptools compile backend/requirements-benchmark.in --generate-hashes --output-file backend/requirements-benchmark.txt
py -3.12 -m pip install --require-hashes -r backend/requirements-offline.txt
py -3.12 -m pip check
```

Expected: both lock files are generated, installation succeeds without contacting the public internet when pointed at the internal wheel mirror, and `pip check` exits 0.

- [ ] **Step 6: Run focused tests and commit**

```powershell
Set-Location backend
py -m unittest tests.test_offline_artifacts tests.test_parser_runtime -v
Set-Location ..
git add .gitignore backend/requirements-offline.in backend/requirements-benchmark.in backend/requirements-offline.txt backend/requirements-benchmark.txt backend/app/offline_artifacts.py backend/app/parser_runtime.py backend/tests/test_offline_artifacts.py backend/tests/test_parser_runtime.py deploy/offline/artifacts.schema.json
git commit -m "build: lock offline data platform dependencies"
```

### Task 3: Define the single-server Compose contract

**Files:**
- Modify: `.gitignore`
- Create: `deploy/offline/compose.yaml`
- Create: `deploy/offline/.env.example`
- Create: `deploy/docker/backend.Dockerfile`
- Create: `deploy/docker/worker.Dockerfile`
- Create: `deploy/docker/embedding.Dockerfile`
- Create: `deploy/offline/README.md`
- Create: `tools/prepare_offline_env.ps1`
- Create: `tools/tests/test_compose_contract.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Create: `backend/alembic/versions/20260715_00_existing_schema.py`
- Create: `backend/tests/test_alembic_baseline.py`
- Create: `backend/app/migration_entrypoint.py`
- Create: `backend/tests/test_migration_entrypoint.py`

- [ ] **Step 1: Write a static Compose contract test**

```python
from pathlib import Path
import unittest


class ComposeContractTest(unittest.TestCase):
    def test_declares_required_offline_services_and_private_network(self) -> None:
        text = Path("deploy/offline/compose.yaml").read_text(encoding="utf-8")
        for service in (
            "postgres:",
            "clickhouse:",
            "qdrant:",
            "redis:",
            "clamav:",
            "schema-migration:",
            "embedding-service:",
            "api:",
            "ingestion-worker:",
            "llama:",
        ):
            self.assertIn(service, text)
        self.assertIn("internal: true", text)
        self.assertNotIn("api.openai.com", text)
        self.assertIn("OFFLINE_MODE: \"true\"", text)
        self.assertIn("condition: service_completed_successfully", text)
        self.assertIn("profiles: [\"indexing\"]", text)
        self.assertIn("PYTHON_BASE_IMAGE:", text)
        self.assertIn('CLAMAV_NO_FRESHCLAMD: "true"', text)
        self.assertNotIn("wget -qO- http://127.0.0.1:6333", text)
        self.assertNotIn("/var/lib/clamav:ro", text)

    def test_environment_preparation_is_non_destructive(self) -> None:
        text = Path("tools/prepare_offline_env.ps1").read_text(encoding="utf-8")
        self.assertIn("Test-Path", text)
        self.assertIn("RandomNumberGenerator", text)
        self.assertIn("artifacts/secrets/", Path(".gitignore").read_text(encoding="utf-8"))
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_compose_contract -v
```

Expected: FAIL because `deploy/offline/compose.yaml` does not exist.

- [ ] **Step 3: Add a resource-bounded Compose topology**

The Compose file must include:

```yaml
name: dc-agent-offline
services:
  postgres:
    image: ${POSTGRES_IMAGE}
    environment:
      POSTGRES_DB: dc_agent
      POSTGRES_USER: dc_agent
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets: [postgres_password]
    volumes:
      - ${DATA_ROOT}/postgres:/var/lib/postgresql/data
    networks: [offline]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dc_agent -d dc_agent"]
      interval: 10s
      timeout: 5s
      retries: 12

  clickhouse:
    image: ${CLICKHOUSE_IMAGE}
    volumes:
      - ${DATA_ROOT}/clickhouse:/var/lib/clickhouse
    networks: [offline]
    mem_limit: ${CLICKHOUSE_MEMORY_LIMIT}
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://127.0.0.1:8123/ping"]

  qdrant:
    image: ${QDRANT_IMAGE}
    volumes:
      - ${DATA_ROOT}/qdrant:/qdrant/storage
    networks: [offline]
    mem_limit: ${QDRANT_MEMORY_LIMIT}
    healthcheck:
      test: ["CMD", "bash", "-ec", ": >/dev/tcp/127.0.0.1/6333"]

  redis:
    image: ${REDIS_IMAGE}
    command: ["redis-server", "--appendonly", "yes"]
    volumes:
      - ${DATA_ROOT}/redis:/data
    networks: [offline]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  clamav:
    image: ${CLAMAV_IMAGE}
    environment:
      CLAMAV_NO_FRESHCLAMD: "true"
    networks: [offline]
    healthcheck:
      test: ["CMD-SHELL", "clamdscan --ping 1"]

  schema-migration:
    build:
      context: ../..
      dockerfile: deploy/docker/backend.Dockerfile
      args:
        PYTHON_BASE_IMAGE: ${PYTHON_BASE_IMAGE}
    command: ["python", "-m", "app.migration_entrypoint"]
    environment:
      DATABASE_URL_FILE: /run/secrets/database_url
    depends_on:
      postgres: {condition: service_healthy}
    secrets: [database_url]
    networks: [offline]

  embedding-service:
    build:
      context: ../..
      dockerfile: deploy/docker/embedding.Dockerfile
      args:
        PYTHON_BASE_IMAGE: ${PYTHON_BASE_IMAGE}
    environment:
      OFFLINE_MODE: "true"
      EMBEDDING_MODEL_ROOT: /models/${EMBEDDING_MODEL_DIR}
      EMBEDDING_MODEL_SHA256: ${EMBEDDING_MODEL_SHA256}
    volumes:
      - ${MODEL_ROOT}:/models:ro
    networks: [offline]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/readyz')"]

  api:
    build:
      context: ../..
      dockerfile: deploy/docker/backend.Dockerfile
      args:
        PYTHON_BASE_IMAGE: ${PYTHON_BASE_IMAGE}
    environment:
      OFFLINE_MODE: "true"
      DATABASE_URL_FILE: /run/secrets/database_url
      CLICKHOUSE_URL: http://clickhouse:8123
      QDRANT_URL: http://qdrant:6333
      REDIS_URL: redis://redis:6379/0
      CLAMAV_HOST: clamav
      EMBEDDING_SERVICE_URL: http://embedding-service:8081
      LLAMA_SERVER_URL: http://llama:8080
      MODEL_SLOTS: ${MODEL_SLOTS}
      LLM_PROVIDER: ${LLM_PROVIDER}
      LLM_API_BASE: ${LLM_API_BASE}
      LLM_API_KEY: ${LLM_API_KEY}
      LLM_MODEL: ${LLM_MODEL}
      RAW_DATA_ROOT: /data/raw
      PARQUET_ROOT: /data/parquet
      MODEL_ROOT: /models
      DOCLING_ARTIFACTS_PATH: /models/docling
      PADDLEOCR_HOME: /models/paddleocr
      LIBREOFFICE_BIN: /usr/bin/libreoffice
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      HF_HUB_DISABLE_TELEMETRY: "1"
      PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK: "true"
    secrets: [database_url]
    volumes:
      - ${DATA_ROOT}/raw:/data/raw
      - ${DATA_ROOT}/parquet:/data/parquet
      - ${MODEL_ROOT}:/models:ro
    ports:
      - "127.0.0.1:8000:8000"
    depends_on:
      schema-migration: {condition: service_completed_successfully}
      clickhouse: {condition: service_healthy}
      qdrant: {condition: service_healthy}
      redis: {condition: service_healthy}
      clamav: {condition: service_healthy}
      embedding-service: {condition: service_healthy}
    networks: [offline]

  ingestion-worker:
    build:
      context: ../..
      dockerfile: deploy/docker/worker.Dockerfile
      args:
        PYTHON_BASE_IMAGE: ${PYTHON_BASE_IMAGE}
    environment:
      OFFLINE_MODE: "true"
      DATABASE_URL_FILE: /run/secrets/database_url
      CLICKHOUSE_URL: http://clickhouse:8123
      QDRANT_URL: http://qdrant:6333
      REDIS_URL: redis://redis:6379/0
      CLAMAV_HOST: clamav
      EMBEDDING_SERVICE_URL: http://embedding-service:8081
      RAW_DATA_ROOT: /data/raw
      PARQUET_ROOT: /data/parquet
      MODEL_ROOT: /models
      DOCLING_ARTIFACTS_PATH: /models/docling
      PADDLEOCR_HOME: /models/paddleocr
      LIBREOFFICE_BIN: /usr/bin/libreoffice
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      HF_HUB_DISABLE_TELEMETRY: "1"
      PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK: "true"
    secrets: [database_url]
    volumes:
      - ${DATA_ROOT}/raw:/data/raw
      - ${DATA_ROOT}/parquet:/data/parquet
      - ${MODEL_ROOT}:/models:ro
    depends_on:
      schema-migration: {condition: service_completed_successfully}
      clickhouse: {condition: service_healthy}
      qdrant: {condition: service_healthy}
      redis: {condition: service_healthy}
      clamav: {condition: service_healthy}
      embedding-service: {condition: service_healthy}
    networks: [offline]
    profiles: ["indexing"]

  llama:
    image: ${LLAMA_IMAGE}
    command: ["--model", "/models/${LLAMA_MODEL_FILE}", "--host", "0.0.0.0", "--port", "8080", "--parallel", "${MODEL_SLOTS}"]
    volumes:
      - ${MODEL_ROOT}:/models:ro
    networks: [offline]
    profiles: ["generation"]

networks:
  offline:
    internal: true

secrets:
  postgres_password:
    file: ${POSTGRES_PASSWORD_FILE}
  database_url:
    file: ${DATABASE_URL_SECRET_FILE}
```

Use secrets or mounted files for passwords. Do not publish ClickHouse, Qdrant, Redis, or llama.cpp ports outside the private Compose network in the production profile.

`deploy/offline/.env.example` must define the image digests, `DATA_ROOT`, `ARTIFACT_ROOT`, `MODEL_ROOT`, `POSTGRES_PASSWORD_FILE`, `DATABASE_URL_SECRET_FILE`, `POSTGRES_IMAGE`, `CLICKHOUSE_IMAGE`, `QDRANT_IMAGE`, `REDIS_IMAGE`, `CLAMAV_IMAGE`, `LLAMA_IMAGE`, `LLAMA_MODEL_FILE`, `MODEL_SLOTS`, `EMBEDDING_MODEL_DIR`, `EMBEDDING_MODEL_SHA256`, `PYTHON_BASE_IMAGE`, `CLICKHOUSE_MEMORY_LIMIT`, `QDRANT_MEMORY_LIMIT`, `LLM_PROVIDER=template`, `LLM_API_BASE=http://llama:8080/v1`, `LLM_API_KEY=local-offline`, and `LLM_MODEL=<locked-local-model-name>`. Phase 3 changes `LLM_PROVIDER` to `openai_compatible` only when the generation profile is enabled; the configured base remains private. Both secret files are created locally with restrictive permissions and are not committed. `CLAMAV_IMAGE` is an internally built, digest-pinned image containing the mirrored signature set; do not mount over `/var/lib/clamav`, so the baked signatures remain visible and the container layer stays writable for the official startup ownership check. `CLAMAV_NO_FRESHCLAMD=true` prevents runtime updates/network access, and the artifact lock records the image and signature checksums.

`tools/prepare_offline_env.ps1` copies `.env.example` only when `.env` is absent, creates both secrets with `RandomNumberGenerator` only when neither exists, refuses a partially configured pair, restricts the secret directory to the current account, and never overwrites or prints credentials unless `-RotateSecrets` is explicitly supplied. Add `deploy/offline/.env` and `artifacts/secrets/` to `.gitignore`.

```powershell
param([switch]$RotateSecrets)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$envPath = Join-Path $repo "deploy/offline/.env"
$secretDir = Join-Path $repo "artifacts/secrets"
$passwordPath = Join-Path $secretDir "postgres-password"
$databasePath = Join-Path $secretDir "database-url"
if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item (Join-Path $repo "deploy/offline/.env.example") $envPath
}
New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
$present = @(@($passwordPath, $databasePath) | Where-Object { Test-Path -LiteralPath $_ }).Count
if ($present -eq 1) { throw "Both offline secret files must exist together; refusing partial configuration" }
if ($RotateSecrets -or $present -eq 0) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    $password = [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=')
    Set-Content -LiteralPath $passwordPath -Value $password -NoNewline -Encoding Ascii
    Set-Content -LiteralPath $databasePath -Value "postgresql+psycopg://dc_agent:$password@postgres/dc_agent" -NoNewline -Encoding Ascii
}
if ($env:OS -eq "Windows_NT") {
    icacls $secretDir /inheritance:r /grant:r "${env:USERNAME}:(OI)(CI)F" | Out-Null
} else {
    chmod 700 $secretDir
    chmod 600 $passwordPath $databasePath
}
```

- [ ] **Step 4: Add Python 3.12 API and worker images**

Both Dockerfiles must use the same internal Python 3.12 base image, install `requirements-offline.txt` with `--require-hashes`, copy only required application files, create a non-root user, and set separate commands:

```dockerfile
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
WORKDIR /app
COPY artifacts/wheels /wheels
COPY backend/requirements.txt backend/requirements-offline.txt ./
RUN python -m pip install --no-index --find-links=/wheels --require-hashes -r requirements-offline.txt
COPY backend/app ./app
COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic
RUN useradd --create-home dcagent
USER dcagent
CMD ["python", "-m", "uvicorn", "app.main:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

The worker image changes only the final command to `python -m app.ingestion_worker` after that module is introduced in Phase 2; its `indexing` profile stays disabled until that task lands. The internal Python base image is built with the locked LibreOffice/Poppler/native runtime and no package-manager network access; the worker validates those versions against the artifact lock before starting. The Embedding image uses the same locked Python base and wheel set, copies only its service/runtime files, and runs `uvicorn app.embedding_service:create_production_app --factory --host 0.0.0.0 --port 8081` as a non-root user. The frozen `20260715_00_existing_schema.py` uses explicit `op.create_table()`/`op.create_index()` calls and must not import live `Base.metadata` or call `create_all()`.

`20260715_00_existing_schema.py` has `down_revision = None`. `migration_entrypoint.py` handles both deployment states: for an empty database it runs `upgrade head`; for an existing pre-Alembic DC-Agent database it first validates the exact baseline table/column/index fingerprint, refuses any mismatch, stamps `20260715_00`, then upgrades head. `test_alembic_baseline.py` covers an empty database; `test_migration_entrypoint.py` creates the current pre-Alembic schema, verifies stamp-plus-upgrade, verifies rerun idempotency, and verifies a missing/extra column blocks stamping. The runbook includes backup and rollback before the first stamp. Production API/worker depend on this migration service.

- [ ] **Step 5: Validate and commit**

```powershell
py -m unittest tools.tests.test_compose_contract -v
Set-Location backend
py -m unittest tests.test_alembic_baseline tests.test_migration_entrypoint -v
Set-Location ..
& tools/prepare_offline_env.ps1
docker compose --env-file deploy/offline/.env -f deploy/offline/compose.yaml config --quiet
git add .gitignore deploy/offline/compose.yaml deploy/offline/.env.example deploy/offline/README.md deploy/docker/backend.Dockerfile deploy/docker/worker.Dockerfile deploy/docker/embedding.Dockerfile tools/prepare_offline_env.ps1 backend/alembic.ini backend/alembic/env.py backend/alembic/script.py.mako backend/alembic/versions/20260715_00_existing_schema.py backend/app/migration_entrypoint.py backend/tests/test_alembic_baseline.py backend/tests/test_migration_entrypoint.py tools/tests/test_compose_contract.py
git commit -m "build: add offline compose topology"
```

Expected: static test PASS. The Docker command must PASS on the target host; the current development machine may record it as environment-blocked because Docker is not installed.

### Task 4: Add the private shared Embedding service

**Files:**
- Create: `backend/app/embedding_contracts.py`
- Create: `backend/app/embedding_service.py`
- Create: `backend/app/embedding_client.py`
- Create: `backend/tests/test_embedding_service.py`
- Create: `backend/tests/test_embedding_client.py`

- [ ] **Step 1: Write failing service and client contract tests**

```python
class EmbeddingServiceTest(unittest.TestCase):
    def test_returns_pinned_model_metadata_with_vectors(self) -> None:
        app = create_embedding_app(
            backend=FakeEmbeddingBackend(dimensions=4),
            metadata=EmbeddingModelMetadata(
                "bge-test", "1", "a" * 64, 4, True, "e" * 64, "1"
            ),
        )
        response = TestClient(app).post(
            "/v1/embeddings",
            json={"texts": ["甲", "乙"], "purpose": "document"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["modelChecksum"], "a" * 64)
        self.assertEqual(len(response.json()["vectors"]), 2)


class EmbeddingClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_public_endpoint_and_metadata_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            HttpEmbeddingClient("https://public.example/v1")
        client = HttpEmbeddingClient(
            "http://embedding-service:8081",
            transport=FakeTransport(
                checksum="b" * 64,
                encoding_profile_sha256="e" * 64,
                protocol_version="1",
            ),
        )
        with self.assertRaises(EmbeddingModelMismatch):
            await client.embed(
                ["甲"],
                expected=EmbeddingModelMetadata(
                    "bge-test", "1", "a" * 64, 4, True, "e" * 64, "1"
                ),
                purpose="query",
            )
```

The test modules define their small fakes in the same files: `FakeEmbeddingBackend.embed(texts)` records batch sizes and returns four-float vectors; `FakeTransport.post_json()` returns the configured metadata and vectors.

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_embedding_service tests.test_embedding_client -v
```

Expected: FAIL because neither module exists.

- [ ] **Step 3: Implement one bounded offline model runtime**

`embedding_contracts.py` defines `EmbeddingPurpose = Literal["query", "document"]`, `EmbeddingModelMetadata(name, version, sha256, dimensions, normalized, encoding_profile_sha256, protocol_version)`, and a structural `EmbeddingMetadataExpectation` protocol exposing those seven read-only fields. It also defines the async `EmbeddingClient.embed(..., expected=EmbeddingMetadataExpectation, purpose=...)` protocol and the request/response DTOs. `HttpEmbeddingClient` is its only production implementation.

`embedding_service.py` loads one local BGE/ONNX model only during service startup, verifies the configured SHA-256, sets every loader to local-files-only mode, disables telemetry, and exposes `GET /readyz`, `GET /v1/metadata`, and `POST /v1/embeddings`. Reject more than 64 texts, any single text over 16 KiB, or a request over 256 KiB. The request requires `purpose`; the response always includes model name, version, checksum, dimensions, normalization, encoding-profile checksum, protocol version, and vectors.

`HttpEmbeddingClient.embed()` accepts texts, purpose, and the expected `EmbeddingMetadataExpectation`, splits requests into bounded batches, validates the private URL, and rejects any response whose name/version/checksum/dimensions/normalization/encoding profile/protocol differs from the expectation. Phase 2's `EmbeddingArtifactRef` satisfies this protocol structurally, so no Phase 1 module imports a later-phase type. Both ingestion workers and online queries depend on the `EmbeddingClient` protocol; they never load separate model weights.

The Phase 1 contract is concrete:

```python
from typing import Literal, Protocol, Sequence, runtime_checkable

EmbeddingPurpose = Literal["query", "document"]

@runtime_checkable
class EmbeddingMetadataExpectation(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def sha256(self) -> str: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def normalized(self) -> bool: ...
    @property
    def encoding_profile_sha256(self) -> str: ...
    @property
    def protocol_version(self) -> str: ...

class EmbeddingClient(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
        expected: EmbeddingMetadataExpectation,
    ) -> list[list[float]]: ...
```

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tests.test_embedding_service tests.test_embedding_client -v
Set-Location ..
git add backend/app/embedding_contracts.py backend/app/embedding_service.py backend/app/embedding_client.py backend/tests/test_embedding_service.py backend/tests/test_embedding_client.py
git commit -m "feat: add private shared embedding service"
```

### Task 5: Add lazy infrastructure health checks

**Files:**
- Create: `backend/app/infra/__init__.py`
- Create: `backend/app/infra/health.py`
- Create: `backend/tests/test_infra_health.py`
- Create: `backend/tests/test_lazy_startup.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes.py`

- [ ] **Step 1: Write tests for liveness and readiness separation**

```python
import unittest

from fastapi.testclient import TestClient

from app.infra.health import DependencyCheck, DependencyHealthRegistry
from app.main import create_app
from app.repository import InMemoryChatRepository
from app.seed import build_seed_state


class InfraHealthTest(unittest.TestCase):
    def test_liveness_does_not_require_external_services(self) -> None:
        registry = DependencyHealthRegistry(
            [DependencyCheck("qdrant", lambda: (False, "unavailable"))]
        )
        client = TestClient(create_app(InMemoryChatRepository(build_seed_state()), health_registry=registry))

        self.assertEqual(client.get("/api/healthz").status_code, 200)
        self.assertEqual(client.get("/api/readyz").status_code, 503)


class LazyStartupTest(unittest.TestCase):
    def test_importing_main_does_not_connect_or_create_schema(self) -> None:
        with patch("app.database.Database.__init__", side_effect=AssertionError("connected")):
            module = importlib.import_module("app.main")
            importlib.reload(module)
        self.assertTrue(callable(module.create_production_app))
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_infra_health tests.test_lazy_startup -v
```

Expected: FAIL because `health_registry` and the endpoints do not exist.

- [ ] **Step 3: Implement dependency health contracts without import-time connections**

```python
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    name: str
    check: Callable[[], tuple[bool, str]]


class DependencyHealthRegistry:
    def __init__(self, checks: list[DependencyCheck] | None = None) -> None:
        self._checks = checks or []

    def report(self) -> dict[str, dict[str, str | bool]]:
        return {
            item.name: {"ok": ok, "detail": detail}
            for item in self._checks
            for ok, detail in [item.check()]
        }

    def ready(self) -> bool:
        return all(bool(item["ok"]) for item in self.report().values())
```

Remove the module-level `app = create_app()` side effect. Add `create_production_app()` as the Uvicorn factory. Add `health_registry` as an optional `create_app()` argument. `/api/healthz` always reports process liveness. `/api/readyz` returns 200 only when PostgreSQL schema revision, ClickHouse, Qdrant, Redis, ClamAV, and the shared Embedding service are ready; llama.cpp is required only when generation is enabled. Unit tests inject fakes; production clients and checks are created in the FastAPI lifespan and never connect or call `create_schema()` at import time.

- [ ] **Step 4: Run focused and API regression tests, then commit**

```powershell
py -m unittest tests.test_infra_health tests.test_lazy_startup tests.test_api_contract -v
Set-Location ..
git add backend/app/infra/__init__.py backend/app/infra/health.py backend/app/main.py backend/app/routes.py backend/tests/test_infra_health.py backend/tests/test_lazy_startup.py
git commit -m "feat: expose offline dependency readiness"
```

### Task 6: Define benchmark manifests and deterministic fixture streams

**Files:**
- Create: `tools/benchmarks/__init__.py`
- Create: `tools/benchmarks/manifest.py`
- Create: `tools/benchmarks/fixtures.py`
- Create: `tools/benchmarks/manifests/smoke.json`
- Create: `tools/benchmarks/manifests/acceptance-30m-5m.json`
- Create: `tools/tests/test_benchmark_manifest.py`
- Create: `tools/tests/test_benchmark_fixtures.py`

- [ ] **Step 1: Write failing manifest and streaming-fixture tests**

```python
import unittest

from tools.benchmarks.fixtures import iter_qdrant_points
from tools.benchmarks.manifest import BenchmarkManifest


class BenchmarkManifestTest(unittest.TestCase):
    def test_acceptance_manifest_matches_approved_capacity_profile(self) -> None:
        manifest = BenchmarkManifest.load("tools/benchmarks/manifests/acceptance-30m-5m.json")
        self.assertEqual(manifest.clickhouse_rows, 30_000_000)
        self.assertEqual(manifest.qdrant_points, 5_000_000)
        self.assertEqual(manifest.vector_dimension_candidates, (512, 768))
        self.assertEqual(manifest.virtual_users, 15)
        self.assertEqual(manifest.think_time_seconds, 5)
        self.assertEqual(manifest.dense_candidates, 50)
        self.assertEqual(manifest.sparse_candidates, 50)
        self.assertTrue(manifest.include_sparse_vectors)

    def test_qdrant_points_are_generated_in_bounded_batches(self) -> None:
        batches = list(iter_qdrant_points(total=7, dimensions=4, batch_size=3, seed=42))
        self.assertEqual([len(batch) for batch in batches], [3, 3, 1])
        self.assertEqual(batches[0][0]["id"], 0)
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_benchmark_manifest tools.tests.test_benchmark_fixtures -v
```

Expected: FAIL because the benchmark modules do not exist.

- [ ] **Step 3: Implement an immutable JSON manifest**

```python
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    clickhouse_rows: int
    qdrant_points: int
    vector_dimension_candidates: tuple[int, ...]
    virtual_users: int
    think_time_seconds: int
    duration_seconds: int
    request_mix: dict[str, int]
    filter_selectivity: list[float]
    dense_candidates: int
    sparse_candidates: int
    fused_evidence_limit: int
    context_tokens: int
    output_tokens: int
    include_sparse_vectors: bool
    gate_profiles: dict[str, tuple[dict[str, object], ...]]

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["vector_dimension_candidates"] = tuple(payload["vector_dimension_candidates"])
        payload["gate_profiles"] = {
            name: tuple(items) for name, items in payload["gate_profiles"].items()
        }
        manifest = cls(**payload)
        if sum(manifest.request_mix.values()) != 100:
            raise ValueError("request mix must total 100")
        if manifest.virtual_users < 1 or not manifest.vector_dimension_candidates:
            raise ValueError("benchmark dimensions and users must be positive")
        return manifest
```

The acceptance JSON is a model-neutral template: 30,000,000 ClickHouse rows, 5,000,000 Qdrant points, dimension candidates 512 and 768, 15 users, five-second think time, 1/10/50 percent filters, 1,800 seconds, a 40/40/20 structured/document/mixed mix, 50 dense and 50 sparse candidates, at most ten fused chunks, 2,048 context tokens, 256 output tokens, and sparse vectors enabled. It contains separate numeric `gate_profiles` for `online-cold`, `online-warm`, `batch-initial`, `batch-daily`, and `batch-weekly`; every online or batch-under-load profile includes `queue_feedback_p95_ms <= 2000` (or `online_queue_feedback_p95_ms <= 2000` for batch modes), and available-slot generation profiles include `first_token_p95_ms <= 10000`. Model dimensions and slots are selected later by the 32GB/64GB profile and recorded in the rendered report. The smoke manifest uses 10,000 rows, 2,000 points, dimension candidate 32, three users, 120 seconds, and only service round-trip gates that Phase 1 can actually produce.

- [ ] **Step 4: Implement bounded deterministic fixture generators**

```python
from collections.abc import Iterator
import random


def iter_qdrant_points(
    total: int,
    dimensions: int,
    batch_size: int,
    seed: int,
) -> Iterator[list[dict]]:
    randomizer = random.Random(seed)
    batch: list[dict] = []
    for point_id in range(total):
        vector = [randomizer.uniform(-1.0, 1.0) for _ in range(dimensions)]
        batch.append(
            {
                "id": point_id,
                "vector": {
                    "dense": vector,
                    "bm25": {
                        "indices": [point_id % 257, (point_id + 17) % 257],
                        "values": [1.0, 0.5],
                    },
                },
                "payload": {
                    "tenant_id": point_id % 100,
                    "department_id": point_id % 20,
                    "classification": point_id % 3,
                },
            }
        )
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
```

ClickHouse fixtures must be generated server-side with `numbers()` or streamed as bounded Arrow batches; never materialize 30 million Python objects.

- [ ] **Step 5: Run tests and commit**

```powershell
py -m unittest tools.tests.test_benchmark_manifest tools.tests.test_benchmark_fixtures -v
git add tools/benchmarks/__init__.py tools/benchmarks/manifest.py tools/benchmarks/fixtures.py tools/benchmarks/manifests/smoke.json tools/benchmarks/manifests/acceptance-30m-5m.json tools/tests/test_benchmark_manifest.py tools/tests/test_benchmark_fixtures.py
git commit -m "test: add offline capacity benchmark manifests"
```

### Task 7: Add benchmark execution and reporting

**Files:**
- Create: `tools/benchmarks/report.py`
- Create: `tools/benchmarks/run_capacity_benchmark.py`
- Create: `tools/benchmarks/locustfile.py`
- Create: `tools/tests/test_benchmark_report.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write a failing capacity-gate report test**

```python
import unittest

from tools.benchmarks.report import MetricGate, evaluate_capacity


class BenchmarkReportTest(unittest.TestCase):
    def test_rejects_profile_when_document_p95_exceeds_five_seconds(self) -> None:
        gates = (
            MetricGate("document_p95_ms", "lte", 5000),
            MetricGate("error_rate", "lte", 0.01),
            MetricGate("warm_cache_hit_rate", "gte", 0.20),
        )
        result = evaluate_capacity(
            gates,
            {
                "document_p95_ms": 6200,
                "error_rate": 0.05,
                "warm_cache_hit_rate": 0.10,
            },
        )
        self.assertFalse(result.passed)
        self.assertEqual(
            result.failures,
            ["document_p95_ms", "error_rate", "warm_cache_hit_rate"],
        )
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_benchmark_report -v
```

Expected: FAIL because the report module does not exist.

- [ ] **Step 3: Implement deterministic gate evaluation and JSON output**

```python
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class MetricGate:
    name: str
    operator: Literal["lte", "gte"]
    limit: float


@dataclass(frozen=True, slots=True)
class CapacityResult:
    passed: bool
    failures: list[str]


def evaluate_capacity(
    gates: tuple[MetricGate, ...],
    metrics: dict[str, float],
) -> CapacityResult:
    failures: list[str] = []
    for gate in gates:
        value = metrics.get(gate.name)
        if value is None:
            failures.append(gate.name)
        elif gate.operator == "lte" and value > gate.limit:
            failures.append(gate.name)
        elif gate.operator == "gte" and value < gate.limit:
            failures.append(gate.name)
    return CapacityResult(passed=not failures, failures=failures)


def write_report(path: Path, manifest: dict, profile: dict, mode: str, gates: tuple[MetricGate, ...], hardware: dict, metrics: dict, result: CapacityResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "profile": profile,
                "mode": mode,
                "gates": [asdict(gate) for gate in gates],
                "hardware": hardware,
                "metrics": metrics,
                "gateResult": asdict(result),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
```

`run_capacity_benchmark.py` must record CPU model, logical and physical cores, total RAM, free RAM, disk device, selected vector dimension/model slots, software versions, manifest/profile checksums, mode/cache label, and command exit codes. It selects the exact numeric gates for the current mode and never reports PASS when a gated metric is missing or violates its `lte`/`gte` direction. Phase 1 smoke gates are separate from Phase 4 online and batch gates; `not_available` is never treated as a passing number.

- [ ] **Step 4: Add a 40/40/20 Locust workload**

The Locust file must define structured, document, and mixed tasks with weights 4/4/2, five-second wait time from the manifest, trusted identity headers for principals pre-seeded in PostgreSQL authorization tables, and event listeners that record queue feedback and first-token timing. It never sends tenant/classification scopes directly. It targets the Phase 3 API through the configured trusted reverse-proxy path; before Phase 3, smoke uses separate dependency round-trip gates instead of fabricating query metrics.

- [ ] **Step 5: Ignore generated artifacts, run tests, and commit**

Add `artifacts/benchmarks/` to `.gitignore`, then run:

```powershell
py -m unittest tools.tests.test_benchmark_report -v
git add .gitignore tools/benchmarks/report.py tools/benchmarks/run_capacity_benchmark.py tools/benchmarks/locustfile.py tools/tests/test_benchmark_report.py
git commit -m "test: add capacity gate reporting"
```

### Task 8: Benchmark local Embedding and generation candidates

**Files:**
- Create: `tools/benchmarks/model_probe.py`
- Create: `tools/tests/test_model_probe.py`

- [ ] **Step 1: Write a failing model-gate test with fake runtimes**

```python
import unittest

from tools.benchmarks.model_probe import ModelGate, evaluate_model_probe


class ModelProbeTest(unittest.TestCase):
    def test_rejects_candidate_when_first_token_gate_is_missed(self) -> None:
        result = evaluate_model_probe(
            ModelGate(
                max_query_embedding_p95_ms=1500,
                max_queue_feedback_p95_ms=2000,
                max_first_token_p95_ms=10000,
            ),
            {
                "query_embedding_p95_ms": 700,
                "queue_feedback_p95_ms": 900,
                "first_token_p95_ms": 13000,
            },
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failures, ["first_token_p95_ms"])
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_model_probe -v
```

Expected: FAIL because `tools.benchmarks.model_probe` does not exist.

- [ ] **Step 3: Implement offline-only candidate probes**

The probe accepts the Compose file, service names, a discovery label, and one candidate artifact-lock entry. It starts one candidate at a time inside the private Compose network, queries service metadata with `docker compose exec`, verifies checksums against the artifact lock, and records:

- BGE small/base batch documents per second for 300-800-token chunks;
- query Embedding p50/p95 under 1, 5, and 15 concurrent callers;
- model resident memory;
- optional local BGE Reranker top-20-to-10 p50/p95, resident memory, model checksum, and busy/timeout rate; candidates that miss the 1.5-second gate are recorded as `disabled`;
- llama.cpp queue wait, first-token p50/p95, output tokens per second, and failure rate for Qwen-family 1.5B and 3B Q4 candidates;
- context lengths of 512, 1,024, and 2,048 tokens with output capped at 256 tokens.

The command must fail if a loader attempts a public network connection or a required metric is missing. Candidate evaluation treats queue/degradation feedback p95 above 2,000 ms, available-slot first-token p95 above 10,000 ms, or query Embedding p95 above 1,500 ms as a failed gate.

- [ ] **Step 4: Run unit and target-host probes**

```powershell
py -m unittest tools.tests.test_model_probe -v
docker compose --env-file deploy/offline/.env -f deploy/offline/compose.yaml up -d --wait embedding-service
docker compose --env-file deploy/offline/.env -f deploy/offline/compose.yaml --profile generation up -d --wait llama
py tools/benchmarks/model_probe.py --compose deploy/offline/compose.yaml --embedding-service embedding-service --llama-service llama --candidate-lock deploy/offline/artifacts.lock.json --label discovery
```

Expected: unit test PASS. Target-host output records candidate metrics only and never claims a 32GB/64GB selection. Phase 4 reruns the selected candidates with the matching locked profile manifest and Compose resource override before either profile can pass; do not choose a model by reputation alone.

- [ ] **Step 5: Commit**

```powershell
git add tools/benchmarks/model_probe.py tools/tests/test_model_probe.py
git commit -m "test: benchmark local model candidates"
```

### Task 9: Add a real-service Compose smoke command

**Files:**
- Create: `tools/compose_smoke.py`
- Create: `tools/tests/test_compose_smoke.py`
- Modify: `deploy/offline/README.md`

- [ ] **Step 1: Write a failing command-construction test**

```python
import unittest

from tools.compose_smoke import build_compose_command


class ComposeSmokeTest(unittest.TestCase):
    def test_starts_api_and_its_core_dependencies_without_fake_profile(self) -> None:
        self.assertEqual(
            build_compose_command("up"),
            [
                "docker",
                "compose",
                "--env-file",
                "deploy/offline/.env",
                "-f",
                "deploy/offline/compose.yaml",
                "up",
                "-d",
                "--build",
                "--wait",
                "--remove-orphans",
                "api",
            ],
        )
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_compose_smoke -v
```

Expected: FAIL because `tools.compose_smoke` does not exist.

- [ ] **Step 3: Implement safe orchestration**

`tools/compose_smoke.py` must:

1. Run `docker compose ... config --quiet`.
2. Run `docker compose up -d --build --wait --remove-orphans api`; API dependencies start PostgreSQL, migration, ClickHouse, Qdrant, Redis, ClamAV, and Embedding without starting worker or llama.cpp.
3. Verify internal-only services through `docker compose exec` rather than assuming their ports are published: PostgreSQL `SELECT 1` and Alembic head, ClickHouse `/ping`, Qdrant `/readyz`, Redis `PING`, ClamAV ping, and Embedding `/readyz` plus `/v1/metadata`.
4. Verify the loopback-published API `/api/readyz`.
5. Print component versions and write `artifacts/benchmarks/compose-smoke.json`.
6. Stop containers in `finally`; preserve volumes unless `--remove-volumes` is explicitly passed.

Use `subprocess.run(..., check=True, shell=False)` with fixed argument arrays. Do not construct shell command strings from environment values.

- [ ] **Step 4: Run the unit test and target-host smoke**

```powershell
py -m unittest tools.tests.test_compose_smoke -v
py tools/compose_smoke.py
```

Expected: unit test PASS everywhere. Real smoke PASS only on a host with Docker and mirrored images.

- [ ] **Step 5: Commit**

```powershell
git add tools/compose_smoke.py tools/tests/test_compose_smoke.py deploy/offline/README.md
git commit -m "test: add offline compose smoke"
```

### Task 10: Run Phase 1 regression and record the platform gate

**Files:**
- Modify: `README.md`
- Create: `docs/offline-platform-runbook.md`

- [ ] **Step 1: Document exact offline setup and limitations**

The runbook must include:

- Python 3.12 requirement and hashed wheel installation.
- Artifact manifest generation and license review.
- Compose service profiles and memory budgets.
- The fact that the current development machine has no Docker and cannot provide the target-host integration result.
- Smoke and acceptance benchmark commands.
- 32GB and 64GB results stored separately.

- [ ] **Step 2: Run all local verification**

```powershell
Set-Location backend
py -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..
py -m unittest discover -s tools/tests -p "test_*.py" -v
git diff --check
```

Expected: all backend and tools unit tests PASS; `git diff --check` has no new findings.

- [ ] **Step 3: Run the target-host gate**

```powershell
py tools/compose_smoke.py
py tools/benchmarks/run_capacity_benchmark.py --manifest tools/benchmarks/manifests/smoke.json --profile smoke
```

Expected: Compose smoke exits 0 and produces a hardware/version report. Do not run the 30M/5M acceptance profile until Phases 2 and 3 provide real ingestion and query endpoints.

- [ ] **Step 4: Commit Phase 1 documentation**

```powershell
git add README.md docs/offline-platform-runbook.md
git commit -m "docs: add offline platform runbook"
```

## Phase 1 completion gate

- Existing API behavior and tests remain unchanged.
- Offline settings reject public model endpoints.
- The existing OpenAI-compatible LLM path also rejects public endpoints whenever offline mode is enabled.
- Dependency locks install with hashes from the internal mirror on Python 3.12.
- Compose passes connection variables, shared volumes, build args, and a schema-migration gate; the worker remains disabled until Phase 2.
- API and worker use one private checksum-pinned Embedding service.
- Health endpoints distinguish liveness from readiness without import-time connections.
- Benchmark manifests, fixture streams, and report gates are deterministic and unit tested.
