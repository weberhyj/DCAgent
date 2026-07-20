# Backend UV and Ruff Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Replace all backend requirements-file dependency management with a Python 3.12 `pyproject.toml` and `uv.lock`, migrate Docker and developer commands to UV, and establish Ruff formatting/linting across the backend.

**Architecture:** Keep `backend` as a non-packaged, standalone UV project. Runtime dependencies live in the project table; offline, benchmark, and development-only packages live in PEP 735 dependency groups. Docker uses an approved UV binary in the pinned internal Python image and synchronizes a frozen `/app/.venv` from the lock without pip or network access.

**Tech Stack:** Python 3.12, UV 0.11.29, Ruff 0.15.22, Docker/Compose, `tomllib`, `unittest`.

---

### Task 1: Add the migration contract tests first

**Files:**
- Create: `tools/tests/test_backend_uv_contract.py`
- Test inputs: `backend/requirements.txt`, `backend/requirements-offline.in`, `backend/requirements-benchmark.in`, `backend/pyproject.toml`, `backend/uv.lock`, `deploy/docker/*.Dockerfile`, `.dockerignore`, `README.md`, `deploy/offline/README.md`, `docs/offline-platform-runbook.md`

- [ ] **Step 1: Write the failing tests**

Create a contract test module that uses only the standard library:

```python
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BASE_PACKAGES = {
    "fastapi",
    "httpx",
    "langgraph",
    "openpyxl",
    "psycopg",
    "pypdf",
    "python-docx",
    "python-multipart",
    "sqlalchemy",
    "uvicorn",
}
OFFLINE_PACKAGES = {
    "alembic",
    "clickhouse-connect",
    "qdrant-client",
    "redis",
    "polars",
    "pyarrow",
    "pyxlsb",
    "docling",
    "paddlepaddle",
    "paddleocr",
    "jieba",
    "sqlglot",
    "flagembedding",
    "onnxruntime",
    "psutil",
}


def package_names(requirements: list[object]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        if isinstance(requirement, str):
            names.add(re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower())
    return names


def load_project() -> dict[str, object]:
    project_path = BACKEND / "pyproject.toml"
    if not project_path.is_file():
        raise AssertionError("backend/pyproject.toml must exist")
    return tomllib.loads(project_path.read_text(encoding="utf-8"))


class BackendUvContractTest(unittest.TestCase):
    def test_pyproject_declares_python_groups_and_all_existing_packages(self) -> None:
        project = load_project()
        self.assertEqual(project["project"]["requires-python"], ">=3.12,<3.13")
        self.assertEqual(
            set(project["dependency-groups"]),
            {"offline", "benchmark", "dev"},
        )
        self.assertIn({"include-group": "offline"}, project["dependency-groups"]["benchmark"])

        self.assertEqual(package_names(project["project"]["dependencies"]), BASE_PACKAGES)
        self.assertEqual(package_names(project["dependency-groups"]["offline"]), OFFLINE_PACKAGES)
        self.assertEqual(package_names(project["dependency-groups"]["benchmark"]), {"locust"})
        self.assertIn("alembic", package_names(project["dependency-groups"]["dev"]))
        self.assertIn("ruff", package_names(project["dependency-groups"]["dev"]))

    def test_requirements_inputs_are_removed_after_migration(self) -> None:
        for name in ("requirements.txt", "requirements-offline.in", "requirements-benchmark.in"):
            self.assertFalse((BACKEND / name).exists(), name)

    def test_lock_and_docker_contract_are_present(self) -> None:
        self.assertTrue((BACKEND / "uv.lock").is_file())
        for name in ("backend.Dockerfile", "embedding.Dockerfile", "worker.Dockerfile"):
            text = (ROOT / "deploy" / "docker" / name).read_text(encoding="utf-8")
            self.assertIn("pyproject.toml", text)
            self.assertIn("uv.lock", text)
            self.assertIn("uv sync --frozen", text)
            self.assertNotIn("requirements.txt", text)
            self.assertNotIn("pip install", text)
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn("!backend/pyproject.toml", dockerignore)
        self.assertIn("!backend/uv.lock", dockerignore)

    def test_ruff_contract_and_documentation_are_current(self) -> None:
        project = load_project()
        self.assertEqual(project["tool"]["ruff"]["target-version"], "py312")
        self.assertEqual(project["tool"]["ruff"]["line-length"], 100)
        self.assertEqual(
            project["tool"]["ruff"]["lint"]["select"],
            ["E4", "E7", "E9", "F", "I", "UP"],
        )
        for path in (ROOT / "README.md", ROOT / "deploy" / "offline" / "README.md", ROOT / "docs" / "offline-platform-runbook.md"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("requirements-offline.txt", text)
            self.assertNotIn("requirements-benchmark.txt", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
py -m unittest tools.tests.test_backend_uv_contract -v
```

Expected: failures because `backend/pyproject.toml`, `backend/uv.lock`, the UV Docker contract, and the new Ruff configuration do not exist yet.

- [ ] **Step 3: Commit the RED contract**

```powershell
git add tools/tests/test_backend_uv_contract.py
git commit -m "test: define backend uv migration contract"
```

### Task 2: Create the backend project metadata and UV lock

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/uv.lock`
- Delete: `backend/requirements.txt`
- Delete: `backend/requirements-offline.in`
- Delete: `backend/requirements-benchmark.in`

- [ ] **Step 1: Install the pinned local UV tool without trusting an unverified binary**

Use the official GitHub release archive for UV 0.11.29 on Windows x86_64, verify the archive against the matching `.sha256` release asset, and put `uv.exe` on the current process `PATH`. In a disconnected environment, retrieve the same archive through the approved internal mirror and retain the checksum record. Do not install from an unapproved public package index.

```powershell
$uvVersion = "0.11.29"
$uvDirectory = Join-Path $env:LOCALAPPDATA "DC-Agent\tools\uv-$uvVersion"
$uvArchive = Join-Path $env:TEMP "uv-$uvVersion-x86_64-pc-windows-msvc.zip"
$uvChecksum = "$uvArchive.sha256"
$uvRelease = "https://github.com/astral-sh/uv/releases/download/$uvVersion"

curl.exe -fL "$uvRelease/uv-x86_64-pc-windows-msvc.zip" -o $uvArchive
curl.exe -fL "$uvRelease/uv-x86_64-pc-windows-msvc.zip.sha256" -o $uvChecksum
$expected = ((Get-Content -Raw $uvChecksum).Trim() -split "\s+")[0].ToLowerInvariant()
$actual = (Get-FileHash -Algorithm SHA256 $uvArchive).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "UV archive checksum mismatch" }
New-Item -ItemType Directory -Force $uvDirectory | Out-Null
Expand-Archive -LiteralPath $uvArchive -DestinationPath $uvDirectory -Force
$env:PATH = "$uvDirectory;$env:PATH"
```

Verify:

```powershell
uv --version
```

Expected: `uv 0.11.29`.

- [ ] **Step 2: Write the minimal project metadata**

Create `backend/pyproject.toml` with this structure, copying the exact version constraints from the three existing requirements inputs:

```toml
[project]
name = "dc-agent-backend"
version = "0.1.0"
description = "DC-Agent backend service"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi>=0.116.0",
  "httpx>=0.28.0",
  "langgraph>=0.2.0,<2.0.0",
  "openpyxl>=3.1.0",
  "psycopg[binary]>=3.2.0",
  "pypdf>=5.0.0",
  "python-docx>=1.1.0",
  "python-multipart>=0.0.20",
  "sqlalchemy>=2.0.0",
  "uvicorn[standard]>=0.35.0",
]

[dependency-groups]
offline = [
  "alembic>=1.16,<2",
  "clickhouse-connect>=0.8,<1",
  "qdrant-client>=1.14,<2",
  "redis>=5,<7",
  "polars>=1.30,<2",
  "pyarrow>=19,<22",
  "pyxlsb>=1.0,<2",
  "docling>=2.40,<3",
  "paddlepaddle>=3,<4",
  "paddleocr>=3,<4",
  "jieba>=0.42,<1",
  "sqlglot>=27,<30",
  "FlagEmbedding>=1.3,<2",
  "onnxruntime>=1.22,<2",
  "psutil>=7,<8",
]
benchmark = [
  { include-group = "offline" },
  "locust>=2.37,<3",
]
dev = [
  "alembic>=1.16,<2",
  "ruff==0.15.22",
]

[tool.uv]
package = false
default-groups = ["dev"]

[tool.ruff]
target-version = "py312"
line-length = 100
extend-exclude = ["uploads"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "UP"]
```

Do not carry `pip-tools` into the benchmark group; UV replaces that workflow.

- [ ] **Step 3: Generate and validate the lock**

Run:

```powershell
uv lock --project backend --python 3.12
uv lock --project backend --check
uv tree --project backend --all-groups
```

Expected: `backend/uv.lock` exists, `uv lock --check` exits 0, and the tree includes base, offline, benchmark, and dev dependencies without unresolved packages.

- [ ] **Step 4: Run the contract tests and commit**

```powershell
py -m unittest `
  tools.tests.test_backend_uv_contract.BackendUvContractTest.test_pyproject_declares_python_groups_and_all_existing_packages `
  tools.tests.test_backend_uv_contract.BackendUvContractTest.test_requirements_inputs_are_removed_after_migration `
  -v
git add backend/pyproject.toml backend/uv.lock backend/requirements.txt backend/requirements-offline.in backend/requirements-benchmark.in tools/tests/test_backend_uv_contract.py
git commit -m "build: migrate backend dependencies to uv"
```

### Task 3: Migrate Dockerfiles and build-context allowlists

**Files:**
- Modify: `deploy/docker/backend.Dockerfile`
- Modify: `deploy/docker/embedding.Dockerfile`
- Modify: `deploy/docker/worker.Dockerfile`
- Modify: `.dockerignore`
- Test: `tools/tests/test_backend_uv_contract.py`

- [ ] **Step 1: Update each Dockerfile to synchronize the frozen project environment**

Replace the requirements/pip installation block in each Dockerfile with this common contract:

```dockerfile
WORKDIR /app

COPY artifacts/wheels /wheels
COPY backend/pyproject.toml backend/uv.lock ./

ENV UV_NO_INDEX=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

RUN uv --version \
    && uv sync --frozen --no-install-project --no-dev --group offline --find-links=/wheels \
    && rm -rf /root/.cache/uv

ENV PATH="/app/.venv/bin:$PATH"
```

Keep the existing source copies, UID/GID checks, non-root user, and service-specific commands. The internal digest-pinned base image must provide `uv`; do not add an unpinned download or public package index to any Dockerfile.

- [ ] **Step 2: Update `.dockerignore`**

Replace the old allowlist entries for requirements files with:

```text
!backend/pyproject.toml
!backend/uv.lock
```

Keep the existing exclusions for uploads, models, benchmarks, secrets, environments, node modules, and Python caches.

- [ ] **Step 3: Run the Docker contract tests and commit**

```powershell
py -m unittest tools.tests.test_backend_uv_contract tools.tests.test_compose_contract -v
git diff --check
git add deploy/docker/backend.Dockerfile deploy/docker/embedding.Dockerfile deploy/docker/worker.Dockerfile .dockerignore tools/tests/test_backend_uv_contract.py
git commit -m "build: synchronize offline images with uv"
```

### Task 4: Run Ruff RED → GREEN across the backend

**Files:**
- Modify: `backend/pyproject.toml`
- Format: `backend/app/**/*.py`
- Format: `backend/tests/**/*.py`
- Format: `backend/alembic/**/*.py`

- [ ] **Step 1: Verify the pre-format baseline is RED**

Run:

```powershell
uv run --project backend --group dev ruff check backend
uv run --project backend --group dev ruff format --check backend
```

Expected: at least one existing lint or formatting failure, proving the new Ruff gate exercises the current source.

- [ ] **Step 2: Apply the requested automated changes**

```powershell
uv run --project backend --group dev ruff check backend --fix
uv run --project backend --group dev ruff format backend
```

Only accept changes under `backend/app`, `backend/tests`, and `backend/alembic`; do not format uploads or generated artifacts.

- [ ] **Step 3: Verify GREEN**

```powershell
uv run --project backend --group dev ruff check backend
uv run --project backend --group dev ruff format --check backend
git diff --check
```

Expected: both Ruff commands exit 0 and the diff has no whitespace errors.

- [ ] **Step 4: Commit the formatting baseline**

```powershell
git add backend/pyproject.toml backend/app backend/tests backend/alembic
git commit -m "style: format backend with ruff"
```

### Task 5: Update developer scripts and offline documentation

**Files:**
- Modify: `README.md`
- Modify: `deploy/offline/README.md`
- Modify: `docs/offline-platform-runbook.md`
- Modify: `tools/start_smoke_backend.cmd`
- Modify: `tools/tests/test_backend_uv_contract.py`

- [ ] **Step 1: Replace developer installation and startup instructions**

Use these commands in the root README and backend smoke script:

```powershell
uv sync --project backend --group dev
Set-Location backend
uv run --project . --group dev python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Set-Location ..
```

The smoke command must set the existing SQLite/template environment variables, then invoke `uv run --project . --group dev python -m uvicorn app.main:app --host 127.0.0.1 --port 8015` from `backend`.

- [ ] **Step 2: Rewrite offline installation guidance**

Replace pip-tools and requirements-file instructions with:

```powershell
uv lock --project backend --python 3.12
uv sync --project backend --frozen --group offline --no-dev --no-index --find-links artifacts/wheels
uv sync --project backend --frozen --no-default-groups --group benchmark --no-index --find-links artifacts/wheels
```

Document that `backend/uv.lock` is the only dependency lock, the wheelhouse must contain every locked artifact, and `UV_PYTHON_DOWNLOADS=never` is required on offline hosts. Remove claims that hashed requirements files are generated.

- [ ] **Step 3: Add documentation contract assertions**

Extend `test_backend_uv_contract.py` to require `uv sync --frozen`, `uv run`, `uv.lock`, and `ruff format` in the relevant documentation, and to reject active references to deleted requirements files.

- [ ] **Step 4: Run documentation and script tests, then commit**

```powershell
py -m unittest tools.tests.test_backend_uv_contract tools.tests.test_ui_smoke -v
git diff --check
git add README.md deploy/offline/README.md docs/offline-platform-runbook.md tools/start_smoke_backend.cmd tools/tests/test_backend_uv_contract.py
git commit -m "docs: document uv and ruff backend workflow"
```

### Task 6: Run the complete migration verification

**Files:**
- Verify: all files changed by Tasks 1–5

- [ ] **Step 1: Validate the lock and synchronized groups**

```powershell
uv lock --project backend --check
uv sync --project backend --frozen --group dev
uv sync --project backend --frozen --group offline --no-dev --no-index --find-links artifacts/wheels
uv sync --project backend --frozen --no-default-groups --group benchmark --no-index --find-links artifacts/wheels
```

If the local wheelhouse is unavailable, run the dev sync and record offline sync as a target-host gate; do not substitute a public index.

- [ ] **Step 2: Run Ruff gates**

```powershell
uv run --project backend --group dev ruff check backend
uv run --project backend --group dev ruff format --check backend
```

- [ ] **Step 3: Run backend and tools tests**

```powershell
Set-Location backend
uv run --project . --group dev python -m unittest discover -s tests -p "test_*.py" -v
uv run --project . --group offline python -m unittest discover -s tests -p "test_offline*.py" -v
Set-Location ..
py -m unittest discover -s tools/tests -p "test_*.py" -v
```

- [ ] **Step 4: Run static repository checks**

```powershell
py -m unittest tools.tests.test_backend_uv_contract -v
py -m compileall -q backend tools
git diff --check
rg -n "requirements(-offline|-benchmark)?\\.(txt|in)|pip install|pip-tools" README.md deploy tools backend --glob '!docs/superpowers/plans/**' --glob '!docs/superpowers/specs/**'
```

Expected: no active requirements/pip workflow references outside historical plan/spec documents, all tests pass, Ruff is clean, and no Docker claim is made locally.

- [ ] **Step 5: Commit final verification evidence**

```powershell
git status --short
git log --oneline -5
```

Keep the worktree clean. Real Docker builds, Compose smoke, and target-host offline wheel/model validation remain explicit follow-up gates.
