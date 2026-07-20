# Backend UV and Ruff Migration Design

## Goal

Replace the backend requirements-file dependency workflow with a backend-local `pyproject.toml` and `uv.lock`, use UV for local and offline dependency synchronization, and establish Ruff as the backend linting and formatting baseline without changing application behavior.

## Scope

The migration covers the complete backend dependency chain:

- delete `backend/requirements.txt`;
- delete `backend/requirements-offline.in`;
- delete `backend/requirements-benchmark.in`;
- create `backend/pyproject.toml`;
- create and commit `backend/uv.lock`;
- update all Dockerfiles, scripts, contract tests, README files, and runbooks that reference requirements files or pip-tools;
- apply Ruff lint fixes and formatting to `backend/app`, `backend/tests`, and `backend/alembic`.

The migration does not add new application features, alter API contracts, run real Docker integration checks, or generate target-host benchmark results.

## Project and Dependency Structure

`backend` becomes an independent UV project. `backend/pyproject.toml` declares:

- project metadata and `requires-python = ">=3.12,<3.13"`;
- the current lightweight backend runtime packages in `project.dependencies`;
- an `offline` dependency group for database services, parsers, OCR, local embedding, and offline operational dependencies;
- a `benchmark` dependency group that includes the offline group and adds Locust and benchmark-only tooling;
- a `dev` dependency group containing Ruff and developer-only tooling.

The backend remains a non-packaged application project. UV creates and synchronizes `backend/.venv`; application imports continue to resolve from the backend working directory rather than from an installed wheel.

`backend/uv.lock` is the only committed dependency lock. Normal development and verification use `uv sync --frozen` or `uv run`, so commands fail instead of silently changing the lock. The lock targets Python 3.12 exclusively to match the offline Linux production contract.

## Offline and Docker Installation

The digest-pinned internal Python 3.12 base image must contain an approved, pinned UV binary. Docker builds do not download UV or Python and do not call pip.

Each backend Dockerfile:

1. copies the reviewed local wheelhouse;
2. copies `backend/pyproject.toml` and `backend/uv.lock`;
3. sets `UV_PYTHON_DOWNLOADS=never`, `UV_NO_INDEX=1`, and a copy-based link mode suitable for container layers;
4. runs `uv sync --frozen --no-install-project --no-dev --group offline --find-links=/wheels`;
5. exposes `/app/.venv/bin` through `PATH`;
6. starts the API, embedding service, or worker with the synchronized virtual environment.

The Compose and Docker contract tests must reject any remaining requirements-file or pip-install references. Target-host documentation must require the internal wheelhouse to contain every locked artifact for the selected dependency group.

## Ruff Baseline

Ruff configuration lives in `backend/pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 100
extend-exclude = ["uploads"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "UP"]
```

The initial migration runs Ruff across `backend/app`, `backend/tests`, and `backend/alembic`:

```powershell
uv run --project backend --group dev ruff check backend --fix
uv run --project backend --group dev ruff format backend
uv run --project backend --group dev ruff check backend
uv run --project backend --group dev ruff format --check backend
```

Generated uploads are excluded. Ruff changes are limited to import ordering, selected safe lint fixes, Python 3.12 syntax normalization, and deterministic formatting. Application behavior changes are outside scope.

## Developer Commands

The documented local workflow becomes:

```powershell
uv sync --project backend --group dev
Set-Location backend
uv run --project . --group dev python -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..
uv run --project backend --group dev uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Offline and benchmark environments use explicit groups and frozen synchronization. Scripts may invoke `uv run --project backend` or the synchronized `backend/.venv` interpreter, but must not fall back to an unmanaged system interpreter for backend runtime commands.

## Tests and Verification

Migration contract tests are written before implementation and must initially fail because `pyproject.toml`, `uv.lock`, and the new Docker contract are absent. They cover:

- exact Python version range;
- preservation of every dependency currently declared across the three requirements inputs;
- offline, benchmark, and dev group boundaries;
- absence of requirements-file references from active build and runtime files;
- frozen, no-index Docker synchronization through UV;
- Ruff target version, source scope, and lint selection;
- README and offline runbook command updates.

The final verification sequence is:

```powershell
uv lock --project backend --check
uv sync --project backend --frozen --group dev
uv run --project backend --group dev ruff check backend
uv run --project backend --group dev ruff format --check backend
Set-Location backend
uv run --project . --group dev python -m unittest discover -s tests -p "test_*.py" -v
uv run --project . --group offline python -m unittest discover -s tests -p "test_offline*.py" -v
Set-Location ..
py -m unittest discover -s tools/tests -p "test_*.py" -v
git diff --check
```

Real Docker smoke, internal image verification, and large capacity benchmarks remain target-host gates because the current development machine does not provide the approved internal base image, complete wheelhouse, models, or Docker environment.

## Commit Strategy

The implementation is split into reviewable commits:

1. dependency metadata, UV lock, Docker migration, and dependency contract tests;
2. Ruff configuration, automated lint fixes, and full backend formatting;
3. scripts and documentation updates plus final regression evidence.

Each commit must leave its focused tests passing. The final branch must pass the full backend and tools suites before it is pushed or merged.
