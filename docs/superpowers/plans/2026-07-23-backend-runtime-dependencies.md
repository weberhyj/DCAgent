# Backend Runtime Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open backend CORS to all origins, configure Asynctor access logging, and reorganize production and development dependencies.

**Architecture:** Keep application-wide HTTP configuration in the existing `_build_app` factory so every FastAPI entrypoint behaves consistently. Keep deployment libraries in project dependencies, developer CLI tooling in the dev dependency group, and Ruff as an independently installed uv tool.

**Tech Stack:** Python 3.12, FastAPI, Starlette CORS middleware, Asynctor, Gunicorn, uv, pytest, Ruff

---

### Task 1: Lock expected application configuration

**Files:**
- Create: `backend/tests/test_app_configuration.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing CORS and access-log tests**

```python
from unittest.mock import patch

from fastapi.middleware.cors import CORSMiddleware

from app.main import _build_app


def test_build_app_allows_all_cors_origins() -> None:
    application = _build_app()
    cors = next(item for item in application.user_middleware if item.cls is CORSMiddleware)
    assert cors.kwargs["allow_origins"] == ["*"]


def test_build_app_configures_asynctor_access_log() -> None:
    with patch("app.main.config_access_log") as configure:
        application = _build_app()
    configure.assert_called_once_with(application)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run pytest tests/test_app_configuration.py -v`

Expected: FAIL because CORS contains only localhost origins and `app.main.config_access_log` does not exist.

- [ ] **Step 3: Implement application configuration**

```python
from asynctor.contrib.fastapi import config_access_log


def _build_app(*, lifespan: Any | None = None) -> FastAPI:
    app = FastAPI(...)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    config_access_log(app)
    return app
```

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `uv run pytest tests/test_app_configuration.py -v`

Expected: PASS.

### Task 2: Reorganize backend dependencies

**Files:**
- Create: `backend/tests/test_project_dependencies.py`
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`

- [ ] **Step 1: Write failing dependency placement test**

```python
from pathlib import Path
import tomllib


def test_runtime_and_development_dependencies_are_grouped_correctly() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    runtime = project["project"]["dependencies"]
    development = project["dependency-groups"]["dev"]
    assert any(item.startswith("gunicorn") for item in runtime)
    assert any(item.startswith("asynctor") for item in runtime)
    assert any(item.startswith("fastapi-cli") for item in development)
    assert not any(item.startswith("ruff") for item in development)
```

- [ ] **Step 2: Run dependency test to verify RED**

Run: `uv run pytest tests/test_project_dependencies.py -v`

Expected: FAIL because the requested dependencies are not yet grouped correctly.

- [ ] **Step 3: Update pyproject dependencies and lockfile**

Add `gunicorn` and `asynctor` under `[project].dependencies`, replace Ruff with `fastapi-cli` in `[dependency-groups].dev`, and run `uv lock` to regenerate `backend/uv.lock`.

- [ ] **Step 4: Install Ruff as a uv tool**

Run: `uv tool install ruff`

Expected: Ruff is available independently of the backend virtual environment.

- [ ] **Step 5: Run dependency test to verify GREEN**

Run: `uv run pytest tests/test_project_dependencies.py -v`

Expected: PASS.

### Task 3: Full verification

**Files:**
- Verify: `backend/app/main.py`
- Verify: `backend/pyproject.toml`
- Verify: `backend/uv.lock`
- Verify: `backend/tests/test_app_configuration.py`
- Verify: `backend/tests/test_project_dependencies.py`

- [ ] **Step 1: Run focused configuration tests**

Run: `uv run pytest tests/test_app_configuration.py tests/test_project_dependencies.py -v`

Expected: PASS.

- [ ] **Step 2: Run the backend test suite**

Run: `uv run pytest`

Expected: PASS.

- [ ] **Step 3: Run Ruff from the global uv tool installation**

Run: `ruff check app tests`

Expected: PASS.

- [ ] **Step 4: Review the final diff**

Run: `git diff --check` and `git status --short`

Expected: no whitespace errors; only the planned files are modified or added.
