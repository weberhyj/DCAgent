# Unified Project Version Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Manage DC-Agent backend and frontend semantic versions independently with a repeatable component-scoped bump command.

**Architecture:** `backend/app/__init__.py` is the backend version source consumed by PDM Backend and FastAPI. Each frontend keeps its own npm version. A standard-library Python tool bumps exactly one selected component, and repository contract tests prevent cross-component changes.

**Tech Stack:** Python 3.12, PDM Backend, uv, npm, pytest

---

### Task 1: Add the version contract

**Files:**
- Create: `tools/tests/test_version_contract.py`

- [x] **Step 1: Write failing tests for repository consistency and manifest synchronization**
- [x] **Step 2: Run `uv run --project backend --no-sync pytest tools/tests/test_version_contract.py -q` and verify the expected failures**

### Task 2: Implement synchronized semantic versioning

**Files:**
- Create: `tools/bump_version.py`
- Modify: `backend/app/__init__.py`
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Modify: `admin-frontend/package.json`
- Modify: `admin-frontend/package-lock.json`

- [x] **Step 1: Implement strict `X.Y.Z` parsing and patch/minor/major calculation**
- [x] **Step 2: Synchronize one selected application's manifests atomically**
- [x] **Step 3: Configure PDM Backend to read `app.__version__`**
- [x] **Step 4: Bump the backend from `0.1.1` to `0.1.2`, leaving both frontends at `0.1.0`**
- [x] **Step 5: Run `uv lock --project backend` and verify the backend lock check**
- [x] **Step 6: Re-run the focused test and verify it passes**

### Task 3: Document and verify the release workflow

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document default patch bumps and explicit minor/major usage**
- [x] **Step 2: Run focused Ruff, backend build, version tests, both frontend test suites, and both frontend builds**
- [x] **Step 3: Review `git diff` and commit only task files**
