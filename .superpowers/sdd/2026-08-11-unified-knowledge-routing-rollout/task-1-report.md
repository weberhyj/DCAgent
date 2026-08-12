# Task 1 Report: Unified Knowledge Routing

## Status

Completed. The reusable route contract and centralized routers are implemented without changing repository constructors, service implementations, application wiring, or deployment documentation.

## Commit

`9cc94ab87dff832dff516b374a02d39df54c530b` (`feat: centralize knowledge answer routing`).

## RED

Command run from `backend` (using the existing repository virtual environment because `uv` is unavailable):

```powershell
& 'E:\project\DCAgent\backend\.venv\Scripts\python.exe' -m unittest tests.test_knowledge_router -v
```

Observed output (exit code 1):

```text
ModuleNotFoundError: No module named 'app.knowledge_route_models'
```

This confirmed the initial test suite failed because the new route contract did not exist.

The requested `uv run --project . --group dev ...` command was also attempted and failed because `uv` is not installed or on `PATH`.

## GREEN

Command run from `backend`:

```powershell
& 'E:\project\DCAgent\backend\.venv\Scripts\python.exe' -m unittest tests.test_knowledge_router tests.test_agent -v
```

Observed output (exit code 0):

```text
Ran 22 tests in 0.073s

OK
```

## Verification

- Added `KnowledgeRouteType` and validated, immutable `KnowledgeRouteMetadata` serialization.
- Added backward-compatible default route fields at the end of `AgentRunResult` and `AgentRunAudit`; `to_audit()` preserves both.
- Added explicit greeting → Excel → Word → classified document routing with terminal service results.
- Added rollback `LegacyKnowledgeAnswerRouter` with greeting → Excel → document routing and no Word factual service.
- Verified source/open-summary classification and standard document-QA classification.
- Ran `python -m compileall -q app tests/test_knowledge_router.py` successfully.
- Ran `git diff --check` successfully.

## Concerns

- `uv` and `ruff` are unavailable in the supplied environment, so verification used `E:\project\DCAgent\backend\.venv\Scripts\python.exe`; no lint command could be run.
- Service-specific route metadata tagging and repository/router wiring are intentionally deferred to Task 2, per scope.

## Fix Round 1

### RED

Added focused tests covering oversized scalar values, oversized list counts/items, strict boolean fields, and strict `origin_route` input types. Before the fix, this command failed (exit code 1):

```powershell
& 'E:\project\DCAgent\backend\.venv\Scripts\python.exe' -m unittest tests.test_knowledge_router.KnowledgeRouteMetadataTests -v
```

Observed: 5 failures because oversized metadata was serialized/accepted and non-boolean/non-string values were coerced.

### GREEN

After adding bounded validation and strict type checks, this command passed (exit code 0):

```powershell
& 'E:\project\DCAgent\backend\.venv\Scripts\python.exe' -m unittest tests.test_knowledge_router tests.test_agent -v
```

Observed: `Ran 27 tests in 0.081s` / `OK`.

Additional verification passed:

```powershell
& 'E:\project\DCAgent\backend\.venv\Scripts\python.exe' -m compileall -q app tests/test_knowledge_router.py
git diff --check
```

Metadata limits are named and shared: scalar strings max 256 characters, list items max 128 characters, and each metadata list max 32 items. Unknown mapping keys remain ignored by `from_dict`.
