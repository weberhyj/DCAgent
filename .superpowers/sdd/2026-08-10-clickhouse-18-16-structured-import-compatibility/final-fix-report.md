# Final fix report — ClickHouse 18.16 structured import compatibility

Date: 2026-08-11
Worktree: `E:\project\DCAgent\.worktrees\clickhouse-18-16-compatibility`
Branch: `codex/clickhouse-18-16-compatibility`
Implementation commit: `7d88be0` (`fix: close ClickHouse 18.16 acceptance gaps`)

## Scope closed

- Legacy query settings now use `readonly=2`, which keeps the query account read-only while
  allowing ClickHouse 18.16 to accept the bounded settings. Modern remains `readonly=1`.
- Legacy Decimal preflight now requires the exact canonical result `1.000000000`; modern preflight
  behavior is unchanged.
- Ubuntu XML, runbook, design/plan notes, deployment contracts, and gateway tests agree on the
  account policy. The runbook and acceptance guard require exactly `18.16.1`; suffixes and other
  patch releases are rejected.
- Opt-in acceptance collection now skips only when `RUN_CLICKHOUSE_18_16` is absent. With opt-in,
  malformed mode/URL/account/password-file configuration or missing `pyarrow`, `openpyxl`, or
  `clickhouse_connect` raises a collection/setup error before any client connection.
- The real production planner/executor acceptance now covers datetime range/count, equality,
  `gt`/`gte`/`lt`/`lte` comparisons, sum, average, minimum, and maximum. Its small workbook uses
  decimal inputs `1` and `1.2` and asserts stored and `toString` output at scale 9, digest
  agreement, and scale-9 aggregate values.
- Corrected the safe indentation-only issue in `structured_answer.py`. No RAG, embedding,
  reranker, or unrelated Supervisor code was changed.

## TDD evidence

### RED

Before production changes, the new gateway contracts failed as intended:

```text
test_legacy_gateway_uses_datetime_and_never_emits_forbidden_tokens ... FAIL
AssertionError: 1 != 2
test_legacy_preflight_requires_exact_scale_nine_decimal_rendering ... FAIL
AssertionError: StructuredStorageError not raised
```

The updated deployment contract likewise failed on the old XML value (`'1' != '2'`). The exact
version and opt-in guard cases were added before the implementation was finalized.

### GREEN / executable focused checks

- Bundled-Python unittest: ClickHouse gateway focused contracts — **2/2 passed**.
- Bundled-Python unittest: all `test_clickhouse_gateway` tests — **20/20 passed** (with a dotenv
  import stub because project dependencies are unavailable locally).
- Bundled-Python unittest: ClickHouse compatibility profile tests — **4/4 passed** (same stub).
- Bundled-Python unittest: deployment contract module — **19/19 passed**.
- Bundled-Python guard/version harness — **PASS**: only non-opt-in can skip; opt-in invalid target
  and missing dependency fail; only exact `18.16.1` is accepted.
- Bundled-Python `compileall -q backend/app backend/tests tools/tests` — **PASS**.
- XML parse of `deploy/ubuntu/clickhouse-18.16-users.xml.example` — **PASS**.
- `clickhouse-connect` lock consistency (`pyproject.toml` and `uv.lock` both `1.6.0`) — **PASS**.
- `git diff --check` — **PASS**.

Commands that could not run: `uv`/`pytest` are not installed in this environment, and neither
`pyarrow` nor a ClickHouse 18.16.1 server is available. Therefore no pytest result or real-server
acceptance result is claimed.

## Self-audit

- Changed files are limited to the gateway, one safe indentation fix, the legacy acceptance test,
  its unit/deployment contracts, Ubuntu XML/runbook, and compatibility design/plan notes.
- No secrets, real password hashes, DSNs, or credentials were added.
- Worker ingest settings remain unchanged; production legacy compatibility validation remains
  intentionally broad at `18.16.x` while the real acceptance target is exact `18.16.1`.
- No push, merge, or remote operation was performed.

## Concerns

1. Real ClickHouse 18.16.1 acceptance remains outstanding because the target server and locked
   dependencies are unavailable here. Run the opt-in integration command on the Ubuntu target.
2. Full pytest/ruff/type-check coverage remains outstanding for the same local dependency gap.
