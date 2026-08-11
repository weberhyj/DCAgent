# Legacy acceptance opt-in guard fix report

Date: 2026-08-11
Worktree: `E:\project\DCAgent\.worktrees\clickhouse-18-16-compatibility`

## Change

`_target_skip_reason` now returns the explicit `RUN_CLICKHOUSE_18_16=1` skip reason
immediately whenever the opt-in variable is not exactly `1`. Target URL, account,
password-file, and dependency validation runs only for an opted-in target; the
client is still constructed only after those checks.

## TDD and verification

- RED focused harness: the complete non-opt-in target attempted target validation and
  failed with `AssertionError: unexpected target validation`.
- GREEN focused harness: the same complete target returned the explicit skip reason,
  without invoking target validation or dependency imports.
- Bundled Python `-m compileall -q backend/tests/integration/test_clickhouse_legacy_18_16.py`: PASS.
- `git diff --check`: PASS.
- `uv` and `pytest` are unavailable locally; `pyarrow`, `clickhouse_connect`, and a
  real ClickHouse 18.16.1 server are also unavailable, so no pytest/server result is claimed.

## Concerns

Real opt-in acceptance remains dependent on the Ubuntu ClickHouse 18.16.1 target and
locked acceptance dependencies.
