# ClickHouse 18.16 Structured Import Compatibility Design

**Date:** 2026-08-10

## Context

The production intranet environment runs ClickHouse 18.16.1 installed through Ubuntu apt.
The database version is a fixed constraint and cannot be upgraded. Structured XLSX publication
currently remains queued or fails because the application generates ClickHouse features that the
server does not support, including `DateTime64(3)`, `toDecimalString(...)`, and modern SQL RBAC
commands.

Datetime values may be stored with second precision. Milliseconds do not need to be preserved.

## Goals

- Publish existing and new structured XLSX jobs to ClickHouse 18.16.1.
- Preserve the current staging-table, validation, atomic-promotion, lease, and retry model.
- Keep modern ClickHouse behavior available for other environments.
- Make version and capability failures visible before the worker claims a queued job.
- Allow an existing `queued`, `checkpointRow=0`, `attempt=0` job to continue without re-upload.

## Non-goals

- Supporting arbitrary historical ClickHouse releases.
- Preserving sub-second datetime precision in legacy mode.
- Replacing ClickHouse with another structured storage engine.
- Refactoring the unrelated embedding, reranker, or narrative RAG pipelines.
- Automatically rewriting the system-owned `/etc/clickhouse-server/users.xml` file.

## Selected Approach

Add an explicit compatibility profile selected by configuration:

```env
CLICKHOUSE_COMPATIBILITY_MODE=legacy_18_16
```

The default remains `modern`. The two accepted values are `modern` and `legacy_18_16`; any other
value is a startup configuration error. An immutable compatibility object centralizes SQL type and
expression choices so that ingestion and query generation cannot drift apart.

## Architecture

Add a small compatibility module with one responsibility: map a selected compatibility mode to the
ClickHouse syntax supported by that mode. It provides:

- the storage type for each structured column type;
- the query parameter type for each structured column type;
- the canonical SQL rendering expression used by publication validation;
- the bounded query settings allowed for that server generation;
- server version validation.

The compatibility object is created from `OfflineSettings` and injected explicitly into:

- `ClickHouseGateway`, for DDL, publication validation, and query settings;
- `StructuredQueryPlanner`, for parameterized filter types;
- `StructuredAnswerService`, so plan generation and execution-time plan regeneration use the same
  profile;
- `_LazyStructuredQueryGateway`, so the API query connection performs the same version preflight;
- `build_structured_worker`, so publication does not begin before compatibility checks pass.

No module reads the environment variable independently after settings construction.

## Legacy 18.16 SQL Behavior

In `legacy_18_16` mode:

| Logical type | ClickHouse storage type | Query parameter type |
| --- | --- | --- |
| string | `Nullable(String)` | `String` |
| integer | `Nullable(Int64)` | `Int64` |
| decimal | `Nullable(Decimal(38, 9))` | `Decimal(38, 9)` |
| date | `Nullable(Date)` | `Date` |
| datetime | `Nullable(DateTime)` | `DateTime` |
| boolean | `Nullable(UInt8)` | `UInt8` |

Datetime values are normalized to whole seconds before Arrow insertion and before query parameter
binding. Modern mode retains `DateTime64(3)` and millisecond behavior.

Publication content validation uses `toString(decimal_column)` instead of
`toDecimalString(decimal_column, 9)` in legacy mode. Decimal input remains quantized to scale 9,
and the legacy integration test verifies that ClickHouse 18.16 returns the same fixed-scale string
used by the Python-side canonical digest. Other values continue to use `toString(...)`.

The legacy profile sends exactly `max_execution_time`, `max_memory_usage`, `max_result_rows`, and
`result_overflow_mode=break`. Startup preflight executes `SELECT 1` with these settings so a local
server policy that makes one readonly or unavailable is reported before a publication is claimed.
The obsolete `overflow_mode` key is never sent.

## Startup Preflight

Both the structured worker and the lazy structured-query connection run a preflight before their
first workload:

1. Connect using the configured account.
2. Execute `SELECT version()`.
3. Parse the major and minor version.
4. Require an `18.16.x` server for `legacy_18_16` mode.
5. Execute read-only probes for the SQL expressions and settings used by the selected profile.

A legacy-mode connection to another version fails with an actionable configuration error. Modern
mode logs a warning rather than rejecting a newer unrecognized version. Passwords and authenticated
DSNs are never included in errors or logs.

The worker completes preflight before calling `claim_publication`. Consequently, an environmental
failure leaves an unclaimed job queued with attempt zero. The process exits non-zero so Supervisor
can apply its normal restart and backoff policy.

## Structured Publication Flow

After preflight succeeds, the existing flow remains:

1. Claim one queued job and establish its lease.
2. Read the confirmed worksheet schema and source path from PostgreSQL.
3. Create a uniquely owned staging table using legacy-compatible DDL.
4. Stream XLSX rows in bounded batches; do not load the workbook result set into memory at once.
5. After every successfully inserted batch, renew the lease and persist the cumulative inserted row
   count as `checkpointRow`.
6. Validate row count, column count, null counts, schema, and the canonical content digest.
7. Atomically promote the validated staging table.
8. Complete the PostgreSQL publication record and continue optional retrieval metadata indexing.

An existing queued job requires no migration and no re-upload. A retry creates a new staging owner
and may remove only stale staging tables belonging to an older generation of the same publication.
It never removes an active table or a newer staging generation.

`checkpointRow` is operational progress, not a promise of row-level resume. A failed attempt safely
restarts publication into a new staging table; the database retains the highest observed checkpoint
for diagnosis while `attempt` identifies the retry generation.

## Error Handling and Status

- Connection, version, or capability failures happen before job claim and terminate worker startup.
- Publication-specific failures are stored in `errorMessage` and follow the configured retry delay.
- A failed replacement publication leaves the previous active publication available.
- Staging cleanup remains best-effort and cannot replace the primary failure message.
- Errors identify the rejected ClickHouse type, function, setting, or permission without exposing
  credentials.
- Supervisor owns continuous process operation; `python -m app.structured_worker` is not expected to
  be run interactively in production.

## Ubuntu ClickHouse 18.16 Accounts

The existing modern `deploy/offline/clickhouse-init.sh` remains for environments that support SQL
RBAC. It is not used for the Ubuntu 18.16 deployment.

ClickHouse 18.16 accounts are configured through the server's users configuration. The deployment
materials provide a documented XML example for two distinct accounts:

- a read-only query account restricted to the application database;
- an ingestion account restricted to the application database with the DDL and DML capabilities
  available through the legacy access model.

Because legacy ClickHouse cannot express the same fine-grained SQL grants as modern RBAC, the
ingestion account has broader rights inside the allowed application database. It must not be allowed
to access unrelated databases. Operators merge the reviewed configuration into the system-managed
ClickHouse configuration and restart ClickHouse. Project scripts validate connectivity and effective
capabilities but do not overwrite the system configuration automatically.

## Dependency Policy

The first implementation pins `clickhouse-connect==1.6.0` because the observed failure occurs after
a successful connection and is caused by generated SQL. The exact pin prevents an offline rebuild
from silently selecting a newer, untested client.

A separate raw-HTTP legacy client is introduced only if the 18.16 integration test proves that
`clickhouse-connect` uses an unavoidable unsupported protocol feature. It is not part of the initial
implementation.

## Test Strategy

### Unit and contract tests

- Settings accept only `modern` and `legacy_18_16`.
- Modern mode preserves `DateTime64(3)`.
- Legacy mode generates `DateTime` for DDL and query parameters.
- Legacy SQL contains neither `DateTime64` nor `toDecimalString`.
- Datetime values are truncated to whole seconds in legacy mode.
- Worker preflight failure occurs before repository job claim.
- Existing queued jobs are claimed after successful preflight.
- Every completed insert batch persists its cumulative row count as job progress.
- Query plan generation and execution-time regeneration share one compatibility profile.
- Retry and staging-generation safety tests continue to pass.

### ClickHouse 18.16.1 acceptance tests

Run against the actual intranet server or an equivalent isolated 18.16.1 instance:

- verify the server version and both service accounts;
- publish a worksheet containing string, integer, decimal, date, datetime, boolean, null, and Chinese
  text values;
- verify canonical digest agreement, including `Decimal(38, 9)` rendering;
- execute equality, range, ordering, count, sum, average, minimum, and maximum queries;
- publish a worksheet with at least 100,000 rows and confirm bounded memory behavior;
- restart the worker during a publication and verify safe lease recovery without duplicate promotion.

Modern-mode unit and existing integration tests remain regression coverage for newer ClickHouse
deployments.

## Acceptance Criteria

- The structured worker remains `RUNNING` under Supervisor after preflight.
- The current job advances from `queued`; `attempt` and `checkpointRow` become observable as work
  progresses.
- The job reaches `published` without re-uploading the XLSX file.
- Published row/record counts are non-zero and match the worksheet data rows.
- Structured filtering and aggregation work for datetime and decimal columns on ClickHouse 18.16.1.
- Failures expose an actionable status and log message.
- Narrative RAG, embedding, and BGE reranker behavior is unchanged.

## Rollout

1. Deploy the compatibility code and bounded Python dependencies.
2. Configure the legacy query and ingestion accounts in ClickHouse users configuration.
3. Set `CLICKHOUSE_COMPATIBILITY_MODE=legacy_18_16` in the API and structured-worker Supervisor
   environments.
4. Restart ClickHouse only if its users configuration changed.
5. Restart the API and structured worker.
6. Verify preflight logs, then monitor the existing structured job status.
7. Run the small acceptance workbook before allowing the queued large workbook to publish.

Rollback consists of stopping the worker and restoring the previous application build. Existing
active publications remain available, and unclaimed queued jobs remain recoverable.
