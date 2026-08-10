# ClickHouse result overflow setting compatibility fix

## Problem

Structured XLSX publication fails before importing rows because `ClickHouseGateway` sends the
unknown setting `overflow_mode=break`. The configured `max_result_rows` limit is governed by
ClickHouse's `result_overflow_mode` setting instead.

## Design

- Replace `overflow_mode` with `result_overflow_mode` in the shared ClickHouse request settings.
- Keep the existing value `break`, execution-time limit, memory limit, result-row limit, and
  read-only query setting unchanged.
- Apply the corrected settings consistently to DDL, inserts, validation queries, and structured
  read queries through the existing `ClickHouseGateway` boundary.
- Update focused unit tests to reject the obsolete key and require the corrected key.

## Compatibility and deployment

This is a request-setting correction only. It requires no PostgreSQL migration, ClickHouse schema
change, or XLSX re-upload. After deploying the backend change, restart the API and structured worker.
The failed publication job may be retried by the existing retry scheduler; operators can also enqueue
the source publication again if the previous job no longer has a scheduled retry.

## Verification

- Run the focused ClickHouse gateway tests.
- Run the structured ingestion and structured worker tests.
- Confirm a gateway request contains `result_overflow_mode=break` and does not contain
  `overflow_mode`.
