# Offline single-server topology

This Compose project is the private, single-server deployment contract for DC-Agent. It exposes only the API on `127.0.0.1:8000`; PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, the embedding service, and optional llama.cpp service remain on the internal Compose network.

## Prepare local configuration

Run `tools/prepare_offline_env.ps1` from the repository root. The script copies `.env.example` only when `.env` is absent and creates the PostgreSQL password/database URL secret pair only when neither file exists. It also preserves valid existing ClickHouse role passwords or generates missing 43-character URL-safe passwords at the fixed repository-managed paths. It refuses partial path configuration and never prints secret values. Secret files are staged, validated, permission-restricted, and published without allowing `.env` to redirect them outside `artifacts/secrets`.

An older `.env` with neither ClickHouse password-file key is upgraded in place with the two fixed relative paths while `STRUCTURED_QUERY_ENABLED` remains unchanged (and therefore remains `false` for legacy deployments). If exactly one key exists, preparation fails closed instead of guessing. Existing valid secret files are never overwritten.

The supported production host contract is **local rootful Linux Compose v2**. The same non-root deployment account must prepare configuration, build the three Python images, and start Compose. `tools/invoke_offline_compose.ps1` is the only supported Compose entry point; do not invoke `docker compose` directly. The wrapper removes every `.env` key and Compose model-selector variable from the child process environment, fixes and inspects the local `default` Docker context, renders every profile with `config --format json`, validates the fixed project name, internal digest-pinned images, approved bind/secret paths, and only then executes the requested Compose arguments. For example, run `& tools/invoke_offline_compose.ps1 up -d`. Configuration/project overrides, one-off `run`, `create`, `start`, `restart`, build-argument overrides, and `up` flags that skip recreation, builds, dependencies, or alter scale are rejected; use `up` to reconcile stopped services with the validated model. On first generation the preparation script records the account's `id -u` and `id -g` as `DCAGENT_UID` and `DCAGENT_GID`; an existing `.env` and any shell overrides must match those exact non-zero numeric values. The locked `PYTHON_BASE_IMAGE` must be a Debian-family image that provides `groupadd` and `useradd`, with the `dcagent` name and selected IDs unused. The Dockerfiles create and verify `dcagent` with those IDs and still finish as `USER dcagent`; rebuild these host-bound images when the deployment UID/GID changes. Host secret files remain mode `0600`; the secret directory and writable `raw`/`parquet` directories remain owned by the deployment account at mode `0700`.

Every host bind uses `create_host_path: false`, and every Compose interpolation is required with `${VAR:?message}`, so missing or empty values fail configuration instead of falling back to paths such as `/postgres`. Preparation creates only the deployment-account-owned `raw` and `parquet` directories without deleting existing contents. It refuses to continue unless the PostgreSQL, ClickHouse, Qdrant, Redis, and model bind sources already exist, every existing ancestor of the data/model/secret targets is a non-link path, the secret directory is a directory, and an existing secret pair consists of matching regular non-link files. Before startup, inspect the locked vendor images to obtain their actual runtime UID/GID, then pre-create and verify ownership and modes for `${DATA_ROOT}/postgres`, `clickhouse`, `qdrant`, and `redis`; also verify the locked llama image can read `${MODEL_ROOT}`. A mismatch must stop deployment rather than be repaired by broad permissions. The repository-root `.dockerignore` is an allowlist for the wheelhouse, backend runtime/migrations, and Dockerfiles; local secrets, models, uploads, benchmarks, dependency trees, Git metadata, and other artifacts must remain outside the build context.

rootless Docker, Docker `userns` remapping, remote Docker engines/contexts, Windows container UID semantics, SELinux labels, and NFS ownership or root-squash behavior are not supported by this direct UID mapping contract. Treat each as a target-host fail-fast gate. Verify a local default rootful daemon, inspect `docker info`, and use `stat` to confirm owner/mode values before running `& tools/invoke_offline_compose.ps1 up -d`.

`-RotateSecrets` is a **pre-initialization only** operation. `DATA_ROOT` and `MODEL_ROOT` must be unquoted explicit paths or the exact unquoted `${VAR}` form whose dedicated host variable exists; use names such as `${HOST_DATA_ROOT}` rather than a self-reference such as `${DATA_ROOT}`, because `.env` keys are deliberately removed before Compose starts. The script rejects single-quoted and double-quoted path values rather than interpreting them with semantics that differ from Compose. A missing environment variable, unresolved value, unsupported Compose expansion, invalid path, or mismatching shell override is rejected before any secret or data-directory mutation. The script refuses rotation when `${DATA_ROOT}/postgres/PG_VERSION` exists, because changing files alone cannot change the password stored in an initialized PostgreSQL role. Rotation after initialization requires a controlled maintenance procedure: stop dependent services, run a reviewed `ALTER ROLE`, update both secret files together, restart services, and verify connectivity. That coordinated workflow is intentionally outside this phase.

Before deployment, replace every placeholder digest and model checksum in `deploy/offline/.env` with the approved values from the offline artifact lock and internal registry. Do not replace digest references with floating public tags. The digest-pinned PYTHON_BASE_IMAGE must use an approved Debian-family image whose reviewed `uv 0.11.29` binary is preinstalled on PATH. The Dockerfiles do not download uv: they run `uv --version` and then perform the frozen offline sync. On the target host, run `uv --version` from the internal reviewed image before building; all three real image builds remain target-host gates.

## Migration safety

Back up PostgreSQL and verify a tested restore procedure before the first `schema-migration` run. An existing pre-Alembic database is stamped only when its tables, columns, keys, defaults, and indexes exactly match the frozen `20260715_00` baseline. Historical self-healed variants can retain obsolete columns, server defaults, nullable sequence fields, or missing indexes; these are deliberately rejected and must be normalized through a reviewed, backed-up manual procedure before stamping. A mismatch does not stamp or modify the database.

Rollback of the first stamp means restoring the database backup; do not run the baseline downgrade against production data. Subsequent schema changes require their own migration-specific rollback plan.

## Profiles

- The default topology starts data services, schema migration, the embedding service, and API.
- `--profile generation` enables the private llama.cpp service after its locked local model is installed.
- `--profile indexing` enables the structured spreadsheet worker (`app.structured_worker`). Keep it disabled while `STRUCTURED_QUERY_ENABLED=false`.

## Structured aggregation rollout and rollback

Structured Excel/CSV aggregation is disabled by default. Keep
`STRUCTURED_QUERY_ENABLED=false` until all of the following gates are complete:

1. Back up PostgreSQL, verify restore, and let the one-shot `schema-migration` service apply the
   structured metadata migration.
2. Run `tools/prepare_offline_env.ps1`. It fixes `CLICKHOUSE_QUERY_PASSWORD_FILE` and
   `CLICKHOUSE_INGEST_PASSWORD_FILE` to repository-managed files under `artifacts/secrets`, creates
   missing values with a CSPRNG, preserves valid existing values, and restricts them to mode `0600`
   under the deployment account. Never put either password directly in `.env`.
3. The version-controlled `clickhouse-init.sh` creates or updates separate least-privilege accounts
   named by `CLICKHOUSE_QUERY_USER` and `CLICKHOUSE_INGEST_USER`. The query account receives only
   `SELECT` on `default.*`; the ingestion account receives the table publication privileges,
   including `SHOW COLUMNS` for governed `DESCRIBE TABLE`, plus read access to `system.tables`. A
   fresh ClickHouse data directory runs the bootstrap through
   `/docker-entrypoint-initdb.d`. For an existing data directory, reconcile the same idempotent
   bootstrap before enabling structured queries:

   ```powershell
   & tools/invoke_offline_compose.ps1 up -d clickhouse
   & tools/invoke_offline_compose.ps1 exec -T clickhouse /bin/sh /docker-entrypoint-initdb.d/010-dcagent-structured-users.sh
   ```

   Do not enable the feature if this command fails. `-RotateSecrets` deliberately does not rotate
   ClickHouse passwords because changing a file without updating an initialized account would break
   authentication. Each container receives only its own role-specific password file under
   `/run/secrets`.
4. Start the default topology once so migration succeeds:

   ```powershell
   & tools/invoke_offline_compose.ps1 up -d
   ```

5. In the administrator UI, upload the XLSX/CSV file, inspect inferred types and aliases, and save a
   confirmed schema. Unconfirmed datasets cannot be published or queried.
6. Set `STRUCTURED_QUERY_ENABLED=true`, then reconcile the API and start the worker with the
   indexing profile:

   ```powershell
   & tools/invoke_offline_compose.ps1 --profile indexing up -d
   ```

Wait for the selected publication to reach `published` before exposing aggregate questions. A
confirmed schema by itself is not queryable; the indexing worker profile must successfully promote
an immutable ClickHouse publication.

For the smoke aggregate gate, use a small reviewed worksheet with known values and nulls. Confirm
its schema, publish it, ask for `avg`, `sum`, `count`, `min`, and `max`, and compare the answer value,
source file, worksheet, total/valid/null counts, schema version, and publication ID with the known
fixture. The gate fails if an aggregate invokes Physoc/template generation or is calculated from
document slices.

If ClickHouse is unavailable or a structured query times out, the API must return an explicit
structured-data unavailable response. It must not fall back to slice arithmetic or the legacy RAG
path for that aggregate question. `STRUCTURED_QUERY_TIMEOUT_SECONDS=4` applies only to the API's
ClickHouse connect/read path. The indexing worker does not inherit that limit; publication retains
the storage gateway's independent 30-second execution default until a dedicated publish setting is
introduced.

Rollback is configuration-only and preserves published data. Set
`STRUCTURED_QUERY_ENABLED=false`, stop the current topology, and restart without the indexing
profile. The worker refuses to start while the feature flag is false, so rollback cannot continue
publishing in the background:

```powershell
& tools/invoke_offline_compose.ps1 down
& tools/invoke_offline_compose.ps1 up -d
```

Verify ordinary document questions still use the legacy/template path and structured upload routes
are no longer active. Do not delete Parquet parts, ClickHouse tables, or structured metadata during
rollback; retaining them permits a reviewed re-enable.

## Current development gates

`backend/uv.lock` is the only backend Python/uv dependency lock. Python 3.12 must be preinstalled on the target host; uv is forbidden from downloading or installing Python. From the repository root, resolve the lock, then verify both offline groups only against the reviewed wheelhouse:

```powershell
$env:UV_PYTHON_DOWNLOADS = "never"
uv lock --project backend --python 3.12
uv sync --project backend --frozen --offline --group offline --no-dev --no-index --find-links artifacts/wheels
uv sync --project backend --frozen --offline --no-default-groups --group benchmark --no-index --find-links artifacts/wheels
```

The wheelhouse must contain all wheels and other artifacts required by `backend/uv.lock` for the target Linux platform and Python 3.12, together with approved checksum evidence. Offline hosts must set `UV_PYTHON_DOWNLOADS=never`; neither sync command may fall back to a public package index.

This development machine has neither Docker nor a complete target wheelhouse. Real offline sync, all three image builds, Compose rendering, and Compose smoke therefore remain target-host gates. Validate them on the approved Linux host before deployment.

## Offline Compose smoke check

After preparing the offline environment, run `python tools/compose_smoke.py` from the repository root. The smoke runner uses the supported `tools/invoke_offline_compose.ps1` wrapper for `config`, `up`, `exec`, and `down`; it starts only `api` and its declared core dependencies, leaving the indexing worker and generation profile disabled. It validates PostgreSQL/Alembic, ClickHouse, Qdrant, Redis, ClamAV, the embedding service metadata, and the host-published API readiness endpoint, then atomically writes the deterministic audit report to `artifacts/benchmarks/compose-smoke.json`. A failed command, missing executable, malformed response, non-loopback endpoint, or non-200 readiness response is a failed smoke check. The runner always attempts `down --remove-orphans` in cleanup, preserves data volumes by default, and removes them only when `--remove-volumes` is explicitly supplied. Docker is not available on this development machine, so this check is a target-host gate and must not be reported as passed locally.

The locked PostgreSQL image must be PostgreSQL 15 or newer (`POSTGRES_MIN_MAJOR=15`) because exact index validation uses `pg_index.indnullsnotdistinct`. Startup rejects older or unreported server versions before catalog inspection with an explicit `PostgreSQL 15+ required` error; there is no compatibility fallback. The PostgreSQL target host must also run the real baseline/stamp and drift-rejection tests against the approved PostgreSQL version. Local unit tests validate catalog-row normalization and advisory lock orchestration, but they do not prove the live `pg_catalog` queries, session advisory lock concurrency, or rollback behavior on the target server. A Docker build of all three offline images and the Compose configuration check remain target-host gates.
