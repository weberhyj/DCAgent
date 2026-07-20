# Offline single-server topology

This Compose project is the private, single-server deployment contract for DC-Agent. It exposes only the API on `127.0.0.1:8000`; PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, the embedding service, and optional llama.cpp service remain on the internal Compose network.

## Prepare local configuration

Run `tools/prepare_offline_env.ps1` from the repository root. The script copies `.env.example` only when `.env` is absent and creates the PostgreSQL password/database URL secret pair only when neither file exists. It refuses a partial pair and never prints secret values. Secret files are staged, validated, permission-restricted, and published as a recoverable pair.

The supported production host contract is **local rootful Linux Compose v2**. The same non-root deployment account must prepare configuration, build the three Python images, and start Compose. `tools/invoke_offline_compose.ps1` is the only supported Compose entry point; do not invoke `docker compose` directly. The wrapper removes every `.env` key and Compose model-selector variable from the child process environment, fixes and inspects the local `default` Docker context, renders every profile with `config --format json`, validates the fixed project name, internal digest-pinned images, approved bind/secret paths, and only then executes the requested Compose arguments. For example, run `& tools/invoke_offline_compose.ps1 up -d`. Configuration/project overrides, one-off `run`, `create`, `start`, `restart`, build-argument overrides, and `up` flags that skip recreation, builds, dependencies, or alter scale are rejected; use `up` to reconcile stopped services with the validated model. On first generation the preparation script records the account's `id -u` and `id -g` as `DCAGENT_UID` and `DCAGENT_GID`; an existing `.env` and any shell overrides must match those exact non-zero numeric values. The locked `PYTHON_BASE_IMAGE` must be a Debian-family image that provides `groupadd` and `useradd`, with the `dcagent` name and selected IDs unused. The Dockerfiles create and verify `dcagent` with those IDs and still finish as `USER dcagent`; rebuild these host-bound images when the deployment UID/GID changes. Host secret files remain mode `0600`; the secret directory and writable `raw`/`parquet` directories remain owned by the deployment account at mode `0700`.

Every host bind uses `create_host_path: false`, and every Compose interpolation is required with `${VAR:?message}`, so missing or empty values fail configuration instead of falling back to paths such as `/postgres`. Preparation creates only the deployment-account-owned `raw` and `parquet` directories without deleting existing contents. It refuses to continue unless the PostgreSQL, ClickHouse, Qdrant, Redis, and model bind sources already exist, every existing ancestor of the data/model/secret targets is a non-link path, the secret directory is a directory, and an existing secret pair consists of matching regular non-link files. Before startup, inspect the locked vendor images to obtain their actual runtime UID/GID, then pre-create and verify ownership and modes for `${DATA_ROOT}/postgres`, `clickhouse`, `qdrant`, and `redis`; also verify the locked llama image can read `${MODEL_ROOT}`. A mismatch must stop deployment rather than be repaired by broad permissions. The repository-root `.dockerignore` is an allowlist for the wheelhouse, backend runtime/migrations, and Dockerfiles; local secrets, models, uploads, benchmarks, dependency trees, Git metadata, and other artifacts must remain outside the build context.

rootless Docker, Docker `userns` remapping, remote Docker engines/contexts, Windows container UID semantics, SELinux labels, and NFS ownership or root-squash behavior are not supported by this direct UID mapping contract. Treat each as a target-host fail-fast gate. Verify a local default rootful daemon, inspect `docker info`, and use `stat` to confirm owner/mode values before running `& tools/invoke_offline_compose.ps1 up -d`.

`-RotateSecrets` is a **pre-initialization only** operation. `DATA_ROOT` and `MODEL_ROOT` must be unquoted explicit paths or the exact unquoted `${VAR}` form whose dedicated host variable exists; use names such as `${HOST_DATA_ROOT}` rather than a self-reference such as `${DATA_ROOT}`, because `.env` keys are deliberately removed before Compose starts. The script rejects single-quoted and double-quoted path values rather than interpreting them with semantics that differ from Compose. A missing environment variable, unresolved value, unsupported Compose expansion, invalid path, or mismatching shell override is rejected before any secret or data-directory mutation. The script refuses rotation when `${DATA_ROOT}/postgres/PG_VERSION` exists, because changing files alone cannot change the password stored in an initialized PostgreSQL role. Rotation after initialization requires a controlled maintenance procedure: stop dependent services, run a reviewed `ALTER ROLE`, update both secret files together, restart services, and verify connectivity. That coordinated workflow is intentionally outside this phase.

Before deployment, replace every placeholder digest and model checksum in `deploy/offline/.env` with the approved values from the offline artifact lock and internal registry. Do not replace digest references with floating public tags.

## Migration safety

Back up PostgreSQL and verify a tested restore procedure before the first `schema-migration` run. An existing pre-Alembic database is stamped only when its tables, columns, keys, defaults, and indexes exactly match the frozen `20260715_00` baseline. Historical self-healed variants can retain obsolete columns, server defaults, nullable sequence fields, or missing indexes; these are deliberately rejected and must be normalized through a reviewed, backed-up manual procedure before stamping. A mismatch does not stamp or modify the database.

Rollback of the first stamp means restoring the database backup; do not run the baseline downgrade against production data. Subsequent schema changes require their own migration-specific rollback plan.

## Profiles

- The default topology starts data services, schema migration, the embedding service, and API.
- `--profile generation` enables the private llama.cpp service after its locked local model is installed.
- `--profile indexing` is reserved for the Phase 2 ingestion worker. The image command intentionally points to the future `app.ingestion_worker` module, so leave this profile disabled until Phase 2 lands.

## Current development gates

`backend/uv.lock` is the only dependency lock. From the repository root, resolve it for Python 3.12, then verify both offline groups only against the reviewed wheelhouse:

```powershell
uv lock --project backend --python 3.12
$env:UV_PYTHON_DOWNLOADS = "never"
uv sync --project backend --frozen --group offline --no-dev --no-index --find-links artifacts/wheels
uv sync --project backend --frozen --no-default-groups --group benchmark --no-index --find-links artifacts/wheels
```

The wheelhouse must contain all wheels and other artifacts required by `backend/uv.lock` for the target Linux platform and Python 3.12, together with approved checksum evidence. Offline hosts must set `UV_PYTHON_DOWNLOADS=never`; neither sync command may fall back to a public package index.

This development machine has neither Docker nor a complete target wheelhouse. Real offline sync, all three image builds, Compose rendering, and Compose smoke therefore remain target-host gates. Validate them on the approved Linux host before deployment.

## Offline Compose smoke check

After preparing the offline environment, run `python tools/compose_smoke.py` from the repository root. The smoke runner uses the supported `tools/invoke_offline_compose.ps1` wrapper for `config`, `up`, `exec`, and `down`; it starts only `api` and its declared core dependencies, leaving the indexing worker and generation profile disabled. It validates PostgreSQL/Alembic, ClickHouse, Qdrant, Redis, ClamAV, the embedding service metadata, and the host-published API readiness endpoint, then atomically writes the deterministic audit report to `artifacts/benchmarks/compose-smoke.json`. A failed command, missing executable, malformed response, non-loopback endpoint, or non-200 readiness response is a failed smoke check. The runner always attempts `down --remove-orphans` in cleanup, preserves data volumes by default, and removes them only when `--remove-volumes` is explicitly supplied. Docker is not available on this development machine, so this check is a target-host gate and must not be reported as passed locally.

The locked PostgreSQL image must be PostgreSQL 15 or newer (`POSTGRES_MIN_MAJOR=15`) because exact index validation uses `pg_index.indnullsnotdistinct`. Startup rejects older or unreported server versions before catalog inspection with an explicit `PostgreSQL 15+ required` error; there is no compatibility fallback. The PostgreSQL target host must also run the real baseline/stamp and drift-rejection tests against the approved PostgreSQL version. Local unit tests validate catalog-row normalization and advisory lock orchestration, but they do not prove the live `pg_catalog` queries, session advisory lock concurrency, or rollback behavior on the target server. A Docker build of all three offline images and the Compose configuration check remain target-host gates.
