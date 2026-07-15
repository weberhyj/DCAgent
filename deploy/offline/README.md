# Offline single-server topology

This Compose project is the private, single-server deployment contract for DC-Agent. It exposes only the API on `127.0.0.1:8000`; PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, the embedding service, and optional llama.cpp service remain on the internal Compose network.

## Prepare local configuration

Run `tools/prepare_offline_env.ps1` from the repository root. The script copies `.env.example` only when `.env` is absent and creates the PostgreSQL password/database URL secret pair only when neither file exists. It refuses a partial pair and never prints secret values. Use `-RotateSecrets` only for an intentional coordinated rotation of both files.

Before deployment, replace every placeholder digest and model checksum in `deploy/offline/.env` with the approved values from the offline artifact lock and internal registry. Do not replace digest references with floating public tags.

## Migration safety

Back up PostgreSQL and verify a tested restore procedure before the first `schema-migration` run. An existing pre-Alembic database is stamped only when its tables, columns, keys, defaults, and indexes exactly match the frozen `20260715_00` baseline. Historical self-healed variants can retain obsolete columns, server defaults, nullable sequence fields, or missing indexes; these are deliberately rejected and must be normalized through a reviewed, backed-up manual procedure before stamping. A mismatch does not stamp or modify the database.

Rollback of the first stamp means restoring the database backup; do not run the baseline downgrade against production data. Subsequent schema changes require their own migration-specific rollback plan.

## Profiles

- The default topology starts data services, schema migration, the embedding service, and API.
- `--profile generation` enables the private llama.cpp service after its locked local model is installed.
- `--profile indexing` is reserved for the Phase 2 ingestion worker. The image command intentionally points to the future `app.ingestion_worker` module, so leave this profile disabled until Phase 2 lands.

## Current development gates

The hashed `backend/requirements-offline.txt` and `backend/requirements-benchmark.txt` files are intentionally absent until the target Linux Python 3.12 environment and internal wheel mirror are fixed. Dockerfiles already enforce the final contract with `--no-index`, the local wheel directory, and `--require-hashes`; do not fabricate lock files locally.

Docker is not installed on the current development machine, so Compose rendering cannot be verified here. Validate `docker compose --env-file deploy/offline/.env -f deploy/offline/compose.yaml config --quiet` on a host with Docker before deployment.
