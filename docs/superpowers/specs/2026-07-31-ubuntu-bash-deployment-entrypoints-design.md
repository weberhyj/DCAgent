# Ubuntu Bash Deployment Entrypoints Design

## Goal

Make Ubuntu 20.04 the primary production deployment environment for DC-Agent by adding Bash entrypoints for environment preparation and validated Docker Compose execution, while preserving the existing PowerShell entrypoints for Windows development compatibility.

## Scope

This change will:

- add `tools/prepare_offline_env.sh`;
- add `tools/invoke_offline_compose.sh`;
- keep `tools/prepare_offline_env.ps1` and `tools/invoke_offline_compose.ps1` available;
- make Bash the documented production path for Ubuntu 20.04;
- remove PowerShell commands from `docs/intranet-deployment-configuration.md`;
- update Linux-oriented deployment documentation and smoke tooling to select the Bash wrapper on POSIX systems;
- retain the current security and fail-closed behavior of the PowerShell deployment path.

This change will not remove Windows development support, relax Compose validation, enable raw `docker compose` as a supported production path, or change the deployed application topology.

## Architecture

The Bash files will be small executable entrypoints using `#!/usr/bin/env bash` and `set -Eeuo pipefail`. They will resolve the repository root without depending on the caller's current directory and will execute Python 3 helpers that implement the complex validation logic.

Two Python helpers will provide the Ubuntu implementation:

- `tools/offline_env.py`: creates or validates `deploy/offline/.env`, records the current non-root Linux UID/GID, creates the managed secret files atomically, enforces owner/mode rules, validates bind roots, and supports explicit secret rotation.
- `tools/offline_compose.py`: validates the requested Compose arguments, forces the local `default` Docker context, clears conflicting environment overrides, renders every profile as JSON, validates images, networks, bind mounts and secret paths, and only then executes Docker Compose.

The existing PowerShell scripts remain valid compatibility entrypoints. They will not be required on Ubuntu and the Ubuntu path must not invoke `pwsh` internally.

## Command Contracts

Ubuntu environment preparation:

```bash
./tools/prepare_offline_env.sh
./tools/prepare_offline_env.sh --rotate-secrets
```

Ubuntu validated Compose execution:

```bash
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh up -d
./tools/invoke_offline_compose.sh --profile indexing up -d
./tools/invoke_offline_compose.sh exec -T api python -m app.physoc_probe
```

Unknown preparation arguments must fail. Missing Compose arguments must fail. The Compose wrapper must preserve argument boundaries exactly and must reject the same unsafe commands and options rejected by the PowerShell implementation.

## Environment Preparation Behavior

The Ubuntu preparation path must preserve these contracts:

1. Copy `deploy/offline/.env.example` only when `deploy/offline/.env` does not exist.
2. Never overwrite a valid existing `.env` automatically.
3. Record `id -u` and `id -g` on first preparation and reject root UID/GID or later mismatches.
4. Require local rootful Docker with the default context contract used by the existing deployment.
5. Keep repository-managed secrets under `artifacts/secrets` and reject redirected, quoted, unresolved or symbolic-link paths.
6. Keep the secret directory at mode `0700` and secret files at mode `0600` on Linux.
7. Create the PostgreSQL password/database URL pair atomically and reject partial pairs.
8. Create separate ClickHouse query and ingest passwords without printing secret values.
9. Preserve existing valid secrets unless `--rotate-secrets` is explicitly supplied.
10. Validate the data/model bind roots and create only the approved writable `raw` and `parquet` directories.

Secret generation will use Python's `secrets` module so Ubuntu does not require OpenSSL solely for password generation.

## Compose Wrapper Behavior

The Ubuntu Compose wrapper must preserve these contracts:

1. Use only `docker --context default compose` with `deploy/offline/.env` and `deploy/offline/compose.yaml`.
2. Reject remote Docker endpoints, non-default contexts and environment-based Compose project/file/profile overrides.
3. Reject unsupported or dangerous commands and flags, including one-off `run`, direct `create`, `start`, `restart`, scale changes and options that bypass builds, dependencies or recreation.
4. Render all profiles with `config --format json` before executing the requested command.
5. Validate the fixed project name, digest-pinned internal images, internal network isolation, API exposure, approved bind sources and repository-managed secret paths.
6. Restore the caller environment after execution and return the real Docker Compose exit code.
7. Never print secret file contents or resolved database credentials.

## Smoke Tool Selection

`tools/compose_smoke.py` will choose the wrapper by operating system:

- POSIX/Linux: `tools/invoke_offline_compose.sh`;
- Windows: `tools/invoke_offline_compose.ps1`.

An explicitly supplied wrapper path will continue to override the default. Existing Windows process handling remains available, while Linux executes the Bash wrapper directly.

## Documentation Rules

The following documents will use Ubuntu Bash commands as their primary deployment examples:

- `docs/intranet-deployment-configuration.md`;
- `docs/offline-platform-runbook.md`;
- `deploy/offline/README.md`;
- the production deployment sections of `README.md`.

`docs/intranet-deployment-configuration.md` must contain no `.ps1`, `Copy-Item`, PowerShell backtick continuation, or `& tools/...` invocation. Multiline commands will use Bash `\` continuation.

The general and Compose README files may retain one short Windows-development compatibility note pointing to the `.ps1` scripts, but Ubuntu remains the only documented production server route.

## Testing

Testing will follow a contract-first approach:

1. Add failing tests for the two new `.sh` entrypoints, LF line endings, executable Git mode and expected Python helper invocation.
2. Add failing unit tests for environment parsing, duplicate keys, secret path confinement, UID/GID validation, secret pair atomicity, rotation and permissions.
3. Add failing unit tests for Compose argument rejection, environment cleanup, rendered JSON validation, internal image digests, networks, binds and secrets.
4. Add failing tests proving `compose_smoke.py` selects `.sh` on POSIX and `.ps1` on Windows.
5. Add documentation contract tests proving the intranet deployment guide is Bash-only and the supported Ubuntu commands are present.
6. Run the existing Compose, Physoc, structured deployment and smoke test suites to prevent regressions in the PowerShell compatibility path.
7. Run Ruff and `git diff --check`.

Real Docker Compose execution remains a target Ubuntu host gate because the current development machine does not provide the production Docker topology.

## Compatibility and Rollout

- Existing Windows developers can continue using the `.ps1` scripts.
- Ubuntu operators will not need PowerShell 7.
- Existing `.env`, bind data and secret files remain compatible; the Bash preparation command validates and preserves them.
- The first Ubuntu deployment must run the Bash preparation command, render Compose configuration, execute the service smoke checks, and verify the Physoc/Ollama routes before receiving production traffic.
- Behavioral drift between Bash and PowerShell paths is treated as a test failure; security checks cannot be silently omitted from either supported entrypoint.

## Acceptance Criteria

The design is complete when:

- both Bash entrypoints work without `pwsh` on Ubuntu 20.04;
- Bash and PowerShell entrypoints preserve the same deployment safety contract;
- the intranet deployment guide is entirely Bash-based;
- POSIX smoke tooling defaults to the Bash Compose wrapper;
- all affected automated tests, Ruff checks and Markdown/diff checks pass;
- the Ubuntu target-host Compose and connectivity gates are documented as required production verification.
