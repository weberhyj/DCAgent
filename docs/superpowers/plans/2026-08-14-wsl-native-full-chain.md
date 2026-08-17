# WSL Ubuntu Native Full Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute the steps task-by-task with verification checkpoints.

**Goal:** Install and start the DC-Agent native Ubuntu chain inside the WSL distribution stored at `E:\WSL\DCAgentUbuntu`, without Docker and without placing runtime data on C:.

**Architecture:** Ubuntu systemd hosts PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, Supervisor, the FastAPI API, the structured worker, and the two frontend dev/build processes. llama.cpp Embedding and Reranker remain separate services on ports 8083 and 8080; the API uses the project’s existing `backend/.env` contract and Supervisor-managed secrets.

**Tech Stack:** Ubuntu 24.04 (WSL2), apt, PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, Supervisor, Python 3.12, uv, Yarn 4.9.2, llama.cpp GGUF services.

## Global Constraints

- Do not use Docker for the native Ubuntu chain.
- Keep Ubuntu’s VHDX and runtime data under `E:\WSL\DCAgentUbuntu` or other explicitly chosen E: paths.
- Do not reuse the Windows Python virtual environment inside WSL.
- ClickHouse compatibility must match the installed server version; use `legacy_18_16` only for ClickHouse 18.16 and `modern` for newer versions.
- Do not start production LLM routing until a reachable Physoc endpoint is configured.

### Task 1: Inspect host and project prerequisites

- Verify Ubuntu user, systemd, free disk, apt availability, project mount, existing service versions, model files, and Physoc reachability.
- Record missing prerequisites before installing anything.

### Task 2: Install base Ubuntu services

- Install required apt packages, PostgreSQL, Redis, Supervisor, ClamAV, build tools, Python 3.12 tooling, Node/Yarn access, and Qdrant/ClickHouse according to the host’s available repositories.
- Create E-backed `/srv/dcagent` data, model, log, and secret directories with `dcagent` ownership.

### Task 3: Configure project runtime

- Create a Linux-only Python virtual environment in WSL.
- Install backend dependencies from the repository lock/configuration.
- Create a WSL-specific environment file and protected secret files without committing credentials.

### Task 4: Configure Supervisor

- Add programs for API, structured worker, llama.cpp Embedding, llama.cpp Reranker, and frontend services.
- Reload Supervisor and verify each process independently.

### Task 5: Start and verify the chain

- Start databases and supporting services first, then migrations, API, worker, model services, and frontends.
- Probe PostgreSQL, ClickHouse, Qdrant, Redis, Embedding, Reranker, API readiness, and structured-worker status.
- Report any missing model or Physoc dependency as a bounded blocker instead of silently enabling template/mock production behavior.
