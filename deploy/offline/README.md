# Offline single-server topology

This Compose project is the private, single-server deployment contract for DC-Agent. It exposes only the API on `127.0.0.1:8000`; PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, the embedding service, the reranker service, and optional llama.cpp service remain on the internal Compose network.

## Ubuntu 20.04 事务部署与恢复

生产主路径是 Ubuntu 20.04 Bash 的 `prepare_offline_env.sh`、`invoke_offline_compose.sh` 与
`recover_offline_deployment.sh`；PowerShell 仅用于 Windows 开发机。`DEPLOYMENT_STATE_ROOT` 固定为
`DATA_ROOT/.dcagent-deployment-state` 并绑定 data/model/secret roots。普通 prepare/Compose
不隐式创建 identity；更换 `DATA_ROOT` 视为新部署。

新部署先完成公共前置。以下创建块只适用于首次不存在 `deploy/offline/.env` 的情况；若文件已存在会直接退出，已有 `deploy/offline/.env` 不得覆盖，必须人工审阅并把两个 root 改为同样的 `/srv` 路径后再执行后续核验。

```bash
set -Eeuo pipefail
install -d -m 0700 /srv/dcagent/data /srv/dcagent/models
if [[ -e deploy/offline/.env ]]; then
  printf '%s\n' 'deploy/offline/.env already exists; review it instead of overwriting.' >&2
  exit 1
fi
install -m 0600 deploy/offline/.env.example deploy/offline/.env
deployment_uid="$(id -u)"
deployment_gid="$(id -g)"
sed -i \
  -e 's|^DATA_ROOT=.*$|DATA_ROOT=/srv/dcagent/data|' \
  -e 's|^MODEL_ROOT=.*$|MODEL_ROOT=/srv/dcagent/models|' \
  -e "s|^DCAGENT_UID=.*$|DCAGENT_UID=$deployment_uid|" \
  -e "s|^DCAGENT_GID=.*$|DCAGENT_GID=$deployment_gid|" \
  deploy/offline/.env
grep -Fx 'DATA_ROOT=/srv/dcagent/data' deploy/offline/.env
grep -Fx 'MODEL_ROOT=/srv/dcagent/models' deploy/offline/.env
grep -Fx "DCAGENT_UID=$deployment_uid" deploy/offline/.env
grep -Fx "DCAGENT_GID=$deployment_gid" deploy/offline/.env
```

已有 `.env` 经人工审阅两个 root 和 UID/GID 后必须单独核验，不能用模板覆盖：

```bash
set -Eeuo pipefail
install -d -m 0700 /srv/dcagent/data /srv/dcagent/models
deployment_uid="$(id -u)"
deployment_gid="$(id -g)"
grep -Fx 'DATA_ROOT=/srv/dcagent/data' deploy/offline/.env
grep -Fx 'MODEL_ROOT=/srv/dcagent/models' deploy/offline/.env
grep -Fx "DCAGENT_UID=$deployment_uid" deploy/offline/.env
grep -Fx "DCAGENT_GID=$deployment_gid" deploy/offline/.env
```

公共前置完成后，手工路径与推荐 gate 路径二选一。

### 手工路径

固定核心顺序是 prepare → config → 单次 build → up：

```bash
set -Eeuo pipefail
./tools/prepare_offline_env.sh --initialize-state
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh build schema-migration embedding-service reranker-service api ingestion-worker
./tools/invoke_offline_compose.sh up -d
```

### 推荐 gate 路径

gate 自身执行上述 prepare/config/build/up 固定序列及验收；不要先运行手工路径，否则会重复 build/up。config 60 秒、build 1800 秒、up/readyz 300 秒、每个 probe 60 秒、recovery drill 120 秒。

```bash
set -Eeuo pipefail
python3 tools/intranet_deployment_gate.py --mode fresh --report artifacts/benchmarks/intranet-deployment-gate.json
```

旧部署必须先接管，再普通 prepare：

```bash
set -Eeuo pipefail
./tools/recover_offline_deployment.sh adopt-existing --state-root /absolute/data/root/.dcagent-deployment-state
./tools/prepare_offline_env.sh
```

部署锁超时为 30 秒。六个 Compose verb：config/build/up/down/exec/cp。`./tools/invoke_offline_compose.sh up`、
`./tools/invoke_offline_compose.sh exec` 和 `./tools/invoke_offline_compose.sh cp` 在执行前 durable 写入
`deployment-started.json`，失败保留；`./tools/invoke_offline_compose.sh config`、
`./tools/invoke_offline_compose.sh build`、`./tools/invoke_offline_compose.sh down` 不写 marker。marker 存在时普通
`--rotate-secrets` 拒绝；只有经 `recover_offline_deployment.sh clear-start-marker` 且确认无 `PG_VERSION`、无未完成事务后，才可能恢复 pre-init rotation。任意 `PG_VERSION` 存在后永久拒绝；不提供在线 PostgreSQL role 密码修改，也不提供单行删除 marker 命令。

先运行 `./tools/recover_offline_deployment.sh inspect --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>`。
自动回滚成功后可继续；`rollback_failed` 使用
`./tools/recover_offline_deployment.sh resume-rollback --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>`；
`committed_cleanup_required` 使用
`./tools/recover_offline_deployment.sh finalize-cleanup --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id>`；
损坏 journal/quarantine 修复后使用
`./tools/recover_offline_deployment.sh acknowledge-repaired --state-root /absolute/data/root/.dcagent-deployment-state --transaction <transaction-id> --evidence /absolute/path/sanitized-repair-evidence.json`。
人工运行 `./tools/recover_offline_deployment.sh clear-start-marker --state-root /absolute/data/root/.dcagent-deployment-state` 前，确认无 DC-Agent 容器、无 `PG_VERSION`、PostgreSQL 目录不存在或未初始化、无未完成事务。日志和 evidence receipt 不含 secret、数据库 URL、模型正文或原始 SSE。

开发机本地测试不是Ubuntu live gate通过；缺少真实 Ubuntu Docker、Physoc、Ollama 拓扑时，只能记录 live gate 未运行。

## Prepare local configuration

For a new Ubuntu 20.04 deployment, run `./tools/prepare_offline_env.sh --initialize-state` from the repository root after creating the fixed data/model roots above. An adopted existing deployment runs ordinary `./tools/prepare_offline_env.sh` only after `adopt-existing`. The script copies `.env.example` only when `.env` is absent and creates the PostgreSQL password/database URL secret pair only when neither file exists. It also preserves valid existing ClickHouse role passwords or generates missing 43-character URL-safe passwords at the fixed repository-managed paths. It refuses partial path configuration and never prints secret values. Secret files are staged, validated, permission-restricted, and published without allowing `.env` to redirect them outside `artifacts/secrets`.

An older `.env` with neither ClickHouse password-file key is upgraded in place with the two fixed relative paths while `STRUCTURED_QUERY_ENABLED` remains unchanged (and therefore remains `false` for legacy deployments). If exactly one key exists, preparation fails closed instead of guessing. Existing valid secret files are never overwritten.

The supported production host contract is **Ubuntu 20.04 with Bash and local rootful Linux Compose v2**. The same non-root deployment account must prepare configuration, build the four Python images, and start Compose. `./tools/invoke_offline_compose.sh` is the only supported Compose entry point; do not invoke `docker compose` directly. The wrapper removes every `.env` key and Compose model-selector variable from the child process environment, fixes and inspects the local `default` Docker context, renders every profile with `config --format json`, validates the fixed project name, internal digest-pinned images, approved bind/secret paths, and only then executes the requested Compose arguments. For example, run `./tools/invoke_offline_compose.sh up -d`. Configuration/project overrides, one-off `run`, `create`, `start`, `restart`, build-argument overrides, and `up` flags that skip recreation, builds, dependencies, or alter scale are rejected; use `up` to reconcile stopped services with the validated model. On first generation the preparation script records the account's `id -u` and `id -g` as `DCAGENT_UID` and `DCAGENT_GID`; an existing `.env` and any shell overrides must match those exact non-zero numeric values. The locked `PYTHON_BASE_IMAGE` must be a Debian-family image that provides `groupadd` and `useradd`, with the `dcagent` name and selected IDs unused. The Dockerfiles create and verify `dcagent` with those IDs and still finish as `USER dcagent`; rebuild these host-bound images when the deployment UID/GID changes. Host secret files remain mode `0600`; the secret directory and writable `raw`/`parquet` directories remain owned by the deployment account at mode `0700`.

Every host bind uses `create_host_path: false`, and every Compose interpolation is required with `${VAR:?message}`, so missing or empty values fail configuration instead of falling back to paths such as `/postgres`. Preparation creates only the deployment-account-owned `raw` and `parquet` directories without deleting existing contents. It refuses to continue unless the PostgreSQL, ClickHouse, Qdrant, Redis, and model bind sources already exist, every existing ancestor of the data/model/secret targets is a non-link path, the secret directory is a directory, and an existing secret pair consists of matching regular non-link files. Before startup, inspect the locked vendor images to obtain their actual runtime UID/GID, then pre-create and verify ownership and modes for `${DATA_ROOT}/postgres`, `clickhouse`, `qdrant`, and `redis`; also verify the locked llama image can read `${MODEL_ROOT}`. A mismatch must stop deployment rather than be repaired by broad permissions. The repository-root `.dockerignore` is an allowlist for the wheelhouse, backend runtime/migrations, and Dockerfiles; local secrets, models, uploads, benchmarks, dependency trees, Git metadata, and other artifacts must remain outside the build context.

rootless Docker, Docker `userns` remapping, remote Docker engines/contexts, Windows container UID semantics, SELinux labels, and NFS ownership or root-squash behavior are not supported by this direct UID mapping contract. Treat each as a target-host fail-fast gate. Verify a local default rootful daemon, inspect `docker info`, and use `stat` to confirm owner/mode values before running `./tools/invoke_offline_compose.sh up -d`.

`--rotate-secrets` is a **pre-initialization only** operation. `DATA_ROOT` and `MODEL_ROOT` must be unquoted explicit paths or the exact unquoted `${VAR}` form whose dedicated host variable exists; use names such as `${HOST_DATA_ROOT}` rather than a self-reference such as `${DATA_ROOT}`, because `.env` keys are deliberately removed before Compose starts. The script rejects single-quoted and double-quoted path values rather than interpreting them with semantics that differ from Compose. A missing environment variable, unresolved value, unsupported Compose expansion, invalid path, or mismatching shell override is rejected before any secret or data-directory mutation. The script refuses rotation when a start marker or `${DATA_ROOT}/postgres/PG_VERSION` exists. It does not provide online PostgreSQL role-password modification after initialization; preserve the secret pair and use the recovery runbook instead.

Before deployment, replace every placeholder digest and model checksum in `deploy/offline/.env` with the approved values from the offline artifact lock and internal registry. Do not replace digest references with floating public tags. The digest-pinned PYTHON_BASE_IMAGE must use an approved Debian-family image whose reviewed `uv 0.11.29` binary is preinstalled on PATH. The Dockerfiles do not download uv: they run `uv --version` and then perform the frozen offline sync. On the target host, run `uv --version` from the internal reviewed image before building; all four real image builds remain target-host gates.

## Migration safety

Back up PostgreSQL and verify a tested restore procedure before the first `schema-migration` run. An existing pre-Alembic database is stamped only when its tables, columns, keys, defaults, and indexes exactly match the frozen `20260715_00` baseline. Historical self-healed variants can retain obsolete columns, server defaults, nullable sequence fields, or missing indexes; these are deliberately rejected and must be normalized through a reviewed, backed-up manual procedure before stamping. A mismatch does not stamp or modify the database.

Rollback of the first stamp means restoring the database backup; do not run the baseline downgrade against production data. Subsequent schema changes require their own migration-specific rollback plan.

## Profiles

- The default topology starts data services, schema migration, the embedding service, the reranker
  service, and API.
- `--profile generation` enables the private llama.cpp service after its locked local model is installed.
- `--profile indexing` enables the structured spreadsheet worker (`app.structured_worker`). Keep it disabled while `STRUCTURED_QUERY_ENABLED=false`.

## Ollama-backed Qwen2.5 hybrid retrieval rollout and rollback

DC-Agent no longer loads or runs Embedding/Reranker weights. The lightweight adapter services keep
the private `/v1/embeddings` and `/v1/rerank` contracts, while the approved company-intranet Ollama
instance serves `qwen2.5:0.5b` through `/api/embed` and `qwen2.5:3b` through `/api/generate`.
Ollama must be reachable from the adapter containers before startup; this deployment does not use
an external model API. Ordinary document retrieval remains `Qdrant Dense + Sparse/BM25 + RRF`,
followed by bounded reranking. Exact spreadsheet statistics remain on the structured ClickHouse
path and are never calculated from RAG chunks.

The deployment selector is `RETRIEVAL_MODE=legacy|shadow|qwen3`:

- `legacy` returns only the PostgreSQL Legacy path and is the immediate safety rollback.
- `shadow` returns Legacy results while a sampled, bounded background job compares hybrid rankings.
- `qwen3` is retained as the backward-compatible route name; it routes a stable conversation canary
  to the Ollama-backed hybrid path and safely falls back to Legacy on a
  sanitized retrieval failure.

### Pull and probe the target Ollama models

Run these commands on the approved Ollama host. Pulling is an internal staging action and must not
cause the production DC-Agent host to contact a public model service:

#### Linux (Bash)

```bash
set -Eeuo pipefail
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b
ollama_url='http://127.0.0.1:11434'

embed_json="$(curl --fail-with-body --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"qwen2.5:0.5b","input":["dimension-probe"],"truncate":true,"keep_alive":"30m"}' \
  "$ollama_url/api/embed")"
dimensions="$(python3 -c 'import json,sys; body=json.load(sys.stdin); value=len(body["embeddings"][0]); assert value > 0; print(value)' <<<"$embed_json")"
printf 'EMBEDDING_MODEL_DIMENSIONS=%s\n' "$dimensions"

generate_body="$(python3 - <<'PY'
import json
prompt = 'Return only JSON: {"scores":[{"index":0,"score":0.0},{"index":1,"score":0.0}]}. Score passage relevance to the query from 0 to 1. Query: leave policy. Passage 0: annual leave policy. Passage 1: cafeteria menu.'
print(json.dumps({"model": "qwen2.5:3b", "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 128}}))
PY
)"
generate_json="$(curl --fail-with-body --silent --show-error \
  -H 'Content-Type: application/json' --data-binary "$generate_body" \
  "$ollama_url/api/generate")"
python3 -c 'import json,sys; envelope=json.load(sys.stdin); scores=json.loads(envelope["response"])["scores"]; assert len(scores) == 2; print(json.dumps(scores))' <<<"$generate_json"

tags_json="$(curl --fail-with-body --silent --show-error "$ollama_url/api/tags")"
for model in qwen2.5:0.5b qwen2.5:3b; do
  digest="$(python3 -c 'import json,re,sys; model=sys.argv[1]; body=json.load(sys.stdin); matches=[item for item in body["models"] if item.get("name") == model or item.get("model") == model]; len(matches) == 1 or sys.exit(f"expected exactly one model match: {model}"); digest=str(matches[0]["digest"]).removeprefix("sha256:"); re.fullmatch(r"[0-9a-f]{64}", digest) or sys.exit(f"invalid digest: {model}"); print(digest)' "$model" <<<"$tags_json")"
  printf '%s %s\n' "$model" "$digest"
done
```

Set `EMBEDDING_MODEL_DIMENSIONS` to exactly the value printed by the Bash probe. Older Ollama releases that do
not expose `/api/embed` must use `OLLAMA_EMBEDDING_PATH=/api/embeddings` and set
`EMBEDDING_ENCODING_PROFILE_SHA256=23e5b954b6099dcc4427a33745ad03b9ce7dc6fbf2d8fd4728f1d7e1ce7db34c`;
the adapter then sends one legacy `prompt` request per text. Do not silently switch paths after an
arbitrary error.

Probe JSON score generation separately. This only proves the native Ollama model can return the
required shape; the adapter still applies its fixed prompt, index/count checks, finite `[0,1]`
validation, and ordering before returning `/v1/rerank`. The Bash probe above must exit non-zero if the score count is not two.

Read the actual model digests from `/api/tags`, select the exact target model, remove only the
optional `sha256:` prefix, and reject anything other than 64 lowercase hexadecimal characters. The Bash loop above performs these checks for both approved model names.

Copy those two real values into `EMBEDDING_MODEL_SHA256` and `RERANKER_MODEL_SHA256`. Never copy the
`aaaaaaaa...`/`cccccccc...` examples from `.env.example`; they are placeholders, not model digests.
The locked adapter profiles are:

```env
EMBEDDING_MODEL_NAME=qwen2.5:0.5b
EMBEDDING_MODEL_DIMENSIONS=<len(embeddings[0]) from the target /api/embed probe>
EMBEDDING_MODEL_SHA256=<normalized qwen2.5:0.5b digest from /api/tags>
EMBEDDING_ENCODING_PROFILE_SHA256=fc5141eb8e304cacf598a7ad39ba75dbed3f22fa144c81f918ec58cd1efa3d10
RERANKER_MODEL_NAME=qwen2.5:3b
RERANKER_MODEL_SHA256=<normalized qwen2.5:3b digest from /api/tags>
RERANKER_PROMPT_PROFILE_SHA256=e474bae5997a24385e95ae8fb3bef00ac066a9afe3999aa6e89ceae6d1c72bbd
```

After upgrading, legacy retrieval publications without a complete embedding fingerprint remain
unavailable. Rebuild and activate the collection with the current model digest, dimensions,
protocol, and the profile hash for the selected endpoint.

Startup checks `/api/tags` and fails closed when either configured digest differs from the target
Ollama model. The embedding profile hash is derived from the canonical raw-text/normalization
profile; the reranker hash pins the generated-score prompt contract.

### Egress and wheel/image build contract

Only `embedding-service` and `reranker-service` join `ollama-egress`; API, ingestion worker,
databases, Qdrant, Redis, ClamAV, and other services stay off that network. Restrict the target
firewall to the DC-Agent host and approved Ollama IP/port, and use the Ollama-side proxy/service ACL
to allow only `/api/tags`, `/api/embed` (or the configured legacy `/api/embeddings`), and
`/api/generate`. Do not expose unrestricted Ollama access to other company hosts.

The wheel bundle must come from an approved internal staging process and cover the exact
`backend/uv.lock` for target Linux/Python 3.12. This development machine does not have the target
wheelhouse, so local tests cannot satisfy the image gate. In the actual offline build environment,
use the real artifacts/wheels and verify the same flags used by the Dockerfiles:

```bash
set -Eeuo pipefail
export UV_PYTHON_DOWNLOADS=never
uv sync --project backend --frozen --offline --no-install-project --no-dev --group offline --no-index --find-links artifacts/wheels
uv sync --project backend --frozen --offline --no-install-project --no-dev --no-index --find-links artifacts/wheels
./tools/invoke_offline_compose.sh build schema-migration embedding-service reranker-service api ingestion-worker
```

The first sync matches backend/worker images; the second matches the lightweight adapter images.
The Compose build must use the actual reviewed `PYTHON_BASE_IMAGE`, artifacts and wheels. A missing
wheel or failed image build is a failed deployment gate, not permission to contact PyPI.

### Starting CPU profile

The adapters do not hold model weights, but Ollama capacity is still shared and model generation is
the bottleneck. `qwen2.5:3b` generation-based reranking is a compatibility mode, not equivalent to a
purpose-trained cross-encoder. The initial bounded configuration is
`RETRIEVAL_RERANK_TOP_K=8`, `RETRIEVAL_DEGRADED_RERANK_TOP_K=4`,
`RETRIEVAL_FINAL_TOP_K=4`, and `RETRIEVAL_TOTAL_TIMEOUT_SECONDS=20`. Treat this 8/4/4/20 profile as a
starting safety bound, not a quality or throughput guarantee. The mandatory 15-concurrent-user run
must record latency, error rate, adapter 429/503, Ollama saturation and controlled fallback rate.
The private `/v1/rerank` wire contract still accepts 1–32 passages: keep
`RERANKER_BATCH_MAX_ITEMS=32` so one legal request fits the service batcher, while
`OLLAMA_RERANK_BATCH_MAX_ITEMS=8` bounds each generation call and causes larger requests to run as
ordered consecutive sub-batches. Configure at least 64 output tokens per sub-batch item
(`OLLAMA_RERANK_NUM_PREDICT=512` for the default eight); changing either setting requires rerunning
the target-host capacity gate. `RETRIEVAL_RERANK_TOP_K=8` is a retrieval policy, not the wire limit.

### Build, validate, and activate

Keep the API in `shadow` with `RETRIEVAL_SHADOW_PERCENT=0` and
`RETRIEVAL_CANARY_PERCENT=0` while creating the first index. Prepare, build, and start the validated
topology:

```bash
set -Eeuo pipefail
install -d -m 0700 /srv/dcagent/data /srv/dcagent/models
./tools/prepare_offline_env.sh --initialize-state
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh build schema-migration embedding-service reranker-service api ingestion-worker
./tools/invoke_offline_compose.sh up -d
```

Every existing Word, PDF, TXT, and Excel-derived text chunk must be embedded again. Never reuse
Qwen3 or any other model's old dense vectors: the embedding space and measured dimension belong to
the exact Ollama model/profile/digest combination.

Run the first full rebuild without `--activate`. The worker creates an immutable versioned
collection, validates dimensions, point count, filters, Dense/Sparse search and sample query hits,
then records the publication as `validated` in PostgreSQL:

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh exec -T api \
  python -m app.retrieval_index_worker --collection knowledge_chunks_qwen3_v1
./tools/invoke_offline_compose.sh exec -T postgres \
  psql -U dc_agent -d dc_agent -c "SELECT collection_name, alias_name, status, dimensions, point_count, embedding_model_version, sparse_profile_sha256 FROM retrieval_publications ORDER BY created_at DESC;"
```

Review the publication row and verify the exact embedding model metadata, measured dimensions,
normalized vectors, encoding profile hash, model digest, sparse profile, point count, filters and
fixed quality set. Keep `knowledge_chunks_qwen3_vN` collection names and `RETRIEVAL_MODE=qwen3` for
backward compatibility even though the dense/rerank backends now use Qwen2.5. Collections are
immutable and names are unique.

Run Shadow/Canary and the 15-user gate in an isolated acceptance deployment that uses its own
PostgreSQL audit database and `QDRANT_COLLECTION_ALIAS=knowledge_chunks_acceptance`; it must not
change the production `knowledge_chunks_current` Alias. Build and activate the next immutable
version there through the same publication fence:

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh exec -T api \
  python -m app.retrieval_index_worker --collection knowledge_chunks_qwen3_v2 --activate
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/api/readyz
```

In that acceptance deployment, `--activate` switches only `knowledge_chunks_acceptance`. If
post-switch verification fails, the publisher restores its previous Alias and audit state. Never
mutate a Qdrant Alias directly because that would bypass the PostgreSQL publication fence and make
readiness fail closed.

### Shadow and canary promotion

Use the following sequence, with a configuration reconcile and review at every step:

```text
Shadow 10 -> 50 -> 100
canary 5 -> 25 -> 50 -> 100
```

For Shadow, set `RETRIEVAL_MODE=shadow`, keep `RETRIEVAL_CANARY_PERCENT=0`, and set
`RETRIEVAL_SHADOW_PERCENT` to 10, 50, then 100. For canary, set `RETRIEVAL_MODE=qwen3`, set
`RETRIEVAL_SHADOW_PERCENT=0`, and set `RETRIEVAL_CANARY_PERCENT` to 5, 25, 50, then 100. After each
edit run:

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh up -d
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/api/readyz
```

Stop promotion when any gate fails: Recall@50 below 0.90, NDCG@8 regression, target NDCG gain below
0.05, a critical Top-8 regression, permission leakage, structured aggregate mismatch, retrieval
P95 above 5 seconds, error rate above 1%, or fallback rate above 1%.

At 100% `qwen3` canary, run the mandatory 15-user acceptance command from the repository root. The
benchmark deliberately rejects any mode other than `qwen3` and any canary value other than 100:

```bash
set -Eeuo pipefail
uv run --project backend --group benchmark python tools/hybrid_retrieval_benchmark.py --concurrency 15 --requests 150 --p95-seconds 5 --max-error-rate 0.01 --max-fallback-rate 0.01 --questions-jsonl artifacts/benchmarks/hybrid-questions.jsonl --output-json artifacts/benchmarks/hybrid-retrieval-report.json
```

Preserve the JSON report as deployment evidence and never commit sensitive question text or generated
reports.

Before alias activation, the target-server acceptance set must prove all of the following:

- every indexed and query vector dimension equals the measured/configured dimension;
- adapter `/readyz`, `/v1/metadata`, `/v1/embeddings`, and `/v1/rerank` produce no 5xx;
- every reranked candidate has one finite `[0,1]` score, or the request records a controlled,
  sanitized fallback code;
- a reranker/answer-model failure never returns raw chunks as the final answer;
- Excel aggregation still follows the structured ClickHouse publication path rather than chunk
  arithmetic;
- the 15-user gate meets the latency/error/fallback thresholds approved for the target server.

Only after this set passes may an operator return to the production deployment, confirm
`QDRANT_COLLECTION_ALIAS=knowledge_chunks_current`, build a fresh version, validate it again, and
atomically activate it:

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh exec -T api \
  python -m app.retrieval_index_worker --collection knowledge_chunks_qwen3_v3 --activate
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/api/readyz
```

That final `--activate` atomically switches `knowledge_chunks_current` and the production
PostgreSQL active-publication record. Ollama probes, real image builds, capacity tests, acceptance
promotion, and production Alias activation are manual pre-production gates; this repository change
does not execute them.

### Monitoring

The `retrieval completed` Loguru event contains `request_id`, `mode`, `model_versions`,
`qdrant_alias`, `candidate_counts`, `stage_timings_ms`, `fallback_code`, and `result_count`. Dashboard
the overall and per-stage p50/p95, Qwen/Legacy result counts, fallback codes, and
`fallback_code=circuit_open`. Track Embedding/Reranker HTTP 429 counts as queue saturation, service
readiness/metadata mismatches, Qdrant Alias resolution, host CPU/RSS, and Shadow queue-full counts.
Logs and Shadow audit rows must not contain query text, evidence text, model response text, internal
URLs, or raw exception strings.

### Rollback commands

`RETRIEVAL_MODE=legacy rollback` is the first response to a retrieval incident. Edit exactly that
key in `deploy/offline/.env`, then reconcile and verify readiness:

```bash
set -Eeuo pipefail
grep -q '^RETRIEVAL_MODE=' deploy/offline/.env || {
  echo 'RETRIEVAL_MODE key not found' >&2
  exit 1
}
sed -i 's/^RETRIEVAL_MODE=.*/RETRIEVAL_MODE=legacy/' deploy/offline/.env
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh up -d
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/api/readyz
```

For an `Alias rollback`, remain in Legacy and retain the previous collection plus the previous
approved environment values. Restore the known-good Ollama model digests/profile/dimension values,
reconcile the adapters, then rebuild that known-good model/data combination under a new collection
version and activate it. Do not reuse an existing immutable collection name or point the Alias
directly at an unaudited collection:

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh up -d
./tools/invoke_offline_compose.sh exec -T api \
  python -m app.retrieval_index_worker --collection knowledge_chunks_qwen3_v4 --activate
```

For a model-service image rollback, check out the last approved release in the deployment release
directory, restore its `backend/uv.lock`, Dockerfiles and wheel bundle, then run:

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh build embedding-service reranker-service api ingestion-worker
./tools/invoke_offline_compose.sh up -d
curl --fail-with-body --silent --show-error http://127.0.0.1:8000/api/readyz
```

Only resume Shadow after the Alias, database publication record, model metadata and checksums agree.

## Physoc production gate

生产入口禁止 `template` 和 `mock`；它们只允许用于本地开发和固定测试数据。生产环境的
`deploy/offline/.env` 必须包含经审核的路由值：

```text
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://172.16.0.10:8090
LLM_STREAM_PATH=/api/physoc/deepseeks/stream
LLM_MODEL=my_deepseek_r1_7b
```

`172.16.0.10` 仅为不含凭据的 private IP 示例。先从仓库根目录运行
`./tools/prepare_offline_env.sh`。该脚本只在 `deploy/offline/.env` 不存在时创建配置；已有 `deploy/offline/.env` 不得覆盖。之后操作者只编辑或审核以下 4 个 LLM 路由键：
`LLM_PROVIDER`、`LLM_API_BASE`、`LLM_STREAM_PATH`、`LLM_MODEL`。其他 digest、UID/GID、path 和 secret settings 必须保持原批准值。将 `LLM_API_BASE` 改为容器可达、已批准的 private Physoc
地址后，执行以下切换门禁：

```bash
set -Eeuo pipefail
./tools/prepare_offline_env.sh
# Edit LLM_API_BASE to the approved private Physoc address.
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh up -d
if ! ./tools/invoke_offline_compose.sh exec -T api \
  python -m app.physoc_probe --report /tmp/physoc-probe.json
then
  echo "Physoc probe failed; do not print or persist evidence." >&2
  exit 1
fi
./tools/invoke_offline_compose.sh exec -T api \
  python -c 'import json, pathlib; print(json.dumps(json.loads(pathlib.Path("/tmp/physoc-probe.json").read_text(encoding="utf-8")), indent=2, sort_keys=True))'
mkdir -p artifacts/benchmarks
./tools/invoke_offline_compose.sh cp api:/tmp/physoc-probe.json artifacts/benchmarks/physoc-probe.json
```

只有上一条 probe exit 0 才能执行打印和持久化步骤。复制完成后的 host evidence 为 `artifacts/benchmarks/physoc-probe.json`；不要把容器内 `/tmp` 文件当作持久化审计证据。

通过的 `physoc-probe.json` 形状如下；报告只包含路由、耗时、回答字符数和引用数量，不含
提示词、证据正文或模型回答正文：

```json
{
  "answerChars": 12,
  "citationCount": 1,
  "elapsedMs": 250.0,
  "model": "my_deepseek_r1_7b",
  "passed": true,
  "provider": "physoc_deepseek",
  "streamPath": "/api/physoc/deepseeks/stream"
}
```

只有探针退出成功且报告中的 provider、model 和 streamPath 与部署配置一致时，才允许开放
普通文档问答流量。以下任一情况都必须使探针失败：timeout、non-2xx、wrong content-type、
malformed event JSON、model mismatch、missing done=true 或 empty answer。生产启动若配置为
`template` 或 `mock` 也必须直接失败。

当 Physoc 不可用时，普通文档问题必须返回 HTTP 502，且不得返回检索切片或把切片内容伪装成
模型回答。纯 ClickHouse 结构化统计是确定性计算，不依赖模型路由，不属于 model-route rollback；
ClickHouse 自身失败时仍按结构化统计的显式失败规则处理。

回滚时，把 `LLM_API_BASE` 和 `LLM_MODEL` 恢复为最后已知可用的 Physoc host/model，然后用
`./tools/invoke_offline_compose.sh up -d` 重启并重新执行 probe。禁止把生产环境回滚到
`template`。核心 `offline` 网络保持 `internal`，仅 API 连接 `physoc-egress`；目标主机和防火墙
必须把此出口限制到批准的 private Physoc 地址，不能据此声称核心 internal-only 网络可以访问外部。

## Structured aggregation rollout and rollback

The shipped environment template enables structured Excel/CSV aggregation with
`STRUCTURED_QUERY_ENABLED=true`. Complete all of the following gates before starting from that
template. If the target environment is not ready, explicitly set the flag to `false` and start
without the indexing profile:

1. Back up PostgreSQL, verify restore, and let the one-shot `schema-migration` service apply the
   structured metadata migration.
2. Run `./tools/prepare_offline_env.sh`. It fixes `CLICKHOUSE_QUERY_PASSWORD_FILE` and
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

   ```bash
   set -Eeuo pipefail
   ./tools/invoke_offline_compose.sh up -d clickhouse
   ./tools/invoke_offline_compose.sh exec -T clickhouse /bin/sh /docker-entrypoint-initdb.d/010-dcagent-structured-users.sh
   ```

   Do not enable the feature if this command fails. `--rotate-secrets` deliberately does not rotate
   ClickHouse passwords because changing a file without updating an initialized account would break
   authentication. Each container receives only its own role-specific password file under
   `/run/secrets`.
4. Start the default topology once so migration succeeds:

   ```bash
   set -Eeuo pipefail
   ./tools/invoke_offline_compose.sh up -d
   ```

5. In the administrator UI, upload the XLSX/CSV file, inspect inferred types and aliases, and save a
   confirmed schema. Unconfirmed datasets cannot be published or queried.
6. Verify `STRUCTURED_QUERY_ENABLED=true`, then reconcile the API and start the worker with the
   indexing profile:

   ```bash
   set -Eeuo pipefail
   ./tools/invoke_offline_compose.sh --profile indexing up -d
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

```bash
set -Eeuo pipefail
./tools/invoke_offline_compose.sh down
./tools/invoke_offline_compose.sh up -d
```

Verify ordinary document questions still use the legacy/template path and structured upload routes
are no longer active. Do not delete Parquet parts, ClickHouse tables, or structured metadata during
rollback; retaining them permits a reviewed re-enable.

## Current development gates

`backend/uv.lock` is the only backend Python/uv dependency lock. Python 3.12 must be preinstalled on the target host; uv is forbidden from downloading or installing Python. From the repository root, resolve the lock, then verify both offline groups only against the reviewed wheelhouse:

```bash
set -Eeuo pipefail
export UV_PYTHON_DOWNLOADS=never
uv lock --project backend --python 3.12
uv sync --project backend --frozen --offline --group offline --no-dev --no-index --find-links artifacts/wheels
uv sync --project backend --frozen --offline --no-default-groups --group benchmark --no-index --find-links artifacts/wheels
```

The wheelhouse must contain all wheels and other artifacts required by `backend/uv.lock` for the target Linux platform and Python 3.12, together with approved checksum evidence. Offline hosts must set `UV_PYTHON_DOWNLOADS=never`; neither sync command may fall back to a public package index.

This development machine has neither Docker nor a complete target wheelhouse. Real offline sync, all four Python image builds, Compose rendering, and Compose smoke therefore remain target-host gates. Validate them on the approved Linux host before deployment.

## Offline Compose smoke check

After preparing the offline environment, run `python tools/compose_smoke.py` from the repository root. The smoke runner uses the supported `tools/invoke_offline_compose.sh` wrapper for `config`, `up`, `exec`, and `down`; it starts only `api` and its declared core dependencies, leaving the indexing worker and generation profile disabled. It validates PostgreSQL/Alembic, ClickHouse, Qdrant, Redis, ClamAV, adapter `/readyz`, `/v1/metadata`, `/v1/embeddings`, `/v1/rerank`, and the host-published API readiness endpoint, then atomically writes the audit report to `artifacts/benchmarks/compose-smoke.json`. Adapter results contain only status, latency, vector count/dimension, score count, and a sanitized error code; prompt/query text, document/passages, vector coordinates, raw scores, and generated Ollama response text are never persisted. A failed command, missing executable, malformed response, metadata/dimension mismatch, or non-200 readiness response is a failed smoke check. The runner always attempts `down --remove-orphans` in cleanup, preserves data volumes by default, and removes them only when `--remove-volumes` is explicitly supplied. Docker is not available on this development machine, so this check is a target-host gate and must not be reported as passed locally.

The locked PostgreSQL image must be PostgreSQL 15 or newer (`POSTGRES_MIN_MAJOR=15`) because exact index validation uses `pg_index.indnullsnotdistinct`. Startup rejects older or unreported server versions before catalog inspection with an explicit `PostgreSQL 15+ required` error; there is no compatibility fallback. The PostgreSQL target host must also run the real baseline/stamp and drift-rejection tests against the approved PostgreSQL version. Local unit tests validate catalog-row normalization and advisory lock orchestration, but they do not prove the live `pg_catalog` queries, session advisory lock concurrency, or rollback behavior on the target server. Docker builds of the backend, worker, Embedding, and Reranker images plus the Compose configuration check remain target-host gates.
