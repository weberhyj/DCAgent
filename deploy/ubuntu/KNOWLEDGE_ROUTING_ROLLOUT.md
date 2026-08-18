# Unified Knowledge Routing Rollout (Ubuntu + Supervisor)

This runbook applies to the non-Docker deployment at `/opt/DCAgent`. Keep the structured worker
running throughout the knowledge-routing cutover so Excel imports and ClickHouse publication work
continue normally.

Before this rollout, install and probe the two independent llama.cpp model processes described in
[`LLAMA_CPP_EMBEDDING.md`](./LLAMA_CPP_EMBEDDING.md). Do not rebuild or switch the Qdrant alias until
the BGE-M3 embedding dimension and both GGUF checksums have been recorded.

On a fresh Ubuntu deployment, run the idempotent bootstrap after PostgreSQL, Qdrant, and the
Embedding service are available, but before accepting document questions:

```bash
./tools/bootstrap_retrieval_index.sh
```

The command skips when an active publication already matches the alias, safely skips an empty
knowledge base, and creates the first versioned collection when imported knowledge chunks already
exist. It refuses to continue if the database audit and Qdrant alias disagree.

For Supervisor deployments, use the repository startup wrapper after PostgreSQL, Qdrant, and the
llama.cpp programs have been installed. It starts the model services, performs the bootstrap, and
only then starts the API and structured worker:

```bash
cd /opt/DCAgent
sudo bash tools/start_ubuntu_supervisor_chain.sh
```

The wrapper is fail-closed: if the first Qdrant build cannot complete, the API and structured
worker are not started. On later boots, an already healthy publication is detected and skipped.
If the knowledge base was empty at boot, the first ordinary document import performs the same
initialization automatically before source-level upsert.

The production default is `RETRIEVAL_MODE=qwen3` with `RERANKER_ENABLED=true`. Exact Excel and
Word factual routes remain deterministic and do not call Embedding, Reranker, or LLM; only ordinary
document routes use the BGE-M3 -> Qdrant/BM25 -> BGE-Reranker-v2-M3 -> configured
OpenAI-compatible LLM chain (DeepSeek in the current intranet environment).

## 1. Deploy code, dependencies, and migrations

Run this sequence exactly:

```bash
cd /opt/DCAgent
git pull --ff-only origin main
uv sync --project backend --frozen --offline --no-install-project --no-dev --group offline --no-index --find-links artifacts/wheels
cd /opt/DCAgent/backend
./.venv/bin/python -m app.migration_entrypoint
cd /opt/DCAgent
sudo supervisorctl restart dcagent-api
sudo supervisorctl restart dcagent-ingestion-worker
sudo supervisorctl restart dcagent-structured-worker
sudo supervisorctl status dcagent-api dcagent-ingestion-worker dcagent-structured-worker
```

At this stage keep the API configuration at:

```dotenv
UNIFIED_KNOWLEDGE_ROUTING_ENABLED=false
WORD_FACTUAL_QA_ENABLED=false
```

Confirm all Supervisor programs are `RUNNING`. Do not stop `dcagent-ingestion-worker` or
`dcagent-structured-worker`; they remain responsible for uploaded document parsing and structured
Excel imports.

## 2. Reindex Word sources before enabling factual QA

First create a new Qdrant collection for the BGE-M3 embedding fingerprint. Reindex every ordinary
document into that new collection, validate its vector dimension and sample retrieval results, and
only then atomically switch `QDRANT_COLLECTION_ALIAS`. Never append BGE-M3 vectors to the previous
Ollama/BGE-large collection; keep the old collection and alias target for rollback.

For the first publication, use the bootstrap command above. The lower-level worker remains
available when an operator intentionally chooses a collection name:

```bash
cd /opt/DCAgent/backend
/srv/dcagent/venv/bin/python -m app.retrieval_index_worker \
  --collection knowledge_chunks_qwen3_v1 \
  --activate
```

Keep `WORD_FACTUAL_QA_ENABLED=false`. For every existing Word knowledge source, submit:

```text
POST /api/knowledge/sources/{source_id}/reindex
```

Wait for each source to return to the indexed state. Before cutover, verify:

- every Word source has a non-zero indexed record count where content is present;
- `knowledge_facts` contains the expected entity/field rows for every reindexed Word source;
- the active retrieval publication remains healthy and the Qdrant alias resolves to it;
- no source is left parsing, failed, or pending publication.

Do not enable Word factual QA until the fact counts and retrieval publication health checks pass.

## 3. Cut over the API only

Set:

```dotenv
UNIFIED_KNOWLEDGE_ROUTING_ENABLED=true
WORD_FACTUAL_QA_ENABLED=true
```

Restart the API and ingestion worker, and leave the structured worker running:

```bash
cd /opt/DCAgent
sudo supervisorctl restart dcagent-api dcagent-ingestion-worker
sudo supervisorctl status dcagent-api dcagent-ingestion-worker dcagent-structured-worker
```

The API must refuse startup if `WORD_FACTUAL_QA_ENABLED=true` while
`UNIFIED_KNOWLEDGE_ROUTING_ENABLED=false`.

## 4. Route-audit smoke test

Ask each question and inspect the newest admin Agent run. The expected `routeType` and dependency
use are:

```text
地区为华东的销售额、成本汇总 -> excel_multi_aggregate; no citations; no reranker/LLM
张三几岁 -> word_factual; only age; no reranker/LLM
介绍张三 -> summary_compare; BGE reranker and LLM used
报销流程是什么 -> document_qa; BGE reranker and LLM used
```

Also verify the exact Excel and Word questions still return their expected routes when the LLM is
temporarily unavailable. Excel must not return Word content or invoke document retrieval. The age
answer must contain only the requested age field, even if the source paragraph also contains gender
and job information.

## 5. Rollback

Rollback is configuration-only. Set only these two flags to `false`:

```dotenv
UNIFIED_KNOWLEDGE_ROUTING_ENABLED=false
WORD_FACTUAL_QA_ENABLED=false
```

Then restart only the API:

```bash
cd /opt/DCAgent
sudo supervisorctl restart dcagent-api dcagent-ingestion-worker
sudo supervisorctl status dcagent-api dcagent-ingestion-worker dcagent-structured-worker
```

This restores the legacy greeting -> structured -> Agent order without Word factual routing. Do not
drop or truncate `knowledge_facts`, remove ClickHouse tables or Qdrant publications, delete indexed
data, stop `dcagent-ingestion-worker` / `dcagent-structured-worker`, or perform any destructive
database action.
