# Unified Knowledge Routing Rollout (Ubuntu + Supervisor)

This runbook applies to the non-Docker deployment at `/opt/DCAgent`. Keep the structured worker
running throughout the knowledge-routing cutover so Excel imports and ClickHouse publication work
continue normally.

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
sudo supervisorctl restart dcagent-structured-worker
sudo supervisorctl status dcagent-api dcagent-structured-worker
```

At this stage keep the API configuration at:

```dotenv
UNIFIED_KNOWLEDGE_ROUTING_ENABLED=false
WORD_FACTUAL_QA_ENABLED=false
```

Confirm both Supervisor programs are `RUNNING`. Do not stop `dcagent-structured-worker`; it remains
responsible for structured Excel imports.

## 2. Reindex Word sources before enabling factual QA

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

Restart only the API and leave the structured worker running:

```bash
cd /opt/DCAgent
sudo supervisorctl restart dcagent-api
sudo supervisorctl status dcagent-api dcagent-structured-worker
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
sudo supervisorctl restart dcagent-api
sudo supervisorctl status dcagent-api dcagent-structured-worker
```

This restores the legacy greeting -> structured -> Agent order without Word factual routing. Do not
drop or truncate `knowledge_facts`, remove ClickHouse tables or Qdrant publications, delete indexed
data, stop `dcagent-structured-worker`, or perform any destructive database action.
