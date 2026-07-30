# Ollama Qwen2.5 Retrieval Adapters Design

## Goal

Replace the unavailable Qwen3 retrieval runtimes with models already served by the
company's Ollama instance:

- `qwen2.5:0.5b` supplies dense vectors through Ollama's embedding API.
- `qwen2.5:3b` supplies bounded, generated relevance scores through Ollama's
  generation API.

The rest of DC-Agent keeps its existing private `/v1/embeddings` and `/v1/rerank`
contracts so Qdrant publication, hybrid retrieval, failure handling, audit data, and
the ingestion worker do not depend directly on Ollama's response formats.

## Constraints and Non-goals

- The target server has an older Ollama/runtime stack that cannot load
  `Qwen3-Embedding-0.6B` or `Qwen3-Reranker-0.6B`.
- Ollama is reachable only on an approved loopback or private-network address and
  requires no authentication.
- The Physoc/DeepSeek answer-generation route is unchanged.
- `Qwen2.5-3B` generation is not considered equivalent to a purpose-trained
  cross-encoder reranker. The system must preserve deterministic validation and a
  safe retrieval fallback when generated scores are unavailable or malformed.
- The existing `RETRIEVAL_MODE=qwen3` value remains as a backward-compatible name
  for the dense + sparse + RRF route. Renaming the route, database records, and
  collection publication format is outside this change.
- Existing Qdrant dense vectors cannot be reused because the embedding space and
  likely the vector dimensions change.

## Selected Architecture

Keep the current DC-Agent service boundary and replace only the inference backends:

```text
API / ingestion worker
        |
        +--> POST embedding-service:8081/v1/embeddings
        |         |
        |         +--> POST Ollama /api/embed
        |                    qwen2.5:0.5b
        |
        +--> POST reranker-service:8082/v1/rerank
                  |
                  +--> POST Ollama /api/generate
                             qwen2.5:3b
```

This adapter-first approach is preferred over changing every consumer to Ollama's
native APIs. It preserves the strict metadata contracts in
`embedding_contracts.py` and `reranker_contracts.py`, keeps one place for response
validation, and lets the retrieval path continue to degrade through its existing
router when a dependency is unavailable.

## Embedding Adapter

### Ollama request

The preferred endpoint is `POST /api/embed`:

```json
{
  "model": "qwen2.5:0.5b",
  "input": ["first text", "second text"],
  "truncate": true,
  "keep_alive": "30m"
}
```

Older Ollama installations may expose only `POST /api/embeddings`, whose request
contains one `prompt`. The endpoint is selected explicitly with
`OLLAMA_EMBEDDING_PATH`; allowed values are `/api/embed` and `/api/embeddings`.
For the legacy endpoint, the adapter issues one request per input while retaining
the outer DC-Agent batch contract. It does not silently switch endpoints after
arbitrary failures.

### DC-Agent behavior

The adapter implements the existing `EmbeddingBackend.embed(texts, purpose=...)`
protocol. It extracts Ollama's vectors, converts every coordinate to a finite float,
performs L2 normalization, and verifies that every vector has the configured
dimension. Query and document inputs use the same raw-text profile because the
general Qwen2.5 model has no Qwen3 retrieval instruction contract.

Startup probes exercise both `query` and `document` purposes. A dimension mismatch,
empty vector, zero norm, non-finite coordinate, connection error, or unexpected
Ollama response prevents the adapter service from reporting ready.

The configured `EMBEDDING_MODEL_DIMENSIONS` must be measured from the target Ollama
model by calling the selected endpoint and calculating `len(embeddings[0])`; it is
not assumed from the model name. `retrieval_settings.py` will accept a positive
configured dimension instead of enforcing Qwen3's 1024 dimensions.

## Generative Reranker Adapter

Ollama does not provide the score-vector contract consumed by DC-Agent. The adapter
therefore sends one bounded batch to `POST /api/generate` with:

- model `qwen2.5:3b`;
- `stream: false`;
- JSON output mode when the configured Ollama version supports it;
- temperature zero and a small output-token limit;
- a fixed prompt that requires exactly one `{index, score}` entry for every passage;
- relevance scores in the inclusive range `[0, 1]`.

The required generated object is:

```json
{
  "scores": [
    {"index": 0, "score": 0.92},
    {"index": 1, "score": 0.11}
  ]
}
```

The adapter rejects prose around the JSON, missing or duplicate indices, unknown
indices, non-numeric or non-finite scores, scores outside `[0, 1]`, and count
mismatches. It restores scores to passage order before returning them through the
existing `/v1/rerank` response.

One Ollama call scores a batch; the adapter must not issue one 3B generation call per
passage. Request size remains bounded by the existing contract. The initial
deployment profile reduces generated reranking work:

```env
RETRIEVAL_RERANK_TOP_K=8
RETRIEVAL_DEGRADED_RERANK_TOP_K=4
RETRIEVAL_FINAL_TOP_K=4
RETRIEVAL_TOTAL_TIMEOUT_SECONDS=20
```

These are deployment defaults, not quality guarantees. Capacity tests on the target
server determine whether they can be increased. A timeout, Ollama overload,
malformed JSON, or model failure is surfaced as an unavailable reranker so the
existing retrieval router can fall back instead of returning unvalidated rankings.

## Configuration

The adapter services use these settings:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=qwen2.5:0.5b
OLLAMA_EMBEDDING_PATH=/api/embed
OLLAMA_RERANKER_MODEL=qwen2.5:3b
OLLAMA_GENERATE_PATH=/api/generate
OLLAMA_KEEP_ALIVE=30m
OLLAMA_REQUEST_TIMEOUT_SECONDS=15
OLLAMA_RERANK_FORMAT_JSON=true
```

`OLLAMA_BASE_URL` accepts only loopback/private IP addresses or the explicit internal
service name `ollama`; credentials, query strings, fragments, and public addresses
are rejected. Existing model identity variables remain the source of the strict
DC-Agent metadata response, but their names, versions, dimensions, checksums, and
profile checksums describe the Ollama-backed profiles rather than Qwen3 artifacts.

The embedding and reranker containers no longer load model directories through
Transformers/OpenVINO. They remain lightweight FastAPI adapters. If Ollama runs on
the host or another private machine, only these two services receive a controlled
`ollama-egress` network path; the internal API, databases, Qdrant, and workers do not
receive general egress. The target firewall restricts that path to the approved
Ollama address and port.

## Index Migration

Changing the embedding model requires a full versioned Qdrant rebuild:

1. Pull and probe `qwen2.5:0.5b` in Ollama.
2. Record the returned vector dimension and configure the embedding metadata.
3. Build a new `knowledge_chunks_qwen3_vN` collection using the Ollama adapter.
4. Validate point count, dimensions, sampled vectors, and retrieval results.
5. Atomically switch `knowledge_chunks_current` to the new collection.
6. Keep the previous collection available for rollback until acceptance completes.

No request may query a collection built by another embedding model or dimension.
Reranker changes alone do not require reindexing.

## Error Handling and Observability

- Adapter HTTP clients use persistent connections, bounded timeouts, no redirects,
  and `trust_env=False`.
- Logs contain model names, durations, batch sizes, endpoint paths, and sanitized
  error classes, but never document text, user queries, generated responses, or
  vectors.
- `/readyz` remains the service readiness endpoint and includes the existing pinned
  metadata response.
- Startup performs real model probes rather than checking only `/api/tags`.
- Ollama 429/503/timeouts become dependency-unavailable errors. Invalid generated
  reranker output is treated as a backend failure, never as a zero score.
- `keep_alive` reduces repeated model loads, but deployment must account for the RAM
  required to keep the embedding model, reranker model, and answer model resident.

## Testing

Tests use an injected HTTP transport and do not require a live Ollama server.

1. Embedding adapter tests cover `/api/embed`, the legacy `/api/embeddings` shape,
   batching, normalization, vector counts, dimensions, malformed JSON, and timeouts.
2. Reranker adapter tests cover request prompt construction, JSON score parsing,
   score ordering, duplicate/missing indices, range validation, malformed responses,
   and timeouts.
3. Service tests verify that `/v1/embeddings`, `/v1/rerank`, `/v1/metadata`, and
   `/readyz` retain their existing wire contracts.
4. Settings tests verify private URL enforcement, arbitrary positive embedding
   dimensions, allowed Ollama paths, and configurable Qwen2.5 model identities.
5. Retrieval integration tests verify reranker failure fallback and that embedding
   metadata mismatches fail closed.
6. Compose smoke tests verify the adapter services can reach only the configured
   Ollama endpoint and that the API/worker still reach the adapters on the internal
   network.

## Acceptance Criteria

- A document batch and a query can be embedded through Ollama while consumers still
  use the existing `/v1/embeddings` contract.
- Qdrant rejects no vectors because the configured and actual dimensions match.
- A batch of fused candidates can be scored through `qwen2.5:3b` and returned as one
  finite `[0, 1]` score per passage.
- Malformed or slow Reranker generation causes a controlled fallback and never leaks
  raw chunks as the final answer.
- Existing Physoc/DeepSeek answer generation remains unchanged.
- Unit, integration, Compose validation, index publication validation, and the
  target-server retrieval acceptance set pass before switching the new collection
  alias into production.
