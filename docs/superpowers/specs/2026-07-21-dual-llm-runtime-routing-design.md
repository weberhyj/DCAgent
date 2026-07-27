# Dual LLM Runtime Routing Design

## Goal

Package one DC-Agent backend image that can run through either of these model routes without rebuilding application code:

1. the external DeepSeek OpenAI-compatible API;
2. the internal Physoc `POST /api/physoc/deepseek/stream` SSE API.

The selected route is determined at deployment or process startup through environment variables. Secrets and real private endpoints must not be embedded in an image, committed environment example, or frontend bundle.

## Current State

The backend provider factory already supports both implementations:

- `LLM_PROVIDER=openai_compatible` creates `OpenAICompatibleLLMProvider` and calls `<LLM_API_BASE>/chat/completions` with a bearer API key.
- `LLM_PROVIDER=physoc_deepseek` creates `PhysocDeepSeekLLMProvider` and calls `LLM_API_BASE + LLM_STREAM_PATH` with a POST JSON body containing the complete RAG `query` and `model`.

The browser conversation API, retrieval pipeline, citations, audit records, and persistence are provider-independent.

The remaining deployment gaps are:

- the offline Compose contract unconditionally requires `LLM_API_KEY`;
- it does not pass `LLM_STREAM_PATH` into the API container;
- it fixes `OFFLINE_MODE=true`, which correctly rejects public external model endpoints;
- its network is internal-only, so it cannot be the external DeepSeek deployment route;
- there is no pair of route-specific deployment environment examples or route validation command.

## Decision

Use one route-neutral backend image and keep `LLM_PROVIDER` as the canonical selector. Do not add a second alias such as `LLM_ROUTE`, because it would duplicate state and permit contradictory combinations.

The image contains both provider implementations and the same dependency lock. Route selection happens only when the container or Python process starts.

## Runtime Configuration Contracts

### External DeepSeek route

```env
OFFLINE_MODE=false
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://api.deepseek.com
LLM_API_KEY=<deployment secret>
LLM_MODEL=deepseek-chat
```

Required behavior:

- `LLM_API_KEY` is required and must be non-empty.
- `LLM_API_BASE` and `LLM_MODEL` are required.
- public HTTPS endpoints are allowed only when `OFFLINE_MODE=false`.
- `LLM_STREAM_PATH` is not used.
- the deployment network must allow outbound HTTPS access to the approved DeepSeek endpoint.

### Internal Physoc route

```env
OFFLINE_MODE=true
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://10.0.0.8:8090
LLM_STREAM_PATH=/api/physoc/deepseek/stream
LLM_MODEL=my_deepseek_r1_7b
```

Required behavior:

- `LLM_API_BASE` and `LLM_MODEL` are required.
- `LLM_STREAM_PATH` defaults to `/api/physoc/deepseek/stream` and may be overridden with another validated absolute path.
- `LLM_API_KEY` is neither required nor sent. A stale generic environment value may be ignored.
- the endpoint remains restricted to loopback, RFC1918 IPv4, or IPv6 ULA addresses.
- container deployments must use a container-reachable private address; `127.0.0.1` is valid only when the Physoc service runs in the same network namespace as the backend process.

## Packaging and Deployment Shape

The Docker build remains identical for both routes:

```text
dc-agent-backend:<version>
```

The repository provides two non-secret environment examples:

```text
deploy/llm-profiles/external-deepseek.env.example
deploy/llm-profiles/physoc-deepseek.env.example
```

Operators copy exactly one example outside source control, fill the real values, and select it with the deployment command. The selected environment file configures the running container; it is not copied into the Docker image.

The existing offline Compose topology remains the Physoc/private-network deployment base. It will be updated to pass `LLM_STREAM_PATH` and to make `LLM_API_KEY` optional at Compose rendering time, while backend startup continues to require the key for `openai_compatible`.

The external DeepSeek route must not be enabled by merely editing the offline `.env`. It requires a separate egress-capable deployment overlay or non-offline deployment command that sets `OFFLINE_MODE=false`. This preserves the fail-closed internal network contract of `deploy/offline/compose.yaml`.

## Validation Boundary

A small route validation command will validate an environment mapping before build/deployment orchestration proceeds. It will instantiate or validate the selected provider configuration without sending a model request.

Validation rules:

- reject unsupported `LLM_PROVIDER` values;
- reject missing route-specific required variables;
- reject the external route when `OFFLINE_MODE` is true;
- reject public or DNS Physoc endpoints;
- accept the Physoc route without `LLM_API_KEY`;
- never print `LLM_API_KEY` or other secret values in validation output.

Backend startup remains the final fail-fast enforcement point, so direct Python execution and container execution share the same provider rules.

## Data Flow

Both routes keep the same application flow:

```text
browser conversation request
  -> retrieval and Agent evidence selection
  -> complete guarded RAG prompt
  -> provider selected from LLM_PROVIDER
     -> openai_compatible: POST /chat/completions
     -> physoc_deepseek: POST /api/physoc/deepseek/stream and consume SSE
  -> normalize plain-text answer
  -> attach citations and persist audit/conversation data
  -> existing ConversationBundle response
```

No provider selection or model credential is exposed to the frontend.

## Error Handling

- Invalid or incomplete route configuration stops backend startup with a provider-specific configuration error.
- Upstream timeouts, HTTP failures, malformed JSON/SSE, incorrect content type, compressed raw Physoc responses, and interrupted streams continue to map to user-safe model errors.
- Environment validation reports variable names and expected constraints, never secret values.
- Missing knowledge evidence continues to return the no-evidence response without calling either upstream model.

## Testing

Automated coverage will include:

1. provider factory tests for the two valid environment mappings;
2. failure tests for missing API key, missing model/base, invalid offline/public combinations, and unsafe Physoc targets;
3. deployment contract tests confirming the same Dockerfile/image is used for both routes;
4. Compose contract tests confirming `LLM_STREAM_PATH` is passed and `LLM_API_KEY` is not unconditionally required;
5. environment example tests confirming both profiles contain no real key, token, hostname, or private production address;
6. documentation tests covering the two commands and the external-egress versus Physoc-private-network boundary;
7. complete backend, tool, Ruff, compile, and whitespace gates.

Live external DeepSeek and live private Physoc calls remain target-environment smoke gates because they require real credentials and endpoints.

## Non-Goals

- building two different backend images;
- embedding API keys or real private endpoints during Docker build;
- changing the frontend conversation API;
- switching providers dynamically within a running process;
- automatic fallback from one provider to the other;
- weakening the existing Physoc private-address restrictions or offline network isolation.
