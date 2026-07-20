# Physoc DeepSeek POST SSE Integration Design

## Goal

Replace the backend's direct OpenAI-compatible model call with an internal Physoc DeepSeek streaming endpoint while preserving DC-Agent's existing retrieval, agent, conversation, citation, audit, and frontend contracts.

The upstream contract is:

```http
POST /api/physoc/deepseek/stream
Content-Type: application/json
Accept: text/event-stream
```

```json
{
  "query": "RAG_PROMPT_TEXT",
  "model": "my_deepseek_r1_7b"
}
```

The endpoint does not require authentication. It returns standard server-sent `message` events. Each event's `data` value is JSON:

```json
{
  "model": "my_deepseek_r1_7b",
  "created_at": "2026-07-20T06:21:33.499587311Z",
  "response": "由",
  "done": false
}
```

The stream completes when an event contains `"done": true`.

## Current State

The frontend sends a normal JSON request to:

```text
POST /api/conversations/{conversation_id}/messages
```

The backend performs retrieval and agent inspection, then `OpenAICompatibleLLMProvider` sends a synchronous JSON request to `{LLM_API_BASE}/chat/completions`. The provider expects `choices[0].message.content` in a complete JSON response. The frontend's current streaming effect is a post-response character reveal; it is not an end-to-end SSE connection.

## Chosen Approach

Add a dedicated `PhysocDeepSeekLLMProvider` that consumes the upstream SSE stream inside the backend and buffers its response text. The existing DC-Agent conversation endpoint continues returning a complete `ConversationBundle` after the upstream stream finishes.

This approach is preferred because it:

- preserves the existing frontend API and UI behavior;
- keeps knowledge retrieval, evidence guards, citations, audit records, and persistence in the backend;
- avoids exposing the model endpoint directly to browsers;
- limits the first integration to the provider boundary;
- leaves true browser-visible streaming as a separate future feature.

## Configuration

The new provider is selected with:

```text
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://physoc.internal
LLM_STREAM_PATH=/api/physoc/deepseek/stream
LLM_MODEL=my_deepseek_r1_7b
```

`LLM_STREAM_PATH` defaults to `/api/physoc/deepseek/stream`. `LLM_API_KEY` is not required for this provider.

When `OFFLINE_MODE` is enabled, the combined Physoc URL must pass the existing private-or-loopback URL validation. The configured path must be absolute, must not contain a scheme, host, query, or fragment, and is joined to `LLM_API_BASE` without accepting an arbitrary external URL.

The existing `template` and `openai_compatible` providers remain supported.

## Request Construction

Physoc accepts one `query` string instead of separate system and user messages. The provider constructs that string as:

```text
RAG_SYSTEM_PROMPT

build_prompt(request)
```

This preserves the existing rules that answers must use retrieved evidence only, avoid unsupported claims, return plain text, and omit inline citation markers. The prompt continues to contain bounded evidence snippets, the agent comparison summary, and bounded recent conversation history.

If no reliable knowledge hits exist, the provider returns the existing no-evidence response without calling Physoc.

## SSE Parsing

The provider uses `httpx.Client.stream()` with a POST request, JSON body, and `Accept: text/event-stream`.

The parser:

1. consumes the response incrementally rather than loading the full HTTP body;
2. supports standard SSE records separated by a blank line;
3. accepts `message` events and the SSE default event type when `event:` is omitted;
4. joins multiple `data:` lines according to SSE rules before JSON decoding;
5. ignores comment/heartbeat lines beginning with `:`;
6. requires each decoded payload to be an object with a string `response` and boolean `done`;
7. validates a non-empty upstream `model`, when present, against the configured model;
8. appends each `response` value in arrival order;
9. stops only after receiving `done: true`;
10. rejects a stream that ends before `done: true` or completes with no answer text.

The accumulated answer is passed through the existing `normalize_plain_text_answer()` function before it is converted to `ChatMessageModel`. Existing citation attachment remains unchanged.

The accumulated response is limited to 65,536 Unicode characters so an untrusted or broken stream cannot consume unbounded memory. Exceeding the bound is treated as an invalid upstream response, not a partial success.

## Error Handling

The adapter does not retry generation automatically because retrying a partially consumed model stream can duplicate work and produce inconsistent answers.

Errors map to the existing user-safe `LLMProviderError` categories:

- connection or read timeout: model response timed out;
- non-success HTTP status: model service unavailable;
- malformed SSE, malformed JSON, wrong field types, model mismatch, premature EOF, missing completion marker, empty final answer, or response-size overflow: model response format invalid.

Upstream response bodies, internal URLs, exception details, prompt contents, and retrieved evidence are not included in user-facing errors.

## Application and Frontend Flow

The application flow remains:

```text
Frontend
  -> POST /api/conversations/{id}/messages
  -> repository and read-only knowledge agent
  -> retrieval and evidence inspection
  -> PhysocDeepSeekLLMProvider
  -> POST Physoc SSE endpoint
  -> buffer and normalize answer
  -> persist messages and agent audit
  -> return ConversationBundle
  -> frontend character-reveal animation
```

No frontend route, response schema, conversation schema, or persistence model changes are required in this phase.

## Health and Deployment Boundary

No Physoc health endpoint has been specified. This change therefore does not invent a readiness URL or reuse the unrelated llama.cpp health check. Physoc request failures surface through the existing safe 502 response path.

Deployment documentation and environment examples will describe the new provider variables. Production configuration must use the internal Physoc host; the public DeepSeek API is not used by this provider.

## Testing

Implementation follows test-driven development. Tests are written and observed failing before production code is added.

Provider tests cover:

- exact POST URL, query/model JSON body, and SSE Accept header;
- multiple Unicode response chunks assembled in order;
- default and explicit `message` event types;
- heartbeat comments and multiline `data:` fields;
- `done: true` termination;
- no external call when retrieval returns no evidence;
- timeout and HTTP failures mapped to safe errors;
- malformed JSON and invalid payload field types;
- model mismatch;
- premature EOF without `done: true`;
- empty completed response;
- response-size overflow;
- provider factory configuration, default path, custom path, missing base/model, and private URL enforcement.

Regression verification includes the full backend test suite, Ruff checks, formatting checks, compile checks, and existing tool contracts.

## Non-Goals

- changing the frontend to consume SSE directly;
- changing the public DC-Agent conversation API;
- bypassing RAG and sending the user's raw question directly to the model;
- adding model authentication that the upstream does not require;
- inventing a Physoc readiness endpoint;
- removing the existing template or OpenAI-compatible providers;
- adding retries, fallback to a public model API, or silent partial-answer recovery.

## Known Constraint

The integration depends on the confirmed POST version of `/api/physoc/deepseek/stream`. If the deployed upstream still accepts only GET, the integration must fail configuration or request validation rather than fall back to a long URL query.
