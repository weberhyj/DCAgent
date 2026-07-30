# Ollama Qwen2.5 Retrieval Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route dense embeddings to Ollama `qwen2.5:0.5b` and bounded generative reranking to Ollama `qwen2.5:3b` while preserving DC-Agent's existing private `/v1/embeddings` and `/v1/rerank` contracts.

**Architecture:** Add one strict private Ollama HTTP client plus focused embedding and reranker backend adapters. Keep the existing FastAPI service contracts and hybrid retrieval consumers unchanged, generalize pinned model metadata, and update Compose so only the two adapter services can reach the approved private Ollama endpoint.

**Tech Stack:** Python 3.12, FastAPI, httpx, Pydantic, unittest, Docker Compose, Qdrant, Ollama REST API, uv, Ruff.

---

## File Structure

- Create `backend/app/ollama_client.py`: private URL validation, persistent HTTP transport, bounded JSON calls, and sanitized Ollama exceptions.
- Create `backend/app/ollama_embedding_backend.py`: `/api/embed` and legacy `/api/embeddings` request/response adaptation plus L2 normalization.
- Create `backend/app/ollama_reranker_backend.py`: fixed batch prompt, `/api/generate` call, exact JSON parsing, score validation, and query grouping.
- Create `backend/tests/test_ollama_client.py`: URL and transport behavior.
- Create `backend/tests/test_ollama_embedding_backend.py`: modern and legacy embedding behavior.
- Create `backend/tests/test_ollama_reranker_backend.py`: prompt and generated-score behavior.
- Modify `backend/app/embedding_service.py`: load metadata from environment and construct the Ollama embedding backend at startup.
- Modify `backend/app/reranker_service.py`: load metadata from environment and construct the Ollama reranker backend at startup.
- Modify `backend/app/retrieval_settings.py`: remove Qwen3-only model-name and 1024-dimension locks.
- Modify service/settings/integration tests to retain the current private wire contracts.
- Modify Dockerfiles, Compose, environment examples, deployment validation, and documentation for Ollama-backed adapter services.

### Task 1: Private Ollama HTTP Client

**Files:**
- Create: `backend/app/ollama_client.py`
- Create: `backend/tests/test_ollama_client.py`

- [ ] **Step 1: Write failing URL and transport tests**

Create `backend/tests/test_ollama_client.py` with tests that define the public API before implementation:

```python
from __future__ import annotations

import unittest
from collections.abc import Mapping

import httpx

from app.ollama_client import (
    OllamaBusy,
    OllamaResponseError,
    OllamaServiceError,
    SyncOllamaClient,
)


class RecordingTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, object], float | None]] = []

    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> object:
        self.calls.append((url, payload, timeout_seconds))
        return self.response

    def close(self) -> None:
        pass


class FailingTransport(RecordingTransport):
    def __init__(self, error: Exception) -> None:
        super().__init__({})
        self.error = error

    def post_json(self, *args: object, **kwargs: object) -> object:
        raise self.error


class OllamaClientTest(unittest.TestCase):
    def test_accepts_only_private_ollama_base_urls(self) -> None:
        for value in (
            "http://127.0.0.1:11434",
            "http://10.20.30.40:11434/",
            "http://ollama:11434",
        ):
            with self.subTest(value=value):
                client = SyncOllamaClient(value, transport=RecordingTransport({"ok": True}))
                self.assertEqual(client.post_json("/api/tags", {}), {"ok": True})

        for value in (
            "https://public.example",
            "http://user:secret@127.0.0.1:11434",
            "http://127.0.0.1:11434/path",
            "http://127.0.0.1:11434?x=1",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SyncOllamaClient(value, transport=RecordingTransport({}))

    def test_posts_only_absolute_api_paths_and_requires_object_json(self) -> None:
        transport = RecordingTransport({"model": "qwen2.5:0.5b"})
        client = SyncOllamaClient("http://ollama:11434", transport=transport)

        result = client.post_json("/api/embed", {"input": ["text"]}, timeout_seconds=3.5)

        self.assertEqual(result["model"], "qwen2.5:0.5b")
        self.assertEqual(transport.calls[0][0], "http://ollama:11434/api/embed")
        self.assertEqual(transport.calls[0][2], 3.5)
        with self.assertRaises(ValueError):
            client.post_json("api/embed", {})
        with self.assertRaises(OllamaResponseError):
            SyncOllamaClient(
                "http://ollama:11434", transport=RecordingTransport(["not", "an", "object"])
            ).post_json("/api/embed", {})

    def test_maps_busy_status_and_connection_failures(self) -> None:
        busy_response = httpx.Response(429, request=httpx.Request("POST", "http://ollama/api"))
        busy_error = httpx.HTTPStatusError("busy", request=busy_response.request, response=busy_response)
        with self.assertRaises(OllamaBusy):
            SyncOllamaClient(
                "http://ollama:11434", transport=FailingTransport(busy_error)
            ).post_json("/api/generate", {})

        with self.assertRaises(OllamaServiceError):
            SyncOllamaClient(
                "http://ollama:11434",
                transport=FailingTransport(httpx.ConnectError("secret endpoint failed")),
            ).post_json("/api/embed", {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
uv run --directory backend python -m unittest tests.test_ollama_client -v
```

Expected: import failure for `app.ollama_client` because the module does not exist.

- [ ] **Step 3: Implement the minimal private client**

Create `backend/app/ollama_client.py` with:

```python
from __future__ import annotations

import math
from collections.abc import Mapping
from ipaddress import ip_address
from typing import Protocol
from urllib.parse import urlparse

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaBusy(OllamaError):
    pass


class OllamaServiceError(OllamaError):
    pass


class OllamaResponseError(OllamaError):
    pass


class OllamaTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> object: ...

    def close(self) -> None: ...


class _HttpxTransport:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> object:
        kwargs: dict[str, object] = {}
        if timeout_seconds is not None:
            kwargs["timeout"] = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0))
        response = self.client.post(url, json=dict(payload), **kwargs)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.client.close()


def _private_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("OLLAMA_BASE_URL must be a non-empty URL")
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OLLAMA_BASE_URL must use HTTP or HTTPS and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OLLAMA_BASE_URL must not include credentials")
    if parsed.path.rstrip("/") or parsed.query or parsed.fragment or parsed.params:
        raise ValueError("OLLAMA_BASE_URL must not include a path, query, or fragment")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("OLLAMA_BASE_URL must use a valid port") from error
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("OLLAMA_BASE_URL must use a valid port")
    if parsed.hostname not in {"localhost", "ollama"}:
        try:
            address = ip_address(parsed.hostname)
        except ValueError as error:
            raise ValueError("OLLAMA_BASE_URL must use ollama or a private/loopback IP") from error
        if (
            address.is_unspecified
            or address.is_multicast
            or address.is_link_local
            or not (address.is_private or address.is_loopback)
        ):
            raise ValueError("OLLAMA_BASE_URL must use ollama or a private/loopback IP")
    return value.strip().rstrip("/")


class SyncOllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: OllamaTransport | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.base_url = _private_base_url(base_url)
        self.timeout_seconds = float(timeout_seconds)
        self.transport = transport or _HttpxTransport(
            httpx.Client(
                timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
                follow_redirects=False,
                trust_env=False,
            )
        )
        self.closed = False

    def post_json(
        self,
        path: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, object]:
        if self.closed:
            raise OllamaServiceError("Ollama client is closed")
        if not path.startswith("/api/") or "?" in path or "#" in path:
            raise ValueError("Ollama path must be an absolute /api/ path")
        try:
            raw = self.transport.post_json(
                f"{self.base_url}{path}",
                payload,
                timeout_seconds=timeout_seconds,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                raise OllamaBusy("Ollama is busy") from error
            raise OllamaServiceError("Ollama returned an unsuccessful status") from error
        except (httpx.TimeoutException, httpx.RequestError) as error:
            raise OllamaServiceError("Ollama request failed") from error
        except OllamaError:
            raise
        except Exception as error:
            raise OllamaServiceError("Ollama request failed") from error
        if not isinstance(raw, Mapping):
            raise OllamaResponseError("Ollama returned a non-object JSON payload")
        return raw

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.transport.close()
```

- [ ] **Step 4: Run the client tests and verify GREEN**

Run:

```powershell
uv run --directory backend python -m unittest tests.test_ollama_client -v
ruff check backend/app/ollama_client.py backend/tests/test_ollama_client.py
```

Expected: all Ollama client tests pass and Ruff reports no violations.

- [ ] **Step 5: Commit the client boundary**

```powershell
git add backend/app/ollama_client.py backend/tests/test_ollama_client.py
git commit -m "feat: add private ollama client"
```

### Task 2: Ollama Embedding Backend

**Files:**
- Create: `backend/app/ollama_embedding_backend.py`
- Create: `backend/tests/test_ollama_embedding_backend.py`

- [ ] **Step 1: Write failing modern and legacy endpoint tests**

Create tests whose fake client records real adapter payloads:

```python
from __future__ import annotations

import math
import unittest
from collections.abc import Mapping

from app.ollama_embedding_backend import OllamaEmbeddingBackend


class FakeClient:
    def __init__(self, responses: list[Mapping[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    def post_json(self, path: str, payload: Mapping[str, object], **_: object) -> Mapping[str, object]:
        self.calls.append((path, payload))
        return self.responses.pop(0)

    def close(self) -> None:
        pass


class OllamaEmbeddingBackendTest(unittest.TestCase):
    def test_batches_api_embed_and_l2_normalizes_vectors(self) -> None:
        client = FakeClient([{"embeddings": [[3.0, 4.0], [0.0, 5.0]]}])
        backend = OllamaEmbeddingBackend(
            client, model="qwen2.5:0.5b", path="/api/embed", dimensions=2, keep_alive="30m"
        )

        vectors = backend.embed(["one", "two"], purpose="document")

        self.assertEqual(client.calls[0][0], "/api/embed")
        self.assertEqual(client.calls[0][1]["input"], ["one", "two"])
        self.assertEqual(client.calls[0][1]["model"], "qwen2.5:0.5b")
        self.assertAlmostEqual(vectors[0][0], 0.6)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vectors[1])), 1.0)

    def test_legacy_endpoint_posts_one_prompt_per_text(self) -> None:
        client = FakeClient([{"embedding": [3.0, 4.0]}, {"embedding": [5.0, 12.0]}])
        backend = OllamaEmbeddingBackend(
            client,
            model="qwen2.5:0.5b",
            path="/api/embeddings",
            dimensions=2,
            keep_alive="30m",
        )

        vectors = backend.embed(["one", "two"], purpose="query")

        self.assertEqual([call[1]["prompt"] for call in client.calls], ["one", "two"])
        self.assertEqual(len(vectors), 2)

    def test_rejects_wrong_dimensions_zero_norm_and_nonfinite_values(self) -> None:
        invalid = (
            {"embeddings": [[1.0]]},
            {"embeddings": [[0.0, 0.0]]},
            {"embeddings": [[float("nan"), 1.0]]},
        )
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                OllamaEmbeddingBackend(
                    FakeClient([response]),
                    model="qwen2.5:0.5b",
                    path="/api/embed",
                    dimensions=2,
                    keep_alive="30m",
                ).embed(["text"], purpose="query")
```

- [ ] **Step 2: Run the tests and verify RED**

```powershell
uv run --directory backend python -m unittest tests.test_ollama_embedding_backend -v
```

Expected: import failure for `app.ollama_embedding_backend`.

- [ ] **Step 3: Implement the embedding adapter**

Create `backend/app/ollama_embedding_backend.py` with `OllamaEmbeddingBackend`. Its constructor validates the model, endpoint, dimensions, and keep-alive value. Its `embed()` implementation must:

```python
def embed(self, texts: Sequence[str], *, purpose: EmbeddingPurpose) -> list[list[float]]:
    values = [text for text in texts]
    if self.path == "/api/embed":
        raw = self.client.post_json(
            self.path,
            {
                "model": self.model,
                "input": values,
                "truncate": True,
                "keep_alive": self.keep_alive,
            },
        )
        vectors = raw.get("embeddings")
    else:
        vectors = []
        for text in values:
            raw = self.client.post_json(
                self.path,
                {"model": self.model, "prompt": text, "keep_alive": self.keep_alive},
            )
            vectors.append(raw.get("embedding"))
    return self._normalize(vectors, expected_count=len(values))
```

`_normalize()` must materialize exactly `expected_count` numeric vectors, require exactly `self.dimensions` coordinates per vector, reject booleans/non-finite values/zero norms, and return `coordinate / norm`. `purpose` is validated as `query` or `document` but does not add the Qwen3 query instruction. `close()` delegates to the injected client.

- [ ] **Step 4: Run focused and contract tests**

```powershell
uv run --directory backend python -m unittest tests.test_ollama_embedding_backend tests.test_embedding_service -v
ruff check backend/app/ollama_embedding_backend.py backend/tests/test_ollama_embedding_backend.py
```

Expected: adapter tests and the existing embedding service tests pass.

- [ ] **Step 5: Commit the embedding adapter**

```powershell
git add backend/app/ollama_embedding_backend.py backend/tests/test_ollama_embedding_backend.py
git commit -m "feat: add ollama embedding backend"
```

### Task 3: Wire the Embedding Service to Ollama

**Files:**
- Modify: `backend/app/embedding_service.py`
- Modify: `backend/tests/test_embedding_service.py`

- [ ] **Step 1: Replace local-model startup tests with failing environment-backed tests**

Add a helper and tests to `backend/tests/test_embedding_service.py`:

```python
def ollama_embedding_environment() -> dict[str, str]:
    return {
        "EMBEDDING_MODEL_NAME": "qwen2.5:0.5b",
        "EMBEDDING_MODEL_VERSION": "ollama-qwen25-05b-v1",
        "EMBEDDING_MODEL_SHA256": "a" * 64,
        "EMBEDDING_MODEL_DIMENSIONS": "896",
        "EMBEDDING_MODEL_NORMALIZED": "true",
        "EMBEDDING_ENCODING_PROFILE_SHA256": "b" * 64,
        "EMBEDDING_PROTOCOL_VERSION": "v1",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_EMBEDDING_MODEL": "qwen2.5:0.5b",
        "OLLAMA_EMBEDDING_PATH": "/api/embed",
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS": "15",
    }
```

Test that `create_production_app(environ=..., backend_loader=loader)` calls the loader once with the environment and pinned metadata, runs both query/document startup probes, serves `/readyz`, and closes the backend. Add cases that reject missing metadata, unsupported embedding paths, a model-name mismatch between `EMBEDDING_MODEL_NAME` and `OLLAMA_EMBEDDING_MODEL`, and non-positive dimensions.

- [ ] **Step 2: Run the production startup tests and verify RED**

```powershell
uv run --directory backend python -m unittest tests.test_embedding_service.EmbeddingServiceTest.test_production_app_loads_ollama_backend_from_environment -v
```

Expected: failure because production startup still requires `EMBEDDING_MODEL_ROOT` and a local manifest.

- [ ] **Step 3: Generalize production metadata loading**

In `backend/app/embedding_service.py`:

1. Remove the production import and call to `load_qwen3_embedding_backend`.
2. Add `_embedding_metadata_from_environ(environ)` that constructs `EmbeddingModelMetadata` from the seven existing metadata variables.
3. Require `EMBEDDING_MODEL_NORMALIZED=true`; the adapter always normalizes vectors.
4. Add `_ollama_embedding_backend(environ, metadata)` that creates `SyncOllamaClient` and `OllamaEmbeddingBackend` using the configuration in the design.
5. Change the optional loader signature to:

```python
EmbeddingBackendLoader = Callable[
    [Mapping[str, str], EmbeddingModelMetadata],
    EmbeddingBackend,
]
```

6. In the lifespan, construct metadata, build the backend, run `_validate_embedding_backend_startup`, start batchers, and close both batchers and backend during shutdown.
7. Keep `create_embedding_app`, `create_batched_embedding_app`, the private `/v1` endpoints, and legacy checksum helpers unchanged so their existing unit contracts remain stable.

- [ ] **Step 4: Verify embedding service behavior**

```powershell
uv run --directory backend python -m unittest tests.test_embedding_service tests.test_ollama_embedding_backend -v
ruff check backend/app/embedding_service.py backend/tests/test_embedding_service.py
```

Expected: all embedding adapter and service tests pass; `/v1/embeddings`, `/v1/metadata`, and `/readyz` response shapes remain unchanged.

- [ ] **Step 5: Commit embedding service wiring**

```powershell
git add backend/app/embedding_service.py backend/tests/test_embedding_service.py
git commit -m "feat: serve embeddings through ollama"
```

### Task 4: Ollama Generative Reranker Backend

**Files:**
- Create: `backend/app/ollama_reranker_backend.py`
- Create: `backend/tests/test_ollama_reranker_backend.py`

- [ ] **Step 1: Write failing prompt and score parser tests**

Create `backend/tests/test_ollama_reranker_backend.py` covering these exact behaviors:

```python
from __future__ import annotations

import unittest
from collections.abc import Mapping

from app.ollama_reranker_backend import OllamaGenerativeRerankerBackend


class FakeClient:
    def __init__(self, responses: list[Mapping[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    def post_json(self, path: str, payload: Mapping[str, object], **_: object) -> Mapping[str, object]:
        self.calls.append((path, payload))
        return self.responses.pop(0)

    def close(self) -> None:
        pass


class OllamaGenerativeRerankerBackendTest(unittest.TestCase):
    def test_scores_one_query_batch_in_passage_order(self) -> None:
        client = FakeClient(
            [{"response": '{"scores":[{"index":1,"score":0.2},{"index":0,"score":0.9}]}'}]
        )
        backend = OllamaGenerativeRerankerBackend(
            client,
            model="qwen2.5:3b",
            path="/api/generate",
            keep_alive="30m",
            format_json=True,
            num_predict=256,
        )

        scores = backend.rerank("policy", ["relevant policy", "weather"])

        self.assertEqual(scores, [0.9, 0.2])
        payload = client.calls[0][1]
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")
        self.assertIn("Document 0", str(payload["prompt"]))
        self.assertNotIn("relevant policy", repr(scores))

    def test_score_pairs_groups_distinct_queries_and_restores_positions(self) -> None:
        client = FakeClient(
            [
                {"response": '{"scores":[{"index":0,"score":0.8},{"index":1,"score":0.3}]}'},
                {"response": '{"scores":[{"index":0,"score":0.6}]}'},
            ]
        )
        backend = OllamaGenerativeRerankerBackend(
            client, model="qwen2.5:3b", path="/api/generate", keep_alive="30m"
        )

        scores = backend.score_pairs([("q1", "a"), ("q2", "b"), ("q1", "c")])

        self.assertEqual(scores, [0.8, 0.6, 0.3])
        self.assertEqual(len(client.calls), 2)

    def test_rejects_malformed_generated_scores(self) -> None:
        invalid = (
            '{"scores":[{"index":0,"score":1.1}]}',
            '{"scores":[{"index":0,"score":0.5},{"index":0,"score":0.4}]}',
            '{"scores":[]}',
            'prefix {"scores":[{"index":0,"score":0.5}]}',
        )
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                OllamaGenerativeRerankerBackend(
                    FakeClient([{"response": response}]),
                    model="qwen2.5:3b",
                    path="/api/generate",
                    keep_alive="30m",
                ).rerank("q", ["p"])
```

- [ ] **Step 2: Run reranker adapter tests and verify RED**

```powershell
uv run --directory backend python -m unittest tests.test_ollama_reranker_backend -v
```

Expected: import failure for `app.ollama_reranker_backend`.

- [ ] **Step 3: Implement fixed-prompt generative scoring**

Create `backend/app/ollama_reranker_backend.py` with:

```python
RERANK_PROMPT_PROFILE = """You are a retrieval relevance scorer.
Return exactly one JSON object and no prose.
The object must contain scores, an array with one item for every document.
Each item must contain its zero-based index and a numeric score from 0 to 1.
Judge only whether the document helps answer the query.

Query:
{query}

Documents:
{documents}
"""
```

Implement `OllamaGenerativeRerankerBackend.rerank()` to enumerate passages as `Document N:\n...`, call `/api/generate` once with `stream=False`, `options={"temperature": 0, "num_predict": self.num_predict}`, optional `format="json"`, and parse `response` with `json.loads`. Require the decoded value to be exactly an object with a `scores` list; require each element to have exactly `index` and `score`; reject booleans, duplicate/out-of-range indices, non-finite values, missing indices, and scores outside `[0, 1]`.

Implement `score_pairs()` by grouping pairs by query in insertion order, calling `rerank()` once per distinct query, and restoring scores to original pair positions. Implement `close()` by closing the shared client. Export a SHA-256 hash of the exact fixed prompt profile for `RERANKER_PROMPT_PROFILE_SHA256` configuration.

- [ ] **Step 4: Run reranker adapter tests and Ruff**

```powershell
uv run --directory backend python -m unittest tests.test_ollama_reranker_backend -v
ruff check backend/app/ollama_reranker_backend.py backend/tests/test_ollama_reranker_backend.py
```

Expected: all generative reranker tests pass.

- [ ] **Step 5: Commit the reranker adapter**

```powershell
git add backend/app/ollama_reranker_backend.py backend/tests/test_ollama_reranker_backend.py
git commit -m "feat: add ollama generative reranker"
```

### Task 5: Wire the Reranker Service to Ollama

**Files:**
- Modify: `backend/app/reranker_service.py`
- Modify: `backend/tests/test_reranker_service.py`

- [ ] **Step 1: Add failing environment-backed production tests**

Add this environment helper to `backend/tests/test_reranker_service.py`:

```python
def ollama_reranker_environment() -> dict[str, str]:
    return {
        "RERANKER_MODEL_NAME": "qwen2.5:3b",
        "RERANKER_MODEL_VERSION": "ollama-qwen25-3b-v1",
        "RERANKER_MODEL_SHA256": "c" * 64,
        "RERANKER_PROMPT_PROFILE_SHA256": "d" * 64,
        "RERANKER_PROTOCOL_VERSION": "v1",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_RERANKER_MODEL": "qwen2.5:3b",
        "OLLAMA_GENERATE_PATH": "/api/generate",
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS": "15",
        "OLLAMA_RERANK_FORMAT_JSON": "true",
        "OLLAMA_RERANK_NUM_PREDICT": "256",
        "RERANKER_BATCH_MAX_ITEMS": "8",
        "RERANKER_QUEUE_MAX_ITEMS": "64",
        "RERANKER_BATCH_WAIT_MS": "10",
    }
```

Test that the production app passes environment and pinned metadata to the injected loader, runs the startup relevance probe, preserves `/v1/rerank` and `/v1/metadata`, and closes the backend. Add rejection tests for a missing metadata value, model mismatch, a path other than `/api/generate`, invalid JSON-mode boolean, and non-positive `OLLAMA_RERANK_NUM_PREDICT`.

- [ ] **Step 2: Run the new production test and verify RED**

```powershell
uv run --directory backend python -m unittest tests.test_reranker_service.RerankerServiceTest.test_production_app_loads_ollama_backend_from_environment -v
```

Expected: failure because production startup still requires `RERANKER_MODEL_ROOT` and a local manifest.

- [ ] **Step 3: Replace the production loader while preserving the HTTP service**

In `backend/app/reranker_service.py`:

1. Replace the default Qwen3 local loader with `OllamaGenerativeRerankerBackend` construction.
2. Load `RerankerModelMetadata` directly from the five existing metadata variables.
3. Require `RERANKER_MODEL_NAME == OLLAMA_RERANKER_MODEL`.
4. Validate `/api/generate`, timeout, keep-alive, JSON-mode boolean, and output-token limit.
5. Change the loader type to accept `(Mapping[str, str], RerankerModelMetadata)`.
6. Retain request bounding, dynamic batching, response validation, and the existing `/v1/rerank`, `/v1/metadata`, and `/readyz` routes.
7. Close the adapter backend after closing the dynamic batcher.
8. Treat malformed generated JSON and Ollama request errors as sanitized 503 backend failures; never include query, passage, prompt, or raw generated text in the HTTP detail.

- [ ] **Step 4: Run reranker service and adapter tests**

```powershell
uv run --directory backend python -m unittest tests.test_reranker_service tests.test_ollama_reranker_backend -v
ruff check backend/app/reranker_service.py backend/tests/test_reranker_service.py
```

Expected: all reranker tests pass and existing wire responses remain stable.

- [ ] **Step 5: Commit reranker service wiring**

```powershell
git add backend/app/reranker_service.py backend/tests/test_reranker_service.py
git commit -m "feat: serve reranking through ollama"
```

### Task 6: Generalize Retrieval Metadata and Protect Index Compatibility

**Files:**
- Modify: `backend/app/retrieval_settings.py`
- Modify: `backend/tests/test_retrieval_settings.py`
- Modify: `backend/tests/test_retrieval_publication.py`
- Modify: `backend/tests/integration/test_hybrid_retrieval_e2e.py`

- [ ] **Step 1: Write failing configurable-model tests**

Change `private_qwen_environment()` in `backend/tests/test_retrieval_settings.py` to use:

```python
"EMBEDDING_MODEL_NAME": "qwen2.5:0.5b",
"EMBEDDING_MODEL_DIMENSIONS": "896",
"RERANKER_MODEL_NAME": "qwen2.5:3b",
```

Replace the fixed-identity rejection test with:

```python
def test_qwen_route_accepts_pinned_non_qwen3_model_identities_and_dimensions(self) -> None:
    settings = RetrievalSettings.from_environ(private_qwen_environment())

    self.assertEqual(settings.embedding.name, "qwen2.5:0.5b")
    self.assertEqual(settings.embedding.dimensions, 896)
    self.assertEqual(settings.reranker.name, "qwen2.5:3b")

    for value in ("0", "-1", "not-an-integer"):
        environ = private_qwen_environment()
        environ["EMBEDDING_MODEL_DIMENSIONS"] = value
        with self.subTest(value=value), self.assertRaises(ValueError):
            RetrievalSettings.from_environ(environ)
```

Add a publication regression test that creates a Qdrant collection with the configured 896 dimensions and rejects validation against a collection created with 1024 dimensions.

- [ ] **Step 2: Run settings/publication tests and verify RED**

```powershell
uv run --directory backend python -m unittest tests.test_retrieval_settings tests.test_retrieval_publication -v
```

Expected: settings fail because model names and dimensions are hard-coded to Qwen3/1024.

- [ ] **Step 3: Remove Qwen3-only locks**

In `backend/app/retrieval_settings.py`:

- Remove `_EMBEDDING_NAME`, `_RERANKER_NAME`, and `_QWEN3_DIMENSIONS`.
- Change `_embedding_metadata()` to require a non-empty model name and any positive integer dimension.
- Change `_reranker_metadata()` to require a non-empty model name without comparing it to a constant.
- Keep the strict version, SHA-256, normalized flag, profile SHA-256, and protocol-version checks.
- Keep `RETRIEVAL_MODE=qwen3` parsing unchanged for backward compatibility.

- [ ] **Step 4: Verify retrieval compatibility and fallback**

```powershell
uv run --directory backend python -m unittest tests.test_retrieval_settings tests.test_retrieval_publication tests.integration.test_hybrid_retrieval_e2e -v
ruff check backend/app/retrieval_settings.py backend/tests/test_retrieval_settings.py
```

Expected: configurable model metadata passes, dimension mismatch tests fail closed, and the existing reranker-unavailable fallback integration test passes.

- [ ] **Step 5: Commit generalized retrieval settings**

```powershell
git add backend/app/retrieval_settings.py backend/tests/test_retrieval_settings.py backend/tests/test_retrieval_publication.py backend/tests/integration/test_hybrid_retrieval_e2e.py
git commit -m "feat: generalize hybrid retrieval models"
```

### Task 7: Update Offline Compose for Controlled Ollama Access

**Files:**
- Modify: `deploy/docker/embedding.Dockerfile`
- Modify: `deploy/docker/reranker.Dockerfile`
- Modify: `deploy/offline/compose.yaml`
- Modify: `deploy/offline/.env.example`
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `tools/invoke_offline_compose.ps1`
- Modify: `tools/tests/test_structured_deployment_contract.py`
- Modify: `tools/tests/test_backend_uv_contract.py`
- Modify: `deploy/offline/artifacts.schema.json`

- [ ] **Step 1: Write failing deployment contract assertions**

Update `tools/tests/test_structured_deployment_contract.py` so the rendered Compose contract requires:

```python
self.assertEqual(embedding["environment"]["OLLAMA_EMBEDDING_MODEL"], "qwen2.5:0.5b")
self.assertEqual(reranker["environment"]["OLLAMA_RERANKER_MODEL"], "qwen2.5:3b")
self.assertEqual(embedding["environment"]["OLLAMA_BASE_URL"], values["OLLAMA_BASE_URL"])
self.assertEqual(reranker["environment"]["OLLAMA_BASE_URL"], values["OLLAMA_BASE_URL"])
self.assertEqual(set(embedding["networks"]), {"offline", "ollama-egress"})
self.assertEqual(set(reranker["networks"]), {"offline", "ollama-egress"})
self.assertNotIn("OLLAMA_BASE_URL", api["environment"])
self.assertNotIn("OLLAMA_BASE_URL", worker["environment"])
self.assertNotIn("volumes", embedding)
self.assertNotIn("volumes", reranker)
```

Update `tools/tests/test_backend_uv_contract.py` to require both adapter Dockerfiles to run base dependencies without `--group offline`.

- [ ] **Step 2: Run deployment contract tests and verify RED**

```powershell
uv run python -m unittest tools.tests.test_structured_deployment_contract tools.tests.test_backend_uv_contract -v
```

Expected: failures for missing Ollama variables/network and retained local model mounts/offline dependency group.

- [ ] **Step 3: Make adapter images lightweight**

In both adapter Dockerfiles, change the sync command to:

```dockerfile
uv sync --frozen --offline --no-install-project --no-dev --find-links=/wheels
```

Remove Transformers-specific environment variables from those two images. Keep the non-root user, bounded filesystem permissions, FastAPI command, and base wheel-only installation.

- [ ] **Step 4: Update Compose service boundaries**

For `embedding-service` and `reranker-service` in `deploy/offline/compose.yaml`:

- Remove model-root bind mounts, local runtime variables, and OpenVINO thread variables.
- Add the Ollama variables from the approved design.
- Keep CPU/memory limits and existing health checks.
- Attach both services to `offline` and a new non-internal `ollama-egress` network.
- Do not attach `api`, `ingestion-worker`, Qdrant, databases, Redis, or ClamAV to `ollama-egress`.

Define:

```yaml
  ollama-egress:
    driver: bridge
```

Update `tools/invoke_offline_compose.ps1` so its network allowlist expects the two adapter services on both networks and verifies `ollama-egress.internal` is not true.

- [ ] **Step 5: Update deployment variables and artifact description**

In all three environment examples:

```env
OLLAMA_BASE_URL=http://172.16.0.10:11434
OLLAMA_EMBEDDING_MODEL=qwen2.5:0.5b
OLLAMA_EMBEDDING_PATH=/api/embed
OLLAMA_RERANKER_MODEL=qwen2.5:3b
OLLAMA_GENERATE_PATH=/api/generate
OLLAMA_KEEP_ALIVE=30m
OLLAMA_REQUEST_TIMEOUT_SECONDS=15
OLLAMA_RERANK_FORMAT_JSON=true
OLLAMA_RERANK_NUM_PREDICT=256
```

Set the retrieval profile to `8/4/4/20`, set model identity variables to the Qwen2.5 profiles, and replace `EMBEDDING_MODEL_DIMENSIONS` with the measured value documented as an operator-supplied value. Remove embedding/reranker model-directory and local runtime variables. Update the artifact schema description so embedding/reranker weights are owned by Ollama and the DC-Agent host manifest covers only artifacts mounted into DC-Agent containers.

- [ ] **Step 6: Regenerate the lock contract only if dependency metadata changes**

The production dependency list does not require a new package because `httpx` already exists. Do not edit `pyproject.toml` or `uv.lock` unless the Docker contract test proves a lock change is necessary.

- [ ] **Step 7: Verify Compose and deployment contracts**

```powershell
uv run python -m unittest tools.tests.test_structured_deployment_contract tools.tests.test_backend_uv_contract -v
powershell -ExecutionPolicy Bypass -File tools/invoke_offline_compose.ps1 config
```

Expected: deployment tests pass; rendered Compose validates; only adapter services receive the Ollama network.

- [ ] **Step 8: Commit deployment changes**

```powershell
git add deploy/docker/embedding.Dockerfile deploy/docker/reranker.Dockerfile deploy/offline/compose.yaml deploy/offline/.env.example .env.example backend/.env.example deploy/offline/artifacts.schema.json tools/invoke_offline_compose.ps1 tools/tests/test_structured_deployment_contract.py tools/tests/test_backend_uv_contract.py
git commit -m "feat: route retrieval adapters to ollama"
```

### Task 8: Documentation, Probe Commands, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `deploy/offline/README.md`
- Modify: `tools/compose_smoke.py`
- Modify: `tools/tests/test_compose_smoke.py`

- [ ] **Step 1: Write failing smoke expectations for Qwen2.5 metadata**

Update the smoke test fixture so embedding readiness expects model `qwen2.5:0.5b` and the configured measured dimensions, while reranker readiness expects `qwen2.5:3b`. Preserve checks that smoke output never contains prompts, document contents, vectors, or generated response text.

- [ ] **Step 2: Run smoke tests and verify RED**

```powershell
uv run python -m unittest tools.tests.test_compose_smoke -v
```

Expected: current Qwen3 metadata assumptions fail.

- [ ] **Step 3: Update smoke probes without exposing content**

Keep calls to the adapter services' `/readyz`, `/v1/metadata`, `/v1/embeddings`, and `/v1/rerank` endpoints. Update expected model identities/dimensions and record only status, latency, vector count/dimension, score count, and sanitized error codes.

- [ ] **Step 4: Update operator documentation**

Document these exact deployment steps in both README files:

```bash
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b
curl -s http://127.0.0.1:11434/api/embed \
  -d '{"model":"qwen2.5:0.5b","input":["dimension probe"]}'
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen2.5:3b","stream":false,"format":"json","prompt":"Return exactly {\"scores\":[{\"index\":0,\"score\":1.0}]}"}'
```

Explain how to set `OLLAMA_EMBEDDING_PATH=/api/embeddings` for legacy Ollama, calculate the returned vector length, configure metadata/profile checksums, restrict the Ollama firewall rule, build a new versioned Qdrant collection, validate it, switch the alias, and roll back to the prior collection. State clearly that Qwen2.5 generative reranking is a compatibility mode and must be capacity-tested for 15 concurrent users.

- [ ] **Step 5: Run focused backend and tooling suites**

```powershell
uv run --directory backend python -m unittest tests.test_ollama_client tests.test_ollama_embedding_backend tests.test_ollama_reranker_backend tests.test_embedding_service tests.test_reranker_service tests.test_retrieval_settings tests.test_retrieval_publication tests.integration.test_hybrid_retrieval_e2e -v
uv run python -m unittest tools.tests.test_compose_smoke tools.tests.test_structured_deployment_contract tools.tests.test_backend_uv_contract -v
```

Expected: every listed test passes with zero failures and zero errors.

- [ ] **Step 6: Run full static and repository verification**

```powershell
ruff check backend tools
ruff format --check backend tools
git diff --check
git status -sb
```

Expected: Ruff and whitespace checks pass; Git status contains only the intended implementation and documentation changes.

- [ ] **Step 7: Commit documentation and smoke coverage**

```powershell
git add README.md deploy/offline/README.md tools/compose_smoke.py tools/tests/test_compose_smoke.py
git commit -m "docs: deploy qwen2.5 retrieval through ollama"
```

- [ ] **Step 8: Perform target-server acceptance before alias activation**

On the target server, run the rendered Compose smoke test, build a new Qdrant collection, and exercise a fixed evaluation set containing Word, PDF, TXT, and Excel questions. Require:

- exact vector dimension agreement;
- no adapter 5xx during the acceptance run;
- one finite reranker score per candidate or a recorded controlled fallback;
- no raw chunk-only final answers;
- successful structured Excel aggregation remains handled by the existing structured path;
- the 15-user acceptance gate completes within the approved latency/error thresholds.

Only after these checks pass, activate `knowledge_chunks_current` for the newly built collection. Keep the previous collection and environment file available for rollback.
