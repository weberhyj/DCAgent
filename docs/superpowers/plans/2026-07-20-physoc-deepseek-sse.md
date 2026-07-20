# Physoc DeepSeek POST SSE Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend-only Physoc DeepSeek provider that sends the complete RAG prompt by POST, consumes `message` SSE events, and returns the completed answer through the existing DC-Agent conversation API.

**Architecture:** Keep the public conversation endpoint, repository, agent graph, citations, persistence, and frontend unchanged. Add a focused pure SSE decoder in `backend/app/physoc_sse.py`, then add `PhysocDeepSeekLLMProvider` in the existing provider factory. The provider buffers the upstream stream, validates its protocol, normalizes the final text, and maps failures to the existing safe model errors.

**Tech Stack:** Python 3.12, FastAPI, httpx streaming client, standard-library `json`/`urllib.parse`, unittest, Ruff 0.15.22, UV 0.11.29.

---

### Task 1: Add and test the pure Physoc SSE decoder

**Files:**
- Create: `backend/app/physoc_sse.py`
- Create: `backend/tests/test_physoc_sse.py`

- [ ] **Step 1: Write failing decoder tests**

Create `backend/tests/test_physoc_sse.py`:

```python
from __future__ import annotations

import unittest

from app.physoc_sse import PhysocStreamError, collect_physoc_response


class PhysocSseTest(unittest.TestCase):
    def test_collects_unicode_message_chunks_until_done(self) -> None:
        lines = [
            "event: message",
            'data: {"model":"my_deepseek_r1_7b","response":"由","done":false}',
            "",
            "event: message",
            'data: {"model":"my_deepseek_r1_7b","response":"检索证据生成","done":false}',
            "",
            "event: message",
            'data: {"model":"my_deepseek_r1_7b","response":"回答。","done":true}',
            "",
        ]

        result = collect_physoc_response(lines, expected_model="my_deepseek_r1_7b")

        self.assertEqual(result, "由检索证据生成回答。")

    def test_accepts_default_message_event_comments_and_multiline_data(self) -> None:
        lines = [
            ": heartbeat",
            'data: {"model":"my_deepseek_r1_7b",',
            'data: "response":"完成", "done":true}',
            "",
        ]

        result = collect_physoc_response(lines, expected_model="my_deepseek_r1_7b")

        self.assertEqual(result, "完成")

    def test_rejects_malformed_or_inconsistent_streams(self) -> None:
        cases = {
            "malformed json": ["data: not-json", ""],
            "wrong response type": [
                'data: {"model":"my_deepseek_r1_7b","response":1,"done":true}',
                "",
            ],
            "wrong done type": [
                'data: {"model":"my_deepseek_r1_7b","response":"x","done":"true"}',
                "",
            ],
            "model mismatch": [
                'data: {"model":"another-model","response":"x","done":true}',
                "",
            ],
            "premature eof": [
                'data: {"model":"my_deepseek_r1_7b","response":"x","done":false}',
                "",
            ],
            "empty completed answer": [
                'data: {"model":"my_deepseek_r1_7b","response":"","done":true}',
                "",
            ],
        }
        for label, lines in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(PhysocStreamError):
                    collect_physoc_response(lines, expected_model="my_deepseek_r1_7b")

    def test_rejects_non_message_events_and_response_overflow(self) -> None:
        with self.assertRaises(PhysocStreamError):
            collect_physoc_response(
                ["event: error", 'data: {"response":"ignored","done":true}', ""],
                expected_model="my_deepseek_r1_7b",
            )

        with self.assertRaisesRegex(PhysocStreamError, "response size"):
            collect_physoc_response(
                ['data: {"response":"12345","done":true}', ""],
                expected_model="my_deepseek_r1_7b",
                max_response_chars=4,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the decoder tests and verify RED**

Run from `backend`:

```powershell
uv run --project . --group dev python -m unittest tests.test_physoc_sse -v
```

Expected: import failure because `app.physoc_sse` does not exist.

- [ ] **Step 3: Implement the minimal pure decoder**

Create `backend/app/physoc_sse.py`:

```python
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any


DEFAULT_MAX_RESPONSE_CHARS = 65_536


class PhysocStreamError(ValueError):
    """Raised when the Physoc SSE stream violates its response contract."""


def iter_message_data(lines: Iterable[str]) -> Iterator[str]:
    event_type = "message"
    data_lines: list[str] = []

    def flush() -> str | None:
        nonlocal event_type, data_lines
        data = "\n".join(data_lines) if event_type == "message" and data_lines else None
        event_type = "message"
        data_lines = []
        return data

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if line == "":
            data = flush()
            if data is not None:
                yield data
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_type = value or "message"
        elif field == "data":
            data_lines.append(value)

    data = flush()
    if data is not None:
        yield data


def _decode_payload(data: str, expected_model: str) -> tuple[str, bool]:
    try:
        payload: Any = json.loads(data)
    except (TypeError, ValueError) as error:
        raise PhysocStreamError("invalid Physoc JSON payload") from error
    if not isinstance(payload, dict):
        raise PhysocStreamError("Physoc payload must be an object")

    response = payload.get("response")
    done = payload.get("done")
    model = payload.get("model")
    if not isinstance(response, str) or type(done) is not bool:
        raise PhysocStreamError("invalid Physoc payload fields")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise PhysocStreamError("invalid Physoc model field")
    if isinstance(model, str) and model != expected_model:
        raise PhysocStreamError("Physoc model mismatch")
    return response, done


def collect_physoc_response(
    lines: Iterable[str],
    *,
    expected_model: str,
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
) -> str:
    parts: list[str] = []
    response_chars = 0
    completed = False

    for data in iter_message_data(lines):
        response, done = _decode_payload(data, expected_model)
        response_chars += len(response)
        if response_chars > max_response_chars:
            raise PhysocStreamError("Physoc response size exceeded")
        parts.append(response)
        if done:
            completed = True
            break

    answer = "".join(parts)
    if not completed:
        raise PhysocStreamError("Physoc stream ended before completion")
    if not answer:
        raise PhysocStreamError("Physoc stream completed without an answer")
    return answer
```

- [ ] **Step 4: Run decoder tests and Ruff**

```powershell
uv run --project . --group dev python -m unittest tests.test_physoc_sse -v
uv run --project . --group dev ruff check app/physoc_sse.py tests/test_physoc_sse.py
uv run --project . --group dev ruff format --check app/physoc_sse.py tests/test_physoc_sse.py
```

Expected: 4 tests pass and both Ruff commands exit 0.

- [ ] **Step 5: Commit the decoder**

```powershell
git add backend/app/physoc_sse.py backend/tests/test_physoc_sse.py
git commit -m "feat: decode physoc model streams"
```

### Task 2: Add the Physoc DeepSeek provider success path

**Files:**
- Modify: `backend/app/llm.py`
- Modify: `backend/tests/test_llm_provider.py`

- [ ] **Step 1: Add a failing provider request test**

Add `PhysocDeepSeekLLMProvider` to the imports in `backend/tests/test_llm_provider.py`, then add these fakes and test:

```python
class FakePhysocResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self) -> FakePhysocResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self.lines


class RecordingPhysocClient:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.requests: list[dict] = []

    def __enter__(self) -> RecordingPhysocClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs) -> FakePhysocResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return FakePhysocResponse(self.lines)


def test_physoc_provider_posts_rag_query_and_collects_sse_reply(self) -> None:
    client = RecordingPhysocClient(
        [
            "event: message",
            'data: {"model":"my_deepseek_r1_7b","response":"证据支持",'
            '"done":false}',
            "",
            "event: message",
            'data: {"model":"my_deepseek_r1_7b","response":"该结论。","done":true}',
            "",
        ]
    )
    provider = PhysocDeepSeekLLMProvider(
        api_base="http://127.0.0.1:8090",
        stream_path="/api/physoc/deepseek/stream",
        model="my_deepseek_r1_7b",
    )

    with patch("app.llm.httpx.Client", return_value=client) as client_factory:
        reply = provider.generate_reply(
            LLMRequest(
                content="请说明制度要求",
                mode="source",
                knowledge_hits=[indexed_hit()],
                previous_messages=[],
            )
        )

    client_factory.assert_called_once_with(timeout=45.0)
    self.assertEqual(len(client.requests), 1)
    request = client.requests[0]
    self.assertEqual(request["method"], "POST")
    self.assertEqual(
        request["url"],
        "http://127.0.0.1:8090/api/physoc/deepseek/stream",
    )
    self.assertEqual(request["headers"], {"Accept": "text/event-stream"})
    self.assertEqual(request["json"]["model"], "my_deepseek_r1_7b")
    self.assertIn(RAG_SYSTEM_PROMPT, request["json"]["query"])
    self.assertIn("cashflow.txt", request["json"]["query"])
    self.assertEqual(reply.paragraphs[0].text, "证据支持该结论。")
    self.assertEqual(reply.paragraphs[0].citations[0].source_id, "kb-llm")
```

- [ ] **Step 2: Run the provider test and verify RED**

```powershell
uv run --project . --group dev python -m unittest tests.test_llm_provider.LLMProviderTest.test_physoc_provider_posts_rag_query_and_collects_sse_reply -v
```

Expected: import or constructor failure because `PhysocDeepSeekLLMProvider` does not exist.

- [ ] **Step 3: Implement the provider success path**

In `backend/app/llm.py`, import the decoder:

```python
from .physoc_sse import PhysocStreamError, collect_physoc_response
```

Add the provider after `OpenAICompatibleLLMProvider`:

```python
class PhysocDeepSeekLLMProvider(LLMProvider):
    def __init__(
        self,
        api_base: str,
        stream_path: str,
        model: str,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.stream_path = stream_path
        self.stream_url = f"{self.api_base}{stream_path}"
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
        if not request.knowledge_hits:
            return build_no_evidence_reply()

        query = f"{RAG_SYSTEM_PROMPT}\n\n{build_prompt(request)}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                with client.stream(
                    "POST",
                    self.stream_url,
                    json={"query": query, "model": self.model},
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    response.raise_for_status()
                    content = collect_physoc_response(
                        response.iter_lines(),
                        expected_model=self.model,
                    )
        except httpx.TimeoutException as exc:
            raise LLMProviderError("大模型响应超时，请稍后重试。") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError("大模型服务暂时不可用，请稍后重试。") from exc
        except PhysocStreamError as exc:
            raise LLMProviderError("大模型返回格式异常，请稍后重试。") from exc

        return ChatMessageModel(
            id=f"msg-{uuid4().hex[:8]}",
            role="assistant",
            time=now_label(),
            paragraphs=[
                ResponseParagraphModel(
                    text=normalize_plain_text_answer(content),
                    citations=build_citations(request.knowledge_hits),
                )
            ],
        )
```

- [ ] **Step 4: Run provider and existing LLM tests**

```powershell
uv run --project . --group dev python -m unittest tests.test_llm_provider -v
uv run --project . --group dev ruff check app/llm.py tests/test_llm_provider.py
uv run --project . --group dev ruff format --check app/llm.py tests/test_llm_provider.py
```

Expected: all `test_llm_provider` tests pass.

- [ ] **Step 5: Commit the provider success path**

```powershell
git add backend/app/llm.py backend/tests/test_llm_provider.py
git commit -m "feat: call physoc deepseek stream"
```

### Task 3: Validate Physoc configuration and error behavior

**Files:**
- Modify: `backend/app/llm.py`
- Modify: `backend/tests/test_llm_provider.py`

- [ ] **Step 1: Add failing factory and failure-path tests**

Add tests that assert:

```python
def test_llm_provider_factory_builds_physoc_without_api_key(self) -> None:
    provider = create_llm_provider(
        {
            "LLM_PROVIDER": "physoc-deepseek",
            "LLM_API_BASE": "http://127.0.0.1:8090",
            "LLM_MODEL": "my_deepseek_r1_7b",
        }
    )

    self.assertIsInstance(provider, PhysocDeepSeekLLMProvider)
    self.assertEqual(
        provider.stream_url,
        "http://127.0.0.1:8090/api/physoc/deepseek/stream",
    )


def test_physoc_factory_rejects_public_base_and_invalid_paths(self) -> None:
    with self.assertRaisesRegex(ValueError, "private or loopback"):
        create_llm_provider(
            {
                "OFFLINE_MODE": "false",
                "LLM_PROVIDER": "physoc_deepseek",
                "LLM_API_BASE": "https://public.example.com",
                "LLM_MODEL": "my_deepseek_r1_7b",
            }
        )

    for path in ("relative", "https://other.example/stream", "/stream?token=x"):
        with self.subTest(path=path):
            with self.assertRaisesRegex(ValueError, "LLM_STREAM_PATH"):
                create_llm_provider(
                    {
                        "OFFLINE_MODE": "false",
                        "LLM_PROVIDER": "physoc_deepseek",
                    "LLM_API_BASE": "http://127.0.0.1:8090",
                        "LLM_STREAM_PATH": path,
                        "LLM_MODEL": "my_deepseek_r1_7b",
                    }
                )


def test_physoc_provider_skips_network_without_evidence(self) -> None:
    provider = PhysocDeepSeekLLMProvider(
        api_base="http://127.0.0.1:8090",
        stream_path="/api/physoc/deepseek/stream",
        model="my_deepseek_r1_7b",
    )
    with patch("app.llm.httpx.Client") as client_factory:
        reply = provider.generate_reply(
            LLMRequest(content="unknown", mode="source", knowledge_hits=[])
        )

    client_factory.assert_not_called()
    self.assertEqual(reply.paragraphs[0].text, NO_EVIDENCE_REPLY)


def test_physoc_factory_requires_base_and_model(self) -> None:
    cases = (
        (
            {
                "LLM_PROVIDER": "physoc_deepseek",
                "LLM_MODEL": "my_deepseek_r1_7b",
            },
            "LLM_API_BASE is required",
        ),
        (
            {
                "LLM_PROVIDER": "physoc_deepseek",
                "LLM_API_BASE": "http://127.0.0.1:8090",
            },
            "LLM_MODEL is required",
        ),
    )
    for environ, message in cases:
        with self.subTest(message=message):
            with self.assertRaisesRegex(ValueError, message):
                create_llm_provider(environ)
```

Add these failure fakes next to `RecordingPhysocClient`:

```python
class HttpFailurePhysocResponse(FakePhysocResponse):
    def raise_for_status(self) -> None:
        request = httpx.Request(
            "POST",
            "http://127.0.0.1:8090/api/physoc/deepseek/stream",
        )
        response = httpx.Response(503, request=request)
        raise httpx.HTTPStatusError(
            "secret upstream failure",
            request=request,
            response=response,
        )


class TimeoutPhysocClient(RecordingPhysocClient):
    def stream(self, method: str, url: str, **kwargs):
        request = httpx.Request(method, url)
        raise httpx.ReadTimeout("secret timeout", request=request)


class HttpFailurePhysocClient(RecordingPhysocClient):
    def stream(self, method: str, url: str, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        return HttpFailurePhysocResponse([])
```

Add exact failure assertions:

```python
def test_physoc_provider_maps_upstream_failures_to_safe_errors(self) -> None:
    provider = PhysocDeepSeekLLMProvider(
        api_base="http://127.0.0.1:8090",
        stream_path="/api/physoc/deepseek/stream",
        model="my_deepseek_r1_7b",
    )
    request = LLMRequest(
        content="请说明制度要求",
        mode="source",
        knowledge_hits=[indexed_hit()],
    )
    cases = (
        (
            TimeoutPhysocClient([]),
            "大模型响应超时，请稍后重试。",
        ),
        (
            HttpFailurePhysocClient([]),
            "大模型服务暂时不可用，请稍后重试。",
        ),
        (
            RecordingPhysocClient(["data: secret malformed stream", ""]),
            "大模型返回格式异常，请稍后重试。",
        ),
    )
    for client, expected_message in cases:
        with self.subTest(expected_message=expected_message):
            with patch("app.llm.httpx.Client", return_value=client):
                with self.assertRaises(LLMProviderError) as error:
                    provider.generate_reply(request)
            self.assertEqual(str(error.exception), expected_message)
            self.assertNotIn("127.0.0.1", str(error.exception))
            self.assertNotIn("secret", str(error.exception))
```

- [ ] **Step 2: Run the new tests and verify RED**

```powershell
uv run --project . --group dev python -m unittest tests.test_llm_provider -v
```

Expected: Physoc factory tests fail because the factory does not support `physoc_deepseek` and path validation is absent.

- [ ] **Step 3: Add URL validation and factory support**

In `backend/app/llm.py`, import `urlsplit` and add:

```python
from urllib.parse import urlsplit


DEFAULT_PHYSOC_STREAM_PATH = "/api/physoc/deepseek/stream"


def _validate_physoc_stream_path(path: str) -> str:
    candidate = path.strip()
    parsed = urlsplit(candidate)
    if (
        not candidate.startswith("/")
        or candidate.startswith("//")
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("LLM_STREAM_PATH must be an absolute URL path without query or fragment")
    return candidate
```

Add this factory branch before the unsupported-provider error:

```python
    if provider == "physoc_deepseek":
        api_base = source.get("LLM_API_BASE", "").strip()
        model = source.get("LLM_MODEL", "").strip()
        stream_path = _validate_physoc_stream_path(
            source.get("LLM_STREAM_PATH", DEFAULT_PHYSOC_STREAM_PATH)
        )
        if not api_base:
            raise ValueError("LLM_API_BASE is required")
        if not model:
            raise ValueError("LLM_MODEL is required")
        stream_url = f"{api_base.rstrip('/')}{stream_path}"
        require_private_url(stream_url, "LLM_API_BASE")
        return PhysocDeepSeekLLMProvider(
            api_base=api_base,
            stream_path=stream_path,
            model=model,
        )
```

Keep `LLM_API_KEY` optional and unused for Physoc. Do not add Physoc to `_generation_enabled()` because the existing readiness check targets llama.cpp and no Physoc health endpoint has been specified.

- [ ] **Step 4: Run LLM, startup, and health tests**

```powershell
uv run --project . --group dev python -m unittest tests.test_llm_provider tests.test_lazy_startup tests.test_infra_health -v
uv run --project . --group dev ruff check app/llm.py tests/test_llm_provider.py
uv run --project . --group dev ruff format --check app/llm.py tests/test_llm_provider.py
```

Expected: all tests pass and existing OpenAI/template behavior remains unchanged.

- [ ] **Step 5: Commit configuration and error handling**

```powershell
git add backend/app/llm.py backend/tests/test_llm_provider.py
git commit -m "feat: configure physoc model provider"
```

### Task 4: Document the provider and environment contract

**Files:**
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `deploy/offline/.env.example`
- Modify: `README.md`
- Create: `tools/tests/test_physoc_llm_contract.py`

- [ ] **Step 1: Add the failing documentation contract**

Create `tools/tests/test_physoc_llm_contract.py`:

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PhysocLlmContractTest(unittest.TestCase):
    def test_environment_examples_document_physoc_provider(self) -> None:
        for relative_path in (
            ".env.example",
            "backend/.env.example",
            "deploy/offline/.env.example",
        ):
            with self.subTest(path=relative_path):
                text = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("LLM_PROVIDER=physoc_deepseek", text)
                self.assertIn("LLM_STREAM_PATH=/api/physoc/deepseek/stream", text)
                self.assertIn("LLM_MODEL=my_deepseek_r1_7b", text)

    def test_readme_documents_post_sse_and_no_api_key(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("physoc_deepseek", text)
        self.assertIn("/api/physoc/deepseek/stream", text)
        self.assertIn("POST", text)
        self.assertIn("text/event-stream", text)
        self.assertIn("无需 LLM_API_KEY", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the documentation contract and verify RED**

```powershell
py -m unittest tools.tests.test_physoc_llm_contract -v
```

Expected: failures because the examples and README do not mention Physoc.

- [ ] **Step 3: Update environment examples and README**

Add this commented block near the existing LLM settings in each environment example, preserving `template` as the active default:

```text
# Internal Physoc DeepSeek POST SSE provider:
# LLM_PROVIDER=physoc_deepseek
# LLM_API_BASE=http://127.0.0.1:8090
# LLM_STREAM_PATH=/api/physoc/deepseek/stream
# LLM_MODEL=my_deepseek_r1_7b
# This provider does not use LLM_API_KEY.
```

Add a README subsection showing:

```text
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://127.0.0.1:8090
LLM_STREAM_PATH=/api/physoc/deepseek/stream
LLM_MODEL=my_deepseek_r1_7b
```

State that DC-Agent sends a POST JSON body with `query` and `model`, consumes `text/event-stream` message events, and keeps the browser conversation API unchanged. Include the exact sentence `Physoc 模式无需 LLM_API_KEY。` Do not place a real internal hostname, cookie, token, or secret in committed files.

- [ ] **Step 4: Run documentation and existing contracts**

```powershell
py -m unittest tools.tests.test_physoc_llm_contract tools.tests.test_backend_uv_contract -v
git diff --check
```

Expected: all tests pass and no whitespace errors exist.

- [ ] **Step 5: Commit documentation**

```powershell
git add .env.example backend/.env.example deploy/offline/.env.example README.md tools/tests/test_physoc_llm_contract.py
git commit -m "docs: document physoc model provider"
```

### Task 5: Run complete verification

**Files:**
- Verify all files changed by Tasks 1-4

- [ ] **Step 1: Run focused provider tests**

```powershell
Set-Location backend
uv run --project . --group dev python -m unittest tests.test_physoc_sse tests.test_llm_provider tests.test_lazy_startup tests.test_infra_health -v
Set-Location ..
```

Expected: all focused backend tests pass.

- [ ] **Step 2: Run the complete backend suite**

```powershell
Set-Location backend
uv run --project . --group dev python -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..
```

Expected: the existing 361 tests plus the new Physoc tests pass with no failures.

- [ ] **Step 3: Run Ruff and compile gates**

```powershell
uv run --project backend --group dev ruff check backend
uv run --project backend --group dev ruff format --check backend
py -m compileall -q backend tools
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 4: Run tool contracts and inspect active configuration**

```powershell
py -m unittest discover -s tools/tests -p "test_*.py" -v
rg -n "physoc_deepseek|LLM_STREAM_PATH|/api/physoc/deepseek/stream" README.md .env.example backend/.env.example deploy/offline/.env.example backend/app backend/tests tools/tests
git status --short --branch
```

Expected: tool tests pass; matches exist only in the provider, tests, and documentation; the worktree is clean after commits.

- [ ] **Step 5: Record deployment activation boundary**

Do not commit or print the real Physoc hostname if it is sensitive. Activation requires setting the real private `LLM_API_BASE` in the target runtime environment and confirming the deployed upstream accepts POST with the documented JSON body. A live Physoc integration smoke is a target-environment gate and must not be reported as passed from a development machine without access to that service.

---

After all tasks pass, request a final cross-task code review before integrating or pushing the branch.
