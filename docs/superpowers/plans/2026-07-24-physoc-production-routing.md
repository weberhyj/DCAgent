# Physoc Production Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the production backend fail closed unless a real LLM provider is configured, make offline Compose directly support the keyless Physoc POST/SSE route, and provide a reproducible target-host interoperability gate.

**Architecture:** Keep the existing `PhysocDeepSeekLLMProvider`, bounded SSE decoder, conversation API, citations, and buffered frontend behavior. Add a small production-only provider policy at application startup, wire the offline deployment to `physoc_deepseek`, and add a probe that exercises the same provider with synthetic authorized evidence without persisting or printing prompt/answer contents.

**Tech Stack:** Python 3.12, FastAPI lifespan, httpx, Physoc SSE, Docker Compose, unittest, Ruff, uv.

**Approved Design:** [`../specs/2026-07-24-enterprise-knowledge-base-qa-design.md`](../specs/2026-07-24-enterprise-knowledge-base-qa-design.md)

**Roadmap:** [`2026-07-24-enterprise-knowledge-base-qa-rollout.md`](2026-07-24-enterprise-knowledge-base-qa-rollout.md)

---

## File map

- `backend/app/llm_runtime.py`: normalize the provider name and reject template/mock providers in production.
- `backend/app/main.py`: invoke the production provider policy before constructing databases, clients, repositories, or health checks.
- `backend/app/physoc_probe.py`: execute a real Physoc provider request and write a redacted machine-readable report.
- `backend/tests/test_llm_runtime.py`: pure provider-policy tests.
- `backend/tests/test_lazy_startup.py`: production lifespan ordering and fail-closed startup tests.
- `backend/tests/test_physoc_probe.py`: probe success, failure, output-redaction, and atomic-report tests.
- `backend/tests/test_api_contract.py`: assert model errors never include evidence/slice text.
- `deploy/offline/compose.yaml`: pass the stream path and make the API key optional.
- `deploy/offline/.env.example`: select the Physoc route for the internal production topology.
- `.env.example` and `backend/.env.example`: keep template only as an explicitly development-only default.
- `README.md` and `deploy/offline/README.md`: document development, production, target-host probe, rollback, and failure behavior.
- `tools/tests/test_physoc_llm_contract.py`: environment and documentation contracts.
- `tools/tests/test_compose_contract.py`: rendered Compose contract for keyless Physoc.
- `tools/tests/test_structured_deployment_contract.py`: update the offline provider expectation without changing structured-query guarantees.

---

### Task 1: Add the production LLM provider policy

**Files:**
- Create: `backend/app/llm_runtime.py`
- Create: `backend/tests/test_llm_runtime.py`

- [ ] **Step 1: Write the failing policy tests**

Create `backend/tests/test_llm_runtime.py`:

```python
from __future__ import annotations

import unittest

from app.llm_runtime import normalize_llm_provider, validate_production_llm_provider


class LlmRuntimeTest(unittest.TestCase):
    def test_normalizes_provider_names_without_mutating_environment(self) -> None:
        environ = {"LLM_PROVIDER": "  physoc-deepseek  "}

        provider = normalize_llm_provider(environ)

        self.assertEqual(provider, "physoc_deepseek")
        self.assertEqual(environ, {"LLM_PROVIDER": "  physoc-deepseek  "})

    def test_production_rejects_template_mock_and_empty_provider(self) -> None:
        for value in (None, "", "template", "mock", " TEMPLATE "):
            with self.subTest(value=value):
                environ = {} if value is None else {"LLM_PROVIDER": value}
                with self.assertRaisesRegex(
                    ValueError,
                    "Production runtime requires a real LLM provider",
                ):
                    validate_production_llm_provider(environ)

    def test_production_accepts_supported_real_provider_routes(self) -> None:
        for value, expected in (
            ("physoc_deepseek", "physoc_deepseek"),
            ("physoc-deepseek", "physoc_deepseek"),
            ("openai_compatible", "openai_compatible"),
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    validate_production_llm_provider({"LLM_PROVIDER": value}),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the policy test and verify RED**

Run from the repository root:

```powershell
Push-Location backend
uv run --project . python -m unittest tests.test_llm_runtime -v
Pop-Location
```

Expected: import failure because `app.llm_runtime` does not exist.

- [ ] **Step 3: Implement the minimal production policy**

Create `backend/app/llm_runtime.py`:

```python
from __future__ import annotations

from collections.abc import Mapping


NON_GENERATING_PROVIDERS = frozenset({"", "template", "mock"})


def normalize_llm_provider(environ: Mapping[str, str]) -> str:
    return environ.get("LLM_PROVIDER", "template").strip().lower().replace("-", "_")


def validate_production_llm_provider(environ: Mapping[str, str]) -> str:
    provider = normalize_llm_provider(environ)
    if provider in NON_GENERATING_PROVIDERS:
        raise ValueError(
            "Production runtime requires a real LLM provider; "
            "set LLM_PROVIDER=physoc_deepseek for the internal deployment"
        )
    return provider
```

- [ ] **Step 4: Run the focused test and Ruff**

```powershell
Push-Location backend
uv run --project . python -m unittest tests.test_llm_runtime -v
Pop-Location
ruff check backend/app/llm_runtime.py backend/tests/test_llm_runtime.py
ruff format --check backend/app/llm_runtime.py backend/tests/test_llm_runtime.py
```

Expected: all tests pass and Ruff reports no violations.

- [ ] **Step 5: Commit the production policy**

```powershell
git add backend/app/llm_runtime.py backend/tests/test_llm_runtime.py
git commit -m "feat(backend): reject template providers in production"
```

---

### Task 2: Enforce the policy before production resources are created

**Files:**
- Modify: `backend/app/main.py:20-35,233-253`
- Modify: `backend/tests/test_lazy_startup.py:24-50,80-160`

- [ ] **Step 1: Add failing production lifespan tests**

In `backend/tests/test_lazy_startup.py`, import the development app factory alongside the existing helper:

```python
from app.main import _database_url_with_connect_timeout, create_app
```

Change `private_environment()` so production tests use a valid Physoc configuration by default:

```python
def private_environment(**changes: str) -> dict[str, str]:
    values = {
        "OFFLINE_MODE": "true",
        "DATABASE_URL": "postgresql+psycopg://dc_agent@127.0.0.1/dc_agent",
        "CLICKHOUSE_URL": "http://127.0.0.1:8123",
        "QDRANT_URL": "http://127.0.0.1:6333",
        "REDIS_URL": "redis://127.0.0.1:6379/0",
        "CLAMAV_HOST": "127.0.0.1",
        "EMBEDDING_SERVICE_URL": "http://127.0.0.1:8081",
        "LLAMA_SERVER_URL": "http://127.0.0.1:8080",
        "LLM_PROVIDER": "physoc_deepseek",
        "LLM_API_BASE": "http://127.0.0.1:8090",
        "LLM_STREAM_PATH": "/api/physoc/deepseek/stream",
        "LLM_MODEL": "my_deepseek_r1_7b",
    }
    values.update(changes)
    return values
```

Add these tests to `LazyStartupTest`:

```python
def test_production_startup_rejects_template_before_resource_factories(self) -> None:
    module = importlib.import_module("app.main")
    calls: list[str] = []

    app = module.create_production_app(
        environ=private_environment(LLM_PROVIDER="template"),
        database_factory=lambda _url: calls.append("database"),
        llm_provider_factory=lambda _environment: calls.append("llm"),
    )

    with self.assertRaisesRegex(
        ValueError,
        "Production runtime requires a real LLM provider",
    ):
        with TestClient(app):
            pass

    self.assertEqual(calls, [])

def test_development_app_still_allows_injected_template_repository(self) -> None:
    repository = ClosableFake("development-repository")

    app = create_app(repository=repository)

    with TestClient(app) as client:
        self.assertEqual(client.get("/api/healthz").status_code, 200)
```

Keep the existing health test that explicitly passes `LLM_PROVIDER="template"`; it proves health-check construction itself remains provider-aware outside production startup.

- [ ] **Step 2: Run the startup tests and verify RED**

```powershell
Push-Location backend
uv run --project . python -m unittest tests.test_lazy_startup -v
Pop-Location
```

Expected: `test_production_startup_rejects_template_before_resource_factories` fails because production startup still accepts `template`.

- [ ] **Step 3: Wire the validator into the production lifespan**

In `backend/app/main.py`, add:

```python
from .llm_runtime import validate_production_llm_provider
```

Then update the beginning of the production lifespan immediately after `source` is selected and before `OfflineSettings.from_environ(source)`:

```python
if environment_override is None:
    load_runtime_environment()
    source: Mapping[str, str] = os.environ
else:
    source = environment_override

validate_production_llm_provider(source)
settings = OfflineSettings.from_environ(source)
```

Do not call this validator from `create_app()` or `create_default_repository()`; those factories remain available for local development and unit tests.

- [ ] **Step 4: Run startup and existing provider tests**

```powershell
Push-Location backend
uv run --project . python -m unittest `
  tests.test_llm_runtime `
  tests.test_lazy_startup `
  tests.test_llm_provider -v
Pop-Location
ruff check backend/app/main.py backend/app/llm_runtime.py backend/tests/test_lazy_startup.py
ruff format --check backend/app/main.py backend/app/llm_runtime.py backend/tests/test_lazy_startup.py
```

Expected: all tests pass. Importing `app.main` remains lazy; the rejection occurs only when the production lifespan starts.

- [ ] **Step 5: Commit production startup enforcement**

```powershell
git add backend/app/main.py backend/tests/test_lazy_startup.py
git commit -m "feat(backend): enforce production llm configuration"
```

---

### Task 3: Make offline Compose support keyless Physoc directly

**Files:**
- Modify: `deploy/offline/compose.yaml:164-181`
- Modify: `deploy/offline/.env.example:35-58`
- Modify: `tools/tests/test_compose_contract.py`
- Modify: `tools/tests/test_physoc_llm_contract.py`
- Modify: `tools/tests/test_structured_deployment_contract.py:110-130`

- [ ] **Step 1: Write failing deployment contract assertions**

Update `tools/tests/test_compose_contract.py` with:

```python
def test_api_wires_keyless_physoc_stream_configuration(self) -> None:
    text = (REPO_ROOT / "deploy" / "offline" / "compose.yaml").read_text(
        encoding="utf-8"
    )

    self.assertIn(
        "LLM_STREAM_PATH: ${LLM_STREAM_PATH:-/api/physoc/deepseek/stream}",
        text,
    )
    self.assertIn("LLM_API_KEY: ${LLM_API_KEY:-}", text)
    self.assertNotIn("LLM_API_KEY: ${LLM_API_KEY:?", text)
```

Replace the active-default test in `tools/tests/test_physoc_llm_contract.py` with:

```python
def test_development_examples_keep_template_and_offline_deployment_uses_physoc(self) -> None:
    development_examples = ENV_EXAMPLES[:2]
    for path in development_examples:
        text = path.read_text(encoding="utf-8")
        active_providers = re.findall(
            r"(?m)^\s*LLM_PROVIDER\s*=\s*([^#\s]+)\s*$",
            text,
        )
        with self.subTest(path=path.relative_to(REPO_ROOT)):
            self.assertEqual(["template"], active_providers)
            self.assertIn("development only", text.lower())

    offline = ENV_EXAMPLES[-1].read_text(encoding="utf-8")
    self.assertRegex(offline, r"(?m)^LLM_PROVIDER=physoc_deepseek$")
    self.assertRegex(
        offline,
        r"(?m)^LLM_STREAM_PATH=/api/physoc/deepseek/stream$",
    )
    self.assertNotRegex(offline, r"(?m)^LLM_API_KEY=\S+")
```

Update the offline block assertions so they require these statements instead of claiming Compose is unsupported:

```python
for required_text in (
    "当前 offline Compose 已透传 LLM_STREAM_PATH",
    "Physoc 模式无需 LLM_API_KEY",
    "容器可达的批准 private IP",
    "生产启动会拒绝 template 和 mock",
):
    self.assertIn(required_text, offline)
```

In `tools/tests/test_structured_deployment_contract.py`, replace the offline environment assertion with:

```python
self.assertIn("LLM_PROVIDER=physoc_deepseek", env)
```

- [ ] **Step 2: Run deployment contract tests and verify RED**

```powershell
uv run --project backend python -m unittest `
  tools.tests.test_compose_contract `
  tools.tests.test_physoc_llm_contract `
  tools.tests.test_structured_deployment_contract -v
```

Expected: failures show missing `LLM_STREAM_PATH`, mandatory `LLM_API_KEY`, and the offline template default.

- [ ] **Step 3: Update Compose environment wiring**

In the `api.environment` block of `deploy/offline/compose.yaml`, use:

```yaml
      LLM_PROVIDER: ${LLM_PROVIDER:?LLM_PROVIDER is required}
      LLM_API_BASE: ${LLM_API_BASE:?LLM_API_BASE is required}
      LLM_STREAM_PATH: ${LLM_STREAM_PATH:-/api/physoc/deepseek/stream}
      LLM_API_KEY: ${LLM_API_KEY:-}
      LLM_MODEL: ${LLM_MODEL:?LLM_MODEL is required}
```

Do not add a public port for Physoc and do not add an external DNS name to the checked-in example.

- [ ] **Step 4: Make the offline environment example select Physoc**

Replace the active LLM settings and old unsupported warning in `deploy/offline/.env.example` with:

```dotenv
# The API container must be able to reach this approved private address.
# Replace 172.16.0.10 with the actual Physoc host before rendering Compose.
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://172.16.0.10:8090
LLM_STREAM_PATH=/api/physoc/deepseek/stream
LLM_MODEL=my_deepseek_r1_7b

# BEGIN PHYSOC DEEPSEEK EXAMPLE
# 当前 offline Compose 已透传 LLM_STREAM_PATH。
# Physoc 模式无需 LLM_API_KEY；空值不会阻止 Compose 渲染。
# LLM_API_BASE 必须使用 API 容器可达的批准 private IP，不能使用公网地址。
# 生产启动会拒绝 template 和 mock，防止把检索切片当成回答。
# LLM_PROVIDER=physoc_deepseek
# LLM_API_BASE=http://127.0.0.1:8090
# LLM_STREAM_PATH=/api/physoc/deepseek/stream
# LLM_MODEL=my_deepseek_r1_7b
# END PHYSOC DEEPSEEK EXAMPLE
```

The commented loopback block remains a syntax example only; the active Compose value is the container-reachable private address.

- [ ] **Step 5: Mark template defaults as development-only**

Immediately above the active `LLM_PROVIDER=template` line in both `.env.example` and `backend/.env.example`, add:

```dotenv
# Development only. create_production_app rejects template and mock providers.
```

Do not change local smoke scripts that intentionally use `create_app()` with template fixtures.

- [ ] **Step 6: Run deployment contracts and Compose rendering checks**

```powershell
uv run --project backend python -m unittest `
  tools.tests.test_compose_contract `
  tools.tests.test_physoc_llm_contract `
  tools.tests.test_structured_deployment_contract -v
```

Expected: all tests pass.

On a host with Docker and a copied `deploy/offline/.env`, also run:

```powershell
& tools/invoke_offline_compose.ps1 config
```

Expected: rendered API environment contains `LLM_PROVIDER=physoc_deepseek`, the stream path, an empty API key, and no public service URL.

- [ ] **Step 7: Commit the deployment route**

```powershell
git add deploy/offline/compose.yaml deploy/offline/.env.example .env.example backend/.env.example tools/tests/test_compose_contract.py tools/tests/test_physoc_llm_contract.py tools/tests/test_structured_deployment_contract.py
git commit -m "feat(deploy): enable keyless physoc routing"
```

---

### Task 4: Add a redacted target-host Physoc probe

**Files:**
- Create: `backend/app/physoc_probe.py`
- Create: `backend/tests/test_physoc_probe.py`

- [ ] **Step 1: Write failing probe tests**

Create `backend/tests/test_physoc_probe.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.llm import LLMProviderError, PhysocDeepSeekLLMProvider
from app.models import ChatMessageModel, CitationModel, ResponseParagraphModel
from app.physoc_probe import run_physoc_probe, write_probe_report


class FakePhysocProvider(PhysocDeepSeekLLMProvider):
    def __init__(self) -> None:
        super().__init__(
            api_base="http://127.0.0.1:8090",
            stream_path="/api/physoc/deepseek/stream",
            model="my_deepseek_r1_7b",
        )
        self.requests = []

    def generate_reply(self, request):
        self.requests.append(request)
        return ChatMessageModel(
            id="probe-answer",
            role="assistant",
            time="现在",
            paragraphs=[
                ResponseParagraphModel(
                    text="Physoc 链路正常。",
                    citations=[
                        CitationModel(
                            label="[1] 内部 · physoc-probe.txt",
                            classification="内部",
                            source_id="physoc-probe-source",
                            source_name="physoc-probe.txt",
                            chunk_id="physoc-probe-chunk",
                            chunk_index=0,
                        )
                    ],
                )
            ],
        )


class PhysocProbeTest(unittest.TestCase):
    def test_probe_uses_synthetic_evidence_and_returns_redacted_metrics(self) -> None:
        provider = FakePhysocProvider()

        result = run_physoc_probe(
            {
                "LLM_PROVIDER": "physoc_deepseek",
                "LLM_API_BASE": "http://127.0.0.1:8090",
                "LLM_STREAM_PATH": "/api/physoc/deepseek/stream",
                "LLM_MODEL": "my_deepseek_r1_7b",
            },
            provider_factory=lambda _environment: provider,
            clock_values=iter((10.0, 10.25)),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["provider"], "physoc_deepseek")
        self.assertEqual(result["model"], "my_deepseek_r1_7b")
        self.assertEqual(result["streamPath"], "/api/physoc/deepseek/stream")
        self.assertEqual(result["elapsedMs"], 250.0)
        self.assertEqual(result["answerChars"], len("Physoc 链路正常。"))
        self.assertEqual(result["citationCount"], 1)
        self.assertNotIn("query", result)
        self.assertNotIn("answer", result)
        self.assertEqual(provider.requests[0].knowledge_hits[0].chunk.text, "Physoc 链路正常")

    def test_probe_rejects_non_physoc_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "physoc_deepseek"):
            run_physoc_probe(
                {"LLM_PROVIDER": "template"},
                provider_factory=lambda _environment: object(),
                clock_values=iter((1.0, 2.0)),
            )

    def test_probe_propagates_safe_provider_failure_without_report_secrets(self) -> None:
        class FailingProvider(FakePhysocProvider):
            def generate_reply(self, request):
                raise LLMProviderError("大模型服务暂时不可用，请稍后重试。")

        with self.assertRaisesRegex(LLMProviderError, "大模型服务暂时不可用"):
            run_physoc_probe(
                {
                    "LLM_PROVIDER": "physoc_deepseek",
                    "LLM_MODEL": "my_deepseek_r1_7b",
                },
                provider_factory=lambda _environment: FailingProvider(),
                clock_values=iter((1.0, 2.0)),
            )

    def test_report_write_is_atomic_and_contains_no_prompt_or_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "physoc-probe.json"
            write_probe_report(
                path,
                {
                    "passed": True,
                    "provider": "physoc_deepseek",
                    "model": "my_deepseek_r1_7b",
                    "streamPath": "/api/physoc/deepseek/stream",
                    "elapsedMs": 250.0,
                    "answerChars": 12,
                    "citationCount": 1,
                },
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(payload["passed"])
        self.assertEqual(set(payload), {
            "answerChars",
            "citationCount",
            "elapsedMs",
            "model",
            "passed",
            "provider",
            "streamPath",
        })


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the probe test and verify RED**

```powershell
Push-Location backend
uv run --project . python -m unittest tests.test_physoc_probe -v
Pop-Location
```

Expected: import failure because `app.physoc_probe` does not exist.

- [ ] **Step 3: Implement the provider-backed probe**

Create `backend/app/physoc_probe.py`:

```python
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .llm import LLMRequest, PhysocDeepSeekLLMProvider, create_llm_provider
from .models import KnowledgeChunkModel, KnowledgeSearchHitModel, KnowledgeSourceModel
from .runtime_env import load_runtime_environment


def _probe_hit() -> KnowledgeSearchHitModel:
    source = KnowledgeSourceModel(
        id="physoc-probe-source",
        name="physoc-probe.txt",
        source_type="文档",
        records=1,
        status="已索引",
        updated_at="probe",
        classification="内部",
    )
    chunk = KnowledgeChunkModel(
        id="physoc-probe-chunk",
        source_id=source.id,
        chunk_index=0,
        text="Physoc 链路正常",
        token_count=8,
    )
    return KnowledgeSearchHitModel(
        source=source,
        chunk=chunk,
        score=10.0,
        keyword_score=10.0,
        vector_score=10.0,
        rank=1,
        matched_terms=["Physoc", "链路"],
    )


def run_physoc_probe(
    environ: Mapping[str, str],
    *,
    provider_factory: Callable[[Mapping[str, str]], object] = create_llm_provider,
    clock_values: Iterator[float] | None = None,
) -> dict[str, object]:
    provider = provider_factory(environ)
    if not isinstance(provider, PhysocDeepSeekLLMProvider):
        raise ValueError("Physoc probe requires LLM_PROVIDER=physoc_deepseek")

    clock = clock_values.__next__ if clock_values is not None else time.perf_counter
    started = clock()
    reply = provider.generate_reply(
        LLMRequest(
            content="请仅根据证据说明 Physoc 链路是否正常",
            mode="source",
            knowledge_hits=[_probe_hit()],
            previous_messages=[],
            agent_context="目标服务器 Physoc POST/SSE 互操作探测",
        )
    )
    elapsed_ms = round((clock() - started) * 1000, 3)
    answer = " ".join(paragraph.text for paragraph in reply.paragraphs).strip()
    citations = [citation for paragraph in reply.paragraphs for citation in paragraph.citations]
    if not answer:
        raise ValueError("Physoc probe returned an empty normalized answer")
    if not citations:
        raise ValueError("Physoc probe did not preserve the synthetic evidence citation")

    return {
        "passed": True,
        "provider": "physoc_deepseek",
        "model": provider.model,
        "streamPath": provider.stream_path,
        "elapsedMs": elapsed_ms,
        "answerChars": len(answer),
        "citationCount": len(citations),
    }


def write_probe_report(path: Path, report: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(report), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the configured Physoc POST/SSE route")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/benchmarks/physoc-probe.json"),
    )
    arguments = parser.parse_args(argv)
    load_runtime_environment()
    report = run_physoc_probe(os.environ)
    write_probe_report(arguments.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The report deliberately excludes the API base, prompt, evidence text, answer text, response events, and upstream exception details.

- [ ] **Step 4: Run probe tests and existing SSE/provider tests**

```powershell
Push-Location backend
uv run --project . python -m unittest `
  tests.test_physoc_probe `
  tests.test_physoc_sse `
  tests.test_llm_provider -v
Pop-Location
ruff check backend/app/physoc_probe.py backend/tests/test_physoc_probe.py
ruff format --check backend/app/physoc_probe.py backend/tests/test_physoc_probe.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the probe**

```powershell
git add backend/app/physoc_probe.py backend/tests/test_physoc_probe.py
git commit -m "feat(backend): add physoc interoperability probe"
```

---

### Task 5: Lock the no-slice fallback contract at the API boundary

**Files:**
- Modify: `backend/tests/test_api_contract.py:187-205`
- Modify: `backend/tests/test_llm_provider.py:592-676`

- [ ] **Step 1: Strengthen the failing-model API assertion**

Extend `test_model_failure_returns_user_safe_gateway_error` in `backend/tests/test_api_contract.py`:

```python
self.assertEqual(response.status_code, 502)
self.assertEqual(
    response.json(),
    {"detail": "大模型服务暂时不可用，请稍后重试。"},
)
serialized = response.text
for forbidden in (
    "已检索到知识库中的相关依据",
    "差旅申请",
    "发票",
    "行程单",
    "chunk",
):
    self.assertNotIn(forbidden, serialized)
```

This assertion uses the seeded repository, which contains matching travel-policy evidence; therefore a passing test proves the HTTP error path does not serialize it.

- [ ] **Step 2: Add a provider-level no-template-fallback assertion**

In the existing Physoc HTTP-status failure test in `backend/tests/test_llm_provider.py`, wrap the call with a patched template provider:

```python
with (
    patch("app.llm.httpx.Client", return_value=client),
    patch.object(
        TemplateLLMProvider,
        "generate_reply",
        side_effect=AssertionError("template fallback must not run"),
    ) as template_reply,
):
    with self.assertRaises(LLMProviderError) as error:
        provider.generate_reply(
            LLMRequest(
                content="请分析现金流风险",
                mode="source",
                knowledge_hits=[indexed_hit()],
            )
        )

template_reply.assert_not_called()
self.assertIn("大模型服务暂时不可用", str(error.exception))
```

Add `TemplateLLMProvider` to the test module's existing `from app.llm import (...)` import list if it is not already present.

- [ ] **Step 3: Run the contract tests**

```powershell
Push-Location backend
uv run --project . python -m unittest tests.test_api_contract tests.test_llm_provider -v
Pop-Location
```

Expected: all tests pass and no response includes evidence text after a model failure.

- [ ] **Step 4: Commit the no-fallback contract**

```powershell
git add backend/tests/test_api_contract.py backend/tests/test_llm_provider.py
git commit -m "test(backend): forbid slice fallback on model failure"
```

---

### Task 6: Update deployment and operator documentation

**Files:**
- Modify: `README.md:40-95`
- Modify: `deploy/offline/README.md`
- Modify: `tools/tests/test_physoc_llm_contract.py`

- [ ] **Step 1: Add failing documentation contract strings**

In `test_readme_documents_the_physoc_streaming_contract`, add:

```python
for required_text in (
    "生产入口禁止 template 和 mock",
    "python -m app.physoc_probe",
    "artifacts/benchmarks/physoc-probe.json",
    "不会输出提示词、证据正文或模型回答正文",
):
    self.assertIn(required_text, section)
```

Add a deployment README test:

```python
def test_offline_runbook_documents_physoc_cutover_and_rollback(self) -> None:
    text = (REPO_ROOT / "deploy" / "offline" / "README.md").read_text(
        encoding="utf-8"
    )
    for required_text in (
        "LLM_PROVIDER=physoc_deepseek",
        "LLM_STREAM_PATH=/api/physoc/deepseek/stream",
        "python -m app.physoc_probe",
        "physoc-probe.json",
        "HTTP 502",
        "不得返回检索切片",
        "回滚",
    ):
        self.assertIn(required_text, text)
```

- [ ] **Step 2: Run the documentation contracts and verify RED**

```powershell
uv run --project backend python -m unittest tools.tests.test_physoc_llm_contract -v
```

Expected: failures list the new production and probe instructions.

- [ ] **Step 3: Update the main README**

In the Physoc section of `README.md`, keep the existing protocol contract and add:

````markdown
生产入口禁止 template 和 mock；它们仅用于本地开发和固定测试数据。公司内网部署应设置：

```text
LLM_PROVIDER=physoc_deepseek
LLM_API_BASE=http://172.16.0.10:8090
LLM_STREAM_PATH=/api/physoc/deepseek/stream
LLM_MODEL=my_deepseek_r1_7b
```

部署后必须在 API 的同一运行环境执行：

```powershell
python -m app.physoc_probe --report artifacts/benchmarks/physoc-probe.json
```

探测报告只记录提供器、模型、路径、耗时、回答字符数和引用数，不会输出提示词、证据正文或模型回答正文。
````

Keep the loopback example for same-host development, but state that a container must use a container-reachable approved private address.

- [ ] **Step 4: Add the target-host runbook**

Add a `Physoc production gate` section to `deploy/offline/README.md` with these exact commands:

```powershell
Copy-Item deploy/offline/.env.example deploy/offline/.env
# Edit LLM_API_BASE to the approved private Physoc address.
& tools/invoke_offline_compose.ps1 config
& tools/invoke_offline_compose.ps1 up -d
& tools/invoke_offline_compose.ps1 exec -T api `
  python -m app.physoc_probe --report /tmp/physoc-probe.json
& tools/invoke_offline_compose.ps1 exec -T api `
  python -c "import json; print(json.dumps(json.load(open('/tmp/physoc-probe.json')), ensure_ascii=False, sort_keys=True))"
```

Document the passing shape:

```json
{
  "answerChars": 12,
  "citationCount": 1,
  "elapsedMs": 250.0,
  "model": "my_deepseek_r1_7b",
  "passed": true,
  "provider": "physoc_deepseek",
  "streamPath": "/api/physoc/deepseek/stream"
}
```

Also document these failure rules:

- startup failure for `template` or `mock`;
- probe failure for timeout, non-2xx status, wrong content type, malformed event JSON, model mismatch, missing `done=true`, or empty answer;
- a normal document question returns HTTP 502 when Physoc is unavailable and不得返回检索切片;
- pure ClickHouse structured statistics remain deterministic and are not part of this model-route rollback.

For rollback, restore the last known-good Physoc host/model settings and restart the API. Do not set `LLM_PROVIDER=template` in production.

- [ ] **Step 5: Run documentation contracts**

```powershell
uv run --project backend python -m unittest tools.tests.test_physoc_llm_contract -v
```

Expected: all documentation contract tests pass.

- [ ] **Step 6: Commit the runbook**

```powershell
git add README.md deploy/offline/README.md tools/tests/test_physoc_llm_contract.py
git commit -m "docs: add physoc production cutover gate"
```

---

### Task 7: Run the complete Phase 1 verification and target-host gate

**Files:**
- Verify only; no production file changes unless a failing test exposes a defect.

- [ ] **Step 1: Run focused backend tests**

```powershell
Push-Location backend
uv run --project . --group offline python -m unittest `
  tests.test_llm_runtime `
  tests.test_lazy_startup `
  tests.test_physoc_sse `
  tests.test_llm_provider `
  tests.test_physoc_probe `
  tests.test_api_contract -v
Pop-Location
```

Expected: all tests pass.

- [ ] **Step 2: Run deployment contract tests**

```powershell
uv run --project backend python -m unittest `
  tools.tests.test_compose_contract `
  tools.tests.test_physoc_llm_contract `
  tools.tests.test_structured_deployment_contract -v
```

Expected: all tests pass.

- [ ] **Step 3: Run the complete backend and tool regression suites**

```powershell
Push-Location backend
uv run --project . --group offline python -m unittest discover -s tests -p "test_*.py" -v
Pop-Location
uv run --project backend --group offline python -m unittest discover -s tools/tests -p "test_*.py" -v
```

Expected: both suites pass. Tests that require Docker or target-only artifacts may skip with their documented reason; they must not be reported as locally passed.

- [ ] **Step 4: Run Ruff and compile checks**

```powershell
ruff check backend tools
ruff format --check backend tools
uv run --project backend --group offline python -m compileall -q backend/app tools
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Run the real target-host interoperability gate**

On the approved company server, after setting the real private Physoc address:

```powershell
& tools/invoke_offline_compose.ps1 config
& tools/invoke_offline_compose.ps1 up -d
& tools/invoke_offline_compose.ps1 exec -T api `
  python -m app.physoc_probe --report /tmp/physoc-probe.json
```

Expected: exit code 0 and `passed=true`. Record the container image digests, Physoc model name, report JSON, timestamp, server hardware, and network route in the deployment evidence package.

- [ ] **Step 6: Exercise the model-outage acceptance case**

Temporarily point the deployment copy of `LLM_API_BASE` at an unused approved private address, restart only the API, and submit a document question with known evidence. Expected:

- API returns HTTP 502 with the safe model-unavailable message;
- response body contains no source text, chunk text, internal URL, prompt, or upstream exception;
- the previous successful Physoc configuration is restored immediately after the check;
- a pure published ClickHouse aggregate remains available if ClickHouse is healthy.

Do not run this outage exercise against an uncontrolled shared production window; use the deployment acceptance environment.

- [ ] **Step 7: Commit final verification fixes if any**

If verification required code or documentation corrections:

```powershell
git add backend deploy tools README.md .env.example
git commit -m "fix: close physoc production gate gaps"
```

If no corrections were required, do not create an empty commit.

- [ ] **Step 8: Record the phase completion state**

The Phase 1 exit gate is satisfied only when all local tests pass and the target-host evidence contains:

- a rendered offline Compose configuration using `physoc_deepseek`;
- a successful redacted `physoc-probe.json`;
- a successful real document question;
- a model-outage HTTP 502 result with no slice leakage;
- the restored known-good configuration;
- the exact Git commit and image digests.

After this gate passes, inspect the resulting repository and write the Phase 2 unified document parsing plan before changing parser production code.
