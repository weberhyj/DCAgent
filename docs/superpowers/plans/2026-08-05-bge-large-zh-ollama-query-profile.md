# BGE Large 中文 Ollama 查询配置档 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Ollama Embedding adapter 切换为可显式选择的 `raw`/`bge-large-zh-v1.5` 查询配置档，并让 BGE query 自动使用固定中文检索前缀，同时把 endpoint、配置档和前缀纳入 Embedding 指纹。

**Architecture:** 保留 `POST /v1/embeddings` 对外协议和现有 `EmbeddingBackend.embed(texts, purpose=...)` 接口。在 Ollama adapter 内增加经过白名单校验的 query profile：document 始终原文，BGE query 在发送到 Ollama 前添加固定前缀。服务启动从环境变量读取 profile，并用 endpoint + profile 计算配置档 SHA-256；任何缺失或不匹配都 fail closed。Qdrant 仍通过全新 collection 版本全量重建，不复用旧向量。

**Tech Stack:** Python 3.12、FastAPI、Pydantic、Ollama HTTP API、unittest/pytest、Ruff、Ubuntu Bash Compose 文档。

---

## 文件边界

- Modify: `backend/app/ollama_embedding_backend.py` — query profile 白名单、BGE 前缀转换、四种 endpoint/profile 编码档和哈希。
- Modify: `backend/app/embedding_service.py` — 读取并校验 `OLLAMA_EMBEDDING_QUERY_PROFILE`，把 profile 绑定到元数据和 backend。
- Test: `backend/tests/test_ollama_embedding_backend.py` — profile、前缀、endpoint 请求体和编码档回归测试。
- Test: `backend/tests/test_embedding_service.py` — 启动环境校验、元数据指纹和 backend 构造参数测试。
- Test: `backend/tests/test_offline_artifacts.py` — Compose 与 `.env.example` 的 BGE 配置一致性测试。
- Modify: `deploy/offline/.env.example` — 默认切换 BGE 模型、维度示例、profile 和 BGE `/api/embed` 指纹。
- Modify: `deploy/offline/compose.yaml` — embedding-service 注入必需的 query profile 环境变量。
- Modify: `README.md` — 更新 Ollama 探针、默认模型和索引迁移说明。
- Modify: `deploy/offline/README.md` — 更新 Ubuntu/Bash 部署、探针、profile 指纹和重建步骤。
- Modify: `docs/intranet-deployment-configuration.md` — 更新中文内网配置清单和 BGE 前缀约束。

## 固定协议值

实现必须使用以下值，不能在运行时自由拼接：

```text
OLLAMA_EMBEDDING_QUERY_PROFILE=raw
OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5
BGE query prefix=为这个句子生成表示以用于检索相关文章：
raw prefix sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
BGE prefix sha256=2bb658b7e092d6b4b1dbde4c3fc5f281f9ed9f1ace5b49566fb8b10f57836e48
modern/raw profile sha256=416da888eb3791bac5a5d7d6b8a8c2f521fb208ac2ffa126f542e56130ac7546
modern/BGE profile sha256=3d5db261732d456b51fa4f9aa89cb15054c21772c0809a50a31f0911eb960170
legacy/raw profile sha256=844eb6bdd11c0a7ce0c021b0df27f5159583d366de76bb7a866eec0bfd92666f
legacy/BGE profile sha256=b8e7252a57feef349f02d6b2624ef3f9e8bc9e989d9073e37aa5df424cf26de4
```

### Task 1: Add failing backend profile tests

**Files:**
- Test: `backend/tests/test_ollama_embedding_backend.py`

- [ ] **Step 1: Replace the two single endpoint profile fixtures with four canonical fixtures.**

  Add this independent expected-profile fixture to the test file:

  ```python
  RAW_PREFIX_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  BGE_PREFIX_SHA256 = "2bb658b7e092d6b4b1dbde4c3fc5f281f9ed9f1ace5b49566fb8b10f57836e48"
  BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


  def expected_encoding_profile(path: str, query_profile: str) -> str:
      prefix_sha256 = RAW_PREFIX_SHA256 if query_profile == "raw" else BGE_PREFIX_SHA256
      query_mode = "raw_text" if query_profile == "raw" else "prefixed_text"
      endpoint_lines = (
          ("input=transformed_text_batch", "truncate=true", "output.count=one_per_input")
          if path == "/api/embed"
          else ("prompt=single_transformed_text", "output.count=one_per_input")
      )
      return "\n".join(
          (
              "profile=dc-agent.ollama.embedding",
              "protocol=dc-agent.ollama.embedding.v2",
              f"purpose.query={query_mode}",
              f"purpose.query.profile={query_profile}",
              f"purpose.query.prefix_sha256={prefix_sha256}",
              "purpose.document=raw_text",
              f"path={path}",
              *endpoint_lines,
              "output.dimensions=configured_exact",
              "output.coordinates=finite_numeric",
              "output.vector=nonzero",
              "normalization.algorithm=max_abs_scaled_l2",
              "normalization.output=unit_l2",
          )
      )
  ```

- [ ] **Step 2: Add a failing test for four distinct, reproducible profile hashes.**

  Add this test:

  ```python
  def test_endpoint_and_query_profiles_are_canonical_distinct_and_hash_derived(self) -> None:
      hashes = set()
      for path in ("/api/embed", "/api/embeddings"):
          for query_profile in ("raw", "bge-large-zh-v1.5"):
              with self.subTest(path=path, query_profile=query_profile):
                  expected = expected_encoding_profile(path, query_profile)
                  actual = ollama_embedding_backend.ollama_embedding_encoding_profile(
                      path, query_profile
                  )
                  profile_hash = (
                      ollama_embedding_backend.ollama_embedding_encoding_profile_sha256(
                          path, query_profile
                      )
                  )
                  self.assertEqual(actual, expected)
                  self.assertTrue(actual.isascii())
                  self.assertNotIn("\r", actual)
                  self.assertFalse(actual.endswith("\n"))
                  self.assertTrue(
                      all(line.count("=") == 1 for line in actual.split("\n"))
                  )
                  self.assertEqual(
                      profile_hash,
                      hashlib.sha256(expected.encode("utf-8")).hexdigest(),
                  )
                  hashes.add(profile_hash)
      self.assertEqual(len(hashes), 4)
  ```

- [ ] **Step 3: Add failing tests for BGE query transformation on both endpoints.**

  Add a parameterized unittest loop:

  ```python
  def test_bge_profile_prefixes_only_query_text_for_both_endpoints(self) -> None:
      for path, response, payload_field in (
          ("/api/embed", {"embeddings": [[1, 0]]}, "input"),
          ("/api/embeddings", {"embedding": [1, 0]}, "prompt"),
      ):
          for purpose, expected_text in (
              ("query", f"{BGE_QUERY_PREFIX}原始查询"),
              ("document", "原始查询"),
          ):
              with self.subTest(path=path, purpose=purpose):
                  client = RecordingOllamaClient([response])
                  backend = OllamaEmbeddingBackend(
                      client,
                      model="bge-large-zh-v1.5:latest",
                      path=path,
                      dimensions=2,
                      keep_alive="10m",
                      query_profile="bge-large-zh-v1.5",
                  )
                  backend.embed(["原始查询"], purpose=purpose)
                  payload = client.calls[0][1]
                  assert isinstance(payload, Mapping)
                  expected_payload_value: object = (
                      [expected_text] if path == "/api/embed" else expected_text
                  )
                  self.assertEqual(payload[payload_field], expected_payload_value)
  ```

- [ ] **Step 4: Add failing tests for invalid profiles.**

  Add this test:

  ```python
  def test_constructor_rejects_unknown_query_profiles(self) -> None:
      for query_profile in ("", "   ", "BGE-LARGE-ZH-V1.5", "unknown"):
          with self.subTest(query_profile=query_profile):
              client = RecordingOllamaClient([])
              with self.assertRaisesRegex(ValueError, "query profile"):
                  OllamaEmbeddingBackend(
                      client,
                      model="bge-large-zh-v1.5:latest",
                      path="/api/embed",
                      dimensions=2,
                      keep_alive="10m",
                      query_profile=query_profile,
                  )
              self.assertEqual(client.calls, [])
  ```

- [ ] **Step 5: Run the focused test file and confirm RED.**

  Run:

  ```bash
  cd backend
  .venv/Scripts/python.exe -m pytest tests/test_ollama_embedding_backend.py -q
  ```

  Expected result: failures because the current backend has no `query_profile` argument, no BGE transformation, and only endpoint-only profile hashes.

### Task 2: Implement the minimal Ollama query profile contract

**Files:**
- Modify: `backend/app/ollama_embedding_backend.py`
- Test: `backend/tests/test_ollama_embedding_backend.py`

- [ ] **Step 1: Add the profile constants and strict resolver.**

  Add this contract near the module constants:

  ```python
  OLLAMA_RAW_QUERY_PROFILE = "raw"
  OLLAMA_BGE_LARGE_ZH_V15_QUERY_PROFILE = "bge-large-zh-v1.5"
  BGE_LARGE_ZH_V15_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
  _QUERY_PREFIXES = {
      OLLAMA_RAW_QUERY_PROFILE: "",
      OLLAMA_BGE_LARGE_ZH_V15_QUERY_PROFILE: BGE_LARGE_ZH_V15_QUERY_PREFIX,
  }


  def ollama_embedding_query_prefix(query_profile: str) -> str:
      try:
          return _QUERY_PREFIXES[query_profile]
      except (KeyError, TypeError):
          raise ValueError("unsupported Ollama embedding query profile") from None
  ```

- [ ] **Step 2: Make profile-aware encoding profile generation.**

  Replace the endpoint-only profile mapping with a builder equivalent to:

  ```python
  def ollama_embedding_encoding_profile(path: str, query_profile: str = "raw") -> str:
      prefix = ollama_embedding_query_prefix(query_profile)
      if path == "/api/embed":
          endpoint_lines = (
              "input=transformed_text_batch",
              "truncate=true",
              "output.count=one_per_input",
          )
      elif path == "/api/embeddings":
          endpoint_lines = (
              "prompt=single_transformed_text",
              "output.count=one_per_input",
          )
      else:
          raise ValueError("Ollama embedding path must be /api/embed or /api/embeddings")
      return "\n".join(
          (
              "profile=dc-agent.ollama.embedding",
              "protocol=dc-agent.ollama.embedding.v2",
              f"purpose.query={'prefixed_text' if prefix else 'raw_text'}",
              f"purpose.query.profile={query_profile}",
              "purpose.query.prefix_sha256="
              + hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
              "purpose.document=raw_text",
              f"path={path}",
              *endpoint_lines,
              "output.dimensions=configured_exact",
              "output.coordinates=finite_numeric",
              "output.vector=nonzero",
              "normalization.algorithm=max_abs_scaled_l2",
              "normalization.output=unit_l2",
          )
      )


  def ollama_embedding_encoding_profile_sha256(
      path: str, query_profile: str = "raw"
  ) -> str:
      profile = ollama_embedding_encoding_profile(path, query_profile)
      return hashlib.sha256(profile.encode("utf-8")).hexdigest()
  ```

  Keep `OLLAMA_EMBEDDING_ENCODING_PROFILE` and
  `OLLAMA_EMBEDDING_ENCODING_PROFILE_SHA256` mapped to `/api/embed` + `raw` for import compatibility.

- [ ] **Step 3: Add `query_profile` to `OllamaEmbeddingBackend`.**

  Extend the constructor and request transformation as follows:

  ```python
  def __init__(
      self,
      client: SyncOllamaClient,
      *,
      model: str,
      path: str,
      dimensions: int,
      keep_alive: str,
      query_profile: str,
  ) -> None:
      # retain the existing model/path/dimension/keep_alive validation
      self._query_prefix = ollama_embedding_query_prefix(query_profile)
      self._query_profile = query_profile

  def _request_text(self, text: str, purpose: EmbeddingPurpose) -> str:
      if purpose == "query" and self._query_prefix:
          return f"{self._query_prefix}{text}"
      return text
  ```

  Inside `embed`, after validating `values`, set:

  ```python
  request_texts = [self._request_text(text, purpose) for text in values]
  ```

  Use `request_texts` in the modern `input` array and legacy `prompt` loop. Continue validating the returned vector count against `values` so prefixing cannot alter request cardinality.

- [ ] **Step 4: Run the focused backend tests and confirm GREEN.**

  Update every existing `OllamaEmbeddingBackend(...)` test fixture with
  `query_profile="raw"` unless the test explicitly covers BGE, then run:

  ```bash
  cd backend
  .venv/Scripts/python.exe -m pytest tests/test_ollama_embedding_backend.py -q
  ```

  Expected result: all backend adapter tests pass, including existing normalization, malformed response, legacy endpoint, and close behavior tests.

### Task 3: Bind query profile to embedding service startup

**Files:**
- Modify: `backend/app/embedding_service.py`
- Test: `backend/tests/test_embedding_service.py`

- [ ] **Step 1: Extend the test environment fixture with an explicit raw profile.**

  Add `OLLAMA_EMBEDDING_QUERY_PROFILE="raw"` to `production_environment()` and calculate its default `EMBEDDING_ENCODING_PROFILE_SHA256` with `ollama_embedding_encoding_profile_sha256("/api/embed", "raw")`.

- [ ] **Step 2: Add failing startup tests for profile behavior.**

  Add these cases to the existing startup validation table:

  ```python
  ("OLLAMA_EMBEDDING_QUERY_PROFILE", None, "OLLAMA_EMBEDDING_QUERY_PROFILE"),
  ("OLLAMA_EMBEDDING_QUERY_PROFILE", "BGE-LARGE-ZH-V1.5", "query profile"),
  ("OLLAMA_EMBEDDING_QUERY_PROFILE", "unknown", "query profile"),
  ```

  Add an explicit mismatch test with `OLLAMA_EMBEDDING_QUERY_PROFILE="raw"` and the modern/BGE hash, then add a success test that uses the modern/BGE hash and asserts:

  ```python
  backend_type.assert_called_once_with(
      client,
      model="bge-large-zh-v1.5:latest",
      path="/api/embed",
      dimensions=1024,
      keep_alive="5m",
      query_profile="bge-large-zh-v1.5",
  )
  ```

- [ ] **Step 3: Implement profile loading and fingerprint validation.**

  In `_load_environment_metadata`, use this sequence before constructing `EmbeddingModelMetadata`:

  ```python
  path = _required_environment_value(environ, "OLLAMA_EMBEDDING_PATH")
  query_profile = _required_environment_value(
      environ, "OLLAMA_EMBEDDING_QUERY_PROFILE"
  )
  ollama_embedding_query_prefix(query_profile)
  expected_profile_sha256 = ollama_embedding_encoding_profile_sha256(
      path, query_profile
  )
  if not hmac.compare_digest(encoding_profile_sha256, expected_profile_sha256):
      raise ValueError(
          "EMBEDDING_ENCODING_PROFILE_SHA256 must match the Ollama embedding encoding profile"
      )
  ```

  In `_load_ollama_embedding_backend`, read the same profile and pass
  `query_profile=query_profile` to `OllamaEmbeddingBackend`; do not infer it from `model`.

- [ ] **Step 4: Update startup probe assertions.**

  Keep both `query` and `document` probes. Assert in the test backend/loader that query receives the transformed input while document receives raw input; the response metadata and outer `/v1/embeddings` wire shape remain unchanged.

- [ ] **Step 5: Run the service tests and confirm GREEN.**

  Run:

  ```bash
  .venv/Scripts/python.exe -m pytest tests/test_embedding_service.py tests/test_ollama_embedding_backend.py -q
  ```

  Expected result: all service, profile, startup, response validation, and adapter tests pass.

### Task 4: Migrate Compose and environment templates

**Files:**
- Modify: `deploy/offline/compose.yaml`
- Modify: `deploy/offline/.env.example`
- Test: `backend/tests/test_offline_artifacts.py`

- [ ] **Step 1: Add a failing Compose/environment contract test.**

  Add this test to `OfflineArtifactManifestTest`:

  ```python
  def test_embedding_compose_environment_pins_bge_query_profile(self) -> None:
      repository = Path(__file__).resolve().parents[2]
      compose = (repository / "deploy" / "offline" / "compose.yaml").read_text(
          encoding="utf-8"
      )
      example = (repository / "deploy" / "offline" / ".env.example").read_text(
          encoding="utf-8"
      )
      self.assertIn(
          "OLLAMA_EMBEDDING_QUERY_PROFILE: "
          "${OLLAMA_EMBEDDING_QUERY_PROFILE:?OLLAMA_EMBEDDING_QUERY_PROFILE is required}",
          compose,
      )
      for line in (
          "EMBEDDING_MODEL_NAME=bge-large-zh-v1.5:latest",
          "EMBEDDING_MODEL_VERSION=ollama-bge-large-zh-v15-v1",
          "EMBEDDING_MODEL_DIMENSIONS=1024",
          "EMBEDDING_ENCODING_PROFILE_SHA256="
          "3d5db261732d456b51fa4f9aa89cb15054c21772c0809a50a31f0911eb960170",
          "OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest",
          "OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5",
      ):
          self.assertIn(line, example)
  ```

  Run `cd backend && .venv/Scripts/python.exe -m pytest tests/test_offline_artifacts.py::OfflineArtifactManifestTest::test_embedding_compose_environment_pins_bge_query_profile -q`. Expected: FAIL because the new variable and BGE defaults are not present.

- [ ] **Step 2: Require the profile in the embedding-service Compose environment.**

  Add exactly this line beside `OLLAMA_EMBEDDING_PATH` in `embedding-service.environment`:

  ```yaml
  OLLAMA_EMBEDDING_QUERY_PROFILE: ${OLLAMA_EMBEDDING_QUERY_PROFILE:?OLLAMA_EMBEDDING_QUERY_PROFILE is required}
  ```

  Do not inject this variable into API, worker, or reranker services because only the embedding adapter transforms query text.

- [ ] **Step 3: Switch the example Embedding model to BGE.**

  Make the Embedding section of `.env.example` contain these checked-in example values:

  ```env
  EMBEDDING_MODEL_NAME=bge-large-zh-v1.5:latest
  EMBEDDING_MODEL_VERSION=ollama-bge-large-zh-v15-v1
  EMBEDDING_MODEL_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  EMBEDDING_MODEL_DIMENSIONS=1024
  EMBEDDING_MODEL_NORMALIZED=true
  EMBEDDING_ENCODING_PROFILE_SHA256=3d5db261732d456b51fa4f9aa89cb15054c21772c0809a50a31f0911eb960170
  EMBEDDING_PROTOCOL_VERSION=v1
  OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
  OLLAMA_EMBEDDING_PATH=/api/embed
  OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5
  ```

  Keep the adjacent comment that the digest and dimension are operator-measured values; the `a...a` digest remains an intentionally invalid example placeholder.

- [ ] **Step 4: Run the config contract test and static consistency check.**

  First run the focused test again. Expected: PASS.

  Run:

  ```bash
  docker compose --env-file deploy/offline/.env.example -f deploy/offline/compose.yaml config
  ```

  If the local Docker gate is unavailable, inspect the rendered environment with `rg` and record the target-host Compose config check as not run; do not claim a live gate passed locally.

### Task 5: Update Ubuntu/Bash deployment documentation

**Files:**
- Modify: `README.md`
- Modify: `deploy/offline/README.md`
- Modify: `docs/intranet-deployment-configuration.md`

- [ ] **Step 1: Update model pull and probe commands.**

  Use these model commands and payload names in all three documents:

  ```bash
  ollama pull bge-large-zh-v1.5:latest
  ollama pull qwen2.5:3b
  embed_json="$(curl --fail-with-body --silent --show-error \
    -H 'Content-Type: application/json' \
    --data-binary '{"model":"bge-large-zh-v1.5:latest","input":["dimension-probe"],"truncate":true,"keep_alive":"30m"}' \
    "$ollama_url/api/embed")"
  ```

  Keep the digest loop but change its model list to
  `bge-large-zh-v1.5:latest qwen2.5:3b`. Keep the explicit legacy `/api/embeddings` fallback and document its BGE profile hash `b8e7252a57feef349f02d6b2624ef3f9e8bc9e989d9073e37aa5df424cf26de4`.

- [ ] **Step 2: Document the query profile and prefix contract.**

  Add this deployment block:

  ```env
  EMBEDDING_MODEL_NAME=bge-large-zh-v1.5:latest
  EMBEDDING_MODEL_VERSION=ollama-bge-large-zh-v15-v1
  EMBEDDING_ENCODING_PROFILE_SHA256=3d5db261732d456b51fa4f9aa89cb15054c21772c0809a50a31f0911eb960170
  OLLAMA_EMBEDDING_MODEL=bge-large-zh-v1.5:latest
  OLLAMA_EMBEDDING_PATH=/api/embed
  OLLAMA_EMBEDDING_QUERY_PROFILE=bge-large-zh-v1.5
  ```

  Explain that only query text receives `为这个句子生成表示以用于检索相关文章：`, document text is unchanged, and `EMBEDDING_MODEL_SHA256`/`EMBEDDING_MODEL_DIMENSIONS` must come from the target Ollama server.

- [ ] **Step 3: Document fingerprint and rebuild requirements.**

  State that the profile hash, endpoint path, model digest, and measured dimension must match `/v1/metadata`; after changing any of them, use a never-used `knowledge_chunks_qwen3_vN` collection and rebuild all Word/PDF/TXT/Excel vectors before activating the Alias.

- [ ] **Step 4: Remove stale claims that Qwen2.5 0.5B is the default Embedding model.**

  Preserve historical compatibility notes where the route name remains `qwen3`, but update current deployment topology and examples to BGE.

### Task 6: Run complete verification and prepare the implementation commit

**Files:**
- Verify `backend/app/ollama_embedding_backend.py`, `backend/app/embedding_service.py`, `backend/tests/test_ollama_embedding_backend.py`, `backend/tests/test_embedding_service.py`, `backend/tests/test_offline_artifacts.py`, `deploy/offline/compose.yaml`, `deploy/offline/.env.example`, `README.md`, `deploy/offline/README.md`, and `docs/intranet-deployment-configuration.md`; no additional production files are in scope.

- [ ] **Step 1: Run focused tests after documentation/config changes.**

  ```bash
  cd backend
  .venv/Scripts/python.exe -m pytest tests/test_ollama_embedding_backend.py tests/test_embedding_service.py -q
  ```

  Expected result: zero failures.

- [ ] **Step 2: Run settings, startup, and deployment-related tests.**

  ```bash
  .venv/Scripts/python.exe -m pytest tests/test_retrieval_settings.py tests/test_lazy_startup.py tests/test_infra_health.py tests/test_offline_artifacts.py -q
  ```

  Expected result: zero failures; any environment-specific Docker gate must be reported separately.

- [ ] **Step 3: Run Ruff and repository-wide backend tests.**

  ```bash
  & "$env:USERPROFILE\.local\bin\ruff.exe" check app tests
  & "$env:USERPROFILE\.local\bin\ruff.exe" format --check app tests
  .venv/Scripts/python.exe -m pytest --import-mode=importlib -q
  git diff --check
  ```

  Expected result: Ruff clean, formatting clean, and all backend tests passing with only the repository's existing skips/warnings.

- [ ] **Step 4: Review the final diff for stale model references and unsafe profile fallbacks.**

  ```bash
  rg -n "qwen2\.5:0\.5b|OLLAMA_EMBEDDING_QUERY_PROFILE|3d5db261|b8e7252a" README.md deploy docs backend/app backend/tests
  git diff --stat
  git status -sb
  ```

  Confirm current deployment examples use BGE, reranker examples remain Qwen2.5 3B, and no code path silently falls back from BGE to raw.

- [ ] **Step 5: Commit the implementation as one coherent change.**

  ```bash
  git add backend/app/ollama_embedding_backend.py backend/app/embedding_service.py backend/tests/test_ollama_embedding_backend.py backend/tests/test_embedding_service.py backend/tests/test_offline_artifacts.py deploy/offline/compose.yaml deploy/offline/.env.example README.md deploy/offline/README.md docs/intranet-deployment-configuration.md docs/superpowers/specs/2026-08-05-bge-large-zh-ollama-query-profile-design.md docs/superpowers/plans/2026-08-05-bge-large-zh-ollama-query-profile.md
  git commit -m "feat: 支持BGE中文Embedding查询配置档"
  ```

  Push only after the user explicitly requests the Git publish step.

## Execution Notes

- Do not use an old collection name after a failed build; the publication audit keeps names unique.
- Do not modify `EMBEDDING_PROTOCOL_VERSION` unless the outer `/v1/embeddings` JSON contract changes; the query profile belongs in `EMBEDDING_ENCODING_PROFILE_SHA256`.
- Do not set `EMBEDDING_MODEL_DIMENSIONS=1024` in a real deployment without measuring the target Ollama response first; `1024` is only the checked-in example for BGE Large Chinese.
- If the target Ollama instance only exposes `/api/embeddings`, use the legacy/BGE hash from the fixed values and let the adapter issue one prompt request per input.
