# Physoc Output Leak Prevention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 阻止嵌套 Physoc/Ollama 事件 JSON 和检索内部元数据进入用户端最终回答。

**Architecture:** 在 `physoc_sse.py` 的协议边界严格识别并解包至多一层嵌套事件，同时验证模型和终止状态一致性；在 `llm.py` 中创建只含片段正文的模型可见上下文，内部 Citation、AgentStep 和检索审计对象保持不变。所有改动通过协议单元测试和 Provider 回归测试驱动完成。

**Tech Stack:** Python 3.12、FastAPI、httpx、SSE、JSON、unittest、pytest、uv、Ruff、ty

---

## 文件结构

- Modify: `backend/app/physoc_sse.py` — 验证标准事件并受限解包一层嵌套事件。
- Modify: `backend/tests/test_physoc_sse.py` — 覆盖正常、嵌套、冲突和二次嵌套协议。
- Modify: `backend/app/llm.py` — 只向模型发送片段正文，不发送来源和审计元数据。
- Modify: `backend/tests/test_llm_provider.py` — 验证提示词与最终回答不泄漏内部字段。
- Modify: `backend/app/__init__.py` — 后端版本升级到 `0.1.10`。
- Refresh: `backend/uv.lock` — 按后端版本管理规则重新锁定。

### Task 1: 支持一层嵌套 Physoc 事件

**Files:**
- Modify: `backend/tests/test_physoc_sse.py`
- Modify: `backend/tests/test_llm_provider.py`
- Modify: `backend/app/physoc_sse.py`

- [ ] **Step 1: 写嵌套事件失败测试**

在测试文件导入 `json`，并增加：

```python
import json

def test_collect_unwraps_one_nested_physoc_event_layer(self) -> None:
    nested_events = [
        {
            "model": "physoc-v1",
            "created_at": "2026-07-20T06:21:33Z",
            "response": "地理",
            "done": False,
        },
        {
            "model": "physoc-v1",
            "created_at": "2026-07-20T06:21:34Z",
            "response": "位置",
            "done": True,
        },
    ]
    lines: list[str] = []
    for event in nested_events:
        outer = {
            "model": "physoc-v1",
            "response": json.dumps(event, ensure_ascii=False),
            "done": event["done"],
        }
        lines.extend([f"data: {json.dumps(outer, ensure_ascii=False)}\n", "\n"])

    result = collect_physoc_response(lines, expected_model="physoc-v1")

    self.assertEqual(result, "地理位置")
    for leaked in ("model", "created_at", "response", "done"):
        self.assertNotIn(leaked, result)
```

同时在 `backend/tests/test_llm_provider.py` 顶部增加 `import json`，并增加 Provider 回归测试：

```python
def test_physoc_provider_unwraps_nested_events_without_exposing_metadata(self) -> None:
    nested = {
        "model": "deepseek-r1",
        "created_at": "2026-07-20T06:21:33Z",
        "response": "XX位于示例区域。",
        "done": True,
    }
    outer = {
        "model": "deepseek-r1",
        "response": json.dumps(nested, ensure_ascii=False),
        "done": True,
    }
    response = FakePhysocResponse(
        [f"data: {json.dumps(outer, ensure_ascii=False)}", ""]
    )
    client = RecordingPhysocClient(response)
    provider = PhysocDeepSeekLLMProvider(
        api_base="http://127.0.0.1:11434",
        stream_path="/private-stream",
        model="deepseek-r1",
    )

    with patch("app.llm.httpx.Client", return_value=client):
        reply = provider.generate_reply(
            LLMRequest(
                content="XX的地理位置",
                mode="source",
                knowledge_hits=[indexed_hit()],
            )
        )

    answer = reply.paragraphs[0].text
    self.assertEqual(answer, "XX位于示例区域。")
    for leaked in ("model", "created_at", "response", "done"):
        self.assertNotIn(leaked, answer)
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
uv run --project backend pytest backend/tests/test_physoc_sse.py::PhysocSseTests::test_collect_unwraps_one_nested_physoc_event_layer backend/tests/test_llm_provider.py::LLMProviderTest::test_physoc_provider_unwraps_nested_events_without_exposing_metadata -q
```

Expected: 两个测试均 FAIL，结果仍是内层 JSON 字符串而不是自然语言正文。

- [ ] **Step 3: 提取统一事件字段验证函数**

在 `backend/app/physoc_sse.py` 增加：

```python
def _event_fields(
    payload: object,
    *,
    expected_model: str,
) -> tuple[str, bool, str | None]:
    if not isinstance(payload, dict):
        raise PhysocStreamError("Physoc payload must be an object")

    response = payload.get("response")
    if not isinstance(response, str):
        raise PhysocStreamError("Physoc response must be a string")

    done = payload.get("done")
    if type(done) is not bool:
        raise PhysocStreamError("Physoc done must be a boolean")

    model: str | None = None
    if "model" in payload:
        raw_model = payload["model"]
        if not isinstance(raw_model, str) or not raw_model:
            raise PhysocStreamError("Physoc model must be a non-empty string")
        if raw_model != expected_model:
            raise PhysocStreamError("Physoc model mismatch")
        model = raw_model

    return response, done, model
```

- [ ] **Step 4: 增加受限嵌套识别函数**

增加：

```python
def _nested_event_payload(response: str) -> dict[str, object] | None:
    candidate = response.strip()
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None
    try:
        decoded = _decode_payload(candidate)
    except PhysocStreamError:
        return None
    if not isinstance(decoded, dict):
        return None

    event_keys = {"response", "done"}.intersection(decoded)
    if not event_keys:
        return None
    if event_keys != {"response", "done"}:
        raise PhysocStreamError("nested Physoc payload is incomplete")
    return decoded
```

增加单层解包：

```python
def _unwrap_nested_event(
    response: str,
    done: bool,
    outer_model: str | None,
    *,
    expected_model: str,
) -> tuple[str, bool]:
    nested = _nested_event_payload(response)
    if nested is None:
        return response, done

    nested_response, nested_done, nested_model = _event_fields(
        nested,
        expected_model=expected_model,
    )
    if outer_model is not None and nested_model is not None and nested_model != outer_model:
        raise PhysocStreamError("nested Physoc model mismatch")
    return nested_response, nested_done
```

- [ ] **Step 5: 在收集器中使用统一验证和单层解包**

把 `collect_physoc_response` 循环内现有字段验证替换为：

```python
payload = _decode_payload(data)
response, done, model = _event_fields(payload, expected_model=expected_model)
response, done = _unwrap_nested_event(
    response,
    done,
    model,
    expected_model=expected_model,
)
```

保留现有响应长度、事件数量、空答案和完成状态检查。

- [ ] **Step 6: 运行嵌套事件测试**

Run:

```bash
uv run --project backend pytest backend/tests/test_physoc_sse.py::PhysocSseTests::test_collect_unwraps_one_nested_physoc_event_layer backend/tests/test_llm_provider.py::LLMProviderTest::test_physoc_provider_unwraps_nested_events_without_exposing_metadata -q
```

Expected: PASS。

### Task 2: 拒绝歧义或多层事件包装

**Files:**
- Modify: `backend/tests/test_physoc_sse.py`
- Modify: `backend/app/physoc_sse.py`

- [ ] **Step 1: 写冲突和深度失败测试**

增加：

```python
def test_collect_rejects_ambiguous_nested_physoc_events(self) -> None:
    invalid_nested_events = {
        "done mismatch": (
            {"response": "ok", "done": True, "model": "physoc-v1"},
            False,
        ),
        "model mismatch": (
            {"response": "ok", "done": True, "model": "other"},
            True,
        ),
        "missing done": (
            {"response": "ok", "model": "physoc-v1"},
            True,
        ),
        "second nested layer": (
            {
                "response": json.dumps(
                    {"response": "secret", "done": True, "model": "physoc-v1"}
                ),
                "done": True,
                "model": "physoc-v1",
            },
            True,
        ),
    }

    for label, (nested, outer_done) in invalid_nested_events.items():
        with self.subTest(label=label):
            outer = {
                "model": "physoc-v1",
                "response": json.dumps(nested),
                "done": outer_done,
            }
            lines = [f"data: {json.dumps(outer)}\n", "\n"]
            with self.assertRaises(PhysocStreamError):
                collect_physoc_response(lines, expected_model="physoc-v1")
```

- [ ] **Step 2: 运行测试并确认失败原因**

Run:

```bash
uv run --project backend pytest backend/tests/test_physoc_sse.py::PhysocSseTests::test_collect_rejects_ambiguous_nested_physoc_events -q
```

Expected: 至少 `done mismatch`、`missing done` 或 `second nested layer` 未被拒绝。

- [ ] **Step 3: 收紧终止状态和二次嵌套检测**

在 `_unwrap_nested_event` 的 `_event_fields` 调用后增加内外终止状态一致性检查：

```python
if nested_done is not done:
    raise PhysocStreamError("nested Physoc done mismatch")
```

如果 `_nested_event_payload(nested_response)` 因无效字段抛出错误，让错误直接传播；不要捕获并把二次事件当作自然语言。保持普通非事件 JSON（不同时包含 `response/done`）作为合法文本。

目标实现保持：

```python
second_layer = _nested_event_payload(nested_response)
if second_layer is not None:
    raise PhysocStreamError("nested Physoc payload depth exceeded")
```

- [ ] **Step 4: 运行完整 Physoc SSE 测试**

Run:

```bash
uv run --project backend pytest backend/tests/test_physoc_sse.py -q
```

Expected: 所有标准流和新嵌套流测试通过。

### Task 3: 从模型提示词移除检索元数据

**Files:**
- Modify: `backend/tests/test_llm_provider.py`
- Modify: `backend/app/llm.py`

- [ ] **Step 1: 把知识上下文测试改为防泄漏契约**

将现有 `test_build_knowledge_context_formats_numbered_evidence` 替换为：

```python
def test_build_knowledge_context_exposes_only_numbered_chunk_text(self) -> None:
    context = build_knowledge_context([indexed_hit(score=8.75, rank=1)])

    self.assertEqual(context, "[知识片段 1]\n现金流风险与回款周期直接相关。")
    for leaked in (
        "cashflow.txt",
        "内部·机密",
        "source=",
        "classification=",
        "rank=",
        "score=",
    ):
        self.assertNotIn(leaked, context)
```

- [ ] **Step 2: 增加 Agent 摘要不进入提示词的失败测试**

将现有 `test_build_prompt_includes_guardrails_evidence_and_recent_history` 替换为：

```python
def test_build_prompt_includes_guardrails_text_and_recent_history(self) -> None:
    prompt = build_prompt(
        LLMRequest(
            content="请分析现金流风险",
            mode="source",
            knowledge_hits=[indexed_hit()],
            previous_messages=[
                ChatMessageModel(
                    id="msg-prev",
                    role="user",
                    time="2026-07-09 09:00:00",
                    content="上一轮问题",
                )
            ],
            agent_context="Agent 已完成 2 轮检索。来源：cashflow.txt。",
        )
    )

    for expected in (
        "请分析现金流风险",
        "仅基于可用知识片段",
        "未检索到足够依据",
        "[知识片段 1]",
        "现金流风险与回款周期直接相关。",
        "上一轮问题",
    ):
        self.assertIn(expected, prompt)
    for leaked in (
        "cashflow.txt",
        "内部·机密",
        "source=",
        "classification=",
        "rank=",
        "score=",
        "Agent 调查摘要",
        "Agent 已完成",
    ):
        self.assertNotIn(leaked, prompt)
```

- [ ] **Step 3: 运行测试并确认失败**

Run:

```bash
uv run --project backend pytest backend/tests/test_llm_provider.py::LLMProviderTest::test_build_knowledge_context_exposes_only_numbered_chunk_text backend/tests/test_llm_provider.py::LLMProviderTest::test_build_prompt_includes_guardrails_text_and_recent_history -q
```

Expected: FAIL，当前提示词仍包含来源、分类、排名、分数或 Agent 摘要。

- [ ] **Step 4: 最小化模型可见知识上下文**

把 `build_knowledge_context` 改为：

```python
def build_knowledge_context(hits: list[KnowledgeSearchHitModel]) -> str:
    return "\n\n".join(
        f"[知识片段 {index}]\n{snippet_text(hit.chunk.text, 500)}"
        for index, hit in enumerate(hits, start=1)
    )
```

- [ ] **Step 5: 从提示词删除 Agent 调查摘要**

把 `build_prompt` 返回内容中的以下段落删除：

```python
f"Agent 调查摘要：\n{request.agent_context or '未启用多步调查'}\n\n"
```

保留 `LLMRequest.agent_context` 字段以及 Agent 内部审计生成逻辑，避免影响管理端审计。

- [ ] **Step 6: 运行提示词测试**

Run:

```bash
uv run --project backend pytest backend/tests/test_llm_provider.py -q
```

Expected: 所有 Provider 和提示词测试通过。

### Task 4: 升级后端版本并完成验证

**Files:**
- Modify: `backend/app/__init__.py`
- Refresh: `backend/uv.lock`
- Verify: `tools/tests/test_version_contract.py`

- [ ] **Step 1: 独立提升后端 patch 版本**

Run:

```bash
uv run --project backend python tools/bump_version.py backend patch
uv lock --project backend
```

Expected:

```text
backend version: 0.1.9 -> 0.1.10
```

用户端和管理端 `package.json` 均保持 `0.1.1`。

- [ ] **Step 2: 运行版本契约与后端完整测试**

Run:

```bash
uv run --project backend pytest tools/tests/test_version_contract.py -q
uv run --project backend pytest backend/tests -q
```

Expected: 版本契约与全部后端测试通过，允许仓库既有 skip 和弃用警告。

- [ ] **Step 3: 运行部署契约测试**

Run:

```bash
uv run --project backend pytest tools/tests/test_compose_contract.py tools/tests/test_structured_deployment_contract.py -q
```

Expected: 全部部署契约测试通过。

- [ ] **Step 4: 运行静态检查和差异检查**

Run:

```bash
fast lint --ty
git diff --check
```

Expected: Ruff、ty 和空白检查均无错误。

- [ ] **Step 5: 检查最终范围**

Run:

```bash
git status --short
git diff -- backend/app/physoc_sse.py backend/app/llm.py backend/app/__init__.py backend/uv.lock backend/tests/test_physoc_sse.py backend/tests/test_llm_provider.py
```

Expected: 只包含本设计的协议解析、提示词最小化、测试和后端版本改动；前端版本文件无变化。
