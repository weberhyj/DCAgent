# Physoc Fragmented Event Output Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 阻止跨多条外层 SSE 消息分片传输的 Physoc 事件 JSON，以及附加在正常答案末尾的 Physoc 事件 JSON，进入用户最终回答。

**Architecture:** 保留现有逐事件单层解包，在外层流声明完成后增加一次严格的最终组装检查。检查器只识别能够从某个 `{` 位置连续解析到文本末尾、且满足 `response`、`done`、`model` 契约的 Physoc 事件序列；普通文本和非协议 JSON 保持不变，不使用通用正则清洗。

**Tech Stack:** Python 3.12、SSE、JSON、unittest、pytest、uv、Ruff、ty

---

## 文件结构

- Modify: `backend/tests/test_physoc_sse.py` — 增加跨事件分片、正文后缀、连续事件和普通 JSON 保留测试。
- Modify: `backend/app/physoc_sse.py` — 增加严格 JSON 前缀解码、事件序列解析和最终组装规范化。
- Modify: `backend/tests/test_llm_provider.py` — 验证 Provider 返回值不包含分片事件元数据。
- Modify: `backend/app/__init__.py` — 后端版本从 `0.1.10` 升级到 `0.1.11`。
- Refresh if changed: `backend/uv.lock` — 执行后端锁文件刷新命令。

### Task 1: 复现跨外层 SSE 分片的内层事件泄漏

**Files:**
- Modify: `backend/tests/test_physoc_sse.py`
- Modify: `backend/tests/test_llm_provider.py`

- [ ] **Step 1: 增加完整内层事件被拆成多个外层 response 的失败测试**

在 `PhysocSseTests` 中增加：

```python
def test_collect_unwraps_nested_event_fragmented_across_outer_events(self) -> None:
    nested = json.dumps(
        {
            "model": "deepseek-llm:7b",
            "created_at": "2026-08-07T00:00:00Z",
            "response": "正确答案",
            "done": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    split_at = nested.index('"response"')
    fragments = (nested[:split_at], nested[split_at:])
    lines: list[str] = []
    for index, fragment in enumerate(fragments):
        outer = {
            "model": "deepseek-llm:7b",
            "response": fragment,
            "done": index == len(fragments) - 1,
        }
        lines.extend([f"data: {json.dumps(outer, ensure_ascii=False)}\n", "\n"])

    result = collect_physoc_response(lines, expected_model="deepseek-llm:7b")

    self.assertEqual(result, "正确答案")
    for leaked in ("model", "created_at", "response", "done"):
        self.assertNotIn(leaked, result)
```

- [ ] **Step 2: 增加正常答案后附加分片事件的失败测试**

```python
def test_collect_removes_fragmented_event_metadata_appended_after_plain_text(self) -> None:
    suffix = json.dumps(
        {
            "model": "deepseek-llm:7b",
            "created_at": "2026-08-07T00:00:00Z",
            "response": "",
            "done": True,
        },
        separators=(",", ":"),
    )
    fragments = ("XX位于示例区域。" + suffix[:24], suffix[24:])
    lines: list[str] = []
    for index, fragment in enumerate(fragments):
        outer = {
            "model": "deepseek-llm:7b",
            "response": fragment,
            "done": index == len(fragments) - 1,
        }
        lines.extend([f"data: {json.dumps(outer, ensure_ascii=False)}\n", "\n"])

    self.assertEqual(
        collect_physoc_response(lines, expected_model="deepseek-llm:7b"),
        "XX位于示例区域。",
    )
```

- [ ] **Step 3: 增加 Provider 级失败测试**

在 `LLMProviderTest` 中增加：

```python
def test_physoc_provider_unwraps_fragmented_nested_event_without_metadata(self) -> None:
    nested = json.dumps(
        {
            "model": "deepseek-r1",
            "created_at": "2026-08-07T00:00:00Z",
            "response": "XX位于示例区域。",
            "done": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    split_at = nested.index('"response"')
    response_lines: list[str] = []
    for index, fragment in enumerate((nested[:split_at], nested[split_at:])):
        outer = {
            "model": "deepseek-r1",
            "response": fragment,
            "done": index == 1,
        }
        response_lines.extend(
            [f"data: {json.dumps(outer, ensure_ascii=False)}", ""]
        )
    response = FakePhysocResponse(response_lines)
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

- [ ] **Step 4: 运行新测试并确认按预期失败**

Run:

```bash
cd backend
uv run --project . python -m pytest \
  tests/test_physoc_sse.py::PhysocSseTests::test_collect_unwraps_nested_event_fragmented_across_outer_events \
  tests/test_physoc_sse.py::PhysocSseTests::test_collect_removes_fragmented_event_metadata_appended_after_plain_text \
  tests/test_llm_provider.py::LLMProviderTest::test_physoc_provider_unwraps_fragmented_nested_event_without_metadata \
  -q
```

Expected: 三个测试均 FAIL，实际答案仍包含 `{\"model\":...}` 事件 JSON。

### Task 2: 在最终组装阶段严格解析 Physoc 事件后缀

**Files:**
- Modify: `backend/app/physoc_sse.py`

- [ ] **Step 1: 复用严格 JSONDecoder 并支持指定位置解码**

在重复键与非标准常量校验函数之后创建严格解码器：

```python
_STRICT_JSON_DECODER = json.JSONDecoder(
    object_pairs_hook=_reject_duplicate_keys,
    parse_constant=_reject_json_constant,
)
```

把 `_decode_payload` 改为使用 `_STRICT_JSON_DECODER.decode(data)`。增加只用于最终组装检查的非抛出式前缀解码：

```python
def _try_decode_payload_at(data: str, start: int) -> tuple[object, int] | None:
    try:
        return _STRICT_JSON_DECODER.raw_decode(data, start)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
```

- [ ] **Step 2: 增加严格事件序列解析函数**

```python
def _event_sequence_response(
    data: str,
    start: int,
    *,
    expected_model: str,
) -> str | None:
    position = start
    response_parts: list[str] = []
    recognized = False
    completed = False

    while position < len(data):
        while position < len(data) and data[position].isspace():
            position += 1
        if position == len(data):
            break

        decoded = _try_decode_payload_at(data, position)
        if decoded is None:
            if recognized:
                raise PhysocStreamError("fragmented Physoc payload is invalid")
            return None
        payload, end = decoded
        if not isinstance(payload, dict) or not {"response", "done"}.issubset(payload):
            if recognized:
                raise PhysocStreamError("fragmented Physoc payload is invalid")
            return None

        if completed:
            raise PhysocStreamError("fragmented Physoc data follows completion")
        response, done, _model = _event_fields(payload, expected_model=expected_model)
        if _nested_event_payload(response) is not None:
            raise PhysocStreamError("nested Physoc payload depth exceeded")
        response_parts.append(response)
        recognized = True
        completed = done
        position = end

        if completed and data[position:].strip():
            raise PhysocStreamError("fragmented Physoc data follows completion")

    if not recognized:
        return None
    if not completed:
        raise PhysocStreamError("fragmented Physoc stream ended before completion")
    return "".join(response_parts)
```

该函数只在候选位置首先解析出同时包含 `response` 和 `done` 的对象时识别协议序列。候选位置不是协议对象时返回 `None`，不修改普通正文。

- [ ] **Step 3: 增加最终组装规范化函数**

```python
def _normalize_assembled_response(data: str, *, expected_model: str) -> str:
    for start, character in enumerate(data):
        if character != "{":
            continue
        event_response = _event_sequence_response(
            data,
            start,
            expected_model=expected_model,
        )
        if event_response is not None:
            return data[:start] + event_response
    return data
```

从左到右选择第一个能够严格解析到文本末尾的合法事件序列，因此既能处理整个累积结果都是事件 JSON，也能处理自然语言前缀后跟事件 JSON。事件对象字符串内部的花括号已被 JSON 转义，不会被误认为新的顶层候选位置。

- [ ] **Step 4: 在返回最终结果前调用组装规范化**

把 `collect_physoc_response` 的完成分支改为：

```python
if done:
    result = _normalize_assembled_response(
        "".join(response_parts),
        expected_model=expected_model,
    )
    if not result:
        raise PhysocStreamError("Physoc response is empty")
    return result
```

- [ ] **Step 5: 运行 Task 1 的三个测试**

Run: Task 1 Step 4 的命令。

Expected: PASS。

### Task 3: 收紧终止顺序并保护普通 JSON 正文

**Files:**
- Modify: `backend/tests/test_physoc_sse.py`
- Modify: `backend/app/physoc_sse.py`

- [ ] **Step 1: 增加连续事件序列测试**

```python
def test_collect_joins_fragmented_physoc_event_sequence(self) -> None:
    sequence = "".join(
        json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        for event in (
            {"model": "physoc-v1", "response": "地理", "done": False},
            {"model": "physoc-v1", "response": "位置", "done": True},
        )
    )
    midpoint = len(sequence) // 2
    lines = []
    for index, fragment in enumerate((sequence[:midpoint], sequence[midpoint:])):
        outer = {"model": "physoc-v1", "response": fragment, "done": index == 1}
        lines.extend([f"data: {json.dumps(outer, ensure_ascii=False)}\n", "\n"])

    self.assertEqual(
        collect_physoc_response(lines, expected_model="physoc-v1"),
        "地理位置",
    )
```

- [ ] **Step 2: 增加普通 JSON 正文保留测试**

```python
def test_collect_preserves_non_physoc_json_in_plain_text(self) -> None:
    answer = '接口示例：{"status":"ok","value":1}'
    outer = {"model": "physoc-v1", "response": answer, "done": True}

    self.assertEqual(
        collect_physoc_response(
            [f"data: {json.dumps(outer, ensure_ascii=False)}\n", "\n"],
            expected_model="physoc-v1",
        ),
        answer,
    )
```

- [ ] **Step 3: 增加非法终止顺序测试**

```python
def test_collect_rejects_fragmented_sequence_without_final_done(self) -> None:
    nested = json.dumps(
        {"model": "physoc-v1", "response": "未完成", "done": False},
        separators=(",", ":"),
    )
    midpoint = len(nested) // 2
    lines: list[str] = []
    for index, fragment in enumerate((nested[:midpoint], nested[midpoint:])):
        outer = {"model": "physoc-v1", "response": fragment, "done": index == 1}
        lines.extend([f"data: {json.dumps(outer)}\n", "\n"])

    with self.assertRaisesRegex(PhysocStreamError, "before completion"):
        collect_physoc_response(
            lines,
            expected_model="physoc-v1",
        )
```

- [ ] **Step 4: 运行新测试并确认失败或覆盖缺口**

Run:

```bash
cd backend
uv run --project . python -m pytest \
  tests/test_physoc_sse.py::PhysocSseTests::test_collect_joins_fragmented_physoc_event_sequence \
  tests/test_physoc_sse.py::PhysocSseTests::test_collect_preserves_non_physoc_json_in_plain_text \
  tests/test_physoc_sse.py::PhysocSseTests::test_collect_rejects_fragmented_sequence_without_final_done \
  -q
```

Expected: 连续序列测试在实现前 FAIL；实现后全部 PASS。

- [ ] **Step 5: 运行完整协议与 Provider 测试**

Run:

```bash
cd backend
uv run --project . python -m pytest tests/test_physoc_sse.py tests/test_llm_provider.py -q
```

Expected: 全部 PASS。

### Task 4: 升级后端版本并完成验证

**Files:**
- Modify: `backend/app/__init__.py`
- Refresh if changed: `backend/uv.lock`

- [ ] **Step 1: 独立提升后端 patch 版本**

Run:

```bash
uv run --project backend python tools/bump_version.py backend patch
uv lock --project backend
```

Expected: `backend version: 0.1.10 -> 0.1.11`。用户端与管理端版本保持 `0.1.1`。

- [ ] **Step 2: 运行版本契约与后端完整测试**

Run:

```bash
cd backend
uv run --project . python -m pytest ../tools/tests/test_version_contract.py -q
uv run --project . python -m pytest tests -q
```

Expected: 版本契约和后端测试全部通过，允许仓库既有 skip 与弃用警告。

- [ ] **Step 3: 运行静态检查和差异检查**

Run:

```bash
cd ..
fast lint --ty
git diff --check
```

Expected: Ruff、ty 和空白检查全部通过。

- [ ] **Step 4: 检查最终范围**

Run:

```bash
git status --short
git diff -- backend/app/physoc_sse.py backend/app/__init__.py backend/tests/test_physoc_sse.py backend/tests/test_llm_provider.py backend/uv.lock
```

Expected: 只包含 Physoc 分片事件解析、回归测试和后端版本变更；前端版本文件无变化。
