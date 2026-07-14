# Plain-Text Answer Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure new OpenAI-compatible assistant answers reach the user frontend as safe, readable plain text without the confirmed Markdown list, bold, or inline-citation markers.

**Architecture:** Keep Vue text interpolation unchanged. Add a conservative, deterministic normalizer in `backend/app/answer_text.py`, call it at the external-model boundary in `backend/app/llm.py`, and strengthen both prompt layers so the model is less likely to emit markup. Protect the behavior with focused `unittest` coverage before running the existing backend and frontend regression suites.

**Tech Stack:** Python 3, `re`, `unittest`, httpx mocks, FastAPI models, Vue 3, Vitest, Vite.

---

## Execution Context

Before Task 1, use the `using-git-worktrees` skill to create `D:\project\DC-Agent\.worktrees\plain-text-answer-formatting` on branch `codex/fix-plain-text-answer-formatting` from the latest `main` commit. Run every task inside that worktree.

The primary workspace currently contains an unrelated uncommitted title change in `frontend/index.html`. Do not copy, modify, stage, or commit that change as part of this plan.

## File Structure

- Create: `backend/tests/test_answer_text.py` — focused unit contract for conservative plain-text normalization and preservation cases.
- Modify: `backend/app/answer_text.py` — own the deterministic answer-boundary normalization helper while retaining citation cleanup.
- Modify: `backend/tests/test_llm_provider.py` — protect plain-text prompt instructions and provider integration.
- Modify: `backend/app/llm.py` — strengthen prompt rules and normalize external model content before constructing `ResponseParagraphModel`.
- Verify only: `frontend/src/components/chat/ChatTranscript.vue` and `frontend/src/components/chat/__tests__/ChatTranscript.spec.ts` — safe Vue interpolation remains unchanged.

No new dependency is required. Do not modify `backend/app/schemas.py`; its existing citation-only projection remains compatible because new external-model answers are normalized before they enter the message model.

### Task 1: Add the Conservative Answer Normalizer

**Files:**

- Create: `backend/tests/test_answer_text.py`
- Modify: `backend/app/answer_text.py:6-12`
- Test: `backend/tests/test_answer_text.py`

- [ ] **Step 1: Write the failing normalization tests**

Create `backend/tests/test_answer_text.py` with this complete content:

```python
from __future__ import annotations

import unittest

from app.answer_text import normalize_plain_text_answer


class PlainTextAnswerNormalizationTest(unittest.TestCase):
    def test_removes_confirmed_list_bold_and_inline_citation_pattern(self) -> None:
        source = (
            "- **数联**：数据要素联通\n"
            "- **智联**：智能与算力连接\n"
            "- **光联**：城市光网支撑。[1]"
        )

        self.assertEqual(
            normalize_plain_text_answer(source),
            "数联：数据要素联通\n智联：智能与算力连接\n光联：城市光网支撑。",
        )

    def test_preserves_plain_text_and_ambiguous_symbols(self) -> None:
        preserved_values = (
            "城市一张网 2.0",
            "2 * 3",
            "user_name",
            "__init__",
            "- 5°C",
        )

        for value in preserved_values:
            with self.subTest(value=value):
                self.assertEqual(normalize_plain_text_answer(value), value)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify the expected red state**

Run from `D:\project\DC-Agent\.worktrees\plain-text-answer-formatting\backend`:

```powershell
py -m unittest tests.test_answer_text -v
```

Expected: FAIL during import because `app.answer_text` does not yet export `normalize_plain_text_answer`. A failure caused by a different import or environment problem must be fixed before continuing.

- [ ] **Step 3: Implement the minimum focused normalizer**

Replace `backend/app/answer_text.py` with:

```python
from __future__ import annotations

import re


_INLINE_CITATION_MARKER = re.compile(r"[ \t]*\[(?:[1-9]\d*)\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([，。；：！？、,.!?;:])")
_FORMATTED_LIST_PREFIX = re.compile(
    r"(?m)^[ \t]*[-+*][ \t]+(?=\*\*\S(?:[^\r\n]*?\S)?\*\*)"
)
_BOLD_MARKER = re.compile(r"\*\*(\S(?:[^\r\n]*?\S)?)\*\*")


def remove_inline_citation_markers(text: str) -> str:
    cleaned = _INLINE_CITATION_MARKER.sub("", text)
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned).strip()


def normalize_plain_text_answer(text: str) -> str:
    cleaned = _FORMATTED_LIST_PREFIX.sub("", text)
    cleaned = _BOLD_MARKER.sub(r"\1", cleaned)
    return remove_inline_citation_markers(cleaned)
```

The list-prefix expression intentionally requires a following `**...**` span. Do not broaden it to every line beginning with `- `, because that would corrupt values such as `- 5°C`. Do not add underscore, heading, link, code-fence, or HTML stripping without a separate failing example and preservation test.

- [ ] **Step 4: Run the focused tests and verify the green state**

Run:

```powershell
py -m unittest tests.test_answer_text -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit the normalization unit**

Run from the worktree root:

```powershell
git add backend/app/answer_text.py backend/tests/test_answer_text.py
git commit -m "fix: normalize plain-text answers"
```

Expected: one commit containing only the normalizer and its unit tests.

### Task 2: Require Plain Text in Both Prompt Layers

**Files:**

- Modify: `backend/tests/test_llm_provider.py:139-162`
- Modify: `backend/app/llm.py:23-29`
- Modify: `backend/app/llm.py:188-203`
- Test: `backend/tests/test_llm_provider.py`

- [ ] **Step 1: Add a failing prompt-contract test**

Inside `LLMProviderTest`, immediately after `test_build_prompt_includes_guardrails_evidence_and_recent_history`, add:

```python
    def test_system_and_user_prompts_require_plain_text_without_markup(self) -> None:
        prompt = build_prompt(
            LLMRequest(
                content="请说明三类连接能力",
                mode="source",
                knowledge_hits=[indexed_hit()],
                previous_messages=[],
            )
        )

        for prompt_layer in (RAG_SYSTEM_PROMPT, prompt):
            with self.subTest(prompt_layer=prompt_layer):
                self.assertIn("纯文本", prompt_layer)
                self.assertIn("Markdown", prompt_layer)
                self.assertIn("HTML", prompt_layer)
                self.assertIn("列表符号", prompt_layer)
                self.assertIn("加粗", prompt_layer)
```

- [ ] **Step 2: Run the prompt test and verify it fails for the missing contract**

Run from `D:\project\DC-Agent\.worktrees\plain-text-answer-formatting\backend`:

```powershell
py -m unittest tests.test_llm_provider.LLMProviderTest.test_system_and_user_prompts_require_plain_text_without_markup -v
```

Expected: FAIL because neither prompt layer currently contains the plain-text and markup restrictions.

- [ ] **Step 3: Add the exact system and user prompt rules**

In `RAG_SYSTEM_PROMPT`, add this sentence after `回答要简洁、审慎、面向业务使用。`:

```python
    "回答必须使用纯文本，不要使用 Markdown 或 HTML，不要输出标题、列表符号、加粗、斜体、代码围栏或链接语法。"
```

In `build_prompt()`, add this line immediately before the existing inline-citation rule:

```python
        "- 只输出纯文本，不要使用 Markdown 或 HTML，不要输出标题、列表符号、加粗、斜体、代码围栏或链接语法。\n"
```

Keep the existing source-grounding, no-evidence, citation, evidence, Agent-context, and history rules unchanged.

- [ ] **Step 4: Run the prompt and provider module tests**

Run:

```powershell
py -m unittest tests.test_llm_provider.LLMProviderTest.test_system_and_user_prompts_require_plain_text_without_markup -v
py -m unittest tests.test_llm_provider -v
```

Expected: the new prompt test and all existing LLM provider tests PASS.

- [ ] **Step 5: Commit the prompt contract**

Run from the worktree root:

```powershell
git add backend/app/llm.py backend/tests/test_llm_provider.py
git commit -m "fix: require plain-text model replies"
```

Expected: one commit containing only prompt-policy changes and their test.

### Task 3: Normalize External Model Output Before Building the Reply

**Files:**

- Modify: `backend/tests/test_llm_provider.py:97-125`
- Modify: `backend/tests/test_llm_provider.py:221-254`
- Modify: `backend/app/llm.py:11`
- Modify: `backend/app/llm.py:102-104`
- Test: `backend/tests/test_llm_provider.py`

- [ ] **Step 1: Make the fake response configurable**

Replace the existing `FakeLLMResponse` and `RecordingHttpClient` definitions in `backend/tests/test_llm_provider.py` with:

```python
class FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": self.content,
                    }
                }
            ]
        }


class RecordingHttpClient:
    def __init__(
        self,
        response_content: str = "根据已检索资料，现金流风险与回款周期相关。[1]",
    ) -> None:
        self.requests: list[dict] = []
        self.response_content = response_content

    def __enter__(self) -> "RecordingHttpClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def post(self, url: str, json: dict, headers: dict) -> FakeLLMResponse:
        self.requests.append({"url": url, "json": json, "headers": headers})
        return FakeLLMResponse(self.response_content)
```

- [ ] **Step 2: Add the failing provider-boundary test**

Inside `LLMProviderTest`, immediately after `test_openai_provider_sends_guarded_rag_payload_and_attaches_citations`, add:

```python
    def test_openai_provider_normalizes_formatted_answer_before_returning(self) -> None:
        client = RecordingHttpClient(
            response_content=(
                "- **数联**：数据要素联通\n"
                "- **智联**：智能与算力连接\n"
                "- **光联**：城市光网支撑。[1]"
            )
        )
        provider = OpenAICompatibleLLMProvider(
            api_base="https://llm.example.test/v1",
            api_key="test-key",
            model="dc-agent-test-model",
        )

        with patch("app.llm.httpx.Client", return_value=client):
            reply = provider.generate_reply(
                LLMRequest(
                    content="请说明三类连接能力",
                    mode="source",
                    knowledge_hits=[indexed_hit()],
                    previous_messages=[],
                )
            )

        self.assertEqual(
            reply.paragraphs[0].text,
            "数联：数据要素联通\n智联：智能与算力连接\n光联：城市光网支撑。",
        )
        self.assertEqual(reply.paragraphs[0].citations[0].source_id, "kb-llm")
```

- [ ] **Step 3: Run the provider-boundary test and verify the expected red state**

Run from `D:\project\DC-Agent\.worktrees\plain-text-answer-formatting\backend`:

```powershell
py -m unittest tests.test_llm_provider.LLMProviderTest.test_openai_provider_normalizes_formatted_answer_before_returning -v
```

Expected: FAIL because the current provider removes `[1]` but still returns the list and `**` markers.

- [ ] **Step 4: Apply the normalizer at the external-model boundary**

Replace the `answer_text` import in `backend/app/llm.py` with:

```python
from .answer_text import normalize_plain_text_answer
```

Replace the model-content cleanup at the end of the HTTP response block with:

```python
            content = normalize_plain_text_answer(
                str(data["choices"][0]["message"]["content"])
            )
```

Do not normalize `request.content`, evidence text, citation metadata, or template-provider evidence. This helper is the boundary for untrusted external model formatting only.

- [ ] **Step 5: Run focused and related backend tests**

Run:

```powershell
py -m unittest tests.test_answer_text tests.test_llm_provider tests.test_api_contract -v
```

Expected: all normalization, provider, prompt, citation-hiding, and API contract tests PASS.

- [ ] **Step 6: Commit provider integration**

Run from the worktree root:

```powershell
git add backend/app/llm.py backend/tests/test_llm_provider.py
git commit -m "fix: sanitize external model formatting"
```

Expected: one commit containing only provider-boundary integration and its regression test.

### Task 4: Run Full Regression and Verify Scope

**Files:**

- Verify: `backend/tests`
- Verify: `frontend/src/components/chat/__tests__/ChatTranscript.spec.ts`
- Verify: `frontend` production build
- Verify: `admin-frontend` test suite and production build
- Verify: repository status and commit history

- [ ] **Step 1: Run the complete backend suite**

Run from `D:\project\DC-Agent\.worktrees\plain-text-answer-formatting\backend`:

```powershell
py -m unittest discover -s tests -p "test_*.py" -v
```

Expected: 0 failures and 0 errors.

- [ ] **Step 2: Verify the unchanged user transcript contract and user build**

Run from `D:\project\DC-Agent\.worktrees\plain-text-answer-formatting\frontend`:

```powershell
npm.cmd run test:run -- src/components/chat/__tests__/ChatTranscript.spec.ts
npm.cmd run test:run
npm.cmd run build
```

Expected: the focused transcript test, full user frontend suite, TypeScript check, and Vite production build all exit with code 0. Inspect `ChatTranscript.vue` and confirm assistant text is still rendered with `{{ paragraph.text }}` and no `v-html` was introduced.

- [ ] **Step 3: Run the management frontend regression**

Run from `D:\project\DC-Agent\.worktrees\plain-text-answer-formatting\admin-frontend`:

```powershell
npm.cmd run test:run
npm.cmd run build
```

Expected: all management frontend tests pass and the production build exits with code 0.

- [ ] **Step 4: Check whitespace, file scope, and commits**

Run from the isolated worktree root:

```powershell
git diff --check main...HEAD
git status --short --branch
git log --oneline main..HEAD
```

Expected:

- `git diff --check` produces no output.
- The worktree is clean.
- The branch contains the three focused commits from Tasks 1-3.
- The diff contains only `backend/app/answer_text.py`, `backend/app/llm.py`, `backend/tests/test_answer_text.py`, and `backend/tests/test_llm_provider.py`.
- No frontend file, especially `frontend/index.html`, is included.

## Completion Criteria

- The exact three-line `- **...**` answer reproducer becomes readable plain text with line order preserved.
- `[1]`-style markers remain absent from the displayed answer while citation metadata remains attached internally.
- `城市一张网 2.0`, `2 * 3`, `user_name`, `__init__`, and `- 5°C` remain unchanged.
- Both prompt layers explicitly require plain text and forbid Markdown and HTML formatting.
- The external provider applies normalization before constructing `ResponseParagraphModel`.
- `ChatTranscript.vue` remains on safe Vue interpolation and no Markdown renderer or `v-html` is added.
- Backend, user frontend, and management frontend regressions and production builds pass.
- The isolated branch is clean and contains no unrelated workspace changes.
