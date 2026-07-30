# Greeting Intent Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pure greeting messages return a deterministic DCAgent welcome response without touching structured data, retrieval, or the LLM.

**Architecture:** Add a small exact-match greeting classifier and Agent result builder to `backend/app/agent.py`. Both repository implementations call the Agent greeting entry before structured-query routing; non-greeting messages retain the existing structured-query-first and RAG behavior.

**Tech Stack:** Python 3.12, FastAPI backend domain models, LangGraph Agent, `unittest`, uv, Ruff

---

## File Structure

- Modify `backend/app/agent.py`: own greeting normalization, exact-match classification, fixed reply construction, and greeting Agent audit result.
- Modify `backend/app/repository.py`: route greetings before the in-memory structured service and retrieval Agent.
- Modify `backend/app/sql_repository.py`: route greetings before the SQL structured service and retrieval Agent.
- Modify `backend/tests/test_agent.py`: unit-test supported greetings, punctuation normalization, dependency bypass, audit metadata, and mixed greeting/question fallthrough.
- Modify `backend/tests/test_sql_repository.py`: verify both repository implementations persist the welcome reply and bypass structured, retrieval, and LLM dependencies.

### Task 1: Add the deterministic greeting Agent behavior

**Files:**
- Modify: `backend/tests/test_agent.py`
- Modify: `backend/app/agent.py`

- [ ] **Step 1: Write failing Agent tests**

Add `GREETING_REPLY` and `is_greeting_message` to the import from `app.agent`, then add these tests to `AgentTest`:

```python
def test_pure_greeting_builds_welcome_run_without_external_dependencies(self) -> None:
    class NeverProvider:
        def generate_reply(self, request: LLMRequest) -> ChatMessageModel:
            raise AssertionError(f"greeting must not call LLM: {request}")

    agent = ReadOnlyKnowledgeAgent(
        tools=KnowledgeAgentTools(
            search_knowledge=lambda query, limit, routing_key: self.fail(
                f"greeting must not search: {query}, {limit}, {routing_key}"
            ),
            inspect_document=lambda source_id: self.fail(
                f"greeting must not inspect: {source_id}"
            ),
        ),
        llm_provider=NeverProvider(),
    )

    result = agent.try_answer_greeting(
        conversation_id="conv-greeting",
        content="  您好！ ",
        mode="quick",
    )

    self.assertIsNotNone(result)
    assert result is not None
    self.assertEqual(result.reply.paragraphs[0].text, GREETING_REPLY)
    self.assertEqual(result.evidence_count, 0)
    self.assertEqual(result.source_count, 0)
    self.assertEqual([step.tool_name for step in result.steps], ["respond_greeting"])
    self.assertTrue(result.steps[0].read_only)

def test_supported_greeting_phrases_are_recognized(self) -> None:
    for content in (
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "在吗",
        "你在吗",
        "你是谁",
        "介绍一下你自己",
    ):
        with self.subTest(content=content):
            self.assertTrue(is_greeting_message(content))

def test_common_unicode_greeting_punctuation_is_ignored(self) -> None:
    for content in (
        "您好…",
        "您好～～",
        "您好；",
        "您好：",
        "“您好”",
        "您好......",
    ):
        with self.subTest(content=content):
            self.assertTrue(is_greeting_message(content))

def test_greeting_with_substantive_question_falls_through(self) -> None:
    agent = ReadOnlyKnowledgeAgent(
        tools=KnowledgeAgentTools(
            search_knowledge=lambda query, limit, routing_key: [],
            inspect_document=lambda source_id: [],
        ),
        llm_provider=RecordingProvider(),
    )

    result = agent.try_answer_greeting(
        conversation_id="conv-greeting",
        content="“你好”，请问报销制度是什么",
        mode="quick",
    )

    self.assertIsNone(result)
```

- [ ] **Step 2: Run the tests and verify RED**

Run from the repository root:

```powershell
uv run --directory backend python -m unittest `
  tests.test_agent.AgentTest.test_pure_greeting_builds_welcome_run_without_external_dependencies `
  tests.test_agent.AgentTest.test_supported_greeting_phrases_are_recognized `
  tests.test_agent.AgentTest.test_greeting_with_substantive_question_falls_through -v
```

Expected: FAIL because `GREETING_REPLY`, `is_greeting_message`, and `try_answer_greeting` do not exist.

- [ ] **Step 3: Implement the minimal Agent greeting classifier and result**

In `backend/app/agent.py`, import `unicodedata` and `ResponseParagraphModel`, then add:

```python
GREETING_REPLY = (
    "您好，我是 DCAgent 企业知识库智能助手。您可以向我询问 Word、Excel、PDF "
    "等知识库资料中的内容，我会检索相关依据并为您汇总回答。"
)
_GREETING_MESSAGES = frozenset(
    {
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "在吗",
        "你在吗",
        "你是谁",
        "介绍一下你自己",
    }
)
def is_greeting_message(content: str) -> bool:
    normalized = "".join(
        character
        for character in content
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
        and character not in {"~", "～"}
    ).casefold()
    return normalized in {item.casefold() for item in _GREETING_MESSAGES}
```

This normalization ignores whitespace, all Unicode punctuation, and the common chat
symbols `~` / `～`. Other symbols and emoji remain significant, so matching still requires
the entire normalized input to equal a greeting phrase.

Add this public method to `ReadOnlyKnowledgeAgent` before `run()`:

```python
def try_answer_greeting(
    self,
    conversation_id: str,
    content: str,
    mode: ComposerMode,
) -> AgentRunResult | None:
    clean_content = content.strip()
    if not is_greeting_message(clean_content):
        return None

    timestamp = now_label()
    reply = ChatMessageModel(
        id=f"msg-{uuid4().hex[:8]}",
        role="assistant",
        time=timestamp,
        paragraphs=[ResponseParagraphModel(text=GREETING_REPLY)],
    )
    step = AgentStep(
        id=f"step-{uuid4().hex[:12]}",
        step_index=0,
        tool_name="respond_greeting",
        status="completed",
        input_summary=clean_content,
        output_summary="已返回固定欢迎词",
        started_at=timestamp,
        completed_at=timestamp,
        read_only=True,
    )
    return AgentRunResult(
        id=f"agent-{uuid4().hex[:12]}",
        conversation_id=conversation_id,
        query=clean_content,
        mode=mode,
        status="completed",
        started_at=timestamp,
        completed_at=timestamp,
        reply=reply,
        steps=[step],
        evidence_count=0,
        source_count=0,
    )
```

- [ ] **Step 4: Run the Agent tests and verify GREEN**

```powershell
uv run --directory backend python -m unittest tests.test_agent -v
```

Expected: all tests in `tests.test_agent` PASS.

- [ ] **Step 5: Run focused Ruff checks**

```powershell
uvx ruff check backend/app/agent.py backend/tests/test_agent.py
uvx ruff format --check backend/app/agent.py backend/tests/test_agent.py
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit the Agent behavior**

```powershell
git add backend/app/agent.py backend/tests/test_agent.py
git commit -m "feat: add deterministic greeting response"
```

### Task 2: Route greetings before structured and retrieval dependencies

**Files:**
- Modify: `backend/tests/test_sql_repository.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`

- [ ] **Step 1: Write the failing repository contract test**

Import `GREETING_REPLY` from `app.agent` in `backend/tests/test_sql_repository.py`, then add this test to `SqlRepositoryTest`:

```python
def test_repositories_route_greeting_before_structured_retrieval_and_llm(self) -> None:
    class NeverStructuredService:
        def try_answer(self, **kwargs):
            raise AssertionError(f"greeting must not access structured service: {kwargs}")

    class NeverRouter:
        def search(self, request):
            raise AssertionError(f"greeting must not access retrieval router: {request}")

    class NeverProvider:
        def generate_reply(self, request):
            raise AssertionError(f"greeting must not call LLM: {request}")

    scope = RetrievalScope("default", ("internal",), "v1")
    repositories = (
        SqlChatRepository(
            self.database,
            llm_provider=NeverProvider(),
            structured_service=NeverStructuredService(),
            retrieval_router=NeverRouter(),
            retrieval_scope=scope,
        ),
        InMemoryChatRepository(
            build_seed_state(),
            llm_provider=NeverProvider(),
            structured_service=NeverStructuredService(),
            retrieval_router=NeverRouter(),
            retrieval_scope=scope,
        ),
    )

    for repository in repositories:
        with self.subTest(repository=type(repository).__name__):
            _, conversation_id, _ = repository.create_conversation()
            _, _, messages = repository.send_message(conversation_id, "你好", "quick")

            self.assertEqual(messages[-1].paragraphs[0].text, GREETING_REPLY)
            run = repository.list_agent_runs(limit=1)[0]
            self.assertEqual(run.evidence_count, 0)
            self.assertEqual(run.source_count, 0)
            self.assertEqual([step.tool_name for step in run.steps], ["respond_greeting"])
```

- [ ] **Step 2: Run the repository test and verify RED**

```powershell
uv run --directory backend python -m unittest `
  tests.test_sql_repository.SqlRepositoryTest.test_repositories_route_greeting_before_structured_retrieval_and_llm -v
```

Expected: FAIL because both repositories call the structured service before checking greeting intent.

- [ ] **Step 3: Reorder the in-memory repository message route**

Replace the `agent_result` initialization in `InMemoryChatRepository.send_message()` with:

```python
agent_result = self._agent.try_answer_greeting(
    conversation_id=conversation_id,
    content=clean_content,
    mode=mode,
)
if agent_result is None and self._structured_service is not None:
    agent_result = self._structured_service.try_answer(
        conversation_id=conversation_id,
        content=clean_content,
        mode=mode,
        previous_messages=previous_messages,
    )
if agent_result is None:
    agent_result = self._agent.run(
        conversation_id=conversation_id,
        content=clean_content,
        mode=mode,
        previous_messages=previous_messages,
    )
```

- [ ] **Step 4: Reorder the SQL repository message route**

Apply the same routing sequence inside `SqlChatRepository.send_message()`:

```python
agent_result = self._agent.try_answer_greeting(
    conversation_id=conversation_id,
    content=clean_content,
    mode=mode,
)
if agent_result is None and self._structured_service is not None:
    agent_result = self._structured_service.try_answer(
        conversation_id=conversation_id,
        content=clean_content,
        mode=mode,
        previous_messages=previous_messages,
    )
if agent_result is None:
    agent_result = self._agent.run(
        conversation_id=conversation_id,
        content=clean_content,
        mode=mode,
        previous_messages=previous_messages,
    )
```

- [ ] **Step 5: Run repository and Agent regression tests**

```powershell
uv run --directory backend python -m unittest tests.test_agent tests.test_sql_repository -v
```

Expected: all tests PASS, including the existing structured-before-retrieval test.

- [ ] **Step 6: Run focused Ruff checks**

```powershell
uvx ruff check `
  backend/app/agent.py `
  backend/app/repository.py `
  backend/app/sql_repository.py `
  backend/tests/test_agent.py `
  backend/tests/test_sql_repository.py
uvx ruff format --check `
  backend/app/agent.py `
  backend/app/repository.py `
  backend/app/sql_repository.py `
  backend/tests/test_agent.py `
  backend/tests/test_sql_repository.py
```

Expected: both commands exit 0.

- [ ] **Step 7: Commit repository routing**

```powershell
git add `
  backend/app/repository.py `
  backend/app/sql_repository.py `
  backend/tests/test_sql_repository.py
git commit -m "feat: route greetings before knowledge dependencies"
```

### Task 3: Run complete backend verification

**Files:**
- Verify only; no production files added.

- [ ] **Step 1: Run the complete backend test suite**

```powershell
uv run --directory backend python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: exit 0 with no failed or errored tests.

- [ ] **Step 2: Run complete backend and tools Ruff checks**

```powershell
uvx ruff check backend tools
uvx ruff format --check backend tools
```

Expected: both commands exit 0.

- [ ] **Step 3: Verify the final diff and repository state**

```powershell
git diff --check
git status --short --branch
git log -3 --oneline
```

Expected: no whitespace errors; only the implementation-plan tracking state may remain uncommitted.

- [ ] **Step 4: Commit plan tracking updates if checkboxes were updated**

```powershell
git add docs/superpowers/plans/2026-07-31-greeting-intent-route.md
git commit -m "docs: record greeting route implementation"
```

Skip this commit only if the plan file was not changed during execution.
