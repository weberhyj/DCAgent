# RAG Answer Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the backend answer pipeline retrieve company knowledge, answer from the retrieved material, cite evidence, and refuse unsupported answers.

**Architecture:** Retrieval remains in `backend/app/repository.py` and `backend/app/sql_repository.py`. Answer policy and OpenAI-compatible prompt construction live in `backend/app/llm.py`. Existing API schemas keep hiding raw knowledge source internals from the user frontend.

**Tech Stack:** Python 3.14, FastAPI, unittest, httpx, existing OpenAI-compatible Chat Completions provider.

---

### Task 1: Guarded Prompt And No-Evidence Policy

**Files:**
- Modify: `backend/tests/test_llm_provider.py`
- Modify: `backend/app/llm.py`

- [ ] Add tests that import `NO_EVIDENCE_REPLY`, `RAG_SYSTEM_PROMPT`, `build_prompt`, and `build_knowledge_context`.
- [ ] Verify `build_knowledge_context([hit])` includes `[1]`, source name, classification, rank, score, and chunk text.
- [ ] Verify `build_prompt()` includes the user question, mode, strict source-grounding rules, numbered evidence, and recent history.
- [ ] Verify `TemplateLLMProvider.generate_reply()` returns `NO_EVIDENCE_REPLY` with no citations when `knowledge_hits=[]`.
- [ ] Run `cd backend; py -m unittest tests.test_llm_provider.LLMProviderTest -v` and confirm the new tests fail because the helpers and guardrail behavior do not exist.
- [ ] Implement constants and helpers in `backend/app/llm.py`.
- [ ] Update `TemplateLLMProvider` to return the deterministic no-evidence reply before building seed artifacts when no hits exist.
- [ ] Re-run `cd backend; py -m unittest tests.test_llm_provider.LLMProviderTest -v` and confirm the tests pass.

### Task 2: OpenAI-Compatible Provider Contract

**Files:**
- Modify: `backend/tests/test_llm_provider.py`
- Modify: `backend/app/llm.py`

- [ ] Add a fake httpx client test for `OpenAICompatibleLLMProvider.generate_reply()`.
- [ ] Assert the request payload contains `model`, `temperature`, a system message equal to `RAG_SYSTEM_PROMPT`, and a user message containing numbered evidence.
- [ ] Assert the returned assistant paragraph attaches citations built from the supplied knowledge hits.
- [ ] Add a test that `OpenAICompatibleLLMProvider.generate_reply()` with no hits returns `NO_EVIDENCE_REPLY` and does not instantiate `httpx.Client`.
- [ ] Run `cd backend; py -m unittest tests.test_llm_provider.LLMProviderTest -v` and confirm the new tests fail.
- [ ] Update `OpenAICompatibleLLMProvider` to use the shared prompt helpers and the no-evidence guard.
- [ ] Re-run `cd backend; py -m unittest tests.test_llm_provider.LLMProviderTest -v` and confirm the tests pass.

### Task 3: Retrieval Limit And Repository Handoff

**Files:**
- Modify: `backend/tests/test_llm_provider.py`
- Modify: `backend/app/repository.py`

- [ ] Add a repository test that indexes six matching knowledge chunks and asserts the injected LLM provider receives five ranked hits.
- [ ] Run `cd backend; py -m unittest tests.test_llm_provider.LLMProviderTest.test_repository_limits_rag_context_to_five_ranked_hits -v` and confirm it fails because only two hits are passed.
- [ ] Change `KNOWLEDGE_SEARCH_LIMIT` from `2` to `5`.
- [ ] Re-run the focused test and confirm it passes.

### Task 4: API Contract Verification

**Files:**
- Modify only if tests expose a regression.

- [ ] Run `cd backend; py -m unittest discover -s tests -v`.
- [ ] Run `cd frontend; npm.cmd run test:run -- --pool=forks --testTimeout=10000 --hookTimeout=10000 --reporter=verbose`.
- [ ] Run `cd admin-frontend; npm.cmd run test:run -- --pool=forks --testTimeout=10000 --hookTimeout=10000 --reporter=verbose`.
- [ ] Run `cd frontend; npm.cmd run build`.
- [ ] Run `cd admin-frontend; npm.cmd run build`.

## Self-Review

- Spec coverage: Tasks cover prompt policy, model provider behavior, retrieval limit, and API contract verification.
- Placeholder scan: No TBD or TODO placeholders remain.
- Type consistency: Tests and implementation use existing `LLMRequest`, `KnowledgeSearchHitModel`, and `ChatMessageModel` types.
- Git constraint: The workspace is not a git repository, so verification checkpoints replace commits.
