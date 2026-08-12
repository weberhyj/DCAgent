# Unified Knowledge Routing final-review fix wave

Date: 2026-08-12
Base: `d0975d4`
Scope: overlong exact Word-fact entities reaching route metadata persistence

This final wave also covers preserving Excel multi-metric route context during
catalog outages, bounding explicit multi-metric route fields, and handling
Word-fact repository outages without API 500/fallback.

## Root cause

The exact Word-fact grammar accepted an entity of any length. A 301-character
entity followed by `鍑犲瞾` therefore produced `WordFactualIntent`, and
`WordFactAnswerService` copied that entity into `KnowledgeRouteMetadata`.
Metadata validation happened later in `KnowledgeRouteMetadata.to_dict()` while
`SqlChatRepository.send_message()` persisted the agent audit. The 256-character
route scalar contract then raised `ValueError`, turning the API request into a
500. The same intent could also perform an unnecessary fact lookup before the
failure.

## Fix

Added `app/bounded_limits.py` as a lower-level shared constants module and made
the route metadata model and Word-fact resolver use the same
`MAX_ROUTE_METADATA_STRING_LENGTH` value. At the exact grammar boundary,
entities above that limit now become a bounded `WordFactClarification` with the
target field preserved, no entity copied into metadata, and no repository
lookup. This is intentionally a terminal clarification: truncation could query
a different entity, while returning `None` would allow document/LLM fallback for
a request already recognized as malformed factual syntax.

Normal exact facts, multi-field/entity clarification, open document/RAG routes,
and exact-route terminal semantics remain unchanged.

Excel catalog outages now parse a warm catalog snapshot before returning the
terminal unavailable result, preserving dataset/source/fields and the parsed
single-vs-multi route. Explicit metric lists above the shared 32-item route list
limit are rejected as a deterministic bounded multi-route clarification;
implicit summaries retain their existing configured cap. Word-fact repository
exceptions now produce terminal `WORD_FACTUAL` results with
`fact_repository_unavailable` and validation false, preventing document or LLM
fallback.

## Regression coverage

- Resolver: 256-character entity remains accepted; 257-character entity is a
  clarification with bounded metadata fields.
- WordFactAnswerService: oversized exact entity returns `CLARIFICATION`, does
  not call the fact repository, and serializes route metadata safely.
- SQL repository: `send_message()` persists and reloads the bounded terminal
  route audit without raising.
- API: oversized exact entity returns HTTP 200 with bounded clarification text;
  a failing LLM provider and retrieval hook prove there is no fallback/retrieval
  path; persisted audit round-trips with `entity=None` and `target_fields=("骞撮緞",)`.

## Verification

Additional regression coverage verifies warm multi-route catalog outages,
explicit 40-metric bounding, and terminal Word-fact repository outages.

Focused routing/Word/SQL/API gate:

```
302 passed, 1 warning, 175 subtests passed
```

New-finding regression gate:

```
15 passed, 4 subtests passed
```

Strict RED captures before implementation recorded four overlong-entity
failures (intent leak, factual service route, SQL `ValueError`, API 500) and
three new-finding failures (lost outage context, explicit 40-field overflow
path, propagated fact repository error).

Backend complete gate (four pre-existing/unrelated failures deselected):

```
1369 passed, 18 skipped, 4 deselected, 1 warning, 1468 subtests passed
```

The complete undeselected backend run also reached 1365 passing and 18 skipped,
with four unrelated failures in legacy ClickHouse acceptance message fixtures,
the existing dependency upper-bound contract, and a structured-worker preflight
fixture. Those files and behaviors were not changed by this fix wave.

Additional checks:

- `python -m compileall -q app tests`: passed.
- `git diff --check`: passed.
- Ruff is unavailable in this environment; no Ruff result is claimed.

## Changed files

- `backend/app/bounded_limits.py`
- `backend/app/knowledge_route_models.py`
- `backend/app/word_facts.py`
- `backend/tests/test_word_facts.py`
- `backend/tests/test_word_fact_answer.py`
- `backend/tests/test_sql_repository.py`
- `backend/tests/test_api_contract.py`
