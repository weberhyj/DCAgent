# RAG Answer Core Design

## Goal

Make DC-Agent behave like a company knowledge search system: retrieve indexed company material, answer from that material, cite the material inside assistant paragraphs, and avoid unsupported answers when the knowledge base has no relevant evidence.

## Scope

This phase changes the backend answer pipeline only. The user frontend and admin frontend keep their current API contracts and visual surfaces.

In scope:
- Stronger RAG prompt construction.
- More knowledge hits passed into the answer pipeline.
- Deterministic no-evidence reply.
- Stronger OpenAI-compatible provider request payload.
- Tests that protect the API contract from exposing admin-only source details to the user frontend.

Out of scope:
- Model configuration UI.
- pgvector or external embedding migration.
- Frontend visual changes.
- Authentication and route permissions.

## Architecture

`repository.py` and `sql_repository.py` keep owning retrieval. They pass ranked `KnowledgeSearchHitModel` items into the configured `LLMProvider`.

`llm.py` owns answer generation policy:
- `TemplateLLMProvider` remains the local fallback.
- `OpenAICompatibleLLMProvider` calls a Chat Completions compatible endpoint.
- Shared helper functions format numbered evidence, strict prompt instructions, citations, and no-evidence replies.

`schemas.py` continues to hide citations and artifact source internals from the user-facing `ConversationBundle`. The model may use source metadata internally, but user responses must not expose raw `sourceId`, `chunkId`, excerpts, or source location controls.

## Answer Policy

When relevant knowledge hits exist:
- The prompt gives the model numbered evidence blocks.
- Each evidence block includes source name, classification, rank, score, and chunk text.
- The assistant should answer in Chinese and cite claims using `[1]`, `[2]`, etc.
- Citations remain attached to paragraphs internally so future admin/debug surfaces can use them.

When no relevant knowledge hits exist:
- The provider returns a deterministic message saying there is not enough indexed company material to answer.
- The provider does not ask the external model to invent an answer.
- The response includes no citations.

## Retrieval

The backend increases the retrieval limit from 2 to 5 hits. This keeps the first implementation simple while giving the model enough evidence to synthesize a useful answer.

## Testing

Backend tests cover:
- Prompt formatting includes strict source-grounding instructions and numbered evidence.
- OpenAI-compatible provider sends the strict system prompt and knowledge context.
- No-evidence requests return a guarded reply without making an external model call.
- Repository passes up to five ranked hits into the LLM provider.
- User API responses still hide raw source metadata.

## Constraints

The current workspace root is not a git repository, so this phase cannot create commits from this directory. Verification replaces commit checkpoints.
