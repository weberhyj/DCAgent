"""Internal contracts for the versioned retrieval pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class RetrievalMode(StrEnum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    QWEN3 = "qwen3"


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    knowledge_base_id: str
    permission_tags: tuple[str, ...]
    publication_version: str

    def __post_init__(self) -> None:
        if not self.knowledge_base_id.strip():
            raise ValueError("knowledge_base_id must not be empty")
        if not self.permission_tags:
            raise ValueError("permission_tags must not be empty")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    limit: int
    routing_key: str
    scope: RetrievalScope


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    source_id: str
    source_name: str
    source_type: str
    classification: str
    chunk_id: str
    chunk_index: int
    text: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    parent_chunk_id: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    mode: RetrievalMode
    candidates: tuple[RetrievalCandidate, ...]
    stage_ms: Mapping[str, float]
    fallback_reason: str | None = None
