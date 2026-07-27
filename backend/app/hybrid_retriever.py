"""Deterministic Dense/Sparse fusion, reranking, and bounded evidence assembly."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
from threading import Lock
from typing import Protocol

from .embedding_contracts import EmbeddingMetadataExpectation
from .models import KnowledgeSearchHitModel, knowledge_search_hit_from_candidate
from .reranker_client import RerankerBusy
from .reranker_contracts import RerankerMetadataExpectation
from .retrieval import (
    DEFAULT_HYBRID_EVIDENCE_CHAR_BUDGET,
    resolve_hybrid_evidence_limit,
)
from .retrieval_models import (
    RetrievalCandidate,
    RetrievalMode,
    RetrievalRequest,
    RetrievalScope,
)
from .retrieval_publication import deterministic_point_id
from .sparse_embedding import SparseVector


class HybridRetrievalError(RuntimeError):
    pass


class HybridRetrievalTimeout(HybridRetrievalError, TimeoutError):
    pass


class DenseEmbeddingClient(Protocol):
    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: str,
        expected: EmbeddingMetadataExpectation,
    ) -> list[list[float]]: ...


class SparseQueryEncoder(Protocol):
    def embed_query(self, query: str) -> SparseVector: ...


class HybridSearchGateway(Protocol):
    def search_dense(
        self,
        vector: Sequence[float],
        *,
        scope: RetrievalScope,
        limit: int,
        collection_name: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...

    def search_sparse(
        self,
        vector: SparseVector,
        *,
        scope: RetrievalScope,
        limit: int,
        collection_name: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...

    def retrieve_points(
        self,
        point_ids: Sequence[int | str],
        *,
        scope: RetrievalScope,
        collection_name: str | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...


class PassageReranker(Protocol):
    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        expected: RerankerMetadataExpectation,
    ) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class HybridRetrievalOutcome:
    """Internal diagnostics plus the existing public evidence-hit representation."""

    mode: RetrievalMode
    candidates: tuple[RetrievalCandidate, ...]
    hits: tuple[KnowledgeSearchHitModel, ...]
    stage_ms: Mapping[str, float]
    fallback_reason: str | None = None


def reciprocal_rank_fusion(
    *,
    dense: Sequence[RetrievalCandidate],
    sparse: Sequence[RetrievalCandidate],
    k: int,
) -> tuple[RetrievalCandidate, ...]:
    """Fuse two ranked lists by chunk ID with deterministic tie breaking."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    dense_ranks = _ranked_candidates(dense, name="dense")
    sparse_ranks = _ranked_candidates(sparse, name="sparse")
    chunk_ids = set(dense_ranks).union(sparse_ranks)
    fused: list[RetrievalCandidate] = []
    for chunk_id in chunk_ids:
        dense_item = dense_ranks.get(chunk_id)
        sparse_item = sparse_ranks.get(chunk_id)
        dense_rank = None if dense_item is None else dense_item[0]
        sparse_rank = None if sparse_item is None else sparse_item[0]
        if dense_item is None:
            payload = sparse_item
        elif sparse_item is None or dense_item[0] <= sparse_item[0]:
            payload = dense_item
        else:
            payload = sparse_item
        assert payload is not None
        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (k + dense_rank)
        if sparse_rank is not None:
            score += 1.0 / (k + sparse_rank)
        fused.append(
            replace(
                payload[1],
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                rrf_score=score,
                rerank_score=None,
            )
        )
    fused.sort(
        key=lambda item: (
            -item.rrf_score,
            item.source_name,
            item.chunk_index,
            item.chunk_id,
        )
    )
    return tuple(fused)


class HybridRetriever:
    def __init__(
        self,
        *,
        embedding: DenseEmbeddingClient,
        sparse: SparseQueryEncoder,
        gateway: HybridSearchGateway,
        reranker: PassageReranker,
        embedding_metadata: EmbeddingMetadataExpectation,
        reranker_metadata: RerankerMetadataExpectation,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        rerank_top_k: int = 24,
        degraded_rerank_top_k: int = 12,
        final_top_k: int = 8,
        rrf_k: int = 60,
        total_timeout_seconds: float = 5.0,
        evidence_char_budget: int = DEFAULT_HYBRID_EVIDENCE_CHAR_BUDGET,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.embedding = _required_dependency(embedding, "embedding")
        self.sparse = _required_dependency(sparse, "sparse")
        self.gateway = _required_dependency(gateway, "gateway")
        self.reranker = _required_dependency(reranker, "reranker")
        self._embedding_metadata = _required_dependency(embedding_metadata, "embedding_metadata")
        self._reranker_metadata = _required_dependency(reranker_metadata, "reranker_metadata")
        self._dense_top_k = _positive_integer(dense_top_k, "dense_top_k")
        self._sparse_top_k = _positive_integer(sparse_top_k, "sparse_top_k")
        self._rerank_top_k = _positive_integer(rerank_top_k, "rerank_top_k")
        self._degraded_rerank_top_k = _positive_integer(
            degraded_rerank_top_k, "degraded_rerank_top_k"
        )
        self._final_top_k = _positive_integer(final_top_k, "final_top_k")
        self._rrf_k = _positive_integer(rrf_k, "rrf_k")
        if self._degraded_rerank_top_k > self._rerank_top_k:
            raise ValueError("degraded_rerank_top_k must not exceed rerank_top_k")
        if self._final_top_k > self._degraded_rerank_top_k:
            raise ValueError("final_top_k must not exceed degraded_rerank_top_k")
        if not math.isfinite(total_timeout_seconds) or total_timeout_seconds <= 0:
            raise ValueError("total_timeout_seconds must be positive and finite")
        self._total_timeout_seconds = float(total_timeout_seconds)
        self._evidence_char_budget = _positive_integer(evidence_char_budget, "evidence_char_budget")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self._monotonic = monotonic
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="hybrid-retrieval",
        )
        self._close_lock = Lock()
        self._closed = False

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def retrieve(self, request: RetrievalRequest) -> HybridRetrievalOutcome:
        if self._closed:
            raise HybridRetrievalError("hybrid retriever is closed")
        query, requested_limit = _validate_request(request)
        deadline = self._monotonic() + self._total_timeout_seconds
        stage_ms: dict[str, float] = {}

        stage_started = self._monotonic()
        dense_vector, sparse_vector = self._run_parallel(
            (
                lambda: self._embed_query(query),
                lambda: self.sparse.embed_query(query),
            ),
            deadline=deadline,
        )
        stage_ms["embedding"] = _elapsed_ms(stage_started, self._monotonic())

        stage_started = self._monotonic()
        dense, sparse = self._run_parallel(
            (
                lambda: self.gateway.search_dense(
                    dense_vector,
                    scope=request.scope,
                    limit=self._dense_top_k,
                ),
                lambda: self.gateway.search_sparse(
                    sparse_vector,
                    scope=request.scope,
                    limit=self._sparse_top_k,
                ),
            ),
            deadline=deadline,
        )
        stage_ms["qdrant"] = _elapsed_ms(stage_started, self._monotonic())

        stage_started = self._monotonic()
        fused = reciprocal_rank_fusion(dense=dense, sparse=sparse, k=self._rrf_k)
        self._require_time(deadline)
        stage_ms["rrf"] = _elapsed_ms(stage_started, self._monotonic())

        stage_started = self._monotonic()
        reranked = self._rerank(query, fused, deadline=deadline)
        stage_ms["reranker"] = _elapsed_ms(stage_started, self._monotonic())

        stage_started = self._monotonic()
        evidence_limit = resolve_hybrid_evidence_limit(
            requested_limit,
            final_top_k=self._final_top_k,
        )
        evidence = self._expand_adjacency(
            reranked[:evidence_limit],
            scope=request.scope,
            limit=evidence_limit,
            deadline=deadline,
        )
        stage_ms["adjacency"] = _elapsed_ms(stage_started, self._monotonic())
        hits = tuple(
            knowledge_search_hit_from_candidate(item, rank=index)
            for index, item in enumerate(evidence, 1)
        )
        return HybridRetrievalOutcome(
            mode=RetrievalMode.QWEN3,
            candidates=evidence,
            hits=hits,
            stage_ms=stage_ms,
        )

    def _embed_query(self, query: str) -> list[float]:
        vectors = self.embedding.embed(
            [query],
            purpose="query",
            expected=self._embedding_metadata,
        )
        if len(vectors) != 1:
            raise ValueError("query embedding returned an unexpected vector count")
        return vectors[0]

    def _rerank(
        self,
        query: str,
        fused: Sequence[RetrievalCandidate],
        *,
        deadline: float,
    ) -> tuple[RetrievalCandidate, ...]:
        candidates = tuple(fused[: self._rerank_top_k])
        if not candidates:
            return ()
        try:
            scores = self._run_one(
                lambda: self.reranker.rerank(
                    query,
                    [item.text for item in candidates],
                    expected=self._reranker_metadata,
                ),
                deadline=deadline,
            )
        except RerankerBusy:
            candidates = candidates[: self._degraded_rerank_top_k]
            scores = self._run_one(
                lambda: self.reranker.rerank(
                    query,
                    [item.text for item in candidates],
                    expected=self._reranker_metadata,
                ),
                deadline=deadline,
            )
        if len(scores) != len(candidates):
            raise ValueError("reranker returned an unexpected score count")
        scored: list[tuple[int, RetrievalCandidate]] = []
        for original_index, (item, raw_score) in enumerate(zip(candidates, scores, strict=True)):
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError("reranker scores must be finite numbers")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError("reranker scores must be finite numbers")
            scored.append((original_index, replace(item, rerank_score=score)))
        scored.sort(
            key=lambda pair: (
                -pair[1].rerank_score,
                pair[0],
                pair[1].source_name,
                pair[1].chunk_index,
                pair[1].chunk_id,
            )
        )
        return tuple(item for _, item in scored)

    def _expand_adjacency(
        self,
        seeds: Sequence[RetrievalCandidate],
        *,
        scope: RetrievalScope,
        limit: int,
        deadline: float,
    ) -> tuple[RetrievalCandidate, ...]:
        if not seeds or limit <= 0:
            return ()
        references: list[tuple[str, str, str]] = []
        requested_point_ids: set[str] = set()
        seed_chunk_ids = {item.chunk_id for item in seeds}
        for seed in seeds:
            for chunk_id in (
                seed.parent_chunk_id,
                seed.previous_chunk_id,
                seed.next_chunk_id,
            ):
                if chunk_id is None or chunk_id in seed_chunk_ids:
                    continue
                point_id = deterministic_point_id(
                    seed.source_id,
                    chunk_id,
                    scope.publication_version,
                )
                if point_id in requested_point_ids:
                    continue
                requested_point_ids.add(point_id)
                references.append((seed.source_id, chunk_id, point_id))
        adjacent: tuple[RetrievalCandidate, ...] = ()
        if references:
            adjacent = self._run_one(
                lambda: self.gateway.retrieve_points(
                    [point_id for _, _, point_id in references],
                    scope=scope,
                ),
                deadline=deadline,
            )
        adjacent_by_key = {(item.source_id, item.chunk_id): item for item in adjacent}
        adjacent_by_key.update({(item.source_id, item.chunk_id): item for item in seeds})
        evidence: list[RetrievalCandidate] = []
        seen_chunk_ids: set[str] = set()
        character_count = 0

        def add(item: RetrievalCandidate) -> bool:
            nonlocal character_count
            if item.chunk_id in seen_chunk_ids:
                return True
            next_count = character_count + len(item.text)
            if next_count > self._evidence_char_budget or len(evidence) >= limit:
                return False
            evidence.append(item)
            seen_chunk_ids.add(item.chunk_id)
            character_count = next_count
            return True

        for seed in seeds:
            if not add(seed):
                break
            for chunk_id in (
                seed.parent_chunk_id,
                seed.previous_chunk_id,
                seed.next_chunk_id,
            ):
                if chunk_id is None:
                    continue
                item = adjacent_by_key.get((seed.source_id, chunk_id))
                if item is not None and not add(item):
                    break
            if len(evidence) >= limit:
                break
        self._require_time(deadline)
        return tuple(evidence)

    def _run_parallel(
        self,
        operations: Sequence[Callable[[], object]],
        *,
        deadline: float,
    ) -> tuple[object, ...]:
        futures = tuple(self._executor.submit(operation) for operation in operations)
        try:
            return tuple(self._future_result(future, deadline) for future in futures)
        except Exception:
            for future in futures:
                future.cancel()
            raise

    def _run_one(self, operation: Callable[[], object], *, deadline: float):
        future = self._executor.submit(operation)
        try:
            return self._future_result(future, deadline)
        except Exception:
            future.cancel()
            raise

    def _future_result(self, future: Future[object], deadline: float):
        remaining = self._require_time(deadline)
        try:
            return future.result(timeout=remaining)
        except TimeoutError:
            raise HybridRetrievalTimeout("hybrid retrieval deadline exceeded") from None

    def _require_time(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise HybridRetrievalTimeout("hybrid retrieval deadline exceeded")
        return remaining


def _ranked_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    name: str,
) -> dict[str, tuple[int, RetrievalCandidate]]:
    if isinstance(candidates, (str, bytes, bytearray)):
        raise TypeError(f"{name} candidates must be a sequence")
    ranked: dict[str, tuple[int, RetrievalCandidate]] = {}
    for rank, item in enumerate(candidates, start=1):
        if not isinstance(item, RetrievalCandidate):
            raise TypeError(f"{name} candidates must contain RetrievalCandidate values")
        chunk_id = item.chunk_id.strip()
        if not chunk_id:
            raise ValueError(f"{name} candidate chunk_id must not be empty")
        if chunk_id in ranked:
            raise ValueError(f"{name} candidates contain duplicate chunk_id {chunk_id}")
        ranked[chunk_id] = (rank, item)
    return ranked


def _validate_request(request: RetrievalRequest) -> tuple[str, int]:
    if not isinstance(request, RetrievalRequest):
        raise TypeError("request must be a RetrievalRequest")
    query = request.query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return query, _positive_integer(request.limit, "request.limit")


def _required_dependency(value, name: str):
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _elapsed_ms(started: float, completed: float) -> float:
    return max(0.0, (completed - started) * 1000.0)


__all__ = [
    "HybridRetrievalError",
    "HybridRetrievalOutcome",
    "HybridRetrievalTimeout",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]
