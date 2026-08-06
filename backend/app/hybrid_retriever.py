"""Deterministic Dense/Sparse fusion, reranking, and bounded evidence assembly."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field, replace
from threading import Condition, RLock
from typing import Protocol

from .embedding_contracts import EmbeddingMetadataExpectation
from .models import KnowledgeSearchHitModel, knowledge_search_hit_from_candidate
from .reranker_client import RerankerBusy, RerankerResponseError, RerankerServiceError
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
        timeout_seconds: float | None = None,
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
        timeout_seconds: float | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...

    def search_sparse(
        self,
        vector: SparseVector,
        *,
        scope: RetrievalScope,
        limit: int,
        collection_name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...

    def retrieve_points(
        self,
        point_ids: Sequence[int | str],
        *,
        scope: RetrievalScope,
        collection_name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...


class PassageReranker(Protocol):
    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        expected: RerankerMetadataExpectation,
        timeout_seconds: float | None = None,
    ) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class HybridRetrievalOutcome:
    """Internal diagnostics plus the existing public evidence-hit representation."""

    mode: RetrievalMode
    candidates: tuple[RetrievalCandidate, ...]
    hits: tuple[KnowledgeSearchHitModel, ...]
    stage_ms: Mapping[str, float]
    fallback_reason: str | None = None


@dataclass(slots=True)
class _ExecutorGeneration:
    generation_id: int
    executor: ThreadPoolExecutor
    active_requests: int = 0
    pending_futures: set[Future[object]] = field(default_factory=set)
    quarantined: bool = False
    shutdown_started: bool = False


_MAX_EXECUTOR_GENERATIONS = 3
_MAX_CLOSE_WAIT_SECONDS = 1.0


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
        reranker: PassageReranker | None,
        embedding_metadata: EmbeddingMetadataExpectation,
        reranker_metadata: RerankerMetadataExpectation | None,
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
        if (reranker is None) != (reranker_metadata is None):
            raise ValueError(
                "reranker and reranker_metadata must both be configured or both be None"
            )
        self.reranker = reranker
        self._embedding_metadata = _required_dependency(embedding_metadata, "embedding_metadata")
        self._reranker_metadata = reranker_metadata
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
        self._lifecycle = Condition(RLock())
        self._next_generation_id = 1
        first_generation = self._new_generation_locked()
        self._executor_generations = [first_generation]
        self._active_generation: _ExecutorGeneration | None = first_generation
        self._executor = first_generation.executor
        self._active_request_count = 0
        self._closing = False
        self._closed = False

    def close(self) -> None:
        close_deadline = time.monotonic() + min(
            self._total_timeout_seconds,
            _MAX_CLOSE_WAIT_SECONDS,
        )
        shutdown: list[ThreadPoolExecutor] = []
        with self._lifecycle:
            if self._closed:
                return
            self._closing = True
            while self._active_request_count > 0 and not self._closed:
                remaining = close_deadline - time.monotonic()
                if remaining <= 0:
                    raise HybridRetrievalError("hybrid retriever close timed out") from None
                self._lifecycle.wait(remaining)
            if self._closed:
                return
            shutdown = self._finalize_close_locked()
        self._shutdown_executors(shutdown)

    def retrieve(self, request: RetrievalRequest) -> HybridRetrievalOutcome:
        generation = self._acquire_request_generation()
        try:
            return self._retrieve(request, generation)
        finally:
            self._release_request_generation(generation)

    def _retrieve(
        self,
        request: RetrievalRequest,
        generation: _ExecutorGeneration,
    ) -> HybridRetrievalOutcome:
        query, requested_limit = _validate_request(request)
        deadline = self._monotonic() + self._total_timeout_seconds
        stage_ms: dict[str, float] = {}

        stage_started = self._monotonic()
        dense_vector, sparse_vector = self._run_parallel(
            (
                lambda: self._embed_query(query, deadline=deadline),
                lambda: self.sparse.embed_query(query),
            ),
            deadline=deadline,
            generation=generation,
        )
        stage_ms["embedding"] = _elapsed_ms(stage_started, self._monotonic())

        stage_started = self._monotonic()
        dense, sparse = self._run_parallel(
            (
                lambda: self.gateway.search_dense(
                    dense_vector,
                    scope=request.scope,
                    limit=self._dense_top_k,
                    timeout_seconds=self._remaining_timeout(deadline),
                ),
                lambda: self.gateway.search_sparse(
                    sparse_vector,
                    scope=request.scope,
                    limit=self._sparse_top_k,
                    timeout_seconds=self._remaining_timeout(deadline),
                ),
            ),
            deadline=deadline,
            generation=generation,
        )
        stage_ms["qdrant"] = _elapsed_ms(stage_started, self._monotonic())

        stage_started = self._monotonic()
        fused = reciprocal_rank_fusion(dense=dense, sparse=sparse, k=self._rrf_k)
        self._require_time(deadline)
        stage_ms["rrf"] = _elapsed_ms(stage_started, self._monotonic())

        if self.reranker is None:
            reranked = fused
            stage_ms["reranker"] = 0.0
        else:
            stage_started = self._monotonic()
            reranked = self._rerank(
                query,
                fused,
                deadline=deadline,
                generation=generation,
            )
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
            generation=generation,
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

    def _embed_query(self, query: str, *, deadline: float | None = None) -> list[float]:
        vectors = self.embedding.embed(
            [query],
            purpose="query",
            expected=self._embedding_metadata,
            timeout_seconds=(None if deadline is None else self._remaining_timeout(deadline)),
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
        generation: _ExecutorGeneration,
    ) -> tuple[RetrievalCandidate, ...]:
        if self.reranker is None or self._reranker_metadata is None:
            raise RuntimeError("reranker is disabled")
        candidates = tuple(fused[: self._rerank_top_k])
        if not candidates:
            return ()
        try:
            scores = self._run_one(
                lambda: self.reranker.rerank(
                    query,
                    [item.text for item in candidates],
                    expected=self._reranker_metadata,
                    timeout_seconds=self._remaining_timeout(deadline),
                ),
                deadline=deadline,
                generation=generation,
            )
        except RerankerBusy:
            candidates = candidates[: self._degraded_rerank_top_k]
            try:
                scores = self._run_one(
                    lambda: self.reranker.rerank(
                        query,
                        [item.text for item in candidates],
                        expected=self._reranker_metadata,
                        timeout_seconds=self._remaining_timeout(deadline),
                    ),
                    deadline=deadline,
                    generation=generation,
                )
            except (RerankerBusy, RerankerResponseError, RerankerServiceError):
                return candidates
        except (RerankerResponseError, RerankerServiceError):
            return candidates[: self._degraded_rerank_top_k]
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
        generation: _ExecutorGeneration,
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
                    timeout_seconds=self._remaining_timeout(deadline),
                ),
                deadline=deadline,
                generation=generation,
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
        generation: _ExecutorGeneration,
    ) -> tuple[object, ...]:
        futures = tuple(self._submit(generation, operation) for operation in operations)
        try:
            return tuple(self._future_result(future, deadline, generation) for future in futures)
        except Exception:
            for future in futures:
                future.cancel()
            raise

    def _run_one(
        self,
        operation: Callable[[], object],
        *,
        deadline: float,
        generation: _ExecutorGeneration,
    ):
        future = self._submit(generation, operation)
        try:
            return self._future_result(future, deadline, generation)
        except Exception:
            future.cancel()
            raise

    def _future_result(
        self,
        future: Future[object],
        deadline: float,
        generation: _ExecutorGeneration,
    ):
        remaining = self._require_time(deadline)
        try:
            return future.result(timeout=remaining)
        except TimeoutError:
            self._quarantine_generation(generation)
            raise HybridRetrievalTimeout("hybrid retrieval deadline exceeded") from None

    def _require_time(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise HybridRetrievalTimeout("hybrid retrieval deadline exceeded")
        return remaining

    def _remaining_timeout(self, deadline: float) -> float:
        return self._require_time(deadline)

    def _acquire_request_generation(self) -> _ExecutorGeneration:
        with self._lifecycle:
            self._prune_generations_locked()
            if self._closed:
                raise HybridRetrievalError("hybrid retriever is closed")
            if self._closing:
                raise HybridRetrievalError("hybrid retriever is closing")
            if (
                self._active_generation is None
                and len(self._executor_generations) < _MAX_EXECUTOR_GENERATIONS
            ):
                replacement = self._new_generation_locked()
                self._executor_generations.append(replacement)
                self._active_generation = replacement
                self._executor = replacement.executor
            generation = self._active_generation
            if generation is None:
                raise HybridRetrievalError("hybrid retrieval worker capacity unavailable")
            generation.active_requests += 1
            self._active_request_count += 1
            return generation

    def _release_request_generation(self, generation: _ExecutorGeneration) -> None:
        shutdown: list[ThreadPoolExecutor] = []
        with self._lifecycle:
            generation.active_requests -= 1
            self._active_request_count -= 1
            if generation.quarantined and generation.active_requests == 0:
                executor = self._start_shutdown_locked(generation)
                if executor is not None:
                    shutdown.append(executor)
            self._prune_generations_locked()
            if self._closing and self._active_request_count == 0 and not self._closed:
                shutdown.extend(self._finalize_close_locked())
            self._lifecycle.notify_all()
        self._shutdown_executors(shutdown)

    def _submit(
        self,
        generation: _ExecutorGeneration,
        operation: Callable[[], object],
    ) -> Future[object]:
        with self._lifecycle:
            if generation.active_requests <= 0 or generation.shutdown_started:
                raise HybridRetrievalError("hybrid retrieval worker unavailable") from None
            try:
                future = generation.executor.submit(operation)
            except RuntimeError:
                raise HybridRetrievalError("hybrid retrieval worker unavailable") from None
            generation.pending_futures.add(future)
            future.add_done_callback(
                lambda completed: self._future_completed(generation, completed)
            )
            return future

    def _future_completed(
        self,
        generation: _ExecutorGeneration,
        future: Future[object],
    ) -> None:
        with self._lifecycle:
            generation.pending_futures.discard(future)
            self._prune_generations_locked()
            self._lifecycle.notify_all()

    def _quarantine_generation(self, generation: _ExecutorGeneration) -> None:
        with self._lifecycle:
            if generation.quarantined:
                return
            generation.quarantined = True
            if self._active_generation is generation:
                self._active_generation = None
            self._prune_generations_locked()
            if (
                not self._closing
                and not self._closed
                and self._active_generation is None
                and len(self._executor_generations) < _MAX_EXECUTOR_GENERATIONS
            ):
                replacement = self._new_generation_locked()
                self._executor_generations.append(replacement)
                self._active_generation = replacement
                self._executor = replacement.executor
            self._lifecycle.notify_all()

    def _new_generation_locked(self) -> _ExecutorGeneration:
        generation_id = self._next_generation_id
        self._next_generation_id += 1
        return _ExecutorGeneration(
            generation_id=generation_id,
            executor=ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix=f"hybrid-retrieval-{generation_id}",
            ),
        )

    def _start_shutdown_locked(
        self,
        generation: _ExecutorGeneration,
    ) -> ThreadPoolExecutor | None:
        if generation.shutdown_started:
            return None
        generation.shutdown_started = True
        return generation.executor

    def _finalize_close_locked(self) -> list[ThreadPoolExecutor]:
        self._closed = True
        self._active_generation = None
        shutdown: list[ThreadPoolExecutor] = []
        for generation in self._executor_generations:
            executor = self._start_shutdown_locked(generation)
            if executor is not None:
                shutdown.append(executor)
        self._lifecycle.notify_all()
        return shutdown

    def _prune_generations_locked(self) -> None:
        self._executor_generations[:] = [
            generation
            for generation in self._executor_generations
            if not (
                generation.shutdown_started
                and generation.active_requests == 0
                and not generation.pending_futures
            )
        ]

    @staticmethod
    def _shutdown_executors(executors: Sequence[ThreadPoolExecutor]) -> None:
        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)


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
