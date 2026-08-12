from __future__ import annotations

import threading
import time
import unittest
from dataclasses import replace

from app.embedding_contracts import EmbeddingModelMetadata
from app.hybrid_retriever import (
    HybridRetrievalError,
    HybridRetrievalTimeout,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from app.models import knowledge_search_hit_from_candidate
from app.reranker_client import RerankerBusy, RerankerResponseError, RerankerServiceError
from app.retrieval_models import (
    EvidenceExpansionPolicy,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalScope,
)
from app.retrieval_publication import deterministic_point_id
from app.retrieval_settings import RerankerModelSettings
from app.sparse_embedding import SparseVector

EMBEDDING = EmbeddingModelMetadata(
    name="Qwen/Qwen3-Embedding-0.6B",
    version="embedding-v1",
    sha256="a" * 64,
    dimensions=3,
    normalized=True,
    encoding_profile_sha256="b" * 64,
    protocol_version="v1",
)
RERANKER = RerankerModelSettings(
    name="Qwen/Qwen3-Reranker-0.6B",
    version="reranker-v1",
    sha256="c" * 64,
    prompt_profile_sha256="d" * 64,
    protocol_version="v1",
)


def candidate(
    chunk_id: str,
    *,
    source_name: str = "policy.txt",
    chunk_index: int = 0,
    text: str | None = None,
    point_id: str | None = None,
    parent_chunk_id: str | None = None,
    previous_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        source_id="source-1",
        source_name=source_name,
        source_type="TXT",
        classification="internal",
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        text=text or f"passage {chunk_id}",
        point_id=point_id or f"point-{chunk_id}",
        parent_chunk_id=parent_chunk_id,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
    )


def request(
    query: str = "policy",
    *,
    limit: int = 8,
    expansion_policy: EvidenceExpansionPolicy = EvidenceExpansionPolicy.BOUNDED_ADJACENCY,
) -> RetrievalRequest:
    return RetrievalRequest(
        query=query,
        limit=limit,
        routing_key="conversation-1",
        scope=RetrievalScope(
            knowledge_base_id="default",
            permission_tags=("internal",),
            publication_version="v7",
        ),
        expansion_policy=expansion_policy,
    )


class RecordingEmbedding:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[tuple[tuple[str, ...], str, object]] = []
        self.timeouts: list[float | None] = []
        self.thread_names: list[str] = []

    def embed(self, texts, *, purpose, expected, timeout_seconds=None):
        self.calls.append((tuple(texts), purpose, expected))
        self.timeouts.append(timeout_seconds)
        self.thread_names.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        return [[0.1, 0.2, 0.3]]


class RecordingSparse:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self.thread_names: list[str] = []

    def embed_query(self, query):
        self.calls.append(query)
        self.thread_names.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        return SparseVector(indices=(1,), values=(1.0,))


class RecordingGateway:
    def __init__(
        self,
        dense: tuple[RetrievalCandidate, ...],
        sparse: tuple[RetrievalCandidate, ...],
        adjacent: tuple[RetrievalCandidate, ...] = (),
        *,
        delay: float = 0.0,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.adjacent = adjacent
        self.delay = delay
        self.dense_calls: list[tuple[object, RetrievalScope, int]] = []
        self.sparse_calls: list[tuple[object, RetrievalScope, int]] = []
        self.retrieve_calls: list[tuple[tuple[str, ...], RetrievalScope]] = []
        self.search_threads: list[str] = []
        self.timeouts: list[float | None] = []

    def search_dense(self, vector, *, scope, limit, collection_name=None, timeout_seconds=None):
        self.dense_calls.append((vector, scope, limit))
        self.timeouts.append(timeout_seconds)
        self.search_threads.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        return self.dense[:limit]

    def search_sparse(self, vector, *, scope, limit, collection_name=None, timeout_seconds=None):
        self.sparse_calls.append((vector, scope, limit))
        self.timeouts.append(timeout_seconds)
        self.search_threads.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        return self.sparse[:limit]

    def retrieve_points(self, point_ids, *, scope, collection_name=None, timeout_seconds=None):
        self.retrieve_calls.append((tuple(point_ids), scope))
        self.timeouts.append(timeout_seconds)
        return self.adjacent


class RecordingReranker:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.passage_count = 0
        self.timeouts: list[float | None] = []

    def rerank(self, query, passages, *, expected, timeout_seconds=None):
        self.batch_sizes.append(len(passages))
        self.timeouts.append(timeout_seconds)
        self.passage_count = len(passages)
        return [1.0 - index / 100 for index in range(len(passages))]


class BusyOnceReranker(RecordingReranker):
    def rerank(self, query, passages, *, expected, timeout_seconds=None):
        self.batch_sizes.append(len(passages))
        self.timeouts.append(timeout_seconds)
        if len(self.batch_sizes) == 1:
            raise RerankerBusy("busy")
        self.passage_count = len(passages)
        return [1.0 - index / 100 for index in range(len(passages))]


class AlwaysBusyReranker(RecordingReranker):
    def rerank(self, query, passages, *, expected, timeout_seconds=None):
        self.batch_sizes.append(len(passages))
        self.timeouts.append(timeout_seconds)
        raise RerankerBusy("busy")


class FailingReranker(RecordingReranker):
    def rerank(self, query, passages, *, expected, timeout_seconds=None):
        self.batch_sizes.append(len(passages))
        self.timeouts.append(timeout_seconds)
        raise RerankerServiceError("unavailable")


class InvalidResponseReranker(RecordingReranker):
    def rerank(self, query, passages, *, expected, timeout_seconds=None):
        self.batch_sizes.append(len(passages))
        self.timeouts.append(timeout_seconds)
        raise RerankerResponseError("invalid response")


class NonCooperativeGate:
    def __init__(self, block_first: int) -> None:
        self.block_first = block_first
        self.release = threading.Event()
        self._condition = threading.Condition()
        self.call_count = 0

    def enter(self) -> None:
        with self._condition:
            self.call_count += 1
            should_block = self.call_count <= self.block_first
            self._condition.notify_all()
        if should_block:
            self.release.wait()

    def wait_for_calls(self, count: int, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self.call_count < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


class GatedEmbedding(RecordingEmbedding):
    def __init__(self, gate: NonCooperativeGate) -> None:
        super().__init__()
        self.gate = gate

    def embed(self, texts, *, purpose, expected, timeout_seconds=None):
        self.gate.enter()
        return super().embed(
            texts,
            purpose=purpose,
            expected=expected,
            timeout_seconds=timeout_seconds,
        )


class GatedSparse(RecordingSparse):
    def __init__(self, gate: NonCooperativeGate) -> None:
        super().__init__()
        self.gate = gate

    def embed_query(self, query):
        self.gate.enter()
        return super().embed_query(query)


class BlockingSearchGateway(RecordingGateway):
    def __init__(self, dense, sparse) -> None:
        super().__init__(dense, sparse)
        self.gate = NonCooperativeGate(2)

    def search_dense(self, vector, *, scope, limit, collection_name=None, timeout_seconds=None):
        self.gate.enter()
        return super().search_dense(
            vector,
            scope=scope,
            limit=limit,
            collection_name=collection_name,
            timeout_seconds=timeout_seconds,
        )

    def search_sparse(self, vector, *, scope, limit, collection_name=None, timeout_seconds=None):
        self.gate.enter()
        return super().search_sparse(
            vector,
            scope=scope,
            limit=limit,
            collection_name=collection_name,
            timeout_seconds=timeout_seconds,
        )


def build_retriever(
    *,
    dense: tuple[RetrievalCandidate, ...] | None = None,
    sparse: tuple[RetrievalCandidate, ...] | None = None,
    adjacent: tuple[RetrievalCandidate, ...] = (),
    embedding: RecordingEmbedding | None = None,
    sparse_encoder: RecordingSparse | None = None,
    reranker: RecordingReranker | None = None,
    reranker_enabled: bool = True,
    gateway_delay: float = 0.0,
    timeout: float = 5.0,
    final_top_k: int = 8,
    evidence_char_budget: int = 24_000,
) -> HybridRetriever:
    dense_values = dense or tuple(
        candidate(f"c{index:02d}", chunk_index=index) for index in range(50)
    )
    sparse_values = sparse or tuple(
        candidate(f"c{index:02d}", chunk_index=index) for index in range(49, -1, -1)
    )
    resolved_reranker = (reranker or RecordingReranker()) if reranker_enabled else None
    return HybridRetriever(
        embedding=embedding or RecordingEmbedding(),
        sparse=sparse_encoder or RecordingSparse(),
        gateway=RecordingGateway(
            dense_values,
            sparse_values,
            adjacent,
            delay=gateway_delay,
        ),
        reranker=resolved_reranker,
        embedding_metadata=EMBEDDING,
        reranker_metadata=RERANKER if reranker_enabled else None,
        dense_top_k=50,
        sparse_top_k=50,
        rerank_top_k=24,
        degraded_rerank_top_k=12,
        final_top_k=final_top_k,
        rrf_k=60,
        total_timeout_seconds=timeout,
        evidence_char_budget=evidence_char_budget,
    )


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_fuses_by_chunk_id_records_one_based_ranks_and_keeps_better_payload(self) -> None:
        dense_b = candidate("b", text="dense b")
        sparse_b = candidate("b", text="sparse b")

        fused = reciprocal_rank_fusion(
            dense=(candidate("a"), dense_b),
            sparse=(sparse_b, candidate("c")),
            k=60,
        )

        self.assertEqual([item.chunk_id for item in fused], ["b", "a", "c"])
        self.assertEqual(fused[0].dense_rank, 2)
        self.assertEqual(fused[0].sparse_rank, 1)
        self.assertEqual(fused[0].text, "sparse b")
        self.assertAlmostEqual(fused[0].rrf_score, 1 / 62 + 1 / 61)

    def test_sorts_exact_score_ties_by_source_name_chunk_index_then_chunk_id(self) -> None:
        fused = reciprocal_rank_fusion(
            dense=(
                candidate("z", source_name="Beta", chunk_index=0),
                candidate("b", source_name="Alpha", chunk_index=2),
                candidate("a", source_name="Alpha", chunk_index=2),
            ),
            sparse=(),
            k=60,
        )

        tied = [item.chunk_id for item in fused if item.dense_rank in {2, 3}]
        self.assertEqual(tied, ["b", "a"])

        exact_ties = reciprocal_rank_fusion(
            dense=(candidate("z", source_name="Beta", chunk_index=0),),
            sparse=(candidate("a", source_name="Alpha", chunk_index=2),),
            k=60,
        )
        self.assertEqual([item.chunk_id for item in exact_ties], ["a", "z"])

    def test_rejects_duplicate_chunk_ids_inside_either_input_list(self) -> None:
        for dense, sparse in (
            ((candidate("a"), candidate("a")), ()),
            ((), (candidate("a"), candidate("a"))),
        ):
            with self.subTest(dense=bool(dense)):
                with self.assertRaisesRegex(ValueError, "duplicate chunk_id"):
                    reciprocal_rank_fusion(dense=dense, sparse=sparse, k=60)

    def test_converts_internal_candidate_to_existing_search_hit_without_diagnostics(self) -> None:
        fused = replace(
            candidate("a", text="authorized evidence"),
            dense_rank=1,
            sparse_rank=2,
            rrf_score=0.25,
            rerank_score=0.91,
        )

        hit = knowledge_search_hit_from_candidate(fused, rank=3)

        self.assertEqual(hit.chunk.id, "a")
        self.assertEqual(hit.source.classification, "internal")
        self.assertEqual(hit.score, 0.91)
        self.assertEqual(hit.rank, 3)
        self.assertFalse(hasattr(hit, "dense_rank"))


class HybridRetrieverTest(unittest.TestCase):
    def addCleanupFor(self, retriever: HybridRetriever) -> HybridRetriever:
        self.addCleanup(retriever.close)
        return retriever

    def test_runs_parallel_dense_and_sparse_stages_then_reranks_top_24(self) -> None:
        retriever = self.addCleanupFor(build_retriever())

        outcome = retriever.retrieve(request())

        self.assertEqual(retriever.reranker.passage_count, 24)
        self.assertLessEqual(len(outcome.candidates), 8)
        self.assertEqual(len(outcome.hits), len(outcome.candidates))
        self.assertEqual(outcome.hits[0].score, outcome.candidates[0].rerank_score)
        self.assertEqual(outcome.hits[0].chunk.id, outcome.candidates[0].chunk_id)
        self.assertEqual(outcome.hits[0].rank, 1)
        self.assertEqual(retriever.gateway.dense_calls[0][2], 50)
        self.assertEqual(retriever.gateway.sparse_calls[0][2], 50)
        self.assertEqual(retriever.gateway.dense_calls[0][1], request().scope)
        self.assertIn("embedding", outcome.stage_ms)
        self.assertIn("qdrant", outcome.stage_ms)
        self.assertIn("rrf", outcome.stage_ms)
        self.assertIn("reranker", outcome.stage_ms)
        self.assertIn("adjacency", outcome.stage_ms)
        self.assertEqual(retriever._executor._max_workers, 4)
        propagated = (
            retriever.embedding.timeouts + retriever.gateway.timeouts + retriever.reranker.timeouts
        )
        self.assertTrue(propagated)
        self.assertTrue(all(timeout is not None and 0 < timeout <= 5.0 for timeout in propagated))

    def test_skips_adjacency_when_expansion_policy_is_none(self) -> None:
        top = candidate("c2", next_chunk_id="c3")
        retriever = self.addCleanupFor(
            build_retriever(
                dense=(top,),
                sparse=(top,),
                adjacent=(candidate("c3", chunk_index=3),),
            )
        )

        outcome = retriever.retrieve(
            request(expansion_policy=EvidenceExpansionPolicy.NONE)
        )

        self.assertEqual(retriever.gateway.retrieve_calls, [])
        self.assertEqual(outcome.stage_ms["adjacency"], 0.0)
        self.assertEqual(retriever.reranker.batch_sizes, [1])

    def test_document_policy_runs_reranker_before_bounded_adjacency(self) -> None:
        top = candidate("c2", next_chunk_id="c3")
        retriever = self.addCleanupFor(
            build_retriever(
                dense=(top,),
                sparse=(top,),
                adjacent=(candidate("c3", chunk_index=3),),
            )
        )

        outcome = retriever.retrieve(
            request(expansion_policy=EvidenceExpansionPolicy.BOUNDED_ADJACENCY)
        )

        self.assertEqual(retriever.reranker.batch_sizes, [1])
        self.assertEqual(len(retriever.gateway.retrieve_calls), 1)
        self.assertGreaterEqual(len(outcome.candidates), 1)

    def test_rrf_only_skips_reranker_and_preserves_fused_order(self) -> None:
        reranker = RecordingReranker()
        retriever = self.addCleanupFor(
            build_retriever(
                reranker=reranker,
                reranker_enabled=False,
                final_top_k=8,
            )
        )

        outcome = retriever.retrieve(request())

        self.assertEqual(reranker.batch_sizes, [])
        self.assertEqual(len(outcome.candidates), 8)
        self.assertTrue(all(item.rerank_score is None for item in outcome.candidates))
        self.assertEqual(outcome.stage_ms["reranker"], 0.0)
        self.assertEqual(
            [item.chunk_id for item in outcome.candidates],
            [
                item.chunk_id
                for item in sorted(outcome.candidates, key=lambda item: -item.rrf_score)
            ],
        )

    def test_requires_reranker_and_metadata_to_be_enabled_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "reranker and reranker_metadata"):
            HybridRetriever(
                embedding=RecordingEmbedding(),
                sparse=RecordingSparse(),
                gateway=RecordingGateway((), (), ()),
                reranker=None,
                embedding_metadata=EMBEDDING,
                reranker_metadata=RERANKER,
            )

    def test_reuses_one_persistent_executor_across_requests(self) -> None:
        retriever = self.addCleanupFor(build_retriever())
        executor = retriever._executor

        retriever.retrieve(request("first"))
        retriever.retrieve(request("second"))

        self.assertIs(retriever._executor, executor)
        self.assertTrue(all(name != "MainThread" for name in retriever.embedding.thread_names))
        self.assertTrue(all(name != "MainThread" for name in retriever.sparse.thread_names))
        self.assertTrue(all(name != "MainThread" for name in retriever.gateway.search_threads))

    def test_retries_busy_reranker_once_with_top_12(self) -> None:
        reranker = BusyOnceReranker()
        retriever = self.addCleanupFor(build_retriever(reranker=reranker))

        outcome = retriever.retrieve(request())

        self.assertEqual(reranker.batch_sizes, [24, 12])
        self.assertTrue(outcome.candidates)

    def test_degrades_to_rrf_candidates_when_busy_retry_is_rejected(self) -> None:
        reranker = AlwaysBusyReranker()
        retriever = self.addCleanupFor(build_retriever(reranker=reranker))

        outcome = retriever.retrieve(request())

        self.assertEqual(reranker.batch_sizes, [24, 12])
        self.assertEqual(len(outcome.candidates), 8)
        self.assertTrue(all(item.rerank_score is None for item in outcome.candidates))

    def test_degrades_to_rrf_candidates_when_reranker_service_is_unavailable(self) -> None:
        reranker = FailingReranker()
        retriever = self.addCleanupFor(build_retriever(reranker=reranker))

        outcome = retriever.retrieve(request())

        self.assertEqual(reranker.batch_sizes, [24])
        self.assertEqual(len(outcome.candidates), 8)
        self.assertTrue(all(item.rerank_score is None for item in outcome.candidates))
        self.assertEqual(outcome.fallback_reason, "reranker_service_error")

    def test_degrades_to_rrf_candidates_when_reranker_response_is_invalid(self) -> None:
        reranker = InvalidResponseReranker()
        retriever = self.addCleanupFor(build_retriever(reranker=reranker))

        outcome = retriever.retrieve(request())

        self.assertEqual(reranker.batch_sizes, [24])
        self.assertEqual(len(outcome.candidates), 8)
        self.assertTrue(all(item.rerank_score is None for item in outcome.candidates))

    def test_uses_one_absolute_deadline_across_encoding_and_search(self) -> None:
        retriever = self.addCleanupFor(
            build_retriever(
                embedding=RecordingEmbedding(delay=0.06),
                sparse_encoder=RecordingSparse(delay=0.06),
                gateway_delay=0.06,
                timeout=0.09,
            )
        )

        started = time.monotonic()
        with self.assertRaises(HybridRetrievalTimeout):
            retriever.retrieve(request())
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.14)

    def test_fetches_all_referenced_neighbors_once_and_deduplicates_evidence(self) -> None:
        top = candidate(
            "c2",
            chunk_index=2,
            parent_chunk_id="parent",
            previous_chunk_id="c1",
            next_chunk_id="c3",
        )
        duplicate_neighbor = replace(candidate("c1", chunk_index=1), text="neighbor c1")
        adjacent = (
            candidate("parent", chunk_index=0),
            duplicate_neighbor,
            candidate("c3", chunk_index=3),
        )
        retriever = self.addCleanupFor(
            build_retriever(
                dense=(top, duplicate_neighbor),
                sparse=(top,),
                adjacent=adjacent,
                final_top_k=4,
            )
        )

        outcome = retriever.retrieve(request(limit=4))

        self.assertEqual(
            [item.chunk_id for item in outcome.candidates], ["c2", "parent", "c1", "c3"]
        )
        self.assertEqual(len(retriever.gateway.retrieve_calls), 1)
        point_ids, scope = retriever.gateway.retrieve_calls[0]
        self.assertEqual(
            point_ids,
            (
                deterministic_point_id("source-1", "parent", "v7"),
                deterministic_point_id("source-1", "c3", "v7"),
            ),
        )
        self.assertEqual(scope, request().scope)
        self.assertEqual(len({item.chunk_id for item in outcome.candidates}), 4)

    def test_stops_at_evidence_count_and_utf8_character_budget(self) -> None:
        top = candidate("c2", text="x" * 10, next_chunk_id="c3")
        adjacent = (candidate("c3", text="y" * 11),)
        retriever = self.addCleanupFor(
            build_retriever(
                dense=(top,),
                sparse=(top,),
                adjacent=adjacent,
                final_top_k=2,
                evidence_char_budget=20,
            )
        )

        outcome = retriever.retrieve(request(limit=2))

        self.assertEqual([item.chunk_id for item in outcome.candidates], ["c2"])
        self.assertLessEqual(sum(len(item.text) for item in outcome.candidates), 20)

    def test_passes_the_exact_scope_to_every_qdrant_operation(self) -> None:
        top = candidate("c2", next_chunk_id="c3")
        retriever = self.addCleanupFor(
            build_retriever(
                dense=(top,),
                sparse=(top,),
                adjacent=(candidate("c3"),),
            )
        )
        expected_scope = request().scope

        retriever.retrieve(request())

        self.assertEqual(retriever.gateway.dense_calls[0][1], expected_scope)
        self.assertEqual(retriever.gateway.sparse_calls[0][1], expected_scope)
        self.assertEqual(retriever.gateway.retrieve_calls[0][1], expected_scope)

    def test_timeout_quarantines_busy_generation_so_a_healthy_request_can_run(self) -> None:
        gate = NonCooperativeGate(4)
        retriever = self.addCleanupFor(
            build_retriever(
                embedding=GatedEmbedding(gate),
                sparse_encoder=GatedSparse(gate),
                timeout=0.05,
            )
        )
        errors: list[Exception] = []

        def run_blocked(query: str) -> None:
            try:
                retriever.retrieve(request(query))
            except Exception as error:
                errors.append(error)

        workers = [
            threading.Thread(target=run_blocked, args=(f"blocked-{index}",)) for index in range(2)
        ]
        for worker in workers:
            worker.start()
        self.assertTrue(gate.wait_for_calls(4))
        for worker in workers:
            worker.join(timeout=0.3)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(isinstance(error, HybridRetrievalTimeout) for error in errors))

        started = time.monotonic()
        try:
            healthy = retriever.retrieve(request("healthy"))
        finally:
            gate.release.set()

        self.assertTrue(healthy.hits)
        self.assertLess(time.monotonic() - started, 0.2)

    def test_executor_quarantine_is_bounded_and_trips_a_sanitized_bulkhead(self) -> None:
        gate = NonCooperativeGate(100)
        retriever = self.addCleanupFor(
            build_retriever(
                embedding=GatedEmbedding(gate),
                sparse_encoder=GatedSparse(gate),
                timeout=0.02,
            )
        )
        try:
            for index in range(3):
                with self.assertRaises(HybridRetrievalTimeout):
                    retriever.retrieve(request(f"stuck-{index}"))

            started = time.monotonic()
            with self.assertRaisesRegex(HybridRetrievalError, "capacity unavailable"):
                retriever.retrieve(request("bulkhead"))
            self.assertLess(time.monotonic() - started, 0.05)
            self.assertLessEqual(len(retriever._executor_generations), 3)
        finally:
            gate.release.set()

    def test_bulkhead_recovers_after_quarantined_workers_finish(self) -> None:
        gate = NonCooperativeGate(100)
        retriever = self.addCleanupFor(
            build_retriever(
                embedding=GatedEmbedding(gate),
                sparse_encoder=GatedSparse(gate),
                timeout=0.02,
            )
        )
        try:
            for index in range(3):
                with self.assertRaises(HybridRetrievalTimeout):
                    retriever.retrieve(request(f"stuck-{index}"))

            with self.assertRaisesRegex(HybridRetrievalError, "capacity unavailable"):
                retriever.retrieve(request("bulkhead"))

            gate.release.set()
            drain_deadline = time.monotonic() + 1.0
            while len(retriever._executor_generations) == 3 and time.monotonic() < drain_deadline:
                time.sleep(0.005)

            healthy = retriever.retrieve(request("recovered"))
        finally:
            gate.release.set()

        self.assertTrue(healthy.hits)

    def test_close_during_search_allows_inflight_request_to_finish_all_stages(self) -> None:
        dense = tuple(candidate(f"c{index}", chunk_index=index) for index in range(24))
        gateway = BlockingSearchGateway(dense, dense)
        retriever = HybridRetriever(
            embedding=RecordingEmbedding(),
            sparse=RecordingSparse(),
            gateway=gateway,
            reranker=RecordingReranker(),
            embedding_metadata=EMBEDDING,
            reranker_metadata=RERANKER,
            total_timeout_seconds=1.0,
        )
        outcomes = []
        retrieve_errors: list[Exception] = []
        close_errors: list[Exception] = []

        def retrieve() -> None:
            try:
                outcomes.append(retriever.retrieve(request()))
            except Exception as error:
                retrieve_errors.append(error)

        def close() -> None:
            try:
                retriever.close()
            except Exception as error:
                close_errors.append(error)

        retrieve_thread = threading.Thread(target=retrieve)
        retrieve_thread.start()
        self.assertTrue(gateway.gate.wait_for_calls(2))
        close_thread = threading.Thread(target=close)
        close_thread.start()
        self.assertTrue(close_thread.is_alive())
        gateway.gate.release.set()
        retrieve_thread.join(timeout=1.0)
        close_thread.join(timeout=1.0)

        self.assertFalse(retrieve_errors)
        self.assertFalse(close_errors)
        self.assertTrue(outcomes[0].hits)
        with self.assertRaisesRegex(HybridRetrievalError, "closed"):
            retriever.retrieve(request("after-close"))

    def test_close_never_waits_for_noncooperative_timed_out_workers(self) -> None:
        gate = NonCooperativeGate(2)
        retriever = build_retriever(
            embedding=GatedEmbedding(gate),
            sparse_encoder=GatedSparse(gate),
            timeout=0.05,
        )
        retrieve_errors: list[Exception] = []
        close_errors: list[Exception] = []

        def retrieve() -> None:
            try:
                retriever.retrieve(request())
            except Exception as error:
                retrieve_errors.append(error)

        def close() -> None:
            try:
                retriever.close()
            except Exception as error:
                close_errors.append(error)

        retrieve_thread = threading.Thread(target=retrieve)
        retrieve_thread.start()
        self.assertTrue(gate.wait_for_calls(2))
        close_thread = threading.Thread(target=close)
        close_thread.start()
        try:
            close_thread.join(timeout=0.2)
            self.assertFalse(close_thread.is_alive())
            self.assertTrue(
                not close_errors
                or all(isinstance(error, HybridRetrievalError) for error in close_errors)
            )
        finally:
            gate.release.set()
            retrieve_thread.join(timeout=1.0)
            close_thread.join(timeout=1.0)
        self.assertTrue(all(isinstance(error, HybridRetrievalTimeout) for error in retrieve_errors))


if __name__ == "__main__":
    unittest.main()
