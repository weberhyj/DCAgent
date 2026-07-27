from __future__ import annotations

import hashlib
import threading
import unittest

from app.hybrid_retriever import HybridRetrievalOutcome, HybridRetrievalTimeout
from app.models import KnowledgeChunkModel, KnowledgeSearchHitModel, KnowledgeSourceModel
from app.reranker_client import RerankerBusy
from app.retrieval_models import (
    RetrievalCandidate,
    RetrievalMode,
    RetrievalRequest,
    RetrievalScope,
)
from app.retrieval_router import RetrievalRouter, stable_percentage_bucket

SCOPE = RetrievalScope("default", ("internal",), "v1")


def request(query: str = "policy", *, routing_key: str = "conv-1") -> RetrievalRequest:
    return RetrievalRequest(query=query, limit=8, routing_key=routing_key, scope=SCOPE)


def hit(chunk_id: str) -> KnowledgeSearchHitModel:
    source = KnowledgeSourceModel(
        id=f"source-{chunk_id}",
        name=f"{chunk_id}.txt",
        source_type="TXT",
        records=1,
        status="indexed",
        updated_at="2026-07-28 10:00:00",
        classification="internal",
    )
    chunk = KnowledgeChunkModel(
        id=chunk_id,
        source_id=source.id,
        chunk_index=0,
        text=f"evidence for {chunk_id}",
        token_count=4,
    )
    return KnowledgeSearchHitModel(source=source, chunk=chunk, score=1.0, rank=1)


def candidate(chunk_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        source_id=f"source-{chunk_id}",
        source_name=f"{chunk_id}.txt",
        source_type="TXT",
        classification="internal",
        chunk_id=chunk_id,
        chunk_index=0,
        text=f"evidence for {chunk_id}",
        rerank_score=0.9,
    )


def hybrid_outcome(*chunk_ids: str) -> HybridRetrievalOutcome:
    return HybridRetrievalOutcome(
        mode=RetrievalMode.QWEN3,
        candidates=tuple(candidate(chunk_id) for chunk_id in chunk_ids),
        hits=tuple(hit(chunk_id) for chunk_id in chunk_ids),
        stage_ms={"embedding": 1.0, "qdrant": 2.0},
    )


class RecordingHybrid:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [hybrid_outcome("qwen-1")])
        self.calls = 0
        self.requests: list[RetrievalRequest] = []
        self._lock = threading.Lock()

    def retrieve(self, retrieval_request: RetrievalRequest) -> HybridRetrievalOutcome:
        with self._lock:
            self.calls += 1
            self.requests.append(retrieval_request)
            outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, HybridRetrievalOutcome)
        return outcome


class RecordingAudit:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def record_shadow(self, **values: object) -> object:
        with self._lock:
            self.records.append(dict(values))
        return values


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds


class BlockingHybrid:
    def __init__(self, first_error: BaseException | None = None) -> None:
        self.first_error = first_error
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def retrieve(self, retrieval_request: RetrievalRequest) -> HybridRetrievalOutcome:
        del retrieval_request
        with self._lock:
            self.calls += 1
            call = self.calls
        if call == 1 and self.first_error is not None:
            raise self.first_error
        self.started.set()
        self.release.wait(2.0)
        return hybrid_outcome("qwen-probe")


class RetrievalRouterTest(unittest.TestCase):
    def build_router(
        self,
        *,
        mode: str = "legacy",
        hybrid: object | None = None,
        audit: RecordingAudit | None = None,
        legacy_hits: list[KnowledgeSearchHitModel] | None = None,
        **options: object,
    ) -> RetrievalRouter:
        router = RetrievalRouter(
            mode=mode,
            legacy_search=lambda query, limit: list(
                [hit("legacy-1")] if legacy_hits is None else legacy_hits
            )[:limit],
            hybrid=hybrid or RecordingHybrid(),
            audit=audit,
            **options,
        )
        self.addCleanup(router.close)
        return router

    def test_stable_bucket_uses_sha256_first_eight_bytes_big_endian(self) -> None:
        routing_key = "conv-stable-7"
        expected = int.from_bytes(
            hashlib.sha256(routing_key.encode()).digest()[:8], "big"
        ) % 100

        self.assertEqual(stable_percentage_bucket(routing_key), expected)

    def test_legacy_never_calls_qwen(self) -> None:
        hybrid = RecordingHybrid()
        router = self.build_router(mode="legacy", hybrid=hybrid)

        result = router.search(request())

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(hybrid.calls, 0)

    def test_shadow_not_selected_returns_legacy_without_queueing(self) -> None:
        hybrid = RecordingHybrid()
        audit = RecordingAudit()
        router = self.build_router(
            mode="shadow", hybrid=hybrid, audit=audit, shadow_percent=0
        )

        result = router.search(request())
        router.shadow_queue.drain_for_test()

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(hybrid.calls, 0)
        self.assertEqual(audit.records, [])

    def test_shadow_selected_returns_legacy_and_records_qwen_off_thread(self) -> None:
        hybrid = RecordingHybrid()
        audit = RecordingAudit()
        router = self.build_router(
            mode="shadow", hybrid=hybrid, audit=audit, shadow_percent=100
        )

        result = router.search(request("policy", routing_key="conv-shadow"))
        router.shadow_queue.drain_for_test()

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(hybrid.calls, 1)
        self.assertEqual(audit.records[0]["status"], "completed")
        self.assertEqual(audit.records[0]["legacy_chunk_ids"], ("legacy-1",))
        self.assertEqual(audit.records[0]["qwen_chunk_ids"], ("qwen-1",))

    def test_canary_assignment_is_stable_and_unselected_returns_legacy(self) -> None:
        hybrid = RecordingHybrid()
        router = self.build_router(mode="qwen3", hybrid=hybrid, canary_percent=0)

        assignments = [router.uses_qwen("conv-7") for _ in range(20)]
        result = router.search(request(routing_key="conv-7"))

        self.assertEqual(assignments, [False] * 20)
        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(hybrid.calls, 0)

    def test_qwen_success_returns_qwen_hits(self) -> None:
        router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([hybrid_outcome("qwen-a", "qwen-b")]),
            canary_percent=100,
        )

        result = router.search(request())

        self.assertEqual(result.mode, RetrievalMode.QWEN3)
        self.assertEqual([item.chunk.id for item in result.hits], ["qwen-a", "qwen-b"])
        self.assertIsNone(result.fallback_reason)

    def test_empty_qwen_results_use_nonempty_legacy_results(self) -> None:
        router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([hybrid_outcome()]),
            canary_percent=100,
        )

        result = router.search(request("known term"))

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(result.fallback_reason, "qwen_empty_legacy_nonempty")

    def test_empty_qwen_and_empty_legacy_remains_qwen_without_fallback(self) -> None:
        router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([hybrid_outcome()]),
            legacy_hits=[],
            canary_percent=100,
        )

        result = router.search(request("unknown term"))

        self.assertEqual(result.mode, RetrievalMode.QWEN3)
        self.assertEqual(result.hits, ())
        self.assertIsNone(result.fallback_reason)

    def test_qwen_failure_falls_back_once_and_opens_circuit(self) -> None:
        hybrid = RecordingHybrid([RuntimeError("secret upstream URL")])
        router = self.build_router(
            mode="qwen3",
            hybrid=hybrid,
            canary_percent=100,
            failure_threshold=2,
        )

        first = router.search(request("one"))
        second = router.search(request("two"))
        third = router.search(request("three"))

        self.assertEqual(first.fallback_reason, "hybrid_unavailable")
        self.assertEqual(second.fallback_reason, "hybrid_unavailable")
        self.assertEqual(third.fallback_reason, "circuit_open")
        self.assertEqual(hybrid.calls, 2)

    def test_success_resets_consecutive_failure_count(self) -> None:
        hybrid = RecordingHybrid(
            [
                RuntimeError("first failure"),
                hybrid_outcome("qwen-success-1"),
                RuntimeError("second failure"),
                hybrid_outcome("qwen-success-2"),
            ]
        )
        router = self.build_router(
            mode="qwen3",
            hybrid=hybrid,
            canary_percent=100,
            failure_threshold=2,
        )

        reasons = [router.search(request(str(index))).fallback_reason for index in range(4)]

        self.assertEqual(
            reasons,
            ["hybrid_unavailable", None, "hybrid_unavailable", None],
        )
        self.assertEqual(hybrid.calls, 4)

    def test_failed_half_open_probe_reopens_until_a_later_probe_succeeds(self) -> None:
        clock = FakeClock()
        hybrid = RecordingHybrid(
            [
                RuntimeError("initial failure"),
                RuntimeError("probe failure"),
                hybrid_outcome("qwen-recovered"),
            ]
        )
        router = self.build_router(
            mode="qwen3",
            hybrid=hybrid,
            canary_percent=100,
            failure_threshold=1,
            reset_interval_seconds=5,
            monotonic=clock,
        )

        self.assertEqual(router.search(request("initial")).fallback_reason, "hybrid_unavailable")
        clock.advance(5)
        self.assertEqual(router.search(request("probe-fails")).fallback_reason, "hybrid_unavailable")
        self.assertEqual(router.search(request("still-open")).fallback_reason, "circuit_open")
        clock.advance(5)
        self.assertEqual(router.search(request("probe-succeeds")).mode, RetrievalMode.QWEN3)
        self.assertEqual(hybrid.calls, 3)

    def test_timeout_and_busy_use_sanitized_fallback_codes(self) -> None:
        timeout_router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([HybridRetrievalTimeout("private timeout detail")]),
            canary_percent=100,
        )
        busy_router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([RerankerBusy("private queue detail")]),
            canary_percent=100,
        )

        self.assertEqual(timeout_router.search(request()).fallback_reason, "qwen_timeout")
        self.assertEqual(busy_router.search(request()).fallback_reason, "reranker_unavailable")

    def test_half_open_allows_exactly_one_concurrent_probe_and_closes_on_success(self) -> None:
        clock = FakeClock()
        hybrid = BlockingHybrid(RuntimeError("initial failure"))
        router = self.build_router(
            mode="qwen3",
            hybrid=hybrid,
            canary_percent=100,
            failure_threshold=1,
            reset_interval_seconds=5,
            monotonic=clock,
        )
        self.assertEqual(router.search(request("open")).fallback_reason, "hybrid_unavailable")
        clock.advance(5)

        results: list[object] = []
        threads = [
            threading.Thread(
                target=lambda index=index: results.append(
                    router.search(request(f"probe-{index}", routing_key=f"conv-{index}"))
                )
            )
            for index in range(8)
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(hybrid.started.wait(1.0))
        hybrid.release.set()
        for thread in threads:
            thread.join(2.0)

        reasons = [result.fallback_reason for result in results]
        self.assertEqual(hybrid.calls, 2)
        self.assertEqual(reasons.count("circuit_open"), 7)
        self.assertEqual(reasons.count(None), 1)
        self.assertEqual(router.search(request("after-probe")).mode, RetrievalMode.QWEN3)

    def test_shadow_queue_full_is_nonblocking_and_increments_dropped_metric(self) -> None:
        hybrid = BlockingHybrid()
        router = self.build_router(
            mode="shadow",
            hybrid=hybrid,
            audit=RecordingAudit(),
            shadow_percent=100,
            shadow_queue_size=1,
        )

        router.search(request("first"))
        self.assertTrue(hybrid.started.wait(1.0))
        router.search(request("second"))
        router.search(request("third"))

        self.assertEqual(router.shadow_queue.dropped_count, 1)
        hybrid.release.set()
        router.shadow_queue.drain_for_test()

    def test_shadow_audit_contains_only_hashes_ids_timings_and_codes(self) -> None:
        secret_query = "secret passage text"
        secret_routing_key = "secret-conversation"
        secret_error = "http://reranker-service:8082 private exception"
        audit = RecordingAudit()
        router = self.build_router(
            mode="shadow",
            hybrid=RecordingHybrid([RuntimeError(secret_error)]),
            audit=audit,
            shadow_percent=100,
        )

        router.search(request(secret_query, routing_key=secret_routing_key))
        router.shadow_queue.drain_for_test()

        record = audit.records[0]
        self.assertEqual(record["query_hash"], hashlib.sha256(secret_query.encode()).hexdigest())
        self.assertEqual(
            record["routing_key_hash"],
            hashlib.sha256(secret_routing_key.encode()).hexdigest(),
        )
        self.assertEqual(record["fallback_reason"], "hybrid_unavailable")
        serialized = repr(record)
        self.assertNotIn(secret_query, serialized)
        self.assertNotIn(secret_routing_key, serialized)
        self.assertNotIn(secret_error, serialized)
        self.assertNotIn("http://", serialized)

    def test_close_joins_the_single_daemon_worker_and_rejects_new_shadow_work(self) -> None:
        router = self.build_router(
            mode="shadow", audit=RecordingAudit(), shadow_percent=100
        )
        worker = router.shadow_queue.worker
        self.assertTrue(worker.daemon)
        self.assertTrue(worker.is_alive())

        router.close()

        self.assertFalse(worker.is_alive())
        self.assertFalse(router.shadow_queue.submit(request(), (), 0.0))


if __name__ == "__main__":
    unittest.main()
