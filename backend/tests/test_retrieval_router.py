from __future__ import annotations

import hashlib
import io
import threading
import time
import unittest
from unittest.mock import patch

from loguru import logger as loguru_logger

import app.retrieval_router as retrieval_router_module
from app.database import Database
from app.evaluation_batches import build_shadow_report
from app.hybrid_retriever import HybridRetrievalOutcome, HybridRetrievalTimeout
from app.models import KnowledgeChunkModel, KnowledgeSearchHitModel, KnowledgeSourceModel
from app.reranker_client import RerankerBusy
from app.retrieval_audit import RetrievalAuditRepository
from app.retrieval_models import (
    RetrievalCandidate,
    RetrievalMode,
    RetrievalRequest,
    RetrievalScope,
)
from app.retrieval_router import (
    RetrievalRouter,
    RoutedRetrievalOutcome,
    stable_percentage_bucket,
)

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


class BlockingSystemExitHybrid:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def retrieve(self, retrieval_request: RetrievalRequest) -> HybridRetrievalOutcome:
        del retrieval_request
        self.started.set()
        self.release.wait(2.0)
        raise SystemExit(3)


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
        expected = int.from_bytes(hashlib.sha256(routing_key.encode()).digest()[:8], "big") % 100

        self.assertEqual(stable_percentage_bucket(routing_key), expected)

    def test_evaluation_labels_are_bounded_explicit_request_metadata(self) -> None:
        labeled = RetrievalRequest(
            query="policy",
            limit=8,
            routing_key="evaluation-routing",
            scope=SCOPE,
            evaluation_case_id="case-critical",
            relevant_chunk_ids=("chunk-a", "chunk-b"),
        )

        self.assertEqual(labeled.evaluation_case_id, "case-critical")
        self.assertEqual(labeled.relevant_chunk_ids, ("chunk-a", "chunk-b"))
        for override in (
            {"evaluation_case_id": "invalid case id"},
            {"relevant_chunk_ids": ["chunk-a"]},
            {"relevant_chunk_ids": tuple(f"chunk-{index}" for index in range(257))},
            {"relevant_chunk_ids": ("invalid chunk id",)},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                RetrievalRequest(
                    query="policy",
                    limit=8,
                    routing_key="evaluation-routing",
                    scope=SCOPE,
                    **override,
                )

    def test_legacy_never_calls_qwen(self) -> None:
        hybrid = RecordingHybrid()
        router = self.build_router(mode="legacy", hybrid=hybrid)

        result = router.search(request())

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(hybrid.calls, 0)

    def test_shadow_not_selected_returns_legacy_without_queueing(self) -> None:
        hybrid = RecordingHybrid()
        audit = RecordingAudit()
        router = self.build_router(mode="shadow", hybrid=hybrid, audit=audit, shadow_percent=0)

        result = router.search(request())
        router.shadow_queue.drain_for_test()

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(hybrid.calls, 0)
        self.assertEqual(audit.records, [])

    def test_shadow_selected_returns_legacy_and_records_qwen_off_thread(self) -> None:
        hybrid = RecordingHybrid()
        audit = RecordingAudit()
        router = self.build_router(mode="shadow", hybrid=hybrid, audit=audit, shadow_percent=100)

        result = router.search(request("policy", routing_key="conv-shadow"))
        router.shadow_queue.drain_for_test()

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(hybrid.calls, 1)
        self.assertEqual(audit.records[0]["status"], "completed")
        self.assertEqual(audit.records[0]["legacy_chunk_ids"], ("legacy-1",))
        self.assertEqual(audit.records[0]["qwen_chunk_ids"], ("qwen-1",))

    def test_explicit_evaluation_labels_survive_shadow_persistence_and_drive_report(
        self,
    ) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        database.create_schema()
        audit = RetrievalAuditRepository(database)
        router = self.build_router(
            mode="shadow",
            hybrid=RecordingHybrid([hybrid_outcome("qwen-other")]),
            audit=audit,
            shadow_percent=100,
        )

        router.search(
            RetrievalRequest(
                query="policy",
                limit=8,
                routing_key="evaluation-routing",
                scope=SCOPE,
                evaluation_case_id="case-critical",
                relevant_chunk_ids=("legacy-1",),
            )
        )
        router.shadow_queue.drain_for_test()

        stored = audit.list_shadow(limit=1)[0]
        self.assertEqual(stored.evaluation_case_id, "case-critical")
        self.assertEqual(stored.relevant_chunk_ids, ("legacy-1",))
        report = build_shadow_report(
            [stored],
            critical_case_ids={"case-critical"},
        )
        self.assertEqual(report["records"][0]["caseId"], "case-critical")
        self.assertEqual(
            report["quality"]["criticalTop8Regressions"],
            ["case-critical"],
        )

        audit.record_shadow(
            request_id="case-unlabelled",
            evaluation_case_id=None,
            relevant_chunk_ids=("legacy-1",),
            routing_key_hash="b" * 64,
            query_hash="c" * 64,
            legacy_chunk_ids=("legacy-1",),
            qwen_chunk_ids=("qwen-other",),
            legacy_ms=1.0,
            qwen_ms=2.0,
            status="completed",
        )
        unlabelled = audit.list_shadow(limit=1)[0]
        negative = build_shadow_report(
            [unlabelled],
            critical_case_ids={"case-unlabelled"},
        )
        self.assertEqual(negative["records"][0]["caseId"], "redacted")
        self.assertEqual(negative["quality"]["criticalTop8Regressions"], [])

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
        self.assertEqual(
            router.search(request("probe-fails")).fallback_reason, "hybrid_unavailable"
        )
        self.assertEqual(router.search(request("still-open")).fallback_reason, "circuit_open")
        clock.advance(5)
        self.assertEqual(router.search(request("probe-succeeds")).mode, RetrievalMode.QWEN3)
        self.assertEqual(hybrid.calls, 3)

    def test_interrupted_half_open_probe_reopens_and_allows_one_later_probe(self) -> None:
        clock = FakeClock()
        hybrid = RecordingHybrid(
            [
                RuntimeError("initial failure"),
                KeyboardInterrupt("operator interrupt"),
                hybrid_outcome("qwen-recovered"),
                hybrid_outcome("qwen-after-recovery"),
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
        self.assertEqual(router.search(request("open")).fallback_reason, "hybrid_unavailable")
        clock.advance(5)

        with self.assertRaises(KeyboardInterrupt):
            router.search(request("interrupted-probe"))

        self.assertEqual(router.search(request("freshly-reopened")).fallback_reason, "circuit_open")
        self.assertEqual(hybrid.calls, 2)
        clock.advance(5)
        self.assertEqual(router.search(request("later-probe")).mode, RetrievalMode.QWEN3)
        self.assertEqual(hybrid.calls, 3)
        self.assertEqual(router.search(request("closed-again")).mode, RetrievalMode.QWEN3)
        self.assertEqual(hybrid.calls, 4)

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

    def test_keyboard_interrupt_propagates_from_foreground_qwen_retrieval(self) -> None:
        router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([KeyboardInterrupt("operator interrupt")]),
            canary_percent=100,
        )

        with self.assertRaises(KeyboardInterrupt):
            router.search(request())

    def test_shadow_system_exit_terminates_worker_instead_of_becoming_audit_failure(
        self,
    ) -> None:
        audit = RecordingAudit()
        router = self.build_router(
            mode="shadow",
            hybrid=RecordingHybrid([SystemExit(3)]),
            audit=audit,
            shadow_percent=100,
        )

        router.search(request("terminate worker"))
        router.shadow_queue.drain_for_test()

        self.assertFalse(router.shadow_queue.worker.is_alive())
        self.assertEqual(audit.records, [])
        dropped = router.shadow_queue.dropped_count
        self.assertFalse(router.shadow_queue.submit(request(), (), 0.0))
        self.assertEqual(router.shadow_queue.dropped_count, dropped + 1)

    def test_shadow_system_exit_discards_pending_work_before_drain_returns(self) -> None:
        hybrid = BlockingSystemExitHybrid()
        router = self.build_router(
            mode="shadow",
            hybrid=hybrid,
            audit=RecordingAudit(),
            shadow_percent=100,
            shadow_queue_size=2,
        )

        router.search(request("terminate after pending work"))
        self.assertTrue(hybrid.started.wait(1.0))
        self.assertTrue(router.shadow_queue.submit(request("pending work"), (), 0.0))
        hybrid.release.set()
        router.shadow_queue.worker.join(1.0)
        self.assertFalse(router.shadow_queue.worker.is_alive())

        drained = threading.Event()
        drain_thread = threading.Thread(
            target=lambda: (router.shadow_queue.drain_for_test(), drained.set()),
            daemon=True,
        )
        drain_thread.start()

        self.assertTrue(drained.wait(1.0), "pending shadow work was not discarded")
        self.assertEqual(router.shadow_queue.dropped_count, 1)

    def test_ordinary_exception_still_uses_sanitized_fallback(self) -> None:
        router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([RuntimeError("secret internal URL http://qdrant:6333")]),
            canary_percent=100,
        )

        result = router.search(request())

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(result.fallback_reason, "hybrid_unavailable")

    def test_completion_log_is_structured_once_and_does_not_leak_failure_detail(
        self,
    ) -> None:
        secret = "secret passage http://reranker-service:8082"
        events: list[tuple[str, str, dict[str, object]]] = []

        class RecordingLogger:
            def __init__(self, extra: dict[str, object] | None = None) -> None:
                self.extra = dict(extra or {})

            def bind(self, **extra: object) -> RecordingLogger:
                return RecordingLogger({**self.extra, **extra})

            def exception(self, message: str) -> None:
                events.append(("exception", message, dict(self.extra)))

            def info(self, message: str) -> None:
                events.append(("info", message, dict(self.extra)))

        router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([RuntimeError(secret)]),
            canary_percent=100,
            embedding_model_version="embedding-v1",
            reranker_model_version="reranker-v1",
            qdrant_alias="knowledge_chunks_current",
            request_id_factory=lambda: "request-123",
        )

        with patch.object(retrieval_router_module, "logger", RecordingLogger()):
            result = router.search(request("private user query"))
            router.sanitized_log_queue.drain_for_test()

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        completion = [event for event in events if event[1] == "retrieval completed"]
        self.assertEqual(len(completion), 1)
        extra = completion[0][2]
        self.assertEqual(extra["request_id"], "request-123")
        self.assertEqual(extra["mode"], "qwen3")
        self.assertEqual(
            extra["model_versions"],
            {"embedding": "embedding-v1", "reranker": "reranker-v1"},
        )
        self.assertEqual(extra["embedding_model_version"], "embedding-v1")
        self.assertEqual(extra["reranker_model_version"], "reranker-v1")
        self.assertEqual(extra["alias"], "knowledge_chunks_current")
        self.assertEqual(extra["qdrant_alias"], "knowledge_chunks_current")
        self.assertEqual(extra["candidate_counts"], {"qwen": 0, "legacy": 1})
        self.assertEqual(extra["stage_timings"], result.stage_ms)
        self.assertEqual(extra["stage_timings_ms"], result.stage_ms)
        self.assertEqual(extra["fallback_code"], "hybrid_unavailable")
        self.assertEqual(extra["fallback_reason"], "hybrid_unavailable")
        self.assertEqual(extra["result_count"], 1)
        exception_events = [event for event in events if event[0] == "exception"]
        self.assertEqual(len(exception_events), 1)
        self.assertEqual(exception_events[0][1], "hybrid retrieval failed")
        structured = repr([(message, extra) for _, message, extra in events])
        self.assertNotIn(secret, structured)
        self.assertNotIn("http://", structured)
        self.assertNotIn("private user query", structured)

    def test_real_loguru_diagnostics_never_render_foreground_upstream_secrets(
        self,
    ) -> None:
        secret_query = "private query sentinel"
        secret_error = "secret passage http://reranker-service:8082"
        records: list[dict[str, object]] = []
        rendered = io.StringIO()
        router = self.build_router(
            mode="qwen3",
            hybrid=RecordingHybrid([RuntimeError(secret_error)]),
            canary_percent=100,
            request_id_factory=lambda: "request-safe-123",
        )
        record_sink = loguru_logger.add(
            lambda message: records.append(dict(message.record)),
            level="DEBUG",
            backtrace=True,
            diagnose=True,
        )
        rendered_sink = loguru_logger.add(
            rendered,
            format="{message} | {extra}",
            level="DEBUG",
            backtrace=True,
            diagnose=True,
            colorize=False,
        )
        try:
            result = router.search(request(secret_query))
            router.sanitized_log_queue.drain_for_test()
        finally:
            loguru_logger.remove(record_sink)
            loguru_logger.remove(rendered_sink)

        self.assertEqual(result.fallback_reason, "hybrid_unavailable")
        output = rendered.getvalue()
        self.assertIn("request-safe-123", output)
        self.assertIn("hybrid_unavailable", output)
        self.assertNotIn(secret_query, output)
        self.assertNotIn(secret_error, output)
        self.assertNotIn("http://", output)
        failure = next(
            record for record in records if record["message"] == "hybrid retrieval failed"
        )
        exception = failure["exception"].value  # type: ignore[union-attr]
        self.assertIsNone(exception.__cause__)
        self.assertIsNone(exception.__context__)

    def test_real_loguru_diagnostics_never_render_shadow_upstream_secrets(self) -> None:
        secret_query = "private shadow query sentinel"
        secret_error = "secret shadow passage http://embedding-service:8081"
        records: list[dict[str, object]] = []
        rendered = io.StringIO()
        router = self.build_router(
            mode="shadow",
            hybrid=RecordingHybrid([RuntimeError(secret_error)]),
            audit=RecordingAudit(),
            shadow_percent=100,
            request_id_factory=lambda: "request-shadow-123",
        )
        record_sink = loguru_logger.add(
            lambda message: records.append(dict(message.record)),
            level="DEBUG",
            backtrace=True,
            diagnose=True,
        )
        rendered_sink = loguru_logger.add(
            rendered,
            format="{message} | {extra}",
            level="DEBUG",
            backtrace=True,
            diagnose=True,
            colorize=False,
        )
        try:
            router.search(request(secret_query))
            router.shadow_queue.drain_for_test()
            router.sanitized_log_queue.drain_for_test()
        finally:
            loguru_logger.remove(record_sink)
            loguru_logger.remove(rendered_sink)

        output = rendered.getvalue()
        self.assertIn("hybrid_unavailable", output)
        self.assertNotIn(secret_query, output)
        self.assertNotIn(secret_error, output)
        self.assertNotIn("http://", output)
        failure = next(
            record for record in records if record["message"] == "shadow hybrid retrieval failed"
        )
        request_id = failure["extra"]["request_id"]  # type: ignore[index]
        self.assertIsInstance(request_id, str)
        self.assertTrue(request_id)
        exception = failure["exception"].value  # type: ignore[union-attr]
        self.assertIsNone(exception.__cause__)
        self.assertIsNone(exception.__context__)

    def test_sanitized_logging_is_one_bounded_nonblocking_worker_under_failure_load(
        self,
    ) -> None:
        blocked = threading.Event()
        release = threading.Event()

        def blocked_log(**safe_fields: object) -> None:
            self.assertNotIn("query", safe_fields)
            self.assertNotIn("error", safe_fields)
            blocked.set()
            release.wait(2.0)

        with patch.object(
            retrieval_router_module,
            "_log_sanitized_failure_in_isolated_frame",
            side_effect=blocked_log,
        ):
            router = self.build_router(
                mode="qwen3",
                hybrid=RecordingHybrid([RuntimeError("private upstream")]),
                canary_percent=100,
                failure_threshold=100,
                sanitized_log_queue_size=2,
            )
            try:
                results: list[RoutedRetrievalOutcome] = []
                callers = [
                    threading.Thread(
                        target=lambda index=index: results.append(
                            router.search(
                                request(
                                    f"private-query-{index}",
                                    routing_key=f"conversation-{index}",
                                )
                            )
                        )
                    )
                    for index in range(24)
                ]
                for caller in callers:
                    caller.start()
                self.assertTrue(blocked.wait(1.0))
                for caller in callers:
                    caller.join(0.5)

                self.assertTrue(all(not caller.is_alive() for caller in callers))
                self.assertEqual(len(results), 24)
                self.assertTrue(
                    all(result.fallback_reason == "hybrid_unavailable" for result in results)
                )
                self.assertEqual(
                    sum(
                        thread.name == "retrieval-sanitized-log-worker"
                        for thread in threading.enumerate()
                    ),
                    1,
                )
                self.assertGreater(router.sanitized_log_queue.dropped_count, 0)
            finally:
                release.set()

    def test_sanitized_logger_start_failure_never_breaks_legacy_fallback(self) -> None:
        original_start = threading.Thread.start

        def fail_sanitized_worker(thread: threading.Thread) -> None:
            if thread.name == "retrieval-sanitized-log-worker":
                raise RuntimeError("cannot start logger")
            original_start(thread)

        with patch.object(threading.Thread, "start", fail_sanitized_worker):
            router = self.build_router(
                mode="qwen3",
                hybrid=RecordingHybrid([RuntimeError("private upstream")]),
                canary_percent=100,
            )

        result = router.search(request("private-query"))

        self.assertEqual(result.mode, RetrievalMode.LEGACY)
        self.assertEqual(result.fallback_reason, "hybrid_unavailable")
        self.assertEqual(router.sanitized_log_queue.dropped_count, 1)

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
        router = self.build_router(mode="shadow", audit=RecordingAudit(), shadow_percent=100)
        worker = router.shadow_queue.worker
        self.assertTrue(worker.daemon)
        self.assertTrue(worker.is_alive())

        router.close()

        self.assertFalse(worker.is_alive())
        self.assertFalse(router.shadow_queue.submit(request(), (), 0.0))

    def test_legacy_and_qwen_modes_do_not_create_unused_shadow_workers(self) -> None:
        legacy = self.build_router(mode="legacy")
        qwen = self.build_router(mode="qwen3", canary_percent=100)

        self.assertIsNone(legacy.shadow_queue)
        self.assertIsNone(qwen.shadow_queue)
        legacy.close()
        legacy.close()
        qwen.close()
        qwen.close()

    def test_shadow_mode_requires_an_audit_repository(self) -> None:
        router = None
        try:
            router = RetrievalRouter(
                mode="shadow",
                legacy_search=lambda query, limit: [],
                hybrid=RecordingHybrid(),
                audit=None,
                shadow_percent=100,
            )
        except ValueError as error:
            self.assertEqual(str(error), "shadow mode requires an audit repository")
        else:
            self.fail("shadow mode accepted a missing audit repository")
        finally:
            if router is not None:
                router.close()

    def test_full_queue_close_discards_queued_work_and_never_blocks_on_stop_signal(self) -> None:
        hybrid = BlockingHybrid()
        router = self.build_router(
            mode="shadow",
            hybrid=hybrid,
            audit=RecordingAudit(),
            shadow_percent=100,
            shadow_queue_size=1,
            close_timeout_seconds=1.0,
        )
        router.search(request("running"))
        self.assertTrue(hybrid.started.wait(1.0))
        router.search(request("queued"))
        errors: list[BaseException] = []

        closer = threading.Thread(target=lambda: self._capture_close(router, errors))
        closer.start()
        try:
            deadline = time.monotonic() + 0.5
            while router.shadow_queue.dropped_count < 1 and time.monotonic() < deadline:
                threading.Event().wait(0.005)
            self.assertEqual(router.shadow_queue.dropped_count, 1)
        finally:
            hybrid.release.set()
            closer.join(2.0)

        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(hybrid.calls, 1)

    def test_stuck_worker_close_raises_sanitized_error_within_bound(self) -> None:
        hybrid = BlockingHybrid()
        router = self.build_router(
            mode="shadow",
            hybrid=hybrid,
            audit=RecordingAudit(),
            shadow_percent=100,
            close_timeout_seconds=0.05,
        )
        router.search(request("secret stuck query"))
        self.assertTrue(hybrid.started.wait(1.0))
        errors: list[BaseException] = []

        started = time.monotonic()
        closer = threading.Thread(target=lambda: self._capture_close(router, errors))
        closer.start()
        closer.join(0.5)
        elapsed = time.monotonic() - started
        try:
            self.assertFalse(closer.is_alive())
            self.assertLess(elapsed, 0.5)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], retrieval_router_module.ShadowQueueCloseError)
            self.assertEqual(
                str(errors[0]),
                "shadow worker did not stop before close timeout",
            )
            self.assertNotIn("secret", str(errors[0]))
            self.assertFalse(router.shadow_queue.submit(request(), (), 0.0))
        finally:
            hybrid.release.set()
            closer.join(2.0)

        router.close()
        self.assertFalse(router.shadow_queue.worker.is_alive())

    def test_normal_close_is_idempotent_after_shadow_queue_drains(self) -> None:
        router = self.build_router(
            mode="shadow",
            audit=RecordingAudit(),
            shadow_percent=100,
            close_timeout_seconds=0.5,
        )
        router.search(request("normal"))
        router.shadow_queue.drain_for_test()

        router.close()
        router.close()

        self.assertFalse(router.shadow_queue.worker.is_alive())

    def test_concurrent_close_callers_finish_safely(self) -> None:
        hybrid = BlockingHybrid()
        router = self.build_router(
            mode="shadow",
            hybrid=hybrid,
            audit=RecordingAudit(),
            shadow_percent=100,
            close_timeout_seconds=1.0,
        )
        router.search(request("running"))
        self.assertTrue(hybrid.started.wait(1.0))
        errors: list[BaseException] = []
        closers = [
            threading.Thread(target=lambda: self._capture_close(router, errors)) for _ in range(4)
        ]

        for closer in closers:
            closer.start()
        hybrid.release.set()
        for closer in closers:
            closer.join(2.0)

        self.assertTrue(all(not closer.is_alive() for closer in closers))
        self.assertEqual(errors, [])
        self.assertFalse(router.shadow_queue.worker.is_alive())

    @staticmethod
    def _capture_close(router: RetrievalRouter, errors: list[BaseException]) -> None:
        try:
            router.close()
        except BaseException as error:
            errors.append(error)


if __name__ == "__main__":
    unittest.main()
