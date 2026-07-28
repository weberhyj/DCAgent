"""Stable Legacy, Shadow, and Qwen3 retrieval routing."""

from __future__ import annotations

import math
import queue
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from threading import Lock, Thread
from typing import Protocol
from uuid import uuid4

from loguru import logger

from .embedding_client import EmbeddingServiceError
from .hybrid_retriever import HybridRetrievalOutcome, HybridRetrievalTimeout
from .models import KnowledgeSearchHitModel
from .reranker_client import RerankerBusy, RerankerServiceError
from .retrieval_models import RetrievalMode, RetrievalRequest


class HybridRetrieverProtocol(Protocol):
    def retrieve(self, request: RetrievalRequest) -> HybridRetrievalOutcome: ...


class ShadowAuditProtocol(Protocol):
    def record_shadow(
        self,
        *,
        request_id: str,
        evaluation_case_id: str | None,
        relevant_chunk_ids: tuple[str, ...],
        routing_key_hash: str,
        query_hash: str,
        legacy_chunk_ids: tuple[str, ...],
        qwen_chunk_ids: tuple[str, ...],
        legacy_ms: float,
        qwen_ms: float,
        status: str,
        fallback_reason: str | None = None,
    ) -> object: ...


LegacySearch = Callable[[str, int], Sequence[KnowledgeSearchHitModel]]


class ShadowQueueCloseError(RuntimeError):
    """The daemon Shadow worker did not stop within the bounded close interval."""


class RetrievalFallbackReason(StrEnum):
    RETRIEVAL_SCOPE_UNAVAILABLE = "retrieval_scope_unavailable"


@dataclass(frozen=True, slots=True)
class RoutedRetrievalOutcome:
    mode: RetrievalMode
    hits: tuple[KnowledgeSearchHitModel, ...]
    stage_ms: Mapping[str, float]
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _CircuitPermit:
    epoch: int
    half_open_probe: bool = False


class _CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int,
        reset_interval_seconds: float,
        monotonic: Callable[[], float],
    ) -> None:
        if isinstance(failure_threshold, bool) or not isinstance(failure_threshold, int):
            raise ValueError("failure_threshold must be a positive integer")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be a positive integer")
        if not math.isfinite(reset_interval_seconds) or reset_interval_seconds <= 0:
            raise ValueError("reset_interval_seconds must be positive and finite")
        self._failure_threshold = failure_threshold
        self._reset_interval_seconds = float(reset_interval_seconds)
        self._monotonic = monotonic
        self._lock = Lock()
        self._state = "closed"
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._epoch = 0

    def acquire(self) -> _CircuitPermit | None:
        with self._lock:
            if self._state == "closed":
                return _CircuitPermit(self._epoch)
            if self._state == "half_open":
                return None
            if self._monotonic() - self._opened_at < self._reset_interval_seconds:
                return None
            self._state = "half_open"
            return _CircuitPermit(self._epoch, half_open_probe=True)

    def record_success(self, permit: _CircuitPermit) -> None:
        with self._lock:
            if permit.epoch != self._epoch:
                return
            if self._state == "half_open":
                if not permit.half_open_probe:
                    return
                self._state = "closed"
                self._consecutive_failures = 0
                self._epoch += 1
                return
            if self._state == "closed":
                self._consecutive_failures = 0

    def record_failure(self, permit: _CircuitPermit) -> None:
        with self._lock:
            if permit.epoch != self._epoch:
                return
            if self._state == "half_open":
                if not permit.half_open_probe:
                    return
                self._open_locked()
                return
            if self._state != "closed":
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._open_locked()

    def abandon(self, permit: _CircuitPermit) -> None:
        with self._lock:
            if permit.epoch != self._epoch:
                return
            if self._state == "half_open" and permit.half_open_probe:
                self._open_locked()

    def _open_locked(self) -> None:
        self._state = "open"
        self._opened_at = self._monotonic()
        self._consecutive_failures = 0
        self._epoch += 1


@dataclass(frozen=True, slots=True)
class _ShadowTask:
    request: RetrievalRequest
    legacy_hits: tuple[KnowledgeSearchHitModel, ...]
    legacy_ms: float


_STOP = object()
_SANITIZED_FALLBACK_CODES = frozenset(
    {
        "qwen_timeout",
        "embedding_unavailable",
        "reranker_unavailable",
        "hybrid_unavailable",
    }
)


def _new_request_id() -> str:
    return uuid4().hex


class _SanitizedRetrievalFailure(RuntimeError):
    pass


def _log_sanitized_failure_in_isolated_frame(
    *,
    request_id: str,
    mode: str,
    fallback_code: str,
    shadow: bool,
) -> None:
    try:
        raise _SanitizedRetrievalFailure(f"{fallback_code} request_id={request_id}") from None
    except _SanitizedRetrievalFailure:
        logger.bind(
            request_id=request_id,
            mode=mode,
            fallback_code=fallback_code,
            fallback_reason=fallback_code,
        ).exception("shadow hybrid retrieval failed" if shadow else "hybrid retrieval failed")


@dataclass(frozen=True, slots=True)
class _SanitizedLogTask:
    request_id: str
    mode: str
    fallback_code: str
    shadow: bool


class SanitizedLogQueue:
    """One bounded, non-blocking sanitized failure logger per router."""

    def __init__(self, *, max_size: int, close_timeout_seconds: float) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("sanitized log queue size must be a positive integer")
        if not math.isfinite(close_timeout_seconds) or close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be positive and finite")
        self._queue: queue.Queue[_SanitizedLogTask | object] = queue.Queue(maxsize=max_size)
        self._close_timeout_seconds = float(close_timeout_seconds)
        self._lock = Lock()
        self._closing = False
        self._closed = False
        self._started = False
        self._dropped_count = 0
        self.worker = Thread(
            target=self._run,
            name="retrieval-sanitized-log-worker",
            daemon=True,
        )
        try:
            self.worker.start()
        except Exception:
            return
        self._started = True

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def submit(
        self,
        *,
        request_id: str,
        mode: str,
        fallback_code: str,
        shadow: bool = False,
    ) -> bool:
        if fallback_code not in _SANITIZED_FALLBACK_CODES:
            fallback_code = "hybrid_unavailable"
        task = _SanitizedLogTask(
            request_id=request_id,
            mode=mode,
            fallback_code=fallback_code,
            shadow=shadow,
        )
        with self._lock:
            if self._closing or self._closed or not self._started or not self.worker.is_alive():
                self._dropped_count += 1
                return False
            try:
                self._queue.put_nowait(task)
            except queue.Full:
                self._dropped_count += 1
                return False
            return True

    def drain_for_test(self) -> None:
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._closing:
                self._closing = True
                self._dropped_count += self._discard_pending_locked()
                if not self._started or not self.worker.is_alive():
                    self._closed = True
                    return
                self._queue.put_nowait(_STOP)
        self.worker.join(self._close_timeout_seconds)
        with self._lock:
            if not self.worker.is_alive():
                self._closed = True

    def _discard_pending_locked(self) -> int:
        discarded = 0
        while True:
            try:
                task = self._queue.get_nowait()
            except queue.Empty:
                return discarded
            else:
                self._queue.task_done()
                if task is not _STOP:
                    discarded += 1

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is _STOP:
                    return
                assert isinstance(task, _SanitizedLogTask)
                try:
                    _log_sanitized_failure_in_isolated_frame(
                        request_id=task.request_id,
                        mode=task.mode,
                        fallback_code=task.fallback_code,
                        shadow=task.shadow,
                    )
                except Exception:
                    continue
            finally:
                self._queue.task_done()


class ShadowQueue:
    """One bounded, non-blocking Shadow worker for an API process."""

    def __init__(
        self,
        *,
        hybrid: HybridRetrieverProtocol,
        audit: ShadowAuditProtocol | None,
        max_size: int,
        close_timeout_seconds: float,
        monotonic: Callable[[], float],
        sanitized_log_queue: SanitizedLogQueue,
    ) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("shadow queue size must be a positive integer")
        if not math.isfinite(close_timeout_seconds) or close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be positive and finite")
        self._hybrid = hybrid
        self._audit = audit
        self._monotonic = monotonic
        self._close_timeout_seconds = float(close_timeout_seconds)
        self._sanitized_log_queue = sanitized_log_queue
        self._queue: queue.Queue[_ShadowTask | object] = queue.Queue(maxsize=max_size)
        self._lock = Lock()
        self._closing = False
        self._closed = False
        self._dropped_count = 0
        self.worker = Thread(
            target=self._run,
            name="retrieval-shadow-worker",
            daemon=True,
        )
        self.worker.start()

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def submit(
        self,
        request: RetrievalRequest,
        legacy_hits: Sequence[KnowledgeSearchHitModel],
        legacy_ms: float,
    ) -> bool:
        task = _ShadowTask(request, tuple(legacy_hits), float(legacy_ms))
        with self._lock:
            if self._closing or self._closed or not self.worker.is_alive():
                self._dropped_count += 1
                return False
            try:
                self._queue.put_nowait(task)
            except queue.Full:
                self._dropped_count += 1
                return False
            return True

    def drain_for_test(self) -> None:
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._closing:
                self._closing = True
                self._dropped_count += self._discard_pending_locked()
                if not self.worker.is_alive():
                    self._closed = True
                    return
                self._queue.put_nowait(_STOP)
        self.worker.join(self._close_timeout_seconds)
        if self.worker.is_alive():
            raise ShadowQueueCloseError("shadow worker did not stop before close timeout") from None
        with self._lock:
            self._closed = True

    def _discard_pending_locked(self) -> int:
        discarded = 0
        while True:
            try:
                task = self._queue.get_nowait()
            except queue.Empty:
                return discarded
            else:
                self._queue.task_done()
                if task is not _STOP:
                    discarded += 1

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is _STOP:
                    return
                assert isinstance(task, _ShadowTask)
                self._run_task(task)
            finally:
                self._queue.task_done()

    def _run_task(self, task: _ShadowTask) -> None:
        started = self._monotonic()
        request_id = f"shadow-{uuid4().hex}"
        status = "completed"
        fallback_reason: str | None = None
        failure_code: str | None = None
        qwen_chunk_ids: tuple[str, ...] = ()
        try:
            outcome = self._hybrid.retrieve(task.request)
            qwen_chunk_ids = tuple(item.chunk.id for item in outcome.hits)
            if not outcome.hits and task.legacy_hits:
                status = "fallback"
                fallback_reason = "qwen_empty_legacy_nonempty"
        except Exception as error:
            status = "failed"
            failure_code = _fallback_code(error)
            fallback_reason = failure_code
        if failure_code is not None:
            self._sanitized_log_queue.submit(
                request_id=request_id,
                mode=RetrievalMode.SHADOW.value,
                fallback_code=failure_code,
                shadow=True,
            )
        qwen_ms = _elapsed_ms(started, self._monotonic())
        if self._audit is None:
            return
        try:
            self._audit.record_shadow(
                request_id=request_id,
                evaluation_case_id=task.request.evaluation_case_id,
                relevant_chunk_ids=task.request.relevant_chunk_ids,
                routing_key_hash=_sha256_hex(task.request.routing_key),
                query_hash=_sha256_hex(task.request.query),
                legacy_chunk_ids=tuple(item.chunk.id for item in task.legacy_hits),
                qwen_chunk_ids=qwen_chunk_ids,
                legacy_ms=task.legacy_ms,
                qwen_ms=qwen_ms,
                status=status,
                fallback_reason=fallback_reason,
            )
        except Exception:
            return


class RetrievalRouter:
    def __init__(
        self,
        *,
        mode: RetrievalMode | str,
        legacy_search: LegacySearch,
        hybrid: HybridRetrieverProtocol,
        audit: ShadowAuditProtocol | None = None,
        shadow_percent: float = 0.0,
        canary_percent: float = 100.0,
        failure_threshold: int = 3,
        reset_interval_seconds: float = 30.0,
        shadow_queue_size: int = 32,
        sanitized_log_queue_size: int = 64,
        close_timeout_seconds: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
        embedding_model_version: str | None = None,
        reranker_model_version: str | None = None,
        qdrant_alias: str | None = None,
        request_id_factory: Callable[[], str] = _new_request_id,
    ) -> None:
        try:
            self.mode = RetrievalMode(mode)
        except ValueError as error:
            raise ValueError("mode must be legacy, shadow, or qwen3") from error
        if not callable(legacy_search):
            raise TypeError("legacy_search must be callable")
        if hybrid is None or not callable(getattr(hybrid, "retrieve", None)):
            raise TypeError("hybrid must expose retrieve()")
        self._legacy_search = legacy_search
        self._hybrid = hybrid
        self._shadow_percent = _percentage(shadow_percent, "shadow_percent")
        self._canary_percent = _percentage(canary_percent, "canary_percent")
        self._monotonic = monotonic
        self._embedding_model_version = embedding_model_version
        self._reranker_model_version = reranker_model_version
        self._qdrant_alias = qdrant_alias
        if not callable(request_id_factory):
            raise TypeError("request_id_factory must be callable")
        self._request_id_factory = request_id_factory
        if self.mode is RetrievalMode.SHADOW and audit is None:
            raise ValueError("shadow mode requires an audit repository")
        self._circuit = _CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_interval_seconds=reset_interval_seconds,
            monotonic=monotonic,
        )
        self.sanitized_log_queue = SanitizedLogQueue(
            max_size=sanitized_log_queue_size,
            close_timeout_seconds=close_timeout_seconds,
        )
        self.shadow_queue = (
            ShadowQueue(
                hybrid=hybrid,
                audit=audit,
                max_size=shadow_queue_size,
                close_timeout_seconds=close_timeout_seconds,
                monotonic=monotonic,
                sanitized_log_queue=self.sanitized_log_queue,
            )
            if self.mode is RetrievalMode.SHADOW
            else None
        )

    def close(self) -> None:
        try:
            if self.shadow_queue is not None:
                self.shadow_queue.close()
        finally:
            self.sanitized_log_queue.close()

    def uses_qwen(self, routing_key: str) -> bool:
        percentage = (
            self._shadow_percent if self.mode is RetrievalMode.SHADOW else self._canary_percent
        )
        return stable_percentage_bucket(routing_key) < percentage

    def search(self, request: RetrievalRequest) -> RoutedRetrievalOutcome:
        if not isinstance(request, RetrievalRequest):
            raise TypeError("request must be a RetrievalRequest")
        request_id = self._request_id()
        outcome, candidate_counts = self._search(request, request_id=request_id)
        self._log_completion(
            request_id=request_id,
            outcome=outcome,
            candidate_counts=candidate_counts,
        )
        return outcome

    def fallback_to_legacy(
        self,
        *,
        query: str,
        limit: int,
        routing_key: str,
        fallback_reason: RetrievalFallbackReason,
    ) -> RoutedRetrievalOutcome:
        if not isinstance(routing_key, str) or not routing_key.strip():
            raise ValueError("routing_key must be a non-empty string")
        if not isinstance(fallback_reason, RetrievalFallbackReason):
            raise TypeError("fallback_reason must be a RetrievalFallbackReason")
        request_id = self._request_id()
        outcome = self._legacy_query(
            query,
            limit,
            fallback_reason=fallback_reason.value,
        )
        self._log_completion(
            request_id=request_id,
            outcome=outcome,
            candidate_counts={"qwen": 0, "legacy": len(outcome.hits)},
        )
        return outcome

    def _request_id(self) -> str:
        request_id = self._request_id_factory()
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id_factory must return a non-empty string")
        return request_id

    def _log_completion(
        self,
        *,
        request_id: str,
        outcome: RoutedRetrievalOutcome,
        candidate_counts: Mapping[str, int],
    ) -> None:
        logger.bind(
            request_id=request_id,
            mode=self.mode.value,
            embedding_model_version=self._embedding_model_version,
            reranker_model_version=self._reranker_model_version,
            model_versions={
                "embedding": self._embedding_model_version,
                "reranker": self._reranker_model_version,
            },
            alias=self._qdrant_alias,
            qdrant_alias=self._qdrant_alias,
            candidate_counts=candidate_counts,
            stage_timings=dict(outcome.stage_ms),
            stage_timings_ms=dict(outcome.stage_ms),
            fallback_code=outcome.fallback_reason,
            fallback_reason=outcome.fallback_reason,
            result_count=len(outcome.hits),
        ).info("retrieval completed")

    def _search(
        self,
        request: RetrievalRequest,
        *,
        request_id: str,
    ) -> tuple[RoutedRetrievalOutcome, dict[str, int]]:
        if self.mode is RetrievalMode.LEGACY:
            legacy = self._legacy(request)
            return legacy, {"qwen": 0, "legacy": len(legacy.hits)}
        if self.mode is RetrievalMode.SHADOW:
            legacy = self._legacy(request)
            if self.uses_qwen(request.routing_key):
                assert self.shadow_queue is not None
                self.shadow_queue.submit(request, legacy.hits, legacy.stage_ms["legacy"])
            return legacy, {"qwen": 0, "legacy": len(legacy.hits)}
        if not self.uses_qwen(request.routing_key):
            legacy = self._legacy(request)
            return legacy, {"qwen": 0, "legacy": len(legacy.hits)}

        permit = self._circuit.acquire()
        if permit is None:
            legacy = self._legacy(request, fallback_reason="circuit_open")
            return legacy, {"qwen": 0, "legacy": len(legacy.hits)}
        qwen: HybridRetrievalOutcome | None = None
        failure_code: str | None = None
        try:
            qwen = self._hybrid.retrieve(request)
        except Exception as error:
            self._circuit.record_failure(permit)
            failure_code = _fallback_code(error)
        except BaseException:
            self._circuit.abandon(permit)
            raise
        if failure_code is not None:
            self.sanitized_log_queue.submit(
                request_id=request_id,
                mode=self.mode.value,
                fallback_code=failure_code,
            )
            legacy = self._legacy(request, fallback_reason=failure_code)
            return legacy, {"qwen": 0, "legacy": len(legacy.hits)}
        assert qwen is not None
        self._circuit.record_success(permit)
        if qwen.hits:
            return (
                RoutedRetrievalOutcome(
                    mode=RetrievalMode.QWEN3,
                    hits=tuple(qwen.hits),
                    stage_ms=dict(qwen.stage_ms),
                ),
                {"qwen": len(qwen.candidates), "legacy": 0},
            )
        legacy = self._legacy(request)
        if legacy.hits:
            return (
                RoutedRetrievalOutcome(
                    mode=RetrievalMode.LEGACY,
                    hits=legacy.hits,
                    stage_ms=legacy.stage_ms,
                    fallback_reason="qwen_empty_legacy_nonempty",
                ),
                {"qwen": len(qwen.candidates), "legacy": len(legacy.hits)},
            )
        return (
            RoutedRetrievalOutcome(
                mode=RetrievalMode.QWEN3,
                hits=(),
                stage_ms=dict(qwen.stage_ms),
            ),
            {"qwen": len(qwen.candidates), "legacy": 0},
        )

    def _legacy(
        self,
        request: RetrievalRequest,
        *,
        fallback_reason: str | None = None,
    ) -> RoutedRetrievalOutcome:
        return self._legacy_query(
            request.query,
            request.limit,
            fallback_reason=fallback_reason,
        )

    def _legacy_query(
        self,
        query: str,
        limit: int,
        *,
        fallback_reason: str | None = None,
    ) -> RoutedRetrievalOutcome:
        started = self._monotonic()
        hits = tuple(self._legacy_search(query, limit))
        return RoutedRetrievalOutcome(
            mode=RetrievalMode.LEGACY,
            hits=hits,
            stage_ms={"legacy": _elapsed_ms(started, self._monotonic())},
            fallback_reason=fallback_reason,
        )


def stable_percentage_bucket(routing_key: str) -> int:
    if not isinstance(routing_key, str) or not routing_key.strip():
        raise ValueError("routing_key must be a non-empty string")
    digest = sha256(routing_key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


def _sha256_hex(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _percentage(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be between 0 and 100")
    percentage = float(value)
    if not math.isfinite(percentage) or not 0 <= percentage <= 100:
        raise ValueError(f"{name} must be between 0 and 100")
    return percentage


def _elapsed_ms(started: float, completed: float) -> float:
    return max(0.0, (completed - started) * 1000.0)


def _fallback_code(error: Exception) -> str:
    if isinstance(error, HybridRetrievalTimeout):
        return "qwen_timeout"
    if isinstance(error, EmbeddingServiceError):
        return "embedding_unavailable"
    if isinstance(error, (RerankerBusy, RerankerServiceError)):
        return "reranker_unavailable"
    return "hybrid_unavailable"


__all__ = [
    "RetrievalFallbackReason",
    "RetrievalRouter",
    "RoutedRetrievalOutcome",
    "SanitizedLogQueue",
    "ShadowQueue",
    "ShadowQueueCloseError",
    "stable_percentage_bucket",
]
