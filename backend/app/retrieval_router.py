"""Stable Legacy, Shadow, and Qwen3 retrieval routing."""

from __future__ import annotations

import math
import queue
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock, Thread
from typing import Protocol
from uuid import uuid4

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
    ) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("shadow queue size must be a positive integer")
        if not math.isfinite(close_timeout_seconds) or close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be positive and finite")
        self._hybrid = hybrid
        self._audit = audit
        self._monotonic = monotonic
        self._close_timeout_seconds = float(close_timeout_seconds)
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
            if self._closing or self._closed:
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
                self._queue.put_nowait(_STOP)
        self.worker.join(self._close_timeout_seconds)
        if self.worker.is_alive():
            raise ShadowQueueCloseError(
                "shadow worker did not stop before close timeout"
            ) from None
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
        status = "completed"
        fallback_reason: str | None = None
        qwen_chunk_ids: tuple[str, ...] = ()
        try:
            outcome = self._hybrid.retrieve(task.request)
            qwen_chunk_ids = tuple(item.chunk.id for item in outcome.hits)
            if not outcome.hits and task.legacy_hits:
                status = "fallback"
                fallback_reason = "qwen_empty_legacy_nonempty"
        except BaseException as error:
            status = "failed"
            fallback_reason = _fallback_code(error)
        qwen_ms = _elapsed_ms(started, self._monotonic())
        if self._audit is None:
            return
        try:
            self._audit.record_shadow(
                request_id=f"shadow-{uuid4().hex}",
                routing_key_hash=_sha256_hex(task.request.routing_key),
                query_hash=_sha256_hex(task.request.query),
                legacy_chunk_ids=tuple(item.chunk.id for item in task.legacy_hits),
                qwen_chunk_ids=qwen_chunk_ids,
                legacy_ms=task.legacy_ms,
                qwen_ms=qwen_ms,
                status=status,
                fallback_reason=fallback_reason,
            )
        except BaseException:
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
        close_timeout_seconds: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
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
        self._circuit = _CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_interval_seconds=reset_interval_seconds,
            monotonic=monotonic,
        )
        self.shadow_queue = ShadowQueue(
            hybrid=hybrid,
            audit=audit,
            max_size=shadow_queue_size,
            close_timeout_seconds=close_timeout_seconds,
            monotonic=monotonic,
        )

    def close(self) -> None:
        self.shadow_queue.close()

    def uses_qwen(self, routing_key: str) -> bool:
        percentage = (
            self._shadow_percent if self.mode is RetrievalMode.SHADOW else self._canary_percent
        )
        return stable_percentage_bucket(routing_key) < percentage

    def search(self, request: RetrievalRequest) -> RoutedRetrievalOutcome:
        if not isinstance(request, RetrievalRequest):
            raise TypeError("request must be a RetrievalRequest")
        if self.mode is RetrievalMode.LEGACY:
            return self._legacy(request)
        if self.mode is RetrievalMode.SHADOW:
            legacy = self._legacy(request)
            if self.uses_qwen(request.routing_key):
                self.shadow_queue.submit(request, legacy.hits, legacy.stage_ms["legacy"])
            return legacy
        if not self.uses_qwen(request.routing_key):
            return self._legacy(request)

        permit = self._circuit.acquire()
        if permit is None:
            return self._legacy(request, fallback_reason="circuit_open")
        try:
            qwen = self._hybrid.retrieve(request)
        except BaseException as error:
            self._circuit.record_failure(permit)
            return self._legacy(request, fallback_reason=_fallback_code(error))
        self._circuit.record_success(permit)
        if qwen.hits:
            return RoutedRetrievalOutcome(
                mode=RetrievalMode.QWEN3,
                hits=tuple(qwen.hits),
                stage_ms=dict(qwen.stage_ms),
            )
        legacy = self._legacy(request)
        if legacy.hits:
            return RoutedRetrievalOutcome(
                mode=RetrievalMode.LEGACY,
                hits=legacy.hits,
                stage_ms=legacy.stage_ms,
                fallback_reason="qwen_empty_legacy_nonempty",
            )
        return RoutedRetrievalOutcome(
            mode=RetrievalMode.QWEN3,
            hits=(),
            stage_ms=dict(qwen.stage_ms),
        )

    def _legacy(
        self,
        request: RetrievalRequest,
        *,
        fallback_reason: str | None = None,
    ) -> RoutedRetrievalOutcome:
        started = self._monotonic()
        hits = tuple(self._legacy_search(request.query, request.limit))
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


def _fallback_code(error: BaseException) -> str:
    if isinstance(error, HybridRetrievalTimeout):
        return "qwen_timeout"
    if isinstance(error, EmbeddingServiceError):
        return "embedding_unavailable"
    if isinstance(error, (RerankerBusy, RerankerServiceError)):
        return "reranker_unavailable"
    return "hybrid_unavailable"


__all__ = [
    "RetrievalRouter",
    "RoutedRetrievalOutcome",
    "ShadowQueue",
    "ShadowQueueCloseError",
    "stable_percentage_bucket",
]
