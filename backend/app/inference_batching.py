"""Bounded dynamic batching for private CPU inference services."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass


class InferenceQueueFull(RuntimeError):
    """Raised when accepting a request would exceed reserved item capacity."""


class InferenceBatcherClosed(RuntimeError):
    """Raised when work is submitted after shutdown begins."""


@dataclass(slots=True)
class _Pending[InputT, OutputT]:
    items: list[InputT]
    future: asyncio.Future[list[OutputT]]


class DynamicBatcher[InputT, OutputT]:
    """Merge accepted requests while preserving request boundaries and order.

    Queue capacity is accounted in individual inference items, not requests.  An
    accepted request retains its reservation even if its waiter is cancelled;
    capacity is released only after the worker finishes that request's batch.
    """

    def __init__(
        self,
        process: Callable[[list[InputT]], Sequence[OutputT]],
        *,
        max_items: int,
        max_queue_items: int,
        wait_ms: float,
    ) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if max_queue_items <= 0:
            raise ValueError("max_queue_items must be positive")
        if wait_ms < 0:
            raise ValueError("wait_ms must not be negative")
        self._process = process
        self._max_items = max_items
        self._max_queue_items = max_queue_items
        self._wait_seconds = wait_ms / 1000.0
        self._pending: deque[_Pending[InputT, OutputT]] = deque()
        self._reserved_items = 0
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

    @property
    def reserved_items(self) -> int:
        return self._reserved_items

    async def start(self) -> None:
        async with self._condition:
            if self._worker is not None:
                return
            if self._closing:
                raise InferenceBatcherClosed("inference batcher is closed")
            self._worker = asyncio.create_task(self._run(), name="dynamic-inference-batcher")

    async def submit(self, items: Sequence[InputT]) -> list[OutputT]:
        submitted = list(items)
        if not submitted:
            return []
        if len(submitted) > self._max_items:
            raise ValueError("request item count exceeds maximum batch size")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[OutputT]] = loop.create_future()
        async with self._condition:
            if self._closing:
                raise InferenceBatcherClosed("inference batcher is closed")
            if self._worker is None:
                raise RuntimeError("inference batcher has not been started")
            if self._reserved_items + len(submitted) > self._max_queue_items:
                raise InferenceQueueFull("inference queue is full")
            self._reserved_items += len(submitted)
            self._pending.append(_Pending(submitted, future))
            self._condition.notify()
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(_consume_future_exception)
            raise

    async def close(self) -> None:
        async with self._condition:
            self._closing = True
            worker = self._worker
            self._condition.notify_all()
        if worker is not None:
            await worker

    async def _run(self) -> None:
        while True:
            batch = await self._take_batch()
            if not batch:
                return
            flat_items = [item for request in batch for item in request.items]
            error: BaseException | None = None
            results: list[OutputT] = []
            try:
                raw_results = await asyncio.to_thread(self._process, flat_items)
                results = list(raw_results)
                if len(results) != len(flat_items):
                    raise ValueError("inference processor returned an item count mismatch")
            except BaseException as caught:
                error = caught

            offset = 0
            for request in batch:
                count = len(request.items)
                if not request.future.done():
                    if error is None:
                        request.future.set_result(results[offset : offset + count])
                    else:
                        request.future.set_exception(error)
                offset += count

            async with self._condition:
                self._reserved_items -= len(flat_items)
                self._condition.notify_all()

    async def _take_batch(self) -> list[_Pending[InputT, OutputT]]:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._pending) or self._closing)
            if not self._pending:
                return []

            deadline = asyncio.get_running_loop().time() + self._wait_seconds
            while not self._closing and self._batchable_item_count() < self._max_items:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError:
                    break

            selected: list[_Pending[InputT, OutputT]] = []
            item_count = 0
            while self._pending:
                candidate = self._pending[0]
                if selected and item_count + len(candidate.items) > self._max_items:
                    break
                if len(candidate.items) > self._max_items:
                    raise RuntimeError("accepted request exceeds maximum batch size")
                selected.append(self._pending.popleft())
                item_count += len(candidate.items)
                if item_count >= self._max_items:
                    break
            return selected

    def _batchable_item_count(self) -> int:
        total = 0
        for request in self._pending:
            if total and total + len(request.items) > self._max_items:
                break
            total += len(request.items)
            if total >= self._max_items:
                break
        return total


def _consume_future_exception(future: asyncio.Future[object]) -> None:
    if not future.cancelled():
        future.exception()


__all__ = ["DynamicBatcher", "InferenceBatcherClosed", "InferenceQueueFull"]
