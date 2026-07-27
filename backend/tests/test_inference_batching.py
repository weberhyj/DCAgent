from __future__ import annotations

import asyncio
import gc
import threading
import unittest

from app.inference_batching import DynamicBatcher, InferenceQueueFull


class DynamicBatcherTest(unittest.IsolatedAsyncioTestCase):
    async def test_merges_requests_and_splits_results_in_original_order(self) -> None:
        observed: list[list[int]] = []

        def process(items: list[int]) -> list[int]:
            observed.append(items)
            return [item * 10 for item in items]

        batcher = DynamicBatcher(process, max_items=8, max_queue_items=16, wait_ms=10)
        await batcher.start()
        left, right = await asyncio.gather(batcher.submit([1, 2]), batcher.submit([3]))
        await batcher.close()
        self.assertEqual(left, [10, 20])
        self.assertEqual(right, [30])
        self.assertEqual(observed, [[1, 2, 3]])

    async def test_rejects_immediately_when_queue_is_full(self) -> None:
        batcher = DynamicBatcher(lambda items: items, max_items=1, max_queue_items=1, wait_ms=50)
        await batcher.start()
        first = asyncio.create_task(batcher.submit([1]))
        await asyncio.sleep(0)
        with self.assertRaises(InferenceQueueFull):
            await batcher.submit([2])
        await first
        await batcher.close()

    async def test_cancelled_waiter_keeps_capacity_until_worker_finishes(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def process(items: list[int]) -> list[int]:
            entered.set()
            release.wait(timeout=2)
            return items

        batcher = DynamicBatcher(process, max_items=1, max_queue_items=1, wait_ms=0)
        await batcher.start()
        waiter = asyncio.create_task(batcher.submit([1]))
        await asyncio.to_thread(entered.wait, 2)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        with self.assertRaises(InferenceQueueFull):
            await batcher.submit([2])
        release.set()
        await batcher.close()
        self.assertEqual(batcher.reserved_items, 0)

    async def test_close_drains_accepted_work_and_joins_worker(self) -> None:
        batcher = DynamicBatcher(
            lambda items: [item + 1 for item in items], max_items=8, max_queue_items=8, wait_ms=50
        )
        await batcher.start()
        result = asyncio.create_task(batcher.submit([1, 2]))
        await asyncio.sleep(0)
        await batcher.close()
        self.assertEqual(await result, [2, 3])

    async def test_cancelled_failed_request_consumes_internal_future_exception(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        observed_contexts: list[dict[str, object]] = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: observed_contexts.append(context))

        def process(items: list[int]) -> list[int]:
            entered.set()
            release.wait(timeout=2)
            raise RuntimeError("backend failed")

        try:
            batcher = DynamicBatcher(process, max_items=1, max_queue_items=1, wait_ms=0)
            await batcher.start()
            waiter = asyncio.create_task(batcher.submit([1]))
            await asyncio.to_thread(entered.wait, 2)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            release.set()
            await batcher.close()
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        self.assertFalse(
            any(
                context.get("message") == "Future exception was never retrieved"
                for context in observed_contexts
            ),
            observed_contexts,
        )
