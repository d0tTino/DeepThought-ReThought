"""Async helper for queuing responses per channel with cooldowns."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Dict, Hashable

logger = logging.getLogger(__name__)

SendCallable = Callable[[], Awaitable[None]]


@dataclass
class _QueuedResponse:
    created_at: float
    sender: SendCallable


class ResponseQueue:
    """Maintain per-channel queues for sending responses.

    The queue enforces a cooldown between responses per ``key`` (for example,
    a Discord channel ID). When multiple responses queue up for a key, stale
    entries are dropped in favor of newer ones to avoid sending outdated
    messages.
    """

    def __init__(self, cooldown: float, *, stale_after: float = 30.0) -> None:
        self.cooldown = cooldown
        self.stale_after = stale_after
        self._queues: Dict[Hashable, Deque[_QueuedResponse]] = {}
        self._last_sent: Dict[Hashable, float] = {}
        self._tasks: Dict[Hashable, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, key: Hashable, sender: SendCallable) -> None:
        """Queue ``sender`` for ``key`` and start a worker if needed."""

        async with self._lock:
            queue = self._queues.setdefault(key, deque())
            queue.append(_QueuedResponse(time.monotonic(), sender))
            worker = self._tasks.get(key)
            if worker is None or worker.done():
                self._tasks[key] = asyncio.create_task(self._run_queue(key))

    async def stop(self) -> None:
        """Cancel all workers and clear queues."""

        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
            self._queues.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_queue(self, key: Hashable) -> None:
        while True:
            async with self._lock:
                queue = self._queues.get(key)
                if not queue:
                    self._tasks.pop(key, None)
                    return

                now = time.monotonic()
                while len(queue) > 1 and now - queue[0].created_at > self.stale_after:
                    queue.popleft()

                if not queue:
                    continue

                ready_at = self._last_sent.get(key, 0.0) + self.cooldown
                delay = max(0.0, ready_at - now)

                # Defer sending if still in cooldown.
                if delay > 0:
                    # Sleep in short slices so new responses can be checked for staleness.
                    await asyncio.sleep(min(delay, 1.0))
                    # If new items were added while sleeping and the head is stale, loop drops it.
                    continue

                item = queue.popleft()

            # Drop stale items when newer messages exist in the queue.
            if (time.monotonic() - item.created_at > self.stale_after) and self._queues.get(key):
                continue

            try:
                await item.sender()
            except Exception:  # pragma: no cover - logging best effort
                logger.warning("Failed to send queued response for %s", key, exc_info=True)

            self._last_sent[key] = time.monotonic()
