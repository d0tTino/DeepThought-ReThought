from __future__ import annotations

"""Simple priority-based goal scheduler."""

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import List, Optional

from .eda.events import BDIIntentionPayload, EventSubjects
from .eda.publisher import Publisher


@dataclass(order=True)
class ScheduledGoal:
    priority: int
    goal: str = field(compare=False)
    intention_id: Optional[int] = field(default=None, compare=False)


from .services.db_manager import DBManager


class GoalScheduler:
    """Maintain a priority queue of goals."""

    def __init__(self, db_manager: DBManager | None = None) -> None:
        self._heap: List[ScheduledGoal] = []
        self._db_manager = db_manager
        self._load_task: asyncio.Task | None = None
        if self._db_manager is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                self._load_task = loop.create_task(self.load_pending_intentions())
            else:  # pragma: no cover - typically executed outside tests
                asyncio.run(self.load_pending_intentions())
                self._load_task = None

    async def wait_loaded(self) -> None:
        """Wait for pending intentions to be loaded on startup."""
        if self._load_task is not None:
            await self._load_task
            self._load_task = None

    def add_goal(self, goal: str, priority: int) -> None:
        """Schedule ``goal`` with ``priority`` (higher runs first)."""
        heapq.heappush(self._heap, ScheduledGoal(-priority, goal))

    async def queue_intention(self, goal: str, priority: int) -> int | None:
        """Schedule and persist an intention."""
        intention_id: int | None = None
        if self._db_manager is not None:
            intention_id = await self._db_manager.add_intention(goal, priority)
        heapq.heappush(self._heap, ScheduledGoal(-priority, goal, intention_id))
        return intention_id

    async def load_pending_intentions(self) -> int:
        """Load pending intentions from the database into the queue."""
        if self._db_manager is None:
            return 0
        rows = await self._db_manager.list_pending_intentions()
        for iid, goal, priority in rows:
            heapq.heappush(self._heap, ScheduledGoal(-priority, goal, iid))
        return len(rows)

    def next_goal(self) -> str | None:
        """Pop the highest priority goal or ``None``."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap).goal

    def next_intention(self) -> BDIIntentionPayload | None:
        """Pop the next goal and convert it to a BDI intention payload."""
        if not self._heap:
            return None
        scheduled = heapq.heappop(self._heap)
        if self._db_manager is not None and scheduled.intention_id is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._db_manager.mark_intention_done(scheduled.intention_id)
                )
            except RuntimeError:
                asyncio.run(
                    self._db_manager.mark_intention_done(scheduled.intention_id)
                )
        return BDIIntentionPayload(goal=scheduled.goal, priority=-scheduled.priority)

    async def publish_intentions(self, publisher: Publisher) -> int:
        """Publish all queued goals as BDI intentions.

        Returns the number of published intentions.
        """
        count = 0
        while self._heap:
            intention = self.next_intention()
            if intention is None:
                break
            await publisher.publish(
                EventSubjects.BDI_INTENTION, intention, use_jetstream=True
            )
            count += 1
        return count
