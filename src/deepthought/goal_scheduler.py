from __future__ import annotations

"""Simple priority-based goal scheduler.

The scheduler now tracks relationships between goals and their sub-goals and
records completion metadata for later analysis.
"""

import asyncio
import contextlib
import heapq
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, List, Optional

from .eda.events import BDIIntentionPayload, EventSubjects
from .eda.publisher import Publisher


@dataclass(order=True)
class ScheduledGoal:
    priority: int
    goal: str = field(compare=False)
    intention_id: Optional[int] = field(default=None, compare=False)
    sub_goals: List[str] = field(default_factory=list, compare=False)


@dataclass
class GoalRecord:
    """Metadata recorded for each persisted goal."""

    sub_goal_ids: List[int] = field(default_factory=list)
    completed_at: datetime | None = None
    outcome: str | None = None


from .services.db_manager import DBManager


class GoalScheduler:
    """Maintain a priority queue of goals."""

    def __init__(self, db_manager: DBManager | None = None) -> None:
        self._heap: List[ScheduledGoal] = []
        self._db_manager = db_manager
        self._load_task: asyncio.Task | None = None
        self._publish_task: asyncio.Task | None = None
        self._publisher: Publisher | None = None
        self._interval = 1.0
        self._records: Dict[int, GoalRecord] = {}
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

    def add_goal(self, goal: str, priority: int, sub_goals: Optional[List[str]] = None) -> None:
        """Schedule ``goal`` with ``priority`` (higher runs first)."""
        heapq.heappush(
            self._heap,
            ScheduledGoal(-priority, goal, None, list(sub_goals or [])),
        )

    async def queue_intention(self, goal: str, priority: int, sub_goals: Optional[List[str]] = None) -> int | None:
        """Schedule and persist an intention."""
        intention_id: int | None = None
        if self._db_manager is not None:
            intention_id = await self._db_manager.add_intention(goal, priority)
        heapq.heappush(
            self._heap,
            ScheduledGoal(-priority, goal, intention_id, list(sub_goals or [])),
        )
        if intention_id is not None:
            self._records.setdefault(intention_id, GoalRecord())
        return intention_id

    async def queue_sub_goals(self, sub_goals: List[str], priority: int) -> List[int | None]:
        """Persist and enqueue each ``sub_goals`` with ``priority``."""
        ids: List[int | None] = []
        for sub in sub_goals:
            ids.append(await self.queue_intention(sub, priority))
        return ids

    async def expand_goal(self, goal: str, priority: int | None = None) -> List[int | None]:
        """Expand a queued goal into its sub-goals."""
        for idx, scheduled in enumerate(self._heap):
            if scheduled.goal == goal:
                self._heap.pop(idx)
                heapq.heapify(self._heap)
                if self._db_manager is not None and scheduled.intention_id is not None:
                    await self._db_manager.mark_intention_done(scheduled.intention_id)
                prio = priority if priority is not None else -scheduled.priority
                ids = await self.queue_sub_goals(scheduled.sub_goals, prio)
                if scheduled.intention_id is not None:
                    record = self._records.setdefault(scheduled.intention_id, GoalRecord())
                    record.sub_goal_ids = [i for i in ids if i is not None]
                return ids
        return []

    def record_result(self, goal_id: int, outcome: str) -> None:
        """Store ``outcome`` and completion timestamp for ``goal_id``."""
        record = self._records.setdefault(goal_id, GoalRecord())
        record.completed_at = datetime.now(UTC)
        record.outcome = outcome

    def get_record(self, goal_id: int) -> GoalRecord | None:
        """Retrieve stored metadata for ``goal_id`` if present."""
        return self._records.get(goal_id)

    async def load_pending_intentions(self) -> int:
        """Load pending intentions from the database into the queue."""
        if self._db_manager is None:
            return 0
        rows = await self._db_manager.list_pending_intentions()
        for iid, goal, priority in rows:
            heapq.heappush(self._heap, ScheduledGoal(-priority, goal, iid))
            self._records.setdefault(iid, GoalRecord())
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
                loop.create_task(self._db_manager.mark_intention_done(scheduled.intention_id))
            except RuntimeError:
                asyncio.run(self._db_manager.mark_intention_done(scheduled.intention_id))
        return BDIIntentionPayload(goal=scheduled.goal, priority=-scheduled.priority)

    async def emit_next_intention(self, publisher: Publisher) -> bool:
        """Publish the next intention using ``publisher``.

        Returns ``True`` if an intention was published."""
        intention = self.next_intention()
        if intention is None:
            return False
        await publisher.publish(EventSubjects.BDI_INTENTION, intention, use_jetstream=True)
        return True

    async def publish_intentions(self, publisher: Publisher) -> int:
        """Publish all queued goals as BDI intentions.

        Returns the number of published intentions.
        """
        count = 0
        while await self.emit_next_intention(publisher):
            count += 1
        return count

    def start(self, publisher: Publisher, interval: float = 1.0) -> None:
        """Start background task to periodically publish intentions."""
        if self._publish_task is not None:
            return
        self._publisher = publisher
        self._interval = interval
        loop = asyncio.get_running_loop()
        self._publish_task = loop.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background publishing task."""
        if self._publish_task is None:
            return
        self._publish_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._publish_task
        self._publish_task = None
        self._publisher = None

    async def _run(self) -> None:
        if self._publisher is None:
            return
        try:
            while True:
                await self.publish_intentions(self._publisher)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass
