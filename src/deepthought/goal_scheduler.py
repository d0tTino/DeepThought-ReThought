from __future__ import annotations

"""Simple priority-based goal scheduler."""

import heapq
from dataclasses import dataclass, field
from typing import List

from .eda.events import BDIIntentionPayload, EventSubjects
from .eda.publisher import Publisher


@dataclass(order=True)
class ScheduledGoal:
    priority: int
    goal: str = field(compare=False)


class GoalScheduler:
    """Maintain a priority queue of goals."""

    def __init__(self) -> None:
        self._heap: List[ScheduledGoal] = []

    def add_goal(self, goal: str, priority: int) -> None:
        """Schedule ``goal`` with ``priority`` (higher runs first)."""
        heapq.heappush(self._heap, ScheduledGoal(-priority, goal))

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
            await publisher.publish(EventSubjects.BDI_INTENTION, intention, use_jetstream=True)
            count += 1
        return count
