from __future__ import annotations

"""Simple priority-based goal scheduler."""

import heapq
from dataclasses import dataclass, field
from typing import List


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
