"""Simple background scheduler for summaries and reminders."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, List, Optional

from ..eda.events import EventSubjects, ReminderTriggeredPayload
from ..eda.publisher import Publisher
from ..motivate.caption import summarise_message
from examples.social_graph_bot import generate_reflection
from .file_graph_dal import FileGraphDAL
from ..graph.dal import GraphDAL


@dataclass
class ScheduledReminder:
    """Internal structure to hold reminder data."""

    message: str
    when: datetime
    reminder_id: str


class SchedulerService:
    """Background tasks for summaries and scheduled reminders."""

    def __init__(
        self,
        publisher: Publisher,
        memory_dal: FileGraphDAL,
        graph_dal: GraphDAL,
        summary_interval: float = 60.0,
        daily_summary_interval: float = 24 * 60 * 60.0,
        now_func: Callable[[], datetime] | None = None,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._publisher = publisher
        self._memory_dal = memory_dal
        self._graph_dal = graph_dal
        self._summary_interval = summary_interval
        self._daily_summary_interval = daily_summary_interval
        self._now = now_func or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep_func
        self._reminders: List[ScheduledReminder] = []
        self._running = False
        self._summary_task: Optional[asyncio.Task] = None
        self._reminder_task: Optional[asyncio.Task] = None
        self._daily_summary_task: Optional[asyncio.Task] = None

    def schedule_reminder(self, message: str, when: datetime, reminder_id: str) -> None:
        """Schedule a reminder message for the future."""
        self._reminders.append(ScheduledReminder(message, when, reminder_id))
        self._reminders.sort(key=lambda r: r.when)

    async def start(self) -> bool:
        self._running = True
        self._summary_task = asyncio.create_task(self._summary_loop())
        self._daily_summary_task = asyncio.create_task(self._daily_summary_loop())
        self._reminder_task = asyncio.create_task(self._reminder_loop())
        return True

    async def stop(self) -> None:
        self._running = False
        tasks = [
            t
            for t in [self._summary_task, self._daily_summary_task, self._reminder_task]
            if t
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _summary_loop(self) -> None:
        while self._running:
            await self._sleep(self._summary_interval)
            await self._generate_summary()

    async def _daily_summary_loop(self) -> None:
        while self._running:
            await self._sleep(self._daily_summary_interval)
            await self._generate_daily_summary()

    async def _generate_summary(self) -> None:
        facts = self._memory_dal.get_recent_facts()
        text = " ".join(facts)
        summary = summarise_message(text, max_words=10)
        self._graph_dal.add_entity(
            "Note",
            {"text": summary, "timestamp": self._now().isoformat()},
        )

    async def _generate_daily_summary(self) -> None:
        facts = self._memory_dal.get_recent_facts(50)
        text = " ".join(facts)
        summary = generate_reflection(text)
        self._graph_dal.add_entity(
            "DailySummary",
            {"text": summary, "timestamp": self._now().isoformat()},
        )

    async def _reminder_loop(self) -> None:
        while self._running:
            await self._sleep(1.0)
            now = self._now()
            due: List[ScheduledReminder] = [r for r in self._reminders if r.when <= now]
            self._reminders = [r for r in self._reminders if r.when > now]
            for r in due:
                payload = ReminderTriggeredPayload(
                    message=r.message,
                    reminder_id=r.reminder_id,
                    timestamp=now.isoformat(),
                )
                await self._publisher.publish(
                    EventSubjects.REMINDER_TRIGGERED,
                    payload,
                    use_jetstream=True,
                    timeout=10.0,
                )

