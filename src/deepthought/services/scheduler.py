"""Simple background scheduler for summaries and reminders."""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, List, Optional

from ..eda.events import EventSubjects, ReminderTriggeredPayload, TickPayload
from ..eda.publisher import Publisher
from ..graph.dal import GraphDAL
from ..motivate.caption import summarise_message
from .file_graph_dal import FileGraphDAL


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
        chat_summary_interval: float = 300.0,
        summary_db=None,
        now_func: Callable[[], datetime] | None = None,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
        micro_tick_range: tuple[float, float] | None = None,
        daily_standup_interval: float | None = None,
        weekly_planning_interval: float | None = None,
        state_file: str | None = None,
        jitter_fraction: float = 0.1,
    ) -> None:
        self._publisher = publisher
        self._memory_dal = memory_dal
        self._graph_dal = graph_dal
        self._summary_interval = summary_interval
        self._daily_summary_interval = daily_summary_interval
        self._now = now_func or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep_func
        self._summary_db = summary_db
        self._chat_summary_interval = chat_summary_interval
        self._micro_tick_range = micro_tick_range
        self._standup_interval = daily_standup_interval
        self._weekly_planning_interval = weekly_planning_interval
        self._jitter_fraction = jitter_fraction
        self._state_file = state_file
        self._next_runs = self._load_state()
        now = self._now()
        if self._micro_tick_range is not None:
            self._next_runs.setdefault(
                "micro_tick", now + timedelta(seconds=self._next_micro_interval())
            )
        if self._standup_interval is not None:
            self._next_runs.setdefault(
                "daily_standup",
                now + timedelta(seconds=self._jittered(self._standup_interval)),
            )
        if self._weekly_planning_interval is not None:
            self._next_runs.setdefault(
                "weekly_planning",
                now + timedelta(seconds=self._jittered(self._weekly_planning_interval)),
            )
        if self._next_runs:
            self._persist_state()
        self._reminders: List[ScheduledReminder] = []
        self._running = False
        self._summary_task: Optional[asyncio.Task] = None
        self._reminder_task: Optional[asyncio.Task] = None
        self._daily_summary_task: Optional[asyncio.Task] = None
        self._chat_summary_task: Optional[asyncio.Task] = None
        self._goal_task: Optional[asyncio.Task] = None
        self._micro_tick_task: Optional[asyncio.Task] = None
        self._standup_task: Optional[asyncio.Task] = None
        self._planning_task: Optional[asyncio.Task] = None

    def schedule_reminder(self, message: str, when: datetime, reminder_id: str) -> None:
        """Schedule a reminder message for the future."""
        self._reminders.append(ScheduledReminder(message, when, reminder_id))
        self._reminders.sort(key=lambda r: r.when)

    async def start(self) -> bool:
        self._running = True
        self._summary_task = asyncio.create_task(self._summary_loop())
        self._daily_summary_task = asyncio.create_task(self._daily_summary_loop())
        self._reminder_task = asyncio.create_task(self._reminder_loop())
        self._chat_summary_task = asyncio.create_task(self._chat_summary_loop())
        self._goal_task = asyncio.create_task(self._goal_loop())
        if self._micro_tick_range is not None:
            self._micro_tick_task = asyncio.create_task(self._micro_tick_loop())
        if self._standup_interval is not None:
            self._standup_task = asyncio.create_task(self._standup_loop())
        if self._weekly_planning_interval is not None:
            self._planning_task = asyncio.create_task(self._planning_loop())
        return True

    async def stop(self) -> None:
        self._running = False
        tasks = [
            t
            for t in [
                self._summary_task,
                self._daily_summary_task,
                self._reminder_task,
                self._chat_summary_task,
                self._goal_task,
                self._micro_tick_task,
                self._standup_task,
                self._planning_task,
            ]
            if t
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def __aenter__(self) -> "SchedulerService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def _next_micro_interval(self) -> float:
        return random.uniform(*self._micro_tick_range)

    def _jittered(self, base: float) -> float:
        span = base * self._jitter_fraction
        return base + random.uniform(-span, span)

    def _load_state(self) -> dict:
        if not self._state_file or not os.path.exists(self._state_file):
            return {}
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: datetime.fromisoformat(v) for k, v in data.items()}
        except Exception:
            return {}

    def _persist_state(self) -> None:
        if not self._state_file:
            return
        data = {k: v.isoformat() for k, v in self._next_runs.items()}
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

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
        from examples.social_graph_bot import generate_reflection  # local import to avoid circular

        facts = self._memory_dal.get_recent_facts(50)
        text = " ".join(facts)
        summary = generate_reflection(text)
        self._graph_dal.add_entity(
            "DailySummary",
            {"text": summary, "timestamp": self._now().isoformat()},
        )

    async def _generate_chat_summary(self) -> None:
        facts = self._memory_dal.get_recent_facts(20)
        text = " ".join(facts)
        summary = summarise_message(text, max_words=20)
        self._graph_dal.add_entity(
            "ChatSummary",
            {"text": summary, "timestamp": self._now().isoformat()},
        )
        if self._summary_db is not None:
            goal = {
                "due": (self._now() + timedelta(seconds=60)).isoformat(),
                "goal": f"Reflect on: {summary}",
            }
            await self._summary_db.add_summary_goal(0, goal, summary)

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

    async def _chat_summary_loop(self) -> None:
        while self._running:
            await self._sleep(self._chat_summary_interval)
            await self._generate_chat_summary()

    async def _micro_tick_loop(self) -> None:
        while self._running:
            wait = (self._next_runs["micro_tick"] - self._now()).total_seconds()
            if wait > 0:
                await self._sleep(wait)
            now = self._now()
            payload = TickPayload(timestamp=now.isoformat())
            await self._publisher.publish(
                EventSubjects.MICRO_TICK,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            self._next_runs["micro_tick"] = self._now() + timedelta(
                seconds=self._next_micro_interval()
            )
            self._persist_state()

    async def _standup_loop(self) -> None:
        while self._running:
            wait = (self._next_runs["daily_standup"] - self._now()).total_seconds()
            if wait > 0:
                await self._sleep(wait)
            now = self._now()
            payload = TickPayload(timestamp=now.isoformat())
            await self._publisher.publish(
                EventSubjects.DAILY_STANDUP,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            self._next_runs["daily_standup"] = self._now() + timedelta(
                seconds=self._jittered(self._standup_interval)
            )
            self._persist_state()

    async def _planning_loop(self) -> None:
        while self._running:
            wait = (self._next_runs["weekly_planning"] - self._now()).total_seconds()
            if wait > 0:
                await self._sleep(wait)
            now = self._now()
            payload = TickPayload(timestamp=now.isoformat())
            await self._publisher.publish(
                EventSubjects.WEEKLY_PLANNING,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            self._next_runs["weekly_planning"] = self._now() + timedelta(
                seconds=self._jittered(self._weekly_planning_interval)
            )
            self._persist_state()

    async def _goal_loop(self) -> None:
        if self._summary_db is None:
            while self._running:
                await self._sleep(1.0)
            return
        while self._running:
            await self._sleep(1.0)
            rows = await self._summary_db.list_pending_summary_goals()
            now = self._now()
            for task_id, _uid, ctx_json, prompt in rows:
                try:
                    ctx = json.loads(ctx_json)
                except Exception:
                    continue
                due = ctx.get("due")
                message = ctx.get("goal")
                if not due or not message:
                    continue
                try:
                    due_dt = datetime.fromisoformat(due)
                except ValueError:
                    continue
                if due_dt <= now:
                    payload = ReminderTriggeredPayload(
                        message=message,
                        reminder_id=str(task_id),
                        timestamp=now.isoformat(),
                    )
                    await self._publisher.publish(
                        EventSubjects.REMINDER_TRIGGERED,
                        payload,
                        use_jetstream=True,
                        timeout=10.0,
                    )
                    await self._summary_db.mark_summary_goal_done(task_id)
