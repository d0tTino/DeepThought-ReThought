import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from deepthought.eda.events import EventSubjects
from deepthought.services.scheduler import SchedulerService


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))


class DummyMemoryDAL:
    def __init__(self, interactions):
        self.interactions = interactions

    def get_recent_facts(self, count=3):
        return self.interactions[-count:]


class DummyGraphDAL:
    def __init__(self):
        self.entities = []

    def add_entity(self, label, props):
        self.entities.append((label, props))


@pytest.mark.asyncio
async def test_scheduler_summary_and_reminder(monkeypatch):
    current = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now():
        return current

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        nonlocal current
        current += timedelta(seconds=seconds)
        await real_sleep(0)

    publisher = DummyPublisher()
    memory = DummyMemoryDAL(["hello world", "how are you"])
    graph = DummyGraphDAL()

    service = SchedulerService(
        publisher,
        memory,
        graph,
        summary_interval=2.0,
        now_func=now,
        sleep_func=fake_sleep,
    )

    await service.start()
    await fake_sleep(0)  # allow tasks to start
    service.schedule_reminder("ping", now() + timedelta(seconds=3), "r1")

    await fake_sleep(4)
    await service.stop()

    # Summary stored
    assert graph.entities
    label, props = graph.entities[0]
    assert label == "Note"
    assert "timestamp" in props

    # Reminder triggered
    assert publisher.published
    subj, payload = publisher.published[0]
    assert subj == EventSubjects.REMINDER_TRIGGERED
    assert payload.message == "ping"
    assert payload.reminder_id == "r1"


@pytest.mark.asyncio
async def test_scheduler_daily_summary(monkeypatch):
    current = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now():
        return current

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        nonlocal current
        current += timedelta(seconds=seconds)
        await real_sleep(0)

    publisher = DummyPublisher()
    memory = DummyMemoryDAL(["hello", "world", "another"])
    graph = DummyGraphDAL()

    service = SchedulerService(
        publisher,
        memory,
        graph,
        summary_interval=100.0,
        daily_summary_interval=2.0,
        now_func=now,
        sleep_func=fake_sleep,
    )

    await service.start()
    await fake_sleep(0)  # allow tasks to start
    await fake_sleep(3)
    await service.stop()

    daily = [e for e in graph.entities if e[0] == "DailySummary"]
    assert daily
    label, props = daily[0]
    assert label == "DailySummary"
    assert "timestamp" in props
