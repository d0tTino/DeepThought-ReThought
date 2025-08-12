import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("nats")

from deepthought.eda.events import BDIIntentionPayload, EventSubjects
from deepthought.eda.publisher import connect
from deepthought.goal_scheduler import GoalScheduler
from deepthought.services.scheduler import SchedulerService

pytest_plugins = ["tests.helpers"]
pytestmark = pytest.mark.nats


@pytest.mark.asyncio
async def test_goals_persist_and_publish(nats_server):
    publisher = await connect(nats_server)
    nc = publisher._nc
    received: list[BDIIntentionPayload] = []

    async def handler(msg):
        received.append(BDIIntentionPayload.from_json(msg.data.decode()))

    sub = await nc.subscribe(EventSubjects.BDI_INTENTION, cb=handler)
    await asyncio.sleep(0.1)

    sched = GoalScheduler()
    sched.add_goal("low", priority=1)
    sched.add_goal("high", priority=5)
    sched.add_goal("mid", priority=3)

    count = await sched.publish_intentions(publisher)
    await asyncio.sleep(0.5)

    await sub.unsubscribe()
    await nc.close()

    assert count == 3
    assert [p.goal for p in received] == ["high", "mid", "low"]
    assert sched.next_goal() is None


class DummyPublisher:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))


class DummyMemoryDAL:
    def __init__(self, interactions=None):
        self.interactions = interactions or []

    def get_recent_facts(self, count=3):
        return self.interactions[-count:]


class DummyGraphDAL:
    def add_entity(self, label, props):
        pass


@pytest.mark.asyncio
async def test_scheduler_loops_and_jitter(tmp_path):
    current = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now():
        return current

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        nonlocal current
        if seconds < 1:
            current += timedelta(seconds=seconds)
        await real_sleep(0)

    state_file = tmp_path / "state.json"
    pub = DummyPublisher()
    service = SchedulerService(
        pub,
        DummyMemoryDAL([]),
        DummyGraphDAL(),
        summary_interval=1000.0,
        daily_summary_interval=1000.0,
        chat_summary_interval=1000.0,
        now_func=now,
        sleep_func=fake_sleep,
        micro_tick_range=(0.01, 0.02),
        daily_standup_interval=0.03,
        weekly_planning_interval=0.04,
        state_file=str(state_file),
        jitter_fraction=0.0,
    )
    await service.start()
    await fake_sleep(0)
    await fake_sleep(0.1)
    await service.stop()

    micro_times = [
        datetime.fromisoformat(p.timestamp)
        for s, p in pub.published
        if s == EventSubjects.MICRO_TICK
    ]
    assert micro_times
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    next_run = datetime.fromisoformat(state["micro_tick"])
    diff = (next_run - micro_times[-1]).total_seconds()
    assert 0.01 <= diff <= 0.02
    subjects = [s for s, _ in pub.published]
    assert EventSubjects.DAILY_STANDUP in subjects
    assert EventSubjects.WEEKLY_PLANNING in subjects


@pytest.mark.asyncio
async def test_scheduler_persists_next_run(tmp_path):
    current = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now():
        return current

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        nonlocal current
        if seconds < 1:
            current += timedelta(seconds=seconds)
        await real_sleep(0)

    state_file = tmp_path / "state.json"
    pub = DummyPublisher()
    service = SchedulerService(
        pub,
        DummyMemoryDAL([]),
        DummyGraphDAL(),
        summary_interval=1000.0,
        daily_summary_interval=1000.0,
        chat_summary_interval=1000.0,
        now_func=now,
        sleep_func=fake_sleep,
        micro_tick_range=(0.05, 0.05),
        daily_standup_interval=1000.0,
        weekly_planning_interval=1000.0,
        state_file=str(state_file),
        jitter_fraction=0.0,
    )
    await service.start()
    await fake_sleep(0)
    await fake_sleep(0.06)
    await service.stop()

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    next_run = datetime.fromisoformat(state["micro_tick"])

    pub2 = DummyPublisher()
    service2 = SchedulerService(
        pub2,
        DummyMemoryDAL([]),
        DummyGraphDAL(),
        summary_interval=1000.0,
        daily_summary_interval=1000.0,
        chat_summary_interval=1000.0,
        now_func=now,
        sleep_func=fake_sleep,
        micro_tick_range=(0.05, 0.05),
        daily_standup_interval=1000.0,
        weekly_planning_interval=1000.0,
        state_file=str(state_file),
        jitter_fraction=0.0,
    )
    await service2.start()
    await fake_sleep((next_run - current).total_seconds() - 0.01)
    assert not [s for s, _ in pub2.published if s == EventSubjects.MICRO_TICK]
    await fake_sleep(0.02)
    await service2.stop()
    assert [s for s, _ in pub2.published if s == EventSubjects.MICRO_TICK]
