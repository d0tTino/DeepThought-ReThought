import asyncio
import sys
import types

import pytest

# Stub nats modules required by Publisher import in GoalScheduler
fake_nats = types.ModuleType("nats")
fake_nats.aio = types.ModuleType("aio")
fake_nats.aio.client = types.ModuleType("client")
fake_nats.js = types.ModuleType("js")
fake_nats.js.client = types.ModuleType("client")
fake_nats.errors = types.ModuleType("errors")
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_nats.aio.client)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_nats.js.client)
sys.modules.setdefault("nats.errors", fake_nats.errors)

from deepthought.goal_scheduler import GoalScheduler
from deepthought.eda.events import EventSubjects


class DummyPublisher:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))


@pytest.mark.asyncio
async def test_goals_emitted_periodically():
    sched = GoalScheduler()
    pub = DummyPublisher()
    sched.start(pub, interval=0.05)
    try:
        sched.add_goal("first", priority=1)
        await asyncio.sleep(0.1)
        assert [p.goal for _, p in pub.published] == ["first"]

        sched.add_goal("second", priority=2)
        await asyncio.sleep(0.1)
        assert [p.goal for _, p in pub.published] == ["first", "second"]
        assert all(s == EventSubjects.BDI_INTENTION for s, _ in pub.published)
    finally:
        await sched.stop()
