import pytest
import sys
import types

# Stub nats modules required by Publisher import in goal_scheduler
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
from deepthought.eda.events import EventSubjects, BDIIntentionPayload


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))


def test_next_intention_returns_payload():
    sched = GoalScheduler()
    sched.add_goal("alpha", priority=3)
    intention = sched.next_intention()
    assert isinstance(intention, BDIIntentionPayload)
    assert intention.goal == "alpha"
    assert intention.priority == 3


@pytest.mark.asyncio
async def test_publish_intentions(monkeypatch):
    sched = GoalScheduler()
    sched.add_goal("first", priority=5)
    sched.add_goal("second", priority=1)
    pub = DummyPublisher()
    count = await sched.publish_intentions(pub)
    assert count == 2
    assert len(pub.published) == 2
    subj, payload = pub.published[0]
    assert subj == EventSubjects.BDI_INTENTION
    assert payload.goal == "first"
    assert payload.priority == 5
