import asyncio
from datetime import datetime

import pytest

from deepthought.harness import replay as replay_mod
from deepthought.harness.record import TraceEvent


class DummyAgent:
    def __init__(self):
        self.states = []

    async def act(self, state: str) -> str:
        self.states.append(state)
        return "ok"


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))
        return None


@pytest.mark.asyncio
async def test_replay_uses_timestamp_delta(monkeypatch):
    events = [
        TraceEvent(
            state="s1",
            action="a1",
            reward=0.0,
            latency=0.0,
            timestamp=datetime.utcnow(),
            timestamp_delta=0.0,
        ),
        TraceEvent(
            state="s2",
            action="a2",
            reward=0.0,
            latency=0.0,
            timestamp=datetime.utcnow(),
            timestamp_delta=1.0,
        ),
    ]
    agent = DummyAgent()
    publisher = DummyPublisher()

    slept = []

    async def fake_sleep(val):
        slept.append(val)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await replay_mod.replay(events, agent, publisher)

    assert slept == [0.0, 1.0]
    assert agent.states == ["s1", "s2"]
    assert publisher.published == [("chat.raw", "s1"), ("chat.raw", "s2")]