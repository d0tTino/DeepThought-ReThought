import asyncio
from datetime import datetime

import pytest

from deepthought.harness import replay as replay_mod
from deepthought.harness.record import TraceEvent


class DummyAgent:
    def __init__(self) -> None:
        self.states: list[str] = []

    async def act(self, state: str) -> str:
        self.states.append(state)
        return "ok"


class DummyPublisher:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    async def publish(
        self,
        subject: str,
        payload: str,
        use_jetstream: bool = True,
        timeout: float = 10.0,
    ):
        self.published.append((subject, payload))
        return None


@pytest.mark.asyncio
async def test_replay_uses_timestamp_delta(monkeypatch) -> None:
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

    slept: list[float] = []

    async def fake_sleep(val: float) -> None:
        slept.append(val)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await replay_mod.replay(events, agent, publisher)

    assert slept == [0.0, 1.0]
    assert agent.states == ["s1", "s2"]
    assert publisher.published == [("chat.raw", "s1"), ("chat.raw", "s2")]
