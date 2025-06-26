import time
from datetime import datetime

import pytest

from deepthought.harness import replay as harness_replay
from deepthought.harness.record import TraceEvent


class DummyAgent:
    def __init__(self):
        self.calls = []

    async def act(self, state: str) -> str:
        self.calls.append((state, time.monotonic()))
        return "ok"


@pytest.mark.asyncio
async def test_replay_order_and_timing():
    now = datetime.utcnow()
    trace = [
        TraceEvent(state="s1", action="a1", reward=0.0, latency=0.05, timestamp=now),
        TraceEvent(state="s2", action="a2", reward=0.0, latency=0.05, timestamp=now),
        TraceEvent(state="s3", action="a3", reward=0.0, latency=0.0, timestamp=now),
    ]
    agent = DummyAgent()
    start = time.monotonic()
    await harness_replay.replay(trace, agent)
    states = [c[0] for c in agent.calls]
    times = [c[1] for c in agent.calls]

    assert states == ["s1", "s2", "s3"]
    assert times[1] - times[0] >= trace[0].latency
    assert times[2] - times[1] >= trace[1].latency
    assert times[0] >= start
