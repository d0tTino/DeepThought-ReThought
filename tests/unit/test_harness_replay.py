import asyncio
import json
from datetime import datetime

import pytest

import tools.replay as tools_replay
from deepthought.harness import replay as replay_mod
from deepthought.harness import trace as trace_mod
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


@pytest.mark.asyncio
async def test_trace_recorder_output_replays(monkeypatch, tmp_path):
    class DummyNATS:
        is_connected = True

    class DummyJS:
        pass

    class DummySubscriber:
        def __init__(self, *args, **kwargs):
            self.calls = []

        async def subscribe(self, *args, **kwargs):
            self.calls.append((args, kwargs))

        async def unsubscribe_all(self):
            self.calls.clear()

    class DummyMsg:
        def __init__(self, data: str) -> None:
            self.data = data.encode()
            self.acked = False

        async def ack(self):
            self.acked = True

    monkeypatch.setattr(trace_mod, "Subscriber", DummySubscriber)

    outfile = tmp_path / "trace.jsonl"
    recorder = trace_mod.TraceRecorder(DummyNATS(), DummyJS(), str(outfile))

    now = datetime.utcnow().isoformat()
    msg1 = DummyMsg(
        json.dumps(
            {
                "state": "s1",
                "action": "a1",
                "reward": 0.0,
                "latency": 0.0,
                "timestamp": now,
                "timestamp_delta": 0.0,
            }
        )
    )
    msg2 = DummyMsg(
        json.dumps(
            {
                "state": "s2",
                "action": "a2",
                "reward": 0.0,
                "latency": 0.0,
                "timestamp": now,
                "timestamp_delta": 1.0,
            }
        )
    )

    await recorder._handle_input(msg1)
    await recorder._handle_input(msg2)

    events = tools_replay._load_trace(outfile)

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
