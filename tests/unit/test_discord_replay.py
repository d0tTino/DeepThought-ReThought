import asyncio
import json
from pathlib import Path
import types

import pytest

import tools.discord_replay as dr


class DummyMsg:
    def __init__(self, text: str) -> None:
        self.data = json.dumps({"final_response": text}).encode()

    async def ack(self) -> None:
        pass


class DummySubscriber:
    instance = None

    def __init__(self, *args, **kwargs):
        DummySubscriber.instance = self
        self.handler = None

    async def subscribe(self, *args, **kwargs):
        self.handler = kwargs.get("handler") or args[1]
        return True

    async def unsubscribe_all(self):
        pass


class DummyPublisher:
    responses = []

    def __init__(self, *args, **kwargs):
        pass

    async def publish(self, subject, payload, **kwargs):
        response = DummyPublisher.responses.pop(0)
        await DummySubscriber.instance.handler(DummyMsg(response))
        return None


class DummyNATS:
    is_connected = True

    async def connect(self, *args, **kwargs):
        pass

    def jetstream(self):
        return object()

    async def drain(self):
        pass


@pytest.mark.asyncio
async def test_replay_generates_metrics(tmp_path: Path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"event": "CHAT_RAW", "payload": {"text": "Hello"}}) + "\n",
        encoding="utf-8",
    )

    golden = tmp_path / "golden.yaml"
    golden.write_text('- input: "Hello"\n  expected: "Hi"\n  rating: 5\n', encoding="utf-8")

    metrics = tmp_path / "metrics.json"
    output = tmp_path / "out.jsonl"

    DummyPublisher.responses = ["Hi"]
    monkeypatch.setattr(dr, "NATS", DummyNATS)
    monkeypatch.setattr(dr, "Publisher", DummyPublisher)
    monkeypatch.setattr(dr, "Subscriber", DummySubscriber)

    await dr._replay(trace, output, metrics, "nats://localhost:4222", golden)

    data = json.loads(metrics.read_text(encoding="utf-8"))
    assert "bleu" in data
    assert "rouge_l" in data
    assert data.get("avg_rating") == pytest.approx(5.0)


def test_load_golden_returns_ratings(tmp_path: Path):
    sample = tmp_path / "golden.yaml"
    sample.write_text(
        "- input: 'Hi'\n  expected: 'Hello'\n  rating: 4\n- input: 'Bye'\n  expected: 'Goodbye'\n  rating: 2\n",
        encoding="utf-8",
    )

    expected, ratings = dr._load_golden(sample)

    assert expected == ["Hello", "Goodbye"]
    assert ratings == [4.0, 2.0]
