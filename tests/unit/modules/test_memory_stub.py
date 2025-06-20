from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import deepthought.modules.memory_stub as memory_stub
from deepthought.eda.events import EventSubjects, InputReceivedPayload


class DummyNATS:
    def __init__(self):
        self.is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))
        return SimpleNamespace(seq=1, stream="test")


class FailingPublisher(DummyPublisher):
    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        raise RuntimeError("boom")


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


def create_stub(monkeypatch, publisher_cls=DummyPublisher):
    monkeypatch.setattr(memory_stub, "Publisher", publisher_cls)
    monkeypatch.setattr(memory_stub, "Subscriber", DummySubscriber)
    return memory_stub.MemoryStub(DummyNATS(), DummyJS())


@pytest.mark.asyncio
async def test_handle_input_success(monkeypatch):
    stub = create_stub(monkeypatch)
    payload = InputReceivedPayload(user_input="hi", input_id="123")
    msg = DummyMsg(payload.to_json())
    await stub._handle_input_event(msg)

    pub = stub._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "123"
    # Verify the nested retrieved_knowledge structure expected by LLMStub
    assert "retrieved_knowledge" in sent_payload.retrieved_knowledge
    facts = sent_payload.retrieved_knowledge["retrieved_knowledge"].get("facts")
    assert facts and "User asked: hi" in facts
    # Timestamp should be timezone-aware UTC
    ts = sent_payload.timestamp
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo == timezone.utc
    assert msg.acked


@pytest.mark.asyncio
async def test_handle_input_error(monkeypatch):
    stub = create_stub(monkeypatch, FailingPublisher)
    payload = InputReceivedPayload(user_input="fail", input_id="9")
    msg = DummyMsg(payload.to_json())
    await stub._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked
