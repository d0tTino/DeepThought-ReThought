import json
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("nats")

import deepthought.modules.memory_basic as memory_basic
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


class DummyMemory:
    def __init__(self):
        self.stored = []
        self.prompts = []

    def store_interaction(self, text):
        self.stored.append(text)

    def retrieve_context(self, prompt):
        self.prompts.append(prompt)
        return [prompt]


def create_memory(monkeypatch, memory=None, publisher_cls=DummyPublisher):
    monkeypatch.setattr(memory_basic, "Publisher", publisher_cls)
    monkeypatch.setattr(memory_basic, "Subscriber", DummySubscriber)
    mem = memory_basic.BasicMemory(DummyNATS(), DummyJS(), memory=memory)
    return mem


@pytest.mark.asyncio
async def test_handle_input_success(monkeypatch):
    dummy = DummyMemory()
    mem = create_memory(monkeypatch, dummy)
    payload = InputReceivedPayload(user_input="hello", input_id="42")
    msg = DummyMsg(payload.to_json())
    await mem._handle_input_event(msg)

    assert msg.acked
    pub = mem._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "42"
    assert dummy.stored == ["hello"]
    assert dummy.prompts == ["hello"]


@pytest.mark.asyncio
async def test_handle_input_error(monkeypatch, caplog):
    dummy = DummyMemory()
    mem = create_memory(monkeypatch, dummy, FailingPublisher)
    payload = InputReceivedPayload(user_input="boom", input_id="99")
    msg = DummyMsg(payload.to_json())
    with caplog.at_level(logging.ERROR):
        await mem._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked
    assert dummy.stored == ["boom"]


@pytest.mark.asyncio
async def test_handle_input_invalid_payload(monkeypatch):
    mem = create_memory(monkeypatch, DummyMemory())
    msg = DummyMsg("not json")
    await mem._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked


@pytest.mark.asyncio
async def test_handle_input_missing_fields(monkeypatch):
    mem = create_memory(monkeypatch, DummyMemory())
    msg = DummyMsg(json.dumps({"input_id": "1"}))
    await mem._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked


@pytest.mark.asyncio
async def test_start_listening_no_subscriber(monkeypatch, caplog):
    mem = create_memory(monkeypatch, DummyMemory())
    mem._subscriber = None
    with caplog.at_level(logging.ERROR):
        result = await mem.start_listening()

    assert result is False
    assert any("Subscriber not initialized for BasicMemory." in r.getMessage() for r in caplog.records)
