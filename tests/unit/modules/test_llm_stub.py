from types import SimpleNamespace
import pytest

import deepthought.modules.llm_stub as llm_stub
from deepthought.eda.events import EventSubjects, MemoryRetrievedPayload

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

    async def ack(self):
        self.acked = True


def create_stub(monkeypatch, publisher_cls=DummyPublisher):
    monkeypatch.setattr(llm_stub, "Publisher", publisher_cls)
    monkeypatch.setattr(llm_stub, "Subscriber", DummySubscriber)
    return llm_stub.LLMStub(DummyNATS(), DummyJS())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "knowledge",
    [
        {"retrieved_knowledge": {"facts": ["f1"]}},
        {"facts": ["f1"]},
    ],
)
async def test_handle_memory_success(monkeypatch, knowledge):
    stub = create_stub(monkeypatch)
    payload = MemoryRetrievedPayload(retrieved_knowledge=knowledge, input_id="abc")
    msg = DummyMsg(payload.to_json())
    await stub._handle_memory_event(msg)

    pub = stub._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.RESPONSE_GENERATED
    assert sent_payload.input_id == "abc"
    assert msg.acked


@pytest.mark.asyncio
async def test_handle_memory_error(monkeypatch):
    stub = create_stub(monkeypatch, FailingPublisher)
    payload = MemoryRetrievedPayload(retrieved_knowledge={"retrieved_knowledge": {}}, input_id="x")
    msg = DummyMsg(payload.to_json())
    await stub._handle_memory_event(msg)

    assert not msg.acked
