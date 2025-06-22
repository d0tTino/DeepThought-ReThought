import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.services.memory_service import MemoryService


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


class DummySubscriber:
    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyDAL:
    def __init__(self):
        self.interactions = []

    def add_interaction(self, text):
        self.interactions.append(text)

    def get_recent_facts(self, count=3):
        return self.interactions[-count:]


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_input_updates_graph_and_publishes(monkeypatch):
    dal = DummyDAL()
    monkeypatch.setattr(MemoryService, "_publisher", DummyPublisher(DummyNATS(), DummyJS()), raising=False)
    monkeypatch.setattr(MemoryService, "_subscriber", DummySubscriber(), raising=False)
    service = MemoryService(DummyNATS(), DummyJS(), dal)
    # replace publisher and subscriber with dummies
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hello", input_id="x")
    msg = DummyMsg(payload.to_json())
    await service._handle_input(msg)

    assert msg.acked
    assert dal.interactions == ["hello"]
    subject, sent_payload = service._publisher.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "x"
    assert "hello" in sent_payload.retrieved_knowledge["retrieved_knowledge"]["facts"]
    ts = sent_payload.timestamp
    assert datetime.fromisoformat(ts).tzinfo == timezone.utc
