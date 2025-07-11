import json
from types import SimpleNamespace

import pytest

pytest.importorskip("nats")

from deepthought.eda.events import CodeTemplatePayload, EventSubjects
from deepthought.services.code_generation_service import CodeGenerationService


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


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_template_request(monkeypatch):
    service = CodeGenerationService(DummyNATS(), DummyJS())
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = CodeTemplatePayload(template="result = ${x} + ${y}", variables={"x": 1, "y": 2}, input_id="a")
    msg = DummyMsg(payload.to_json())
    await service._handle_template_request(msg)

    assert msg.acked
    pub = service._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.CODE_GENERATED
    assert sent_payload.input_id == "a"
    assert sent_payload.result == "3"
