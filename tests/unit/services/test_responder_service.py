import json

import pytest

from deepthought.eda.events import EventSubjects
from deepthought.services.responder_service import (
    FactualResponderService,
    PersonaResponderService,
    SafetyResponderService,
)


class DummyNATS:
    is_connected = True


class DummyJS:
    pass


class RecordingSubscriber:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def subscribe(self, **kwargs):
        self.calls.append(kwargs)
        return True

    async def unsubscribe_all(self):
        return None


class RecordingPublisher:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.calls.append((subject, payload, use_jetstream))


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type,expected_source,expected_tag",
    [
        (FactualResponderService, "responder:factual", "factual"),
        (PersonaResponderService, "responder:persona", "persona"),
        (SafetyResponderService, "responder:safety", "safety"),
    ],
)
async def test_responder_publishes_candidate_with_metadata(monkeypatch, service_type, expected_source, expected_tag):
    import deepthought.services.responder_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = service_type(DummyNATS(), DummyJS())
    await svc.start()

    context_payload = {
        "input_id": "in-1",
        "user_input": "hello",
        "retrieved_facts": ["fact1"],
        "author_id": "a-1",
    }
    msg = DummyMsg(json.dumps(context_payload))
    await svc._handle_context_event(msg)

    assert msg.acked
    assert svc._publisher.calls[0][0] == EventSubjects.RESPONSE_CANDIDATES
    payload = svc._publisher.calls[0][1]["payload"]
    candidate = payload["candidates"][0]
    assert candidate["source"] == expected_source
    assert expected_tag in candidate["rationale_tags"]
    assert isinstance(candidate["source_metadata"], dict)
    assert isinstance(candidate["source_metadata"].get("calibration"), dict)
