import json

import pytest

from deepthought.eda.events import EventSubjects, ResponseCandidate, ResponseCandidatesPayload
from deepthought.services.selector_service import SelectorService


class DummyNATS:
    is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        self.subscribed = []

    async def subscribe(self, **kwargs):
        self.subscribed.append(kwargs)

    async def unsubscribe_all(self):
        return None


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


@pytest.fixture
def service(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)
    return SelectorService(DummyNATS(), DummyJS())


@pytest.mark.asyncio
async def test_rank_by_confidence(service):
    payload = ResponseCandidatesPayload(
        input_id="1",
        candidates=[
            ResponseCandidate(text="low", confidence=0.2),
            ResponseCandidate(text="high", confidence=0.8),
        ],
    )
    msg = DummyMsg(payload.to_json())
    await service._handle_candidates_event(msg)

    assert msg.acked
    assert not msg.nacked
    subject, ranked = service._publisher.published[0]
    assert subject == EventSubjects.RESPONSE_RANKED
    assert ranked.final_response == "high"


@pytest.mark.asyncio
async def test_empty_candidates_ack_without_publish(service):
    payload = ResponseCandidatesPayload(input_id="2", candidates=[])
    msg = DummyMsg(payload.to_json())
    await service._handle_candidates_event(msg)

    assert msg.acked
    assert service._publisher.published == []


@pytest.mark.asyncio
async def test_invalid_payload_nak(service):
    msg = DummyMsg(json.dumps(["bad"]))
    await service._handle_candidates_event(msg)
    assert msg.nacked
