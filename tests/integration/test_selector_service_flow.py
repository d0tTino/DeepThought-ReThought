import pytest

from deepthought.eda.events import EventSubjects, ResponseCandidate, ResponseCandidatesPayload
from deepthought.services.selector_service import SelectorService


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
async def test_start_uses_durable_and_subject(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = SelectorService(DummyNATS(), DummyJS())
    ok = await svc.start(durable_name="selector_durable")
    assert ok is True
    assert svc._subscriber.calls[0]["subject"] == EventSubjects.RESPONSE_CANDIDATES
    assert svc._subscriber.calls[0]["durable"] == "selector_durable"
    assert svc._subscriber.calls[0]["use_jetstream"] is True


@pytest.mark.asyncio
async def test_candidates_publish_and_ack(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = SelectorService(DummyNATS(), DummyJS())
    payload = ResponseCandidatesPayload(
        input_id="i-1",
        candidates=[
            ResponseCandidate(text="low", confidence=0.2),
            ResponseCandidate(text="high", confidence=0.9),
        ],
    )
    msg = DummyMsg(payload.to_json())

    await svc._handle_candidates_event(msg)

    assert msg.acked
    assert not msg.nacked
    subject, ranked_payload, use_jetstream = svc._publisher.calls[0]
    assert subject == EventSubjects.RESPONSE_RANKED
    assert use_jetstream is True
    assert ranked_payload.final_response == "high"
