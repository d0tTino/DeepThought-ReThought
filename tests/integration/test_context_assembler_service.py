import asyncio
import json

import pytest

from deepthought.eda.events import EventSubjects
from deepthought.services.context_assembler_service import ContextAssemblerService


class DummyNATS:
    is_connected = True


class DummyJS:
    pass


class RecordingPublisher:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.calls.append((subject, payload, use_jetstream, timeout))


class RecordingSubscriber:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def subscribe(self, **kwargs):
        self.calls.append(kwargs)
        return True

    async def unsubscribe_all(self):
        return None


class DummyMsg:
    def __init__(self, payload: dict):
        self.data = json.dumps(payload).encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


@pytest.mark.asyncio
async def test_race_safe_assembly_collects_all_providers(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.12)

    input_msg = DummyMsg(
        {
            "input_id": "i-race",
            "user_input": "hello",
            "author_id": "u1",
            "conversation_window": [{"role": "user", "text": "earlier"}],
        }
    )
    await svc._handle_input_received(input_msg)

    async def send_memory():
        await asyncio.sleep(0.01)
        await svc._handle_provider_response(
            DummyMsg({"input_id": "i-race", "retrieved_knowledge": {"facts": ["f2", "f1"]}}),
            "memory",
        )

    async def send_social():
        await asyncio.sleep(0.03)
        await svc._handle_provider_response(
            DummyMsg({"input_id": "i-race", "social_signals": {"tone": "neutral"}}),
            "social",
        )

    async def send_perception():
        await asyncio.sleep(0.02)
        await svc._handle_provider_response(
            DummyMsg({"input_id": "i-race", "multimodal_interpretations": {"image": "none"}}),
            "perception",
        )

    await asyncio.gather(send_social(), send_memory(), send_perception())
    await asyncio.sleep(0.06)

    assembled = [c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED]
    assert len(assembled) == 1
    payload = assembled[0][1]
    assert payload.input_id == "i-race"
    assert payload.retrieved_facts == ["f2", "f1"]
    assert payload.social_signals == {"tone": "neutral"}
    assert payload.multimodal_interpretations == {"image": "none"}
    assert payload.confidence["partial"] is False
    assert payload.confidence["completed_providers"] == ["memory", "social", "perception"]


@pytest.mark.asyncio
async def test_partial_result_when_provider_missing_is_deterministic(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.05)

    input_msg = DummyMsg({"input_id": "i-partial", "user_input": "hello"})
    await svc._handle_input_received(input_msg)

    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-partial", "retrieved_knowledge": {"facts": ["f1"]}}),
        "memory",
    )
    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-partial", "social_signals": {"sentiment": 0.7}}),
        "social",
    )

    await asyncio.sleep(0.08)

    assembled = [c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED]
    assert len(assembled) == 1
    payload = assembled[0][1]
    assert payload.input_id == "i-partial"
    assert payload.retrieved_facts == ["f1"]
    assert payload.social_signals == {"sentiment": 0.7}
    assert payload.multimodal_interpretations == {}
    assert payload.confidence["partial"] is True
    assert payload.confidence["missing_providers"] == ["perception"]
    assert payload.confidence["completed_providers"] == ["memory", "social"]
