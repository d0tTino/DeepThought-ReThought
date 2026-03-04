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


@pytest.mark.asyncio
async def test_context_assembler_merges_perception_interpretations_within_wait_window(monkeypatch):
    import deepthought.services.context_assembler_service as ca_mod
    import deepthought.services.perception_interpret_service as pi_mod

    class InMemoryBus:
        def __init__(self):
            self.handlers = {}
            self.published = []

        async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
            self.published.append((subject, payload, use_jetstream, timeout))
            for handler in self.handlers.get(subject, []):
                await handler(DummyMsg(payload))

    class BusSubscriber:
        def __init__(self, _nats, _js):
            self.calls = []

        async def subscribe(self, **kwargs):
            self.calls.append(kwargs)
            bus.handlers.setdefault(kwargs["subject"], []).append(kwargs["handler"])
            return True

        async def unsubscribe_all(self):
            return None

    bus = InMemoryBus()

    monkeypatch.setattr(ca_mod, "Publisher", lambda *_args, **_kwargs: bus)
    monkeypatch.setattr(ca_mod, "Subscriber", BusSubscriber)
    monkeypatch.setattr(pi_mod, "Publisher", lambda *_args, **_kwargs: bus)
    monkeypatch.setattr(pi_mod, "Subscriber", BusSubscriber)

    context_service = ca_mod.ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.08)
    perception_service = pi_mod.PerceptionInterpretService(DummyNATS(), DummyJS())
    await context_service.start()
    await perception_service.start()

    await bus.publish(
        EventSubjects.PERCEPTION_EMBEDDINGS,
        {
            "event": EventSubjects.PERCEPTION_EMBEDDINGS,
            "version": 1,
            "payload": {
                "message_id": "m-1",
                "user_id": "u-1",
                "input_id": "i-embed",
                "confidence": 0.77,
                "modality_confidence": {"image": 0.71},
                "by_modality": {
                    "image": {
                        "spans": [[0, 400]],
                        "embeddings": [[0.1, 0.2, 0.3]],
                        "encoders": [],
                    }
                },
            },
        },
    )

    await bus.publish(
        EventSubjects.INPUT_RECEIVED,
        {
            "input_id": "i-embed",
            "user_input": "look at this",
            "attachments": [{"url": "https://example.test/a.png", "content_type": "image/png"}],
        },
    )

    await bus.publish(
        EventSubjects.MEMORY_RETRIEVED,
        {"input_id": "i-embed", "retrieved_knowledge": {"facts": ["fact-a"]}},
    )
    await bus.publish(
        EventSubjects.SOCIAL_UPDATED,
        {"input_id": "i-embed", "social_signals": {"tone": "positive"}},
    )

    await asyncio.sleep(0.12)

    assembled = [item for item in bus.published if item[0] == EventSubjects.CONTEXT_ASSEMBLED]
    assert len(assembled) == 1
    payload = assembled[0][1]
    assert payload.input_id == "i-embed"
    assert payload.multimodal_interpretations["by_modality"]["image"].startswith("image: 1 vectors")
    assert "attachments[image:1]" in payload.multimodal_interpretations["summary"]
    assert payload.confidence["partial"] is False
