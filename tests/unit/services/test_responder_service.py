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
    assert candidate["source_metadata"]["policy_version"] == "v1"
    assert candidate["source_metadata"]["role"] == "specialist_candidate_producer"
    assert candidate["source_metadata"]["is_primary_voice"] is False
    assert candidate["source_metadata"]["source"] == expected_source
    assert len(candidate["safety_metadata"]["policy_artifacts"]) >= 3
    assert candidate["safety_metadata"]["safety_passed"] == candidate["safety_passed"]


@pytest.mark.asyncio
async def test_responder_bounded_social_features_in_candidate_metadata(monkeypatch):
    import deepthought.services.responder_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = PersonaResponderService(DummyNATS(), DummyJS())
    await svc.start()

    context_payload = {
        "input_id": "in-social",
        "user_input": "hello",
        "retrieved_facts": ["fact1"],
        "author_id": "a-1",
        "social_signals": {
            "relationship_status": "friend",
            "familiarity_tier": "high",
            "channel_norms": {
                "interaction_frequency": 42,
                "reciprocity": 0.7,
                "sentiment_trend": "up",
                "extra_large_payload": "x" * 500,
            },
            "raw_blob": {"nested": "ignore me"},
        },
    }
    msg = DummyMsg(json.dumps(context_payload))
    await svc._handle_context_event(msg)

    payload = svc._publisher.calls[0][1]["payload"]
    candidate = payload["candidates"][0]
    features = candidate["source_metadata"]["social_features"]
    assert set(features.keys()) == {"relationship_status", "familiarity_tier", "channel_norms"}
    assert set(features["channel_norms"].keys()) == {
        "interaction_frequency",
        "reciprocity",
        "sentiment_trend",
    }


@pytest.mark.asyncio
async def test_responder_passes_selector_inputs_from_durable_social_model(monkeypatch):
    import deepthought.services.responder_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = PersonaResponderService(DummyNATS(), DummyJS())
    await svc.start()

    context_payload = {
        "input_id": "in-selector",
        "user_input": "hello",
        "retrieved_facts": ["fact1"],
        "author_id": "a-1",
        "social_signals": {
            "selector_inputs": {
                "interaction_policy": {"response_style": "friendly", "ask_clarifying_on_no_safe": True},
                "social_intent_hints": {"preferred_style": "friendly", "high_rapport_expected": True},
                "user_history_affinity": {"default": 0.5, "intent": 0.3},
            }
        },
    }
    msg = DummyMsg(json.dumps(context_payload))
    await svc._handle_context_event(msg)

    payload = svc._publisher.calls[0][1]["payload"]
    assert payload["interaction_policy"]["response_style"] == "friendly"
    assert payload["social_intent_hints"]["high_rapport_expected"] is True
    assert payload["user_history_affinity"]["default"] == pytest.approx(0.5)
