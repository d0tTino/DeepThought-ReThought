import asyncio
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
    return SelectorService(DummyNATS(), DummyJS(), window_seconds=0.0, early_exit_confidence=0.0)


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
    assert ranked["payload"]["final_response"] == "high"
    telemetry_subject, telemetry_payload = service._publisher.published[1]
    assert telemetry_subject == "dtr.telemetry.selector_ranking.v1"
    assert telemetry_payload["chosen_source"] is None


@pytest.mark.asyncio
async def test_empty_candidates_ack_without_publish(service):
    payload = ResponseCandidatesPayload(input_id="2", candidates=[])
    msg = DummyMsg(payload.to_json())
    await service._handle_candidates_event(msg)
    await asyncio.sleep(0.02)

    assert msg.acked
    assert len(service._publisher.published) == 2
    assert service._publisher.published[0][1]["payload"]["source"] == "selector_fallback"


@pytest.mark.asyncio
async def test_invalid_payload_nak(service):
    msg = DummyMsg(json.dumps(["bad"]))
    await service._handle_candidates_event(msg)
    assert msg.nacked


@pytest.mark.asyncio
async def test_window_aggregates_candidates_before_flush(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)
    svc = SelectorService(DummyNATS(), DummyJS(), window_seconds=0.03, early_exit_confidence=0.99)

    msg1 = DummyMsg(
        ResponseCandidatesPayload(
            input_id="window-1",
            candidates=[ResponseCandidate(text="one", confidence=0.6)],
        ).to_json()
    )
    msg2 = DummyMsg(
        ResponseCandidatesPayload(
            input_id="window-1",
            candidates=[ResponseCandidate(text="two", confidence=0.8)],
        ).to_json()
    )
    await svc._handle_candidates_event(msg1)
    await svc._handle_candidates_event(msg2)
    assert svc._publisher.published == []

    await asyncio.sleep(0.05)
    assert svc._publisher.published[0][1]["payload"]["final_response"] == "two"


@pytest.mark.asyncio
async def test_unsafe_only_candidates_use_clarifying_fallback(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)
    svc = SelectorService(
        DummyNATS(),
        DummyJS(),
        early_exit_confidence=0.0,
        safety_filter=lambda _: False,
        window_seconds=0.0,
    )

    msg = DummyMsg(
        ResponseCandidatesPayload(
            input_id="unsafe-1",
            candidates=[ResponseCandidate(text="risky", confidence=0.9, source="x")],
        ).to_json()
    )
    await svc._handle_candidates_event(msg)
    await asyncio.sleep(0.02)

    ranked_payload = svc._publisher.published[0][1]
    assert ranked_payload["payload"]["source"] == "selector_fallback"
    assert "clarify" in ranked_payload["payload"]["final_response"].lower()


@pytest.mark.asyncio
async def test_source_calibration_changes_ranking(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)
    svc = SelectorService(
        DummyNATS(),
        DummyJS(),
        early_exit_confidence=0.0,
        source_confidence_weights={"default": 1.0, "trusted": 1.6},
    )

    msg = DummyMsg(
        ResponseCandidatesPayload(
            input_id="cal-1",
            candidates=[
                ResponseCandidate(text="base", confidence=0.7, source="default"),
                ResponseCandidate(text="weighted", confidence=0.5, source="trusted"),
            ],
        ).to_json()
    )
    await svc._handle_candidates_event(msg)

    assert svc._publisher.published[0][1]["payload"]["final_response"] == "weighted"
    diagnostics = svc._publisher.published[1][1]["diagnostics"]
    assert diagnostics[0]["source"] == "trusted"


@pytest.mark.asyncio
async def test_mixed_source_and_confidence_filters_and_weights(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)
    svc = SelectorService(
        DummyNATS(),
        DummyJS(),
        early_exit_confidence=0.0,
        window_seconds=0.0,
        source_confidence_weights={"default": 1.0, "tool": 1.3, "rule": 0.9},
    )

    msg = DummyMsg(
        ResponseCandidatesPayload(
            input_id="mix-1",
            candidates=[
                ResponseCandidate(text="safe tool", confidence=0.65, source="tool", safety_passed=True),
                ResponseCandidate(text="unsafe rule", confidence=0.95, source="rule", safety_passed=False),
                ResponseCandidate(text="safe default", confidence=0.7, source="default", safety_passed=True),
            ],
        ).to_json()
    )

    await svc._handle_candidates_event(msg)

    ranked = svc._publisher.published[0][1]
    telemetry = svc._publisher.published[1][1]
    assert ranked["payload"]["final_response"] == "safe tool"
    rejected = [item for item in telemetry["diagnostics"] if item["rejection_reasons"]]
    assert rejected and rejected[0]["text"] == "unsafe rule"


@pytest.mark.asyncio
async def test_early_exit_flushes_without_waiting(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)
    svc = SelectorService(
        DummyNATS(),
        DummyJS(),
        window_seconds=10.0,
        early_exit_confidence=0.8,
        source_confidence_weights={"trusted": 1.2},
    )

    msg = DummyMsg(
        ResponseCandidatesPayload(
            input_id="early-1",
            candidates=[ResponseCandidate(text="fast", confidence=0.75, source="trusted")],
        ).to_json()
    )

    await svc._handle_candidates_event(msg)

    assert msg.acked
    assert svc._publisher.published
    assert svc._publisher.published[0][1]["payload"]["final_response"] == "fast"
    assert "early-1" not in svc._pending_by_input


@pytest.mark.asyncio
async def test_context_and_policy_and_affinity_factors_shape_ranking(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)
    svc = SelectorService(
        DummyNATS(),
        DummyJS(),
        early_exit_confidence=0.0,
        window_seconds=0.0,
        source_confidence_weights={"default": 1.0, "remote": 1.0},
    )

    payload = ResponseCandidatesPayload(
        input_id="ctx-pol-1",
        interaction_policy={"response_style": "concise", "ask_clarifying_on_no_safe": True},
        context_confidence={"aggregate": 0.2, "threshold": 0.45, "low_confidence": True},
        social_intent_hints={"preferred_style": "concise", "clarify_preferred": True, "high_rapport_expected": True},
        user_history_affinity={"remote": 0.8, "default": -0.2, "intent": 0.4},
        candidates=[
            ResponseCandidate(text="generic", confidence=0.72, source="default", safety_metadata={"style": "verbose"}),
            ResponseCandidate(text="tailored", confidence=0.65, source="remote", safety_metadata={"style": "concise"}),
        ],
    )

    msg = DummyMsg(payload.to_json())
    await svc._handle_candidates_event(msg)

    ranked = svc._publisher.published[0][1]
    telemetry = svc._publisher.published[1][1]
    assert ranked["payload"]["final_response"] == "tailored"
    assert telemetry["weights"]["policy_fit"] == pytest.approx(0.2)
    assert telemetry["weights"]["history_affinity"] == pytest.approx(0.15)
    assert telemetry["weights"]["context_degradation"] == pytest.approx(0.25)

    top_diag = telemetry["diagnostics"][0]
    assert top_diag["text"] == "tailored"
    assert top_diag["factor_scores"]["policy_fit"] > 0.5
    assert top_diag["factor_scores"]["history_affinity"] > 0.0
    assert top_diag["factor_scores"]["context_degradation"] > 0.0
