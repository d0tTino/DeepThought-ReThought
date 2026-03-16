import pytest

from deepthought.eda.events import ResponseCandidate, ResponseCandidatesPayload
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

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_conflicting_multi_responder_candidates_use_calibrated_selection(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = SelectorService(
        DummyNATS(),
        DummyJS(),
        window_seconds=0.03,
        early_exit_confidence=0.99,
        source_calibration_profiles={
            "responder:factual": {"slope": 1.1, "bias": 0.0},
            "responder:persona": {"slope": 0.95, "bias": 0.0},
            "responder:safety": {"slope": 1.2, "bias": 0.0},
        },
    )

    factual_msg = DummyMsg(
        ResponseCandidatesPayload(
            input_id="integration-1",
            candidates=[
                ResponseCandidate(
                    text="Factual answer.",
                    confidence=0.76,
                    source="responder:factual",
                    source_metadata={"kind": "factual", "calibration": {"slope": 1.05}},
                    rationale_tags=["factual", "grounded"],
                    safety_passed=True,
                )
            ],
        ).to_json()
    )
    persona_msg = DummyMsg(
        ResponseCandidatesPayload(
            input_id="integration-1",
            candidates=[
                ResponseCandidate(
                    text="Friendly answer.",
                    confidence=0.78,
                    source="responder:persona",
                    source_metadata={"kind": "persona"},
                    rationale_tags=["persona", "rapport"],
                    safety_passed=True,
                )
            ],
        ).to_json()
    )
    safety_msg = DummyMsg(
        ResponseCandidatesPayload(
            input_id="integration-1",
            candidates=[
                ResponseCandidate(
                    text="Safety reply.",
                    confidence=0.7,
                    source="responder:safety",
                    source_metadata={"kind": "safety"},
                    rationale_tags=["safety"],
                    safety_passed=True,
                )
            ],
        ).to_json()
    )

    await svc._handle_candidates_event(factual_msg)
    await svc._handle_candidates_event(persona_msg)
    await svc._handle_candidates_event(safety_msg)

    await __import__("asyncio").sleep(0.05)

    ranked_payload = svc._publisher.calls[0][1]["payload"]
    assert ranked_payload["final_response"] == "Factual answer."
    assert factual_msg.acked and persona_msg.acked and safety_msg.acked


@pytest.mark.asyncio
async def test_conflicting_equal_score_candidates_use_deterministic_tie_break(monkeypatch):
    import deepthought.services.selector_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = SelectorService(DummyNATS(), DummyJS(), window_seconds=0.0, early_exit_confidence=0.0)

    msg = DummyMsg(
        ResponseCandidatesPayload(
            input_id="integration-tie-1",
            candidates=[
                ResponseCandidate(
                    text="same",
                    confidence=0.8,
                    source="responder:persona",
                    rationale_tags=["equal"],
                    safety_passed=True,
                ),
                ResponseCandidate(
                    text="same",
                    confidence=0.8,
                    source="responder:factual",
                    rationale_tags=["equal"],
                    safety_passed=True,
                ),
            ],
        ).to_json()
    )

    await svc._handle_candidates_event(msg)

    ranked_payload = svc._publisher.calls[0][1]["payload"]
    assert ranked_payload["source"] == "responder:factual"
