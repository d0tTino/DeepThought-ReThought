import json
import pytest

pytest.importorskip("aiosqlite")

from deepthought.eda.events import EventSubjects
from deepthought.services.feedback_service import FeedbackService


class DummyDB:
    def __init__(self):
        self.affinity_calls = []
        self.confidence_calls = []

    async def adjust_affinity(self, user_id, delta):
        self.affinity_calls.append((user_id, delta))

    async def adjust_theory_confidence(self, user_id, delta):
        self.confidence_calls.append((user_id, delta))


class DummyPublisher:
    def __init__(self):
        self.calls = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.calls.append((subject, payload, use_jetstream))


class DummyMsg:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_feedback_service_tracks_response_and_applies_positive_outcome():
    service = FeedbackService(nats_client=None, js_context=None, db_manager=DummyDB())
    service._publisher = DummyPublisher()

    ranked = DummyMsg(
        {
            "final_response": "hello",
            "input_id": "in-1",
            "user_id": "u-1",
            "author_id": "u-1",
            "source": "responder_a",
            "confidence": 0.7,
        }
    )
    await service._handle_response_ranked(ranked)
    assert ranked.acked

    outcome = DummyMsg({"signal": "positive", "input_id": "in-1"})
    await service._handle_outcome_signal(outcome)

    assert service._db.affinity_calls == [("u-1", 1.0)]
    assert service._db.confidence_calls == [("u-1", 0.1)]
    assert outcome.acked
    assert service._publisher.calls[0][0] == "dtr.telemetry.response_feedback.v1"


@pytest.mark.asyncio
async def test_feedback_service_correction_uses_explicit_deltas():
    service = FeedbackService(nats_client=None, js_context=None, db_manager=DummyDB())
    service._publisher = DummyPublisher()

    correction = DummyMsg(
        {
            "correction": "The city is Rome.",
            "user_id": "u-9",
            "confidence_delta": -0.35,
            "affinity_delta": -0.2,
        }
    )
    await service._handle_correction_signal(correction)

    assert service._db.affinity_calls == [("u-9", -0.2)]
    assert service._db.confidence_calls == [("u-9", -0.35)]


def test_feedback_subjects_are_canonicalized():
    assert EventSubjects.OUTCOME_SIGNAL.endswith(".v1")
    assert EventSubjects.CORRECTION_SIGNAL.endswith(".v1")
    assert EventSubjects.USER_SUMMARY_REFRESH.endswith(".v1")
