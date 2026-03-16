import json
import pytest

pytest.importorskip("aiosqlite")

from deepthought.eda.events import EventSubjects
from deepthought.services.feedback_service import FeedbackService


class DummyDB:
    def __init__(self):
        self.affinity_calls = []
        self.confidence_calls = []
        self.feedback_rows = []

    async def adjust_affinity(self, user_id, delta):
        self.affinity_calls.append((user_id, delta))

    async def adjust_theory_confidence(self, user_id, delta):
        self.confidence_calls.append((user_id, delta))

    async def record_feedback_signal(self, **kwargs):
        self.feedback_rows.append(kwargs)

    async def fetch_high_value_feedback(self, *, limit=100, min_score=0.7):
        high = [r for r in self.feedback_rows if float(r.get("score", 0.0)) >= min_score]
        return high[:limit]


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
async def test_feedback_service_discord_reaction_mapping_and_guardrails():
    service = FeedbackService(nats_client=None, js_context=None, db_manager=DummyDB(), min_confidence=0.5)
    service._publisher = DummyPublisher()

    await service._apply_feedback(
        signal_type="reaction",
        signal="positive",
        input_id="in-2",
        user_id="u-2",
        source="discord",
        model_id="model-a",
        affinity_delta=0.75,
        confidence_delta=0.08,
        signal_confidence=0.9,
        details={"emoji": "👍"},
    )
    assert service._db.affinity_calls[-1] == ("u-2", 0.75)
    assert service._db.feedback_rows[-1]["source"] == "discord"
    assert service._db.feedback_rows[-1]["model_id"] == "model-a"

    prior = len(service._db.feedback_rows)
    await service._apply_feedback(
        signal_type="reaction",
        signal="negative",
        input_id="in-2",
        user_id="u-2",
        source="discord",
        model_id="model-a",
        affinity_delta=-0.6,
        confidence_delta=-0.08,
        signal_confidence=0.2,
        details={"emoji": "👎"},
    )
    assert len(service._db.feedback_rows) == prior


@pytest.mark.asyncio
async def test_feedback_queue_generation_filters_high_value_rows():
    service = FeedbackService(nats_client=None, js_context=None, db_manager=DummyDB())
    service._publisher = DummyPublisher()

    await service._db.record_feedback_signal(
        signal_type="reaction",
        signal="positive",
        input_id="in-h",
        user_id="u-h",
        source="discord",
        model_id="m1",
        confidence=0.9,
        score=0.85,
        details={},
    )
    await service._db.record_feedback_signal(
        signal_type="message_edit",
        signal="edited",
        input_id="in-l",
        user_id="u-l",
        source="discord",
        model_id="m1",
        confidence=0.7,
        score=0.2,
        details={},
    )

    tuples = await service._db.fetch_high_value_feedback(min_score=0.7)
    await service._publisher.publish("dtr.training.feedback_tuples.v1", {"items": tuples})

    assert len(tuples) == 1
    assert tuples[0]["input_id"] == "in-h"
    assert service._publisher.calls[-1][0] == "dtr.training.feedback_tuples.v1"


def test_feedback_subjects_are_canonicalized():
    assert EventSubjects.OUTCOME_SIGNAL.endswith(".v1")
    assert EventSubjects.CORRECTION_SIGNAL.endswith(".v1")
    assert EventSubjects.DISCORD_FEEDBACK_SIGNAL.endswith(".v1")
    assert EventSubjects.USER_SUMMARY_REFRESH.endswith(".v1")
