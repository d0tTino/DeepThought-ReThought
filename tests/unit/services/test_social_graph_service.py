import json
from types import SimpleNamespace

import pytest

from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.services import social_graph_service
from deepthought.services.db_manager import DBManager
from deepthought.services.persona_manager import PersonaManager
from deepthought.services.social_graph_service import SocialGraphService


class DummyMsg:
    def __init__(self, payload: str, headers=None):
        self.data = payload.encode()
        self.headers = headers or {}
        self.acked = False
        self.naked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.naked = True


class RecordingPublisher:
    def __init__(self):
        self.calls = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.calls.append((subject, payload, use_jetstream, timeout))


@pytest.mark.asyncio
async def test_handle_input_resolves_identity_from_payload_and_headers(tmp_path, monkeypatch):
    db = DBManager(str(tmp_path / "sg.db"))
    await db.init_db()
    pm = PersonaManager(db, friendly=1, playful=1)

    monkeypatch.setattr(
        social_graph_service,
        "analyze_social",
        lambda _text: {"flirtation": 0.6, "avoidance": 0.1, "manipulation": 0.0},
    )

    service = SocialGraphService(db_manager=db, persona_manager=pm)
    service._publisher = RecordingPublisher()
    service._subscriber = SimpleNamespace()

    payload = InputReceivedPayload(user_input="hello", input_id="1", author_id="author-42")
    msg = DummyMsg(payload.to_json(), headers={"user_id": "header-user", "channel_id": "chan-1"})

    await service._handle_input(msg)

    assert msg.acked
    assert not msg.naked
    assert await db.get_affinity("author-42") > 0
    assert await pm.get_persona("author-42", channel_id="chan-1") == "friendly"
    topics = [topic for topic, _ in await db.recall_user("author-42")]
    assert "social_perception" in topics
    assert len(service._publisher.calls) == 2

    update_subject, update_payload, _, _ = service._publisher.calls[0]
    assert update_subject == EventSubjects.SOCIAL_UPDATED
    assert update_payload["payload"]["input_id"] == "1"
    assert update_payload["payload"]["affinity"] > 0
    assert update_payload["trace_id"]
    assert update_payload["event_id"]
    assert update_payload["causation_id"]

    event_subject, payload, _, _ = service._publisher.calls[1]
    assert event_subject == EventSubjects.SOCIAL_SIGNALS_RETRIEVED
    assert payload["payload"]["input_id"] == "1"
    assert payload["payload"]["social_signals"]["affinity"] > 0
    assert payload["payload"]["social_signals"]["persona"] == "friendly"
    assert payload["trace_id"]

    await db.close()


@pytest.mark.asyncio
async def test_handle_input_invalid_payload_naks(tmp_path):
    db = DBManager(str(tmp_path / "sg.db"))
    await db.init_db()
    pm = PersonaManager(db)
    service = SocialGraphService(db_manager=db, persona_manager=pm)

    msg = DummyMsg(json.dumps({"input_id": "missing-user-input"}))
    await service._handle_input(msg)

    assert msg.naked
    assert not msg.acked

    await db.close()


@pytest.mark.asyncio
async def test_each_input_updates_social_state_exactly_once(tmp_path, monkeypatch):
    db = DBManager(str(tmp_path / "sg.db"))
    await db.init_db()
    pm = PersonaManager(db)
    service = SocialGraphService(db_manager=db, persona_manager=pm)
    service._publisher = RecordingPublisher()

    adjust_calls = []
    original_adjust = db.adjust_affinity

    async def _counting_adjust(user_id, delta, target_id=None):
        adjust_calls.append((user_id, delta, target_id))
        await original_adjust(user_id, delta, target_id)

    monkeypatch.setattr(db, "adjust_affinity", _counting_adjust)
    monkeypatch.setattr(
        social_graph_service,
        "analyze_social",
        lambda _text: {"flirtation": 0.7, "avoidance": 0.0, "manipulation": 0.0},
    )

    msg = DummyMsg(InputReceivedPayload(user_input="hello", input_id="in-1", author_id="u-1").to_json())
    await service._handle_input(msg)

    assert msg.acked
    assert len(adjust_calls) == 1
    memories = await db.recall_user("u-1")
    assert [topic for topic, _ in memories].count("social_perception") == 1
    assert len(service._publisher.calls) == 2
    assert service._publisher.calls[0][0] == EventSubjects.SOCIAL_UPDATED
    assert service._publisher.calls[1][0] == EventSubjects.SOCIAL_SIGNALS_RETRIEVED
    assert service._publisher.calls[0][1]["payload"]["input_id"] == "in-1"
    assert service._publisher.calls[1][1]["payload"]["input_id"] == "in-1"

    await db.close()


@pytest.mark.asyncio
async def test_social_contract_update_and_retrieval_semantics(tmp_path, monkeypatch):
    db = DBManager(str(tmp_path / "sg.db"))
    await db.init_db()
    service = SocialGraphService(db_manager=db, persona_manager=PersonaManager(db))
    service._publisher = RecordingPublisher()

    monkeypatch.setattr(
        social_graph_service,
        "analyze_social",
        lambda _text: {"flirtation": 0.5, "avoidance": 0.2, "manipulation": 0.0},
    )

    msg = DummyMsg(InputReceivedPayload(user_input="hey", input_id="in-contract", author_id="u-1").to_json())
    await service._handle_social_signals_requested(msg)

    subjects = [call[0] for call in service._publisher.calls]
    assert subjects == [EventSubjects.SOCIAL_UPDATED, EventSubjects.SOCIAL_SIGNALS_RETRIEVED]

    updated = service._publisher.calls[0][1]
    retrieved = service._publisher.calls[1][1]
    assert updated["payload"]["input_id"] == "in-contract"
    assert updated["payload"]["perception"]["flirtation"] == pytest.approx(0.5)
    assert "social_signals" not in updated["payload"]

    assert retrieved["payload"]["input_id"] == "in-contract"
    assert retrieved["payload"]["social_signals"]["perception"]["flirtation"] == pytest.approx(0.5)
    assert retrieved["payload"]["social_signals"]["affinity"] == updated["payload"]["affinity"]
    assert retrieved["payload"]["social_signals"]["persona_state"] in {
        "new_acquaintance",
        "familiar",
        "trusted",
        "repair_mode",
        "uncertain_mode",
    }
    assert isinstance(retrieved["payload"]["social_signals"]["persona_policy_hints"], dict)

    await db.close()


@pytest.mark.asyncio
async def test_social_signals_include_summarized_context_and_channel_norms(tmp_path, monkeypatch):
    db = DBManager(str(tmp_path / "sg.db"))
    await db.init_db()
    service = SocialGraphService(db_manager=db, persona_manager=PersonaManager(db))
    service._publisher = RecordingPublisher()

    monkeypatch.setattr(
        social_graph_service,
        "analyze_social",
        lambda _text: {"flirtation": 0.4, "avoidance": 0.0, "manipulation": 0.0},
    )

    payload = InputReceivedPayload(user_input="hey", input_id="in-social", author_id="u-1").to_json()
    msg = DummyMsg(payload)
    msg.data = json.dumps(
        {
            "user_input": "hey",
            "input_id": "in-social",
            "author_id": "u-1",
            "target_id": "u-2",
            "channel_id": "c-1",
            "thread_participants": ["u-2", "u-3"],
            "co_occurring_users": ["u-4"],
        }
    ).encode()

    await service._handle_social_signals_requested(msg)

    retrieved = service._publisher.calls[1][1]["payload"]["social_signals"]
    assert retrieved["relationship_status"] in {"neutral", "friend", "rival"}
    assert retrieved["familiarity_tier"] in {"low", "medium", "high"}
    assert isinstance(retrieved["channel_norms"]["interaction_frequency"], int)
    assert "reciprocity" in retrieved["channel_norms"]
    assert "sentiment_trend" in retrieved["channel_norms"]


    assert retrieved["durable_user_model"]["version"] == "v2"
    assert "dimensions" in retrieved["durable_user_model"]
    assert set(retrieved["durable_user_model"]["dimensions"]).issuperset({
        "familiarity",
        "trust_rapport",
        "preferred_response_style",
        "topic_affinity",
        "cadence_tolerance",
        "correction_sensitivity",
        "channel_specific_norms",
    })
    selector_inputs = retrieved["selector_inputs"]
    assert set(selector_inputs) == {"social_intent_hints", "user_history_affinity", "interaction_policy"}
    assert selector_inputs["interaction_policy"]["response_style"]
    assert selector_inputs["interaction_policy"]["persona_state"] in {
        "new_acquaintance",
        "familiar",
        "trusted",
        "repair_mode",
        "uncertain_mode",
    }
    assert await db.get_edge_weight("u-1", "u-2", "interaction", channel_id="c-1") > 0
    assert await db.get_edge_weight("u-1", "u-3", "interaction", channel_id="c-1") > 0
    assert await db.get_edge_weight("u-1", "u-4", "interaction", channel_id="c-1") > 0
    await db.close()
