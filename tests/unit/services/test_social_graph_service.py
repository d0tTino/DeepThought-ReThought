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
    assert await db.get_affinity("author-42") == 1
    assert await pm.get_persona("author-42", channel_id="chan-1") == "friendly"
    topics = [topic for topic, _ in await db.recall_user("author-42")]
    assert "social_perception" in topics
    assert len(service._publisher.calls) == 2

    update_subject, update_payload, _, _ = service._publisher.calls[0]
    assert update_subject == EventSubjects.SOCIAL_UPDATED
    assert update_payload["input_id"] == "1"
    assert update_payload["affinity"] == 1

    event_subject, payload, _, _ = service._publisher.calls[1]
    assert event_subject == EventSubjects.SOCIAL_SIGNALS_RETRIEVED
    assert payload["input_id"] == "1"
    assert payload["social_signals"]["affinity"] == 1
    assert payload["social_signals"]["persona"] == "friendly"

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
    assert service._publisher.calls[0][1]["input_id"] == "in-1"
    assert service._publisher.calls[1][1]["input_id"] == "in-1"

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
    assert updated["input_id"] == "in-contract"
    assert updated["perception"]["flirtation"] == pytest.approx(0.5)
    assert "social_signals" not in updated

    assert retrieved["input_id"] == "in-contract"
    assert retrieved["social_signals"]["perception"]["flirtation"] == pytest.approx(0.5)
    assert retrieved["social_signals"]["affinity"] == updated["affinity"]

    await db.close()
