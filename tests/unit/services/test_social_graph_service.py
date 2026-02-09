import json
from types import SimpleNamespace

import pytest

from deepthought.eda.events import InputReceivedPayload
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
    service._publisher = SimpleNamespace()
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
