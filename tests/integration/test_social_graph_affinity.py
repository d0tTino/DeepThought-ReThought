from types import SimpleNamespace

import pytest

from deepthought.eda.events import InputReceivedPayload
from deepthought.services import DBManager, PersonaManager
from deepthought.services.social_graph_service import SocialGraphService


class DummyMsg:
    def __init__(self, payload: InputReceivedPayload) -> None:
        self.data = payload.to_json().encode()
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


@pytest.mark.asyncio
async def test_affinity_changes_after_processing(tmp_path):
    db = DBManager(str(tmp_path / "sg.db"))
    await db.init_db()
    pm = PersonaManager(db, friendly=1, playful=1)
    svc = SocialGraphService(db_manager=db, persona_manager=pm)
    svc._publisher = SimpleNamespace()
    svc._subscriber = SimpleNamespace()

    pos = DummyMsg(InputReceivedPayload(user_input="I love this"))
    await svc._handle_input(pos)
    assert pos.acked
    assert await db.get_affinity("user") == 1
    assert await pm.get_persona("user") == "friendly"

    neg = DummyMsg(InputReceivedPayload(user_input="I hate this"))
    await svc._handle_input(neg)
    assert await db.get_affinity("user") == 0
    assert await pm.get_persona("user") == "snarky"

    await db.close()
