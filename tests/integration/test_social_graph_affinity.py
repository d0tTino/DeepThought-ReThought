from types import SimpleNamespace
import sys
import types

import pytest

from deepthought.eda.events import InputReceivedPayload



class DummyMsg:
    def __init__(self, payload: InputReceivedPayload) -> None:
        self.data = payload.to_json().encode()
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


@pytest.mark.asyncio
async def test_affinity_changes_after_processing(tmp_path, monkeypatch):
    # Stub heavy modules before importing the service
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    sp_stub = types.ModuleType("social_perception")
    sp_stub.analyze = lambda text: {
        "flirtation": 0.0,
        "avoidance": 0.0,
        "manipulation": 0.0,
    }
    monkeypatch.setitem(
        sys.modules,
        "deepthought.perception.social_perception",
        sp_stub,
    )

    from deepthought.services.db_manager import DBManager
    from deepthought.services.persona_manager import PersonaManager
    from deepthought.services.social_graph_service import SocialGraphService
    from deepthought.perception import social_perception
    import deepthought.services.social_graph_service as sgs

    db = DBManager(str(tmp_path / "sg.db"))
    await db.init_db()
    pm = PersonaManager(db, friendly=1, playful=1)
    svc = SocialGraphService(db_manager=db, persona_manager=pm)
    svc._publisher = SimpleNamespace()
    svc._subscriber = SimpleNamespace()

    monkeypatch.setattr(
        sgs,
        "analyze_social",
        lambda text: {"flirtation": 0.6, "avoidance": 0.2, "manipulation": 0.0},
    )
    pos = DummyMsg(InputReceivedPayload(user_input="I love this"))
    await svc._handle_input(pos)
    assert pos.acked
    assert await db.get_affinity("user") == 1
    assert await pm.get_persona("user") == "friendly"

    monkeypatch.setattr(
        sgs,
        "analyze_social",
        lambda text: {"flirtation": 0.0, "avoidance": 0.4, "manipulation": 0.3},
    )
    neg = DummyMsg(InputReceivedPayload(user_input="I hate this"))
    await svc._handle_input(neg)
    assert await db.get_affinity("user") == 0
    assert await pm.get_persona("user") == "snarky"

    memories = await db.recall_user("user")
    topics = [t for t, _ in memories]
    assert topics.count("social_perception") == 2

    await db.close()
