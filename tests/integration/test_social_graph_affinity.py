import sys
import types
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
    svc._publisher = SimpleNamespace(publish=lambda *a, **k: None)
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


class DummyMemory:
    def __init__(self) -> None:
        self.interactions: list[str] = []

    def store_interaction(self, text: str) -> None:
        self.interactions.append(text)

    def retrieve_context(self, prompt: str) -> list[str]:
        return self.interactions[-3:]


@pytest.mark.asyncio
async def test_cognitive_core_affinity_with_mocked_perception(tmp_path, monkeypatch):
    sp_mod = types.ModuleType("deepthought.perception.social_perception")
    sp_mod.analyze = lambda _t: {
        "flirtation": 0.6,
        "avoidance": 0.1,
        "manipulation": 0.0,
    }
    monkeypatch.setitem(sys.modules, "deepthought.perception.social_perception", sp_mod)

    import importlib

    cognitive_core_service = importlib.import_module("deepthought.services.cognitive_core_service")
    importlib.reload(cognitive_core_service)
    CognitiveCoreService = cognitive_core_service.CognitiveCoreService
    Settings = cognitive_core_service.Settings
    from deepthought.services.db_manager import DBManager

    db = DBManager(str(tmp_path / "core.db"))
    await db.init_db()
    svc = CognitiveCoreService(None, None, Settings(), memory=DummyMemory(), db=db)
    svc._publisher = SimpleNamespace(publish=lambda *a, **k: None)
    svc._subscriber = SimpleNamespace()

    msg = DummyMsg(InputReceivedPayload(user_input="hello", input_id="1"))
    await svc._handle_input(msg)

    assert msg.acked
    assert await db.get_affinity("user") == 2

    await db.close()
