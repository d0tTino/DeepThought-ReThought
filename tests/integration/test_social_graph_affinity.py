import sys
import types
from types import SimpleNamespace

import pytest

from deepthought.eda.events import InputReceivedPayload
from deepthought.services.cognitive_core_service import CognitiveCoreService, Settings
from deepthought.services.db_manager import DBManager
from deepthought.services.persona_manager import PersonaManager
from deepthought.services.social_graph_service import SocialGraphService


class DummyMsg:
    def __init__(self, payload: InputReceivedPayload) -> None:
        self.data = payload.to_json().encode()
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


class DummyMemory:
    def __init__(self) -> None:
        self.interactions: list[str] = []

    def store_interaction(self, text: str) -> None:
        self.interactions.append(text)

    def retrieve_context(self, prompt: str) -> list[str]:
        return self.interactions[-3:]


@pytest.mark.asyncio
async def test_affinity_changes_after_processing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    sp_stub = types.ModuleType("social_perception")
    sp_stub.analyze = lambda text: {"flirtation": 0.0, "avoidance": 0.0, "manipulation": 0.0}
    monkeypatch.setitem(sys.modules, "deepthought.perception.social_perception", sp_stub)

    db = DBManager(str(tmp_path / "sg.db"))
    await db.init_db()
    pm = PersonaManager(db, friendly=1, playful=1)
    async def _noop(*a, **k):
        return None

    svc = SocialGraphService(db_manager=db, persona_manager=pm)
    svc._publisher = SimpleNamespace(publish=_noop)
    svc._subscriber = SimpleNamespace()

    import deepthought.services.social_graph_service as sgsvc
    monkeypatch.setattr(
        sgsvc, "analyze_social", lambda _t: {"flirtation": 0.6, "avoidance": 0.2, "manipulation": 0.0}
    )
    pos = DummyMsg(InputReceivedPayload(user_input="I love this", input_id="1"))
    await svc._handle_input(pos)
    assert pos.acked
    assert await db.get_affinity("anonymous") == 1
    assert await pm.get_persona("anonymous") == "friendly"

    monkeypatch.setattr(
        sgsvc, "analyze_social", lambda _t: {"flirtation": 0.0, "avoidance": 0.4, "manipulation": 0.3}
    )
    neg = DummyMsg(InputReceivedPayload(user_input="I hate this", input_id="2"))
    await svc._handle_input(neg)
    assert await db.get_affinity("anonymous") == 0
    assert await pm.get_persona("anonymous") == "snarky"

    memories = await db.recall_user("anonymous")
    topics = [t for t, _ in memories]
    assert topics.count("social_perception") == 2

    await db.close()


@pytest.mark.asyncio
async def test_cognitive_core_does_not_mutate_social_state(tmp_path):
    db = DBManager(str(tmp_path / "core.db"))
    await db.init_db()
    svc = CognitiveCoreService(None, None, Settings(), memory=DummyMemory(), db=db)

    async def _noop(*a, **k):
        return None

    svc._publisher = SimpleNamespace(publish=_noop)
    svc._subscriber = SimpleNamespace()

    msg = DummyMsg(InputReceivedPayload(user_input="hello", input_id="1"))
    await svc._handle_input(msg)

    assert msg.acked
    assert await db.get_affinity("anonymous") == 0
    topics = [topic for topic, _ in await db.recall_user("anonymous")]
    assert topics.count("social_perception") == 0

    await db.close()


class RecordingPublisher:
    def __init__(self) -> None:
        self.calls = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.calls.append((subject, payload, use_jetstream, timeout))


@pytest.mark.asyncio
async def test_social_graph_cross_service_emits_enveloped_payloads(tmp_path, monkeypatch):
    db = DBManager(str(tmp_path / "sg-envelope.db"))
    await db.init_db()
    pm = PersonaManager(db, friendly=1, playful=1)

    svc = SocialGraphService(db_manager=db, persona_manager=pm)
    svc._publisher = RecordingPublisher()
    svc._subscriber = SimpleNamespace()

    import deepthought.services.social_graph_service as sgsvc

    monkeypatch.setattr(
        sgsvc,
        "analyze_social",
        lambda _t: {"flirtation": 0.4, "avoidance": 0.1, "manipulation": 0.0},
    )

    msg = DummyMsg(InputReceivedPayload(user_input="hello", input_id="enveloped-1"))
    await svc._handle_input(msg)

    assert msg.acked
    assert len(svc._publisher.calls) == 2
    for _, payload, _, _ in svc._publisher.calls:
        assert payload["trace_id"]
        assert payload["event_id"]
        assert payload["causation_id"]
        assert payload["payload"]["input_id"] == "enveloped-1"

    await db.close()
