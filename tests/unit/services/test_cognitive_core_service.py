import importlib.machinery
import sys
import types
from types import SimpleNamespace

fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")


fake_prom = types.ModuleType("prometheus_client")


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


fake_prom.Counter = lambda *a, **k: _Metric()
fake_prom.Histogram = lambda *a, **k: _Metric()
fake_prom.REGISTRY = SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
fake_nats = types.ModuleType("nats")
fake_nats.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
fake_nats.aio = types.ModuleType("aio")
client_mod = types.ModuleType("client")
setattr(client_mod, "Client", object)
fake_nats.aio.client = client_mod
msg_mod = types.ModuleType("msg")
setattr(msg_mod, "Msg", object)
fake_nats.aio.msg = msg_mod
fake_nats.js = types.ModuleType("js")
js_client_mod = types.ModuleType("client")
setattr(js_client_mod, "JetStreamContext", object)
fake_nats.js.client = js_client_mod
err_mod = types.ModuleType("errors")
setattr(err_mod, "Error", Exception)
setattr(err_mod, "TimeoutError", Exception)
fake_nats.errors = err_mod
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", client_mod)
sys.modules.setdefault("nats.aio.msg", msg_mod)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", js_client_mod)
sys.modules.setdefault("nats.errors", err_mod)


class DummyBase:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


fake_ps.BaseSettings = DummyBase
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)

import json

import pytest

from deepthought.config import Settings
from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.services.cognitive_core_service import CognitiveCoreService


class DummyNATS:
    def __init__(self):
        self.is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))
        return SimpleNamespace(seq=1, stream="test")


class DummySubscriber:
    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyMemory:
    def __init__(self):
        self.interactions = []

    def store_interaction(self, text):
        self.interactions.append(text)

    def retrieve_context(self, prompt: str):
        return self.interactions[-3:]


class DummyDB:
    def __init__(self):
        self.memories = []
        self.interactions = 0

    async def store_memory(self, user_id, memory, topic="", sentiment_score=None):
        self.memories.append(memory)

    async def log_interaction(self, user_id, target_id=None, sentiment_score=None):
        self.interactions += 1

    async def recall_user(self, user_id):
        return [("", m) for m in self.memories]

    async def close(self):
        pass


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_input_stores_and_publishes(monkeypatch):
    memory = DummyMemory()
    db = DummyDB()
    monkeypatch.setattr(
        CognitiveCoreService,
        "_publisher",
        DummyPublisher(DummyNATS(), DummyJS()),
        raising=False,
    )
    monkeypatch.setattr(CognitiveCoreService, "_subscriber", DummySubscriber(), raising=False)
    settings = Settings()
    service = CognitiveCoreService(DummyNATS(), DummyJS(), settings, memory=memory, db=db)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hello", input_id="x")
    msg = DummyMsg(payload.to_json())
    await service._handle_input(msg)

    assert msg.acked
    assert memory.interactions == ["hello"]
    assert db.memories == ["hello"]
    subject, sent_payload = service._publisher.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "x"
    assert "hello" in sent_payload.retrieved_knowledge["facts"]
