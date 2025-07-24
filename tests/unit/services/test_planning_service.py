import importlib.machinery
import sys
import types
import json

fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")
fake_ps.BaseSettings = object
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)
fake_prom = types.ModuleType("prometheus_client")
fake_prom.Counter = lambda *a, **k: object()
fake_prom.Histogram = lambda *a, **k: object()
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
torch_mod = types.ModuleType("torch")
torch_mod.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None)()
torch_mod.softmax = lambda t, dim=-1: t
sys.modules.setdefault("torch", torch_mod)
tb_mod = types.ModuleType("textblob")
tb_mod.TextBlob = lambda text: types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0))
sys.modules.setdefault("textblob", tb_mod)
tf_mod = types.ModuleType("transformers")
tf_mod.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": classmethod(lambda cls, p: cls())})
tf_mod.AutoModelForSequenceClassification = type(
    "AutoModelForSequenceClassification",
    (),
    {"from_pretrained": classmethod(lambda cls, p: cls()), "__call__": lambda self, **k: types.SimpleNamespace(logits=[[0, 0, 0]])},
)
sys.modules.setdefault("transformers", tf_mod)

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
fake_nats.errors = types.ModuleType("errors")
setattr(fake_nats.errors, "Error", Exception)
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", client_mod)
sys.modules.setdefault("nats.aio.msg", msg_mod)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", js_client_mod)
sys.modules.setdefault("nats.errors", fake_nats.errors)

from deepthought.eda.events import (
    EventSubjects,
    MemoryRetrievedPayload,
    PlanGeneratedPayload,
)
from deepthought.services.planning_service import PlanningService


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, **kw):
        self.published.append((subject, payload))


class DummySubscriber:
    async def subscribe(self, *a, **k):
        return True

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, payload):
        self.data = payload.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


import pytest


@pytest.mark.asyncio
async def test_memory_triggers_plan(tmp_path):
    cfg = tmp_path / "d.json"
    cfg.write_text(json.dumps({"desires": ["do it"]}), encoding="utf-8")

    service = PlanningService(None, None, desires_file=str(cfg))
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = MemoryRetrievedPayload(retrieved_knowledge={"facts": ["hi"]}, input_id="x")
    msg = DummyMsg(payload.to_json())
    await service._handle_memory(msg)

    assert msg.acked
    assert service._publisher.published
    subj, out = service._publisher.published[0]
    assert subj == EventSubjects.PLAN_REQUESTED
    assert out.goal == "do it"
    assert out.input_id == "x"


@pytest.mark.asyncio
async def test_execute_plan():
    service = PlanningService(None, None)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = PlanGeneratedPayload(plan=["step1", "step2"], input_id="y")
    msg = DummyMsg(payload.to_json())
    await service._handle_plan(msg)

    assert msg.acked
    subjects = [s for s, _ in service._publisher.published]
    assert subjects == [EventSubjects.CHAT_RAW, EventSubjects.CHAT_RAW]
