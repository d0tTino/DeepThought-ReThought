import json
import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

sys.modules.setdefault("deepthought.harness", types.ModuleType("harness"))
record_mod = types.ModuleType("record")


class TraceEvent:
    pass


record_mod.TraceEvent = TraceEvent
sys.modules.setdefault("deepthought.harness.record", record_mod)
import importlib.util

if importlib.util.find_spec("networkx") is None:
    fake_nx = types.ModuleType("networkx")
    setattr(fake_nx, "DiGraph", object)
    sys.modules.setdefault("networkx", fake_nx)
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")


class DummyBase:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


fake_ps.BaseSettings = DummyBase
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
fake_nats = types.ModuleType("nats")
import importlib.machinery

fake_nats.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
fake_nats.aio = types.ModuleType("aio")
fake_client_mod = types.ModuleType("client")
setattr(fake_client_mod, "Client", object)
fake_nats.aio.client = fake_client_mod
fake_msg_mod = types.ModuleType("msg")
setattr(fake_msg_mod, "Msg", object)
fake_nats.aio.msg = fake_msg_mod
fake_nats.js = types.ModuleType("js")
fake_js_client_mod = types.ModuleType("client")
setattr(fake_js_client_mod, "JetStreamContext", object)
fake_nats.js.client = fake_js_client_mod
fake_errors_mod = types.ModuleType("errors")
setattr(fake_errors_mod, "Error", Exception)
fake_nats.errors = fake_errors_mod
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_client_mod)
sys.modules.setdefault("nats.aio.msg", fake_msg_mod)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_js_client_mod)
sys.modules.setdefault("nats.errors", fake_errors_mod)
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
sys.modules.setdefault("prometheus_client", fake_prom)

from deepthought.config import Settings
from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.services.memory_service import MemoryService


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


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_input_updates_graph_and_publishes(monkeypatch):
    memory = DummyMemory()
    monkeypatch.setattr(MemoryService, "_publisher", DummyPublisher(DummyNATS(), DummyJS()), raising=False)
    monkeypatch.setattr(MemoryService, "_subscriber", DummySubscriber(), raising=False)
    settings = Settings()
    service = MemoryService(DummyNATS(), DummyJS(), settings, memory)
    # replace publisher and subscriber with dummies
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hello", input_id="x")
    msg = DummyMsg(payload.to_json())
    await service._handle_input(msg)

    assert msg.acked
    assert memory.interactions == ["hello"]
    subject, sent_payload = service._publisher.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "x"
    assert "hello" in sent_payload.retrieved_knowledge["facts"]
    ts = sent_payload.timestamp
    assert datetime.fromisoformat(ts).tzinfo == timezone.utc


def test_init_from_settings(monkeypatch):
    calls = {}
    import deepthought.services.memory_service as ms

    def fake_create_memory_backend(*, settings):
        calls["settings"] = settings
        return object()

    monkeypatch.setattr(ms, "create_memory_backend", fake_create_memory_backend)

    fake_settings = Settings(vector_backend="faiss", vector_use_gpu=True, graph_backend="noop")

    ms.MemoryService(DummyNATS(), DummyJS(), fake_settings)

    assert calls["settings"] is fake_settings


def test_from_config(monkeypatch):
    import deepthought.services.memory_service as ms

    dummy = object()

    def fake_create_memory_backend(*, settings):
        return dummy

    monkeypatch.setattr(ms, "create_memory_backend", fake_create_memory_backend)
    monkeypatch.setattr(ms, "get_settings", lambda: Settings())

    service = ms.MemoryService.from_config(DummyNATS(), DummyJS())

    assert service._memory is dummy


class ClosedNATS(DummyNATS):
    def __init__(self):
        super().__init__()
        self.is_connected = False
        self.drain_called = False

    async def drain(self):
        self.drain_called = True


@pytest.mark.asyncio
async def test_stop_skips_drain_when_not_connected():
    service = MemoryService(DummyNATS(), DummyJS(), Settings(), DummyMemory())
    service._subscriber = DummySubscriber()
    service._publisher = DummyPublisher()
    service._nc = ClosedNATS()

    await service.stop()

    assert not service._nc.drain_called
