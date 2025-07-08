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
fake_nx = types.ModuleType("networkx")
setattr(fake_nx, "DiGraph", object)
sys.modules.setdefault("networkx", fake_nx)
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")
fake_ps.BaseSettings = object
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
fake_nats = types.ModuleType("nats")
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
    service = MemoryService(DummyNATS(), DummyJS(), memory)
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

    def fake_create_vector_store(backend, collection_name, persist_directory=None, use_gpu=False, embedding_function=None):
        calls["vector"] = (backend, use_gpu)
        return object()

    def fake_create_graph_backend(name):
        calls["graph"] = name
        return object()

    class DummyTiered:
        def __init__(self, store, backend, capacity=100, top_k=3):
            calls["tiered"] = (store, backend, capacity, top_k)

    monkeypatch.setattr(ms, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(ms, "create_graph_backend", fake_create_graph_backend)
    monkeypatch.setattr(ms, "TieredMemory", DummyTiered)

    import deepthought.config as config

    config._settings_cache = None
    fake_settings = SimpleNamespace(
        vector_backend="faiss", vector_use_gpu=True, graph_backend="noop"
    )
    monkeypatch.setattr(config, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(ms, "get_settings", lambda: fake_settings)

    ms.MemoryService(DummyNATS(), DummyJS())

    assert calls["vector"] == ("faiss", True)
    assert calls["graph"] == "noop"
    assert calls["tiered"][2:] == (100, 3)
