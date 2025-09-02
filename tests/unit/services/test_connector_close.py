import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("aiosqlite")

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
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)


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
import pytest

from deepthought.config import Settings
from deepthought.services.cognitive_core_service import CognitiveCoreService


class DummyConnector:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class DummyMemory:
    def __init__(self, connector):
        backend = types.SimpleNamespace(_dal=types.SimpleNamespace(_connector=connector))
        self.graph_backend = backend

    def store_interaction(self, text):
        pass

    def retrieve_context(self, prompt):
        return []


class DummyDB:
    def __init__(self):
        self.closed = False

    async def store_memory(self, *a, **k):
        pass

    async def log_interaction(self, *a, **k):
        pass

    async def recall_user(self, user_id):
        return []

    async def close(self):
        self.closed = True


class DummyNATS:
    def __init__(self):
        self.is_connected = True

    async def drain(self):
        pass


class DummyJS:
    pass


class DummySubscriber:
    async def unsubscribe_all(self):
        pass


class DummyPublisher:
    pass


@pytest.mark.asyncio
async def test_cognitive_core_service_stop_closes_connector_and_db():
    connector = DummyConnector()
    memory = DummyMemory(connector)
    db = DummyDB()
    service = CognitiveCoreService(DummyNATS(), DummyJS(), Settings(), memory=memory, db=db)
    service._subscriber = DummySubscriber()
    service._publisher = DummyPublisher()
    await service.stop()
    assert connector.closed
    assert db.closed
