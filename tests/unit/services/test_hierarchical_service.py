import json
import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace

sys.modules.setdefault("deepthought.harness", types.ModuleType("harness"))
record_mod = types.ModuleType("record")


class TraceEvent:
    pass


record_mod.TraceEvent = TraceEvent
sys.modules.setdefault("deepthought.harness.record", record_mod)
sys.modules.setdefault("deepthought.learn", types.ModuleType("learn"))
sys.modules.setdefault("deepthought.modules", types.ModuleType("modules"))
sys.modules.setdefault("deepthought.motivate", types.ModuleType("motivate"))
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
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
import importlib.machinery
import importlib.util

if importlib.util.find_spec("networkx") is None:
    fake_nx = types.ModuleType("networkx")
    setattr(fake_nx, "DiGraph", object)
    fake_nx.__spec__ = importlib.machinery.ModuleSpec("networkx", loader=None)
    sys.modules.setdefault("networkx", fake_nx)
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

from pathlib import Path
from typing import List

import pytest

from deepthought.config import Settings
from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.graph.backend import GraphDALBackend
from deepthought.memory.tiered import TieredMemory
from deepthought.search import OfflineSearch
from deepthought.services.hierarchical_service import HierarchicalService


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


class DummyVector:
    def add_texts(self, texts, ids=None, metadatas=None):
        pass

    def query(self, query_texts, n_results=3):
        return {"documents": [["vec1"], ["vec2"]]}


class DummyDAL:
    def query_subgraph(self, query, params):
        return [{"fact": "graph1"}]


class FailingVector:
    def query(self, *args, **kwargs):
        raise RuntimeError("boom")


class FailingDAL:
    def query_subgraph(self, *args, **kwargs):
        raise RuntimeError("boom")


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_input_publishes_combined_context(monkeypatch):
    vec = DummyVector()
    dal = DummyDAL()
    memory = TieredMemory(vec, GraphDALBackend(dal), top_k=3)
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), memory)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hello", input_id="x")
    msg = DummyMsg(payload.to_json())
    await service._handle_input(msg)

    assert msg.acked
    assert service._publisher.published
    subject, sent_payload = service._publisher.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "x"
    facts = sent_payload.retrieved_knowledge["facts"]
    assert "vec1" in facts and "graph1" in facts
    ts = sent_payload.timestamp
    assert datetime.fromisoformat(ts).tzinfo == timezone.utc


def test_retrieve_context_merges():
    vec = DummyVector()
    dal = DummyDAL()
    memory = TieredMemory(vec, GraphDALBackend(dal), top_k=3)
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), memory)
    ctx = service.retrieve_context("hi")
    assert ctx == ["vec1", "vec2", "graph1"]


def test_retrieve_context_failures():
    memory = TieredMemory(FailingVector(), GraphDALBackend(FailingDAL()), top_k=3)
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), memory)
    ctx = service.retrieve_context("x")
    assert ctx == []


class DummyGraphDAL:
    def query_subgraph(self, query, params):
        return [
            {"src_id": 1, "src": "A", "rel": "KNOWS", "dst_id": 2, "dst": "B"},
            {"src_id": 2, "src": "B", "rel": "LIKES", "dst_id": 3, "dst": "C"},
        ]


class DummyMemory:
    def __init__(self, backend):
        self._graph = backend

    @property
    def graph_backend(self):
        return self._graph


def test_dump_graph(tmp_path):
    dal = DummyGraphDAL()
    memory = DummyMemory(GraphDALBackend(dal))

    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), memory)

    dot_file = service.dump_graph(str(tmp_path))

    assert dot_file == str(tmp_path / "graph.dot")
    contents = (tmp_path / "graph.dot").read_text()
    assert '"A" -> "B" [label="KNOWS"]' in contents
    assert '"B" -> "C" [label="LIKES"]' in contents


def test_dump_graph_no_memory(tmp_path):
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), None)
    # ensure memory is unset even if the service auto-creates one
    service._memory = None
    with pytest.raises(ValueError):
        service.dump_graph(str(tmp_path))


def test_retrieve_context_with_search(tmp_path):
    vec = DummyVector()
    dal = DummyDAL()
    memory = TieredMemory(vec, GraphDALBackend(dal), top_k=3)
    search = OfflineSearch.create_index(
        str(tmp_path / "index.db"),
        [("t1", "search result 1"), ("t2", "search result 2")],
    )
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), memory, search=search)
    ctx = service.retrieve_context("result")
    assert "search result 1" in ctx and "search result 2" in ctx


def test_retrieve_context_from_config(monkeypatch, tmp_path):
    vec = DummyVector()
    dal = DummyDAL()
    memory = TieredMemory(vec, GraphDALBackend(dal), top_k=3)
    db = tmp_path / "conf.db"
    OfflineSearch.create_index(str(db), [("t", "via config")])
    monkeypatch.setenv("DT_SEARCH_DB", str(db))
    import deepthought.config as config

    config._settings_cache = None
    monkeypatch.setattr(config, "get_settings", lambda: Settings(search_db=str(db)))
    import deepthought.services.hierarchical_service as hs

    monkeypatch.setattr(hs, "get_settings", lambda: Settings(search_db=str(db)))
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(search_db=str(db)), memory)
    ctx = service.retrieve_context("config")
    assert "via config" in ctx


def test_retrieve_context_with_example_corpus(tmp_path):
    data_path = Path(__file__).resolve().parents[3] / "examples" / "data" / "sample_docs.json"
    docs = json.loads(data_path.read_text())
    search = OfflineSearch.create_index(
        str(tmp_path / "example.db"),
        [(d["title"], d["content"]) for d in docs],
    )
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), None, search=search)
    ctx = service.retrieve_context("web")
    assert any("web development" in c for c in ctx)


def test_service_uses_public_memory_interface():
    class SpyMemory:
        def __init__(self):
            self.calls = []

        def vector_matches(self, prompt: str) -> List[str]:
            self.calls.append(("vector", prompt))
            return ["v"]

        def graph_facts(self) -> List[str]:
            self.calls.append(("graph", None))
            return ["g"]

    mem = SpyMemory()
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), mem)

    assert service._vector_matches("hi") == ["v"]
    assert service._graph_facts() == ["g"]
    assert mem.calls == [("vector", "hi"), ("graph", None)]


from unittest import mock


def test_retrieve_context_delegates_to_memory():
    mem = mock.MagicMock()
    mem.retrieve_context.return_value = ["m1", "m2"]
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), mem)
    ctx = service.retrieve_context("hello")
    mem.retrieve_context.assert_called_once_with("hello")
    assert ctx == ["m1", "m2"]


def test_retrieve_context_merges_search_results():
    mem = mock.MagicMock()
    mem.retrieve_context.return_value = ["m1", "dup"]
    search = mock.MagicMock()
    search.search.return_value = ["dup", "s2"]
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(memory_top_k=2), mem, search=search)
    ctx = service.retrieve_context("question")
    mem.retrieve_context.assert_called_once_with("question")
    search.search.assert_called_once_with("question", limit=2)
    assert ctx == ["m1", "dup", "s2"]


class ClosedNATS(DummyNATS):
    def __init__(self):
        super().__init__()
        self.is_connected = False
        self.drain_called = False

    async def drain(self):
        self.drain_called = True


@pytest.mark.asyncio
async def test_stop_skips_drain_when_not_connected():
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), None)
    service._subscriber = DummySubscriber()
    service._publisher = DummyPublisher()
    service._nc = ClosedNATS()

    await service.stop()

    assert not service._nc.drain_called


class FailingAckMsg(DummyMsg):
    async def ack(self):
        raise fake_nats.errors.Error("boom")


class FailingNakMsg(DummyMsg):
    async def nak(self):
        raise fake_nats.errors.Error("boom")


class NoAckMsg:
    def __init__(self, data):
        self.data = data.encode()


@pytest.mark.asyncio
async def test_handle_input_ack_error_suppressed():
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), None)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hi", input_id="i1")
    msg = FailingAckMsg(payload.to_json())
    await service._handle_input(msg)


@pytest.mark.asyncio
async def test_handle_input_nak_error_suppressed():
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), None)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    msg = FailingNakMsg('{"bad": "payload"}')
    await service._handle_input(msg)


@pytest.mark.asyncio
async def test_handle_input_without_ack_attribute():
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), None)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hey", input_id="i2")
    msg = NoAckMsg(payload.to_json())
    await service._handle_input(msg)
