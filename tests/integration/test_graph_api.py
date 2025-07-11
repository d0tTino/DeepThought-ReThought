import json
import sys
import types

fake_nats = types.ModuleType("nats")
fake_nats.aio = types.ModuleType("aio")
fake_nats.aio.client = types.ModuleType("client")
fake_nats.aio.msg = types.ModuleType("msg")
fake_nats.errors = types.ModuleType("errors")
fake_nats.js = types.ModuleType("js")
fake_nats.js.client = types.ModuleType("jsclient")
fake_nats.aio.client.Client = object
fake_nats.aio.msg.Msg = object
fake_nats.errors.Error = Exception
fake_nats.js.client.JetStreamContext = object
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_nats.aio.client)
sys.modules.setdefault("nats.aio.msg", fake_nats.aio.msg)
sys.modules.setdefault("nats.errors", fake_nats.errors)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_nats.js.client)

fake_nx = types.ModuleType("networkx")
fake_nx.DiGraph = object
sys.modules.setdefault("networkx", fake_nx)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
fake_prom = types.ModuleType("prometheus_client")
fake_prom.Counter = lambda *a, **k: object()
fake_prom.Histogram = lambda *a, **k: object()
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import deepthought.api.graph as graph
import deepthought.api.server as server


class DummyJS:
    def __init__(self):
        self.published = []

    async def publish(self, subject, data, timeout=10.0):
        self.published.append((subject, data))
        return types.SimpleNamespace(seq=1, stream="test")

    async def subscribe(self, *args, **kwargs):
        return types.SimpleNamespace()


class DummyNATS:
    def __init__(self):
        self.is_connected = True
        self.js = DummyJS()
        self.drained = False

    def jetstream(self):
        return self.js

    async def drain(self):
        self.drained = True


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, *args, **kwargs):
        return None

    async def unsubscribe_all(self):
        return None


class DummyDAL:
    def __init__(self):
        self.entities = []
        self.rels = []
        self.queries = []
        self.result = [{"ok": True}]

    def add_entity(self, label, props):
        self.entities.append((label, props))

    def add_relationship(self, start, end, rel_type, props):
        self.rels.append((start, end, rel_type, props))

    def query_subgraph(self, query, params):
        self.queries.append((query, params))
        return self.result


@pytest.fixture
def dummy_nats(monkeypatch):
    nc = DummyNATS()

    async def fake_connect(*_a, **_k):
        return nc

    monkeypatch.setattr(server.nats, "connect", fake_connect, raising=False)
    monkeypatch.setattr(server, "Subscriber", DummySubscriber)
    return nc


@pytest.fixture
def dummy_dal(monkeypatch):
    dal = DummyDAL()
    monkeypatch.setattr(graph, "_dal", dal, raising=False)
    return dal


@pytest.fixture
def client(dummy_nats, dummy_dal):
    with TestClient(server.app) as client:
        yield client


def test_add_entity(client, dummy_dal):
    resp = client.post("/graph/entity", json={"label": "Person", "props": {"id": "1"}})
    assert resp.status_code == 200
    assert dummy_dal.entities == [("Person", {"id": "1"})]


def test_add_relationship(client, dummy_dal):
    payload = {
        "start_id": "1",
        "end_id": "2",
        "rel_type": "KNOWS",
        "props": {"since": 2024},
    }
    resp = client.post("/graph/relationship", json=payload)
    assert resp.status_code == 200
    assert dummy_dal.rels == [("1", "2", "KNOWS", {"since": 2024})]


def test_query(client, dummy_dal):
    dummy_dal.result = [{"id": 1}]
    resp = client.post("/graph/query", json={"query": "MATCH (n) RETURN n", "params": {}})
    assert resp.status_code == 200
    assert resp.json()["results"] == [{"id": 1}]
    assert dummy_dal.queries == [("MATCH (n) RETURN n", {})]
