import json
import sys
import types

fake_nx = types.ModuleType("networkx")
fake_nx.DiGraph = object
sys.modules.setdefault("networkx", fake_nx)
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import deepthought.api.server as server
from deepthought.eda.events import EventSubjects


class DummyJS:
    def __init__(self):
        self.published = []

    async def publish(self, subject, data, timeout=10.0):
        self.published.append((subject, data))
        return SimpleNamespace(seq=1, stream="test")

    async def subscribe(self, *args, **kwargs):
        return SimpleNamespace()


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


@pytest.fixture
def dummy_nats(monkeypatch):
    nc = DummyNATS()

    async def fake_connect(*args, **kwargs):
        return nc

    monkeypatch.setattr(server.nats, "connect", fake_connect)
    monkeypatch.setattr(server, "Subscriber", DummySubscriber)
    return nc


@pytest.fixture
def client(dummy_nats):
    with TestClient(server.app) as client:
        yield client


def test_memory_add_publishes(client, dummy_nats):
    resp = client.post("/memory/add", json={"text": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "input_id" in data

    js = dummy_nats.js
    subjects = [entry[0] for entry in js.published]
    assert len(subjects) == 2
    assert subjects[0] == EventSubjects.INPUT_RECEIVED
    assert subjects[1] == EventSubjects.CHAT_RAW
    payload = json.loads(js.published[0][1].decode())
    assert payload["user_input"] == "hello"


def test_memory_query_reads_cache(client):
    cache = server.app.state.memory_cache
    cache._cache["abc"] = {"facts": ["f1"], "source": "test"}

    resp = client.post("/memory/query", json={"query": "f1"})
    assert resp.status_code == 200
    assert resp.json()["results"] == [
        {"input_id": "abc", "retrieved_knowledge": {"facts": ["f1"], "source": "test"}}
    ]

def test_memory_query_no_match(client):
    resp = client.post("/memory/query", json={"query": "nomatch"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []

