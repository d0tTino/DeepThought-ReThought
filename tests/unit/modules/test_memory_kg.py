import json
from types import SimpleNamespace

import pytest

import deepthought.modules.memory_kg as memory_kg
from deepthought.eda.events import InputReceivedPayload


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
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


class DummyDAL:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.merge_entity_calls = []
        self.merge_edge_calls = []

    def merge_entity(self, name: str):
        if self.fail:
            raise RuntimeError("boom")
        self.merge_entity_calls.append(name)

    def merge_next_edge(self, src: str, dst: str):
        if self.fail:
            raise RuntimeError("boom")
        self.merge_edge_calls.append((src, dst))


def create_memory(monkeypatch, dal):
    monkeypatch.setattr(memory_kg, "Publisher", DummyPublisher)
    monkeypatch.setattr(memory_kg, "Subscriber", DummySubscriber)
    return memory_kg.KnowledgeGraphMemory(DummyNATS(), DummyJS(), dal)


@pytest.mark.asyncio
async def test_handle_input_creates_nodes_edges(monkeypatch):
    connector = DummyDAL()
    mem = create_memory(monkeypatch, connector)
    payload = InputReceivedPayload(user_input="hello world", input_id="7")
    msg = DummyMsg(payload.to_json())
    await mem._handle_input_event(msg)

    assert msg.acked
    pub = mem._publisher
    assert pub.published
    assert connector.merge_entity_calls == ["hello", "world"]
    assert connector.merge_edge_calls == [("hello", "world")]


@pytest.mark.asyncio
async def test_handle_input_error(monkeypatch):
    connector = DummyDAL(fail=True)
    mem = create_memory(monkeypatch, connector)
    payload = InputReceivedPayload(user_input="fail", input_id="8")
    msg = DummyMsg(payload.to_json())
    await mem._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked
    assert connector.merge_entity_calls == []
    assert connector.merge_edge_calls == []


@pytest.mark.asyncio
async def test_handle_input_invalid_payload(monkeypatch):
    connector = DummyDAL()
    mem = create_memory(monkeypatch, connector)
    msg = DummyMsg("not json")
    await mem._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked


@pytest.mark.asyncio
async def test_handle_input_missing_fields(monkeypatch):
    connector = DummyDAL()
    mem = create_memory(monkeypatch, connector)
    msg = DummyMsg(json.dumps({"user_input": "hi"}))
    await mem._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked


def test_parse_input(monkeypatch):
    dal = DummyDAL()
    mem = create_memory(monkeypatch, dal)
    nodes, edges = mem._parse_input("a b a c")

    assert nodes == ["a", "b", "c"]
    assert edges == [("a", "b"), ("b", "a"), ("a", "c")]


def test_store(monkeypatch):
    dal = DummyDAL()
    mem = create_memory(monkeypatch, dal)
    nodes = ["x", "y"]
    edges = [("x", "y"), ("y", "x")]
    mem._store(nodes, edges)

    assert dal.merge_entity_calls == nodes
    assert dal.merge_edge_calls == edges
