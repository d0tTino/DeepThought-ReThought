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

    async def ack(self):
        self.acked = True


class DummyConnector:
    def __init__(self, fail=False):
        self.fail = fail
        self.executed = []

    def execute(self, query, params=None):
        if self.fail:
            raise RuntimeError("boom")
        self.executed.append((query, params))
        return []


def create_memory(monkeypatch, connector):
    monkeypatch.setattr(memory_kg, "Publisher", DummyPublisher)
    monkeypatch.setattr(memory_kg, "Subscriber", DummySubscriber)
    return memory_kg.KnowledgeGraphMemory(DummyNATS(), DummyJS(), connector)


@pytest.mark.asyncio
async def test_handle_input_creates_nodes_edges(monkeypatch):
    connector = DummyConnector()
    mem = create_memory(monkeypatch, connector)
    payload = InputReceivedPayload(user_input="hello world", input_id="7")
    msg = DummyMsg(payload.to_json())
    await mem._handle_input_event(msg)

    assert msg.acked
    pub = mem._publisher
    assert pub.published
    node_queries = [q for q, _ in connector.executed if "MERGE (:Entity" in q]
    edge_queries = [q for q, _ in connector.executed if "MERGE (a)-[:NEXT]->(b)" in q]
    assert len(node_queries) == 2
    assert len(edge_queries) == 1


@pytest.mark.asyncio
async def test_handle_input_error(monkeypatch):
    connector = DummyConnector(fail=True)
    mem = create_memory(monkeypatch, connector)
    payload = InputReceivedPayload(user_input="fail", input_id="8")
    msg = DummyMsg(payload.to_json())
    await mem._handle_input_event(msg)

    assert msg.acked
    assert connector.executed == []
