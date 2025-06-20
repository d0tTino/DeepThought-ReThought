import json
import logging
from types import SimpleNamespace

import networkx as nx
import pytest

import deepthought.modules.memory_graph as memory_graph


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


def create_memory(monkeypatch, graph_file):
    monkeypatch.setattr(memory_graph, "Publisher", DummyPublisher)
    monkeypatch.setattr(memory_graph, "Subscriber", DummySubscriber)
    return memory_graph.GraphMemory(DummyNATS(), DummyJS(), graph_file=graph_file)


def test_read_graph_invalid_json(tmp_path, monkeypatch, caplog):
    graph_file = tmp_path / "graph.json"
    graph_file.write_text("{ invalid json")

    caplog.set_level(logging.ERROR)
    mem = create_memory(monkeypatch, graph_file)

    assert isinstance(mem._graph, nx.DiGraph)
    assert len(mem._graph.nodes) == 0
    error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("Failed to read graph file" in r.getMessage() for r in error_logs)


def test_init_creates_directory(tmp_path, monkeypatch):
    graph_file = tmp_path / "newdir" / "graph.json"
    mem = create_memory(monkeypatch, graph_file)
    assert graph_file.parent.is_dir()
    assert graph_file.exists()
    assert isinstance(mem._graph, nx.DiGraph)
    with open(graph_file, "r", encoding="utf-8") as f:
        assert isinstance(json.load(f), dict)


@pytest.mark.asyncio
async def test_handle_input_invalid_payload(tmp_path, monkeypatch):
    graph_file = tmp_path / "graph.json"
    mem = create_memory(monkeypatch, graph_file)
    msg = DummyMsg("not json")
    await mem._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked


@pytest.mark.asyncio
async def test_handle_input_missing_fields(tmp_path, monkeypatch):
    graph_file = tmp_path / "graph.json"
    mem = create_memory(monkeypatch, graph_file)
    msg = DummyMsg(json.dumps({"user_input": "hi"}))
    await mem._handle_input_event(msg)

    assert msg.nacked
    assert not msg.acked
