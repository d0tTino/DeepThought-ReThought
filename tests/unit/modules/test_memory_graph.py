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
