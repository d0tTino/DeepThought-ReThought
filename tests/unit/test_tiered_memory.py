import sys
import types

sys.modules.setdefault("deepthought.harness", types.ModuleType("harness"))
sys.modules.setdefault("deepthought.learn", types.ModuleType("learn"))
sys.modules.setdefault("deepthought.modules", types.ModuleType("modules"))
sys.modules.setdefault("deepthought.motivate", types.ModuleType("motivate"))
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import pytest

import deepthought.memory.tiered as tiered
from deepthought.memory.tiered import TieredMemory


class DummyVector:
    def __init__(self):
        self.docs = {}

    def add_texts(self, texts, ids=None, metadatas=None):
        ids = ids or [str(i) for i in range(len(texts))]
        for i, text in zip(ids, texts):
            self.docs[i] = text

    def query(self, query_texts, n_results=3):
        vals = list(self.docs.values())[:n_results]
        return {"documents": [[v] for v in vals]}

    class Collection:
        def __init__(self, outer):
            self.outer = outer

        def delete(self, ids):
            for i in ids:
                self.outer.docs.pop(i, None)

    @property
    def collection(self):
        return DummyVector.Collection(self)


class DummyDAL:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.merged = []

    def query_subgraph(self, query, params):
        limit = params.get("limit", len(self.rows))
        return self.rows[:limit]

    def merge_entity(self, name):
        self.merged.append(name)


class FailingVector(DummyVector):
    def query(self, query_texts, n_results=3):
        raise RuntimeError("vector fail")


class FailingDAL(DummyDAL):
    def query_subgraph(self, query, params):
        raise RuntimeError("graph fail")


def test_eviction_lru():
    vec = DummyVector()
    dal = DummyDAL()
    mem = TieredMemory(vec, dal, capacity=2, top_k=2)
    mem.store_interaction("a")
    mem.store_interaction("b")
    mem.store_interaction("c")
    assert list(mem._lru.keys()) == ["b", "c"]
    assert list(vec.docs.values()) == ["b", "c"]


def test_loads_from_graph():
    vec = DummyVector()
    dal = DummyDAL([{"fact": "g1"}, {"fact": "g2"}])
    mem = TieredMemory(vec, dal, capacity=3, top_k=2)
    ctx = mem.retrieve_context("x")
    assert ctx == ["g1", "g2"]
    assert set(mem._lru.keys()) == {"g1", "g2"}


def test_public_vector_and_graph_methods():
    vec = DummyVector()
    dal = DummyDAL([{"fact": "g1"}, {"fact": "g2"}])
    mem = TieredMemory(vec, dal, capacity=3, top_k=1)
    mem.store_interaction("v1")

    assert mem.vector_matches("whatever") == ["v1"]
    assert mem.graph_facts() == ["g1"]
    assert mem.graph_facts(2) == ["g1", "g2"]


def test_graph_backend_accessor():
    vec = DummyVector()
    dal = DummyDAL()
    mem = TieredMemory(vec, dal)

    assert mem.graph_backend is dal


def test_eviction_cleanup_on_delete_failure(monkeypatch):
    vec = DummyVector()
    dal = DummyDAL()
    mem = TieredMemory(vec, dal, capacity=1, top_k=1)

    def fail_delete(self, ids):
        raise RuntimeError("boom")

    monkeypatch.setattr(DummyVector.Collection, "delete", fail_delete)

    mem.store_interaction("a")
    mem.store_interaction("b")

    assert list(mem._lru.keys()) == ["b"]
    assert list(vec.docs.values()) == ["a", "b"]


def test_eviction_when_delete_missing():
    class NoDeleteVector(DummyVector):
        class Collection:
            def __init__(self, outer):
                self.outer = outer

        @property
        def collection(self):
            return NoDeleteVector.Collection(self)

    vec = NoDeleteVector()
    dal = DummyDAL()
    mem = TieredMemory(vec, dal, capacity=1, top_k=1)

    mem.store_interaction("a")
    mem.store_interaction("b")

    assert list(mem._lru.keys()) == ["b"]
    assert list(vec.docs.values()) == ["a", "b"]


def test_vector_match_logs_error(monkeypatch):
    vec = FailingVector()
    dal = DummyDAL()
    mem = TieredMemory(vec, dal)

    def boom(*args, **kwargs):
        raise RuntimeError("log")

    monkeypatch.setattr(tiered.logger, "error", boom)
    with pytest.raises(RuntimeError, match="log"):
        mem.vector_matches("hi")


def test_graph_facts_logs_error(monkeypatch):
    vec = DummyVector()
    dal = FailingDAL()
    mem = TieredMemory(vec, dal)

    def boom(*args, **kwargs):
        raise RuntimeError("log")

    monkeypatch.setattr(tiered.logger, "error", boom)
    with pytest.raises(RuntimeError, match="log"):
        mem.graph_facts()
