import sys
import types

sys.modules.setdefault("deepthought.harness", types.ModuleType("harness"))
sys.modules.setdefault("deepthought.learn", types.ModuleType("learn"))
sys.modules.setdefault("deepthought.modules", types.ModuleType("modules"))
sys.modules.setdefault("deepthought.motivate", types.ModuleType("motivate"))

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
