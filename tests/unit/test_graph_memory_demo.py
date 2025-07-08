import importlib
import sys
import types

sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import examples.graph_memory_demo as demo


class DummyStore:
    def __init__(self):
        self.added = []

    def add_texts(self, texts, ids=None, metadatas=None):
        self.added.extend(texts)

    def query(self, query_texts, n_results=3):
        return {"documents": [[t] for t in self.added[:n_results]]}

    class Collection:
        def __init__(self, outer):
            self.outer = outer

        def delete(self, ids):
            pass

    @property
    def collection(self):
        return DummyStore.Collection(self)


class DummyBackend:
    def __init__(self):
        self.entities = []

    def query_subgraph(self, query, params):
        return [{"fact": "g1"}]

    def merge_entity(self, name):
        self.entities.append(name)


def test_main(monkeypatch):
    store = DummyStore()
    backend = DummyBackend()
    importlib.reload(demo)
    monkeypatch.setattr(demo, "create_vector_store", lambda *a, **k: store)
    monkeypatch.setattr(demo, "create_graph_backend", lambda *a, **k: backend)

    captured = {}
    monkeypatch.setattr(demo.logger, "info", lambda msg, ctx: captured.setdefault("ctx", ctx))

    demo.main()

    assert store.added
    assert backend.entities
    assert captured.get("ctx")
