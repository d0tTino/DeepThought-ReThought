import importlib
import os
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


def test_main_noop(monkeypatch):
    store = DummyStore()
    monkeypatch.setenv("DT_GRAPH_BACKEND", "noop")
    importlib.reload(demo)
    monkeypatch.setattr(demo, "create_vector_store", lambda *a, **k: store)

    logs = []
    monkeypatch.setattr(demo.logger, "info", lambda msg, *a: logs.append(msg))

    demo.main()

    assert store.added
    assert any("Retrieval latency" in m for m in logs)
