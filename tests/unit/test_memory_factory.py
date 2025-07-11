import sys
import types

fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")


class DummyBase:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


fake_ps.BaseSettings = DummyBase
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import deepthought.memory as memory
from deepthought.config import Settings


def test_factory_uses_settings(monkeypatch):
    calls = {}
    vec_obj = object()
    graph_obj = object()

    def fake_create_vector_store(
        backend, collection_name, persist_directory=None, use_gpu=False, embedding_function=None
    ):
        calls["vector"] = (backend, collection_name, persist_directory, use_gpu)
        return vec_obj

    def fake_create_graph_backend(name):
        calls["graph"] = name
        return graph_obj

    class DummyTiered:
        def __init__(self, store, backend, capacity=100, top_k=3):
            calls["tiered"] = (store, backend, capacity, top_k)

    monkeypatch.setattr(memory, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(memory, "create_graph_backend", fake_create_graph_backend)
    monkeypatch.setattr(memory, "TieredMemory", DummyTiered)

    fake_settings = Settings(vector_backend="faiss", vector_use_gpu=True, graph_backend="noop")
    monkeypatch.setattr(
        memory,
        "load_memory_settings",
        lambda settings=None: memory.MemorySettings(
            vector_backend=fake_settings.vector_backend,
            vector_use_gpu=fake_settings.vector_use_gpu,
            graph_backend=fake_settings.graph_backend,
            memory_capacity=fake_settings.memory_capacity,
            memory_top_k=fake_settings.memory_top_k,
        ),
    )

    memory.create_memory_backend(capacity=42, top_k=7)

    assert calls["vector"] == ("faiss", "deepthought", None, True)
    assert calls["graph"] == "noop"
    assert calls["tiered"] == (vec_obj, graph_obj, 42, 7)
