import sys
import types

sys.modules.pop("numpy", None)
import pytest

pytest.importorskip("numpy")
import numpy as np

fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")
fake_ps.BaseSettings = object
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))

from deepthought.memory import vector_store
from deepthought.memory.vector_store import (
    ChromaVectorStore,
    FaissVectorStore,
    SimpleEmbeddingFunction,
)

sys.modules["numpy"] = np


def test_add_and_query():
    store = ChromaVectorStore(collection_name="test_vs", embedding_function=SimpleEmbeddingFunction())
    store.add_texts(["hello world", "goodbye"], ids=["1", "2"])
    result = store.query(["hello world"], n_results=1)
    assert result["documents"][0][0] == "hello world"


def test_query_multiple_results():
    store = ChromaVectorStore(collection_name="test_vs2", embedding_function=SimpleEmbeddingFunction())
    store.add_texts(["a", "b", "c"])
    res = store.query(["a"], n_results=2)
    assert len(res["documents"][0]) == 2


def test_add_twice_without_ids():
    store = ChromaVectorStore(collection_name="test_vs3", embedding_function=SimpleEmbeddingFunction())
    store.add_texts(["first"])
    store.add_texts(["second"])
    assert store.collection.count() == 2


def test_upsert_vectors_chroma():
    store = ChromaVectorStore(collection_name="test_vs4", embedding_function=SimpleEmbeddingFunction())
    store.upsert_vectors([[0.1, 0.2]], ids=["m1"])
    store.upsert_vectors([[0.2, 0.3]], ids=["m1"])
    assert store.collection.count() == 1


def test_faiss_store():
    import types

    class DummyIndex:
        def __init__(self, dim: int) -> None:
            self.count = 0

        def add_with_ids(self, vecs, ids):
            self.count += len(vecs)

        def remove_ids(self, ids):
            self.count -= len(ids)

        def search(self, vecs, k):
            import numpy as np

            k = min(k, self.count)
            idx = np.arange(k)
            return np.zeros((len(vecs), k), dtype="float32"), np.tile(idx, (len(vecs), 1))

    fake_faiss = types.SimpleNamespace(
        IndexFlatL2=DummyIndex,
        IndexIDMap=lambda idx: idx,
        StandardGpuResources=object,
        index_cpu_to_gpu=lambda res, device, index: index,
    )
    vector_store.faiss = fake_faiss

    store = FaissVectorStore(embedding_function=SimpleEmbeddingFunction())
    store.add_texts(["hello", "world"], ids=["1", "2"])
    result = store.query(["hello"], n_results=1)
    assert result["documents"][0][0] == "hello"


from deepthought.memory.vector_store import create_vector_store


def test_create_vector_store_chroma():
    store = create_vector_store(backend="chroma", collection_name="test")
    assert isinstance(store, ChromaVectorStore)


def test_create_vector_store_faiss(monkeypatch):
    import types

    class DummyIndex:
        def __init__(self, dim: int) -> None:
            self.count = 0

        def add_with_ids(self, vecs, ids):
            self.count += len(vecs)

        def remove_ids(self, ids):
            self.count -= len(ids)

        def search(self, vecs, k):
            import numpy as np

            k = min(k, self.count)
            idx = np.arange(k)
            return np.zeros((len(vecs), k), dtype="float32"), np.tile(idx, (len(vecs), 1))

    fake_faiss = types.SimpleNamespace(
        IndexFlatL2=DummyIndex,
        IndexIDMap=lambda idx: idx,
        StandardGpuResources=object,
        index_cpu_to_gpu=lambda res, device, index: index,
    )
    vector_store.faiss = fake_faiss

    store = create_vector_store(backend="faiss", use_gpu=False)
    assert isinstance(store, FaissVectorStore)


def test_faiss_delete(monkeypatch):
    import types

    class DummyIndex:
        def __init__(self, dim: int) -> None:
            self.ids: list[int] = []

        def add_with_ids(self, vecs, ids):
            for i in ids:
                self.ids.append(int(i))

        def search(self, vecs, k):
            import numpy as np

            ids = self.ids[:k]
            return np.zeros((len(vecs), len(ids)), dtype="float32"), np.tile(
                np.array(ids, dtype="int64"), (len(vecs), 1)
            )

        def remove_ids(self, ids):
            for i in ids:
                if int(i) in self.ids:
                    self.ids.remove(int(i))

    fake_faiss = types.SimpleNamespace(
        IndexFlatL2=lambda dim: DummyIndex(dim),
        IndexIDMap=lambda idx: idx,
        StandardGpuResources=object,
        index_cpu_to_gpu=lambda res, device, index: index,
    )
    vector_store.faiss = fake_faiss

    store = FaissVectorStore(embedding_function=SimpleEmbeddingFunction())
    store.add_texts(["hello", "world"], ids=["1", "2"])
    store.collection.delete(["1"])
    res = store.query(["hello"], n_results=2)
    assert "hello" not in res["documents"][0]


def test_create_vector_store_invalid_backend():
    with pytest.raises(ValueError, match="chroma|faiss"):
        create_vector_store(backend="invalid")
