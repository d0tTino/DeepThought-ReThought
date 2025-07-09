import sys

sys.modules.pop("numpy", None)
import pytest
pytest.importorskip("numpy")
import numpy as np

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


def test_faiss_store():
    import types

    class DummyIndex:
        def __init__(self, dim: int) -> None:
            self.count = 0

        def add(self, vecs):
            self.count += len(vecs)

        def search(self, vecs, k):
            import numpy as np

            k = min(k, self.count)
            idx = np.arange(k)
            return np.zeros((len(vecs), k), dtype="float32"), np.tile(idx, (len(vecs), 1))

    fake_faiss = types.SimpleNamespace(
        IndexFlatL2=DummyIndex,
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

        def add(self, vecs):
            self.count += len(vecs)

        def search(self, vecs, k):
            import numpy as np

            k = min(k, self.count)
            idx = np.arange(k)
            return np.zeros((len(vecs), k), dtype="float32"), np.tile(idx, (len(vecs), 1))

    fake_faiss = types.SimpleNamespace(
        IndexFlatL2=DummyIndex,
        StandardGpuResources=object,
        index_cpu_to_gpu=lambda res, device, index: index,
    )
    vector_store.faiss = fake_faiss

    store = create_vector_store(backend="faiss", use_gpu=False)
    assert isinstance(store, FaissVectorStore)
