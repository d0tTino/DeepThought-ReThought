import types

from deepthought.memory import vector_store
from deepthought.memory.vector_store import FaissVectorStore, SimpleEmbeddingFunction


def test_add_and_query_neighbors():
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
    store.add_texts(["hello world", "goodbye", "hello there"], ids=["1", "2", "3"])
    res = store.query(["hello"], n_results=2)
    assert res["documents"][0][0] in {"hello world", "hello there"}
    assert len(res["documents"][0]) == 2
