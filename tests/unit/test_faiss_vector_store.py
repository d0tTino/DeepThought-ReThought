from deepthought.memory.faiss_vector_store import FaissVectorStore
from deepthought.memory.vector_store import SimpleEmbeddingFunction


def test_add_and_query_neighbors():
    store = FaissVectorStore(embedding_dim=8, embedding_function=SimpleEmbeddingFunction())
    store.add_texts(["hello world", "goodbye", "hello there"], ids=["1", "2", "3"])
    res = store.query(["hello"], n_results=2)
    assert res["documents"][0][0] in {"hello world", "hello there"}
    assert len(res["documents"][0]) == 2
