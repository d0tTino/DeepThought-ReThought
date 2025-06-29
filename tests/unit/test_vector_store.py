from deepthought.memory.vector_store import SimpleEmbeddingFunction, VectorStore


def test_add_and_query():
    store = VectorStore(
        collection_name="test_vs", embedding_function=SimpleEmbeddingFunction()
    )
    store.add_texts(["hello world", "goodbye"], ids=["1", "2"])
    result = store.query(["hello world"], n_results=1)
    assert result["documents"][0][0] == "hello world"


def test_query_multiple_results():
    store = VectorStore(
        collection_name="test_vs2", embedding_function=SimpleEmbeddingFunction()
    )
    store.add_texts(["a", "b", "c"])
    res = store.query(["a"], n_results=2)
    assert len(res["documents"][0]) == 2


def test_add_twice_without_ids():
    store = VectorStore(
        collection_name="test_vs3", embedding_function=SimpleEmbeddingFunction()
    )
    store.add_texts(["first"])
    store.add_texts(["second"])
    assert store.collection.count() == 2
