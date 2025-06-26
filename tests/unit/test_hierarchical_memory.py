from deepthought.memory.hierarchical import HierarchicalMemory


class DummyVector:
    def __init__(self, docs):
        self.docs = docs

    def query(self, query_texts, n_results):
        return self.docs[:n_results]


class DummyDAL:
    def __init__(self, rows):
        self.rows = rows

    def query_subgraph(self, query, params):
        return self.rows[: params["limit"]]


class FailingVector:
    def query(self, *args, **kwargs):
        raise RuntimeError("boom")


class FailingDAL:
    def query_subgraph(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_retrieve_context_merges():
    vector = DummyVector(["a", "b"])
    dal = DummyDAL([{"fact": "b"}, {"fact": "c"}])
    mem = HierarchicalMemory(vector, dal, top_k=2)
    ctx = mem.retrieve_context("hi")
    assert ctx == ["a", "b", "c"]


def test_retrieve_context_failures():
    mem = HierarchicalMemory(FailingVector(), FailingDAL())
    ctx = mem.retrieve_context("x")
    assert ctx == []
