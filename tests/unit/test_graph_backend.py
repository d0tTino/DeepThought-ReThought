from deepthought.graph.backend import GraphDALBackend, NoOpGraphBackend


class DummyDAL:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.merged = []

    def query_subgraph(self, query, params):
        self.called = (query, params)
        return self.rows

    def merge_entity(self, name):
        self.merged.append(name)


def test_graphdal_backend_delegates():
    dal = DummyDAL([{"fact": "x"}])
    backend = GraphDALBackend(dal)
    result = backend.query_subgraph("Q", {"a": 1})
    backend.merge_entity("name")
    assert result == [{"fact": "x"}]
    assert dal.called == ("Q", {"a": 1})
    assert dal.merged == ["name"]


def test_noop_backend():
    backend = NoOpGraphBackend()
    assert backend.query_subgraph("Q", {}) == []
    backend.merge_entity("n")  # should do nothing

