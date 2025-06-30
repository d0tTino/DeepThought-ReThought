import networkx as nx

from deepthought.services.file_graph_dal import FileGraphDAL


def test_add_interaction_creates_next_edge(tmp_path):
    graph_file = tmp_path / "g.json"
    dal = FileGraphDAL(str(graph_file))
    first = dal.add_interaction("hello")
    second = dal.add_interaction("world")
    assert dal._graph.has_edge(first, second)
    assert dal._graph[first][second]["relation"] == "next"


def test_get_recent_facts_returns_latest(tmp_path):
    graph_file = tmp_path / "g.json"
    dal = FileGraphDAL(str(graph_file))
    dal.add_interaction("a")
    dal.add_interaction("b")
    dal.add_interaction("c")
    assert dal.get_recent_facts(2) == ["b", "c"]
