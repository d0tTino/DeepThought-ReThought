import pytest

from deepthought.services.user_graph_dal import UserGraphDAL


@pytest.mark.parametrize("decay", [0.5])
def test_edge_decay(tmp_path, decay):
    dal = UserGraphDAL(str(tmp_path / "g.json"), weight_decay=decay, sentiment_decay=decay)
    dal.add_message("a", "b", sentiment_score=1.0)
    # Simulate time passing
    edge = dal._graph.get_edge_data("a", "b")
    edge["last_interaction"] -= 1
    edge = dal._graph.get_edge_data("b", "a")
    edge["last_interaction"] -= 1
    rel = dal.get_relationship("a", "b")
    assert rel[1] == pytest.approx(decay)
    assert rel[2] == pytest.approx(decay)
    dal.add_message("a", "b", sentiment_score=1.0)
    rel2 = dal.get_relationship("a", "b")
    # Previous values should have decayed before adding new interaction
    expected = decay + 1.0
    assert rel2[1] == pytest.approx(expected)
    assert rel2[2] == pytest.approx(expected)
    assert dal.get_mutual_affinity("a", "b") == pytest.approx(expected)
