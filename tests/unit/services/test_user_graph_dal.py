from deepthought.services.user_graph_dal import UserGraphDAL


def test_add_message_updates_edges(tmp_path):
    path = tmp_path / "g.json"
    dal = UserGraphDAL(str(path))
    dal.add_message("u1", "u2", sentiment_score=0.5)
    dal.add_message("u1", "u2", sentiment_score=-0.2)

    assert dal.get_affinity("u1") == 2
    count, ssum = dal.get_relationship("u1", "u2")
    assert count == 2
    assert ssum == 0.3


def test_friendliness_and_hostility(tmp_path):
    path = tmp_path / "g.json"
    dal = UserGraphDAL(str(path))
    dal.add_message("u1", "u2", sentiment_score=0.5)
    dal.add_message("u1", "u2", sentiment_score=-1.0)

    assert dal.get_friendliness("u1", "u2") == 0.0
    assert dal.get_hostility("u1", "u2") < 0
