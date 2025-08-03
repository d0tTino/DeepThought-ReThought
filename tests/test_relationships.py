import pytest

pytest.importorskip("aiosqlite")
import aiosqlite

import examples.social_graph_bot as sg

pytest.importorskip("nats")
from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory
from deepthought.services.user_graph_dal import UserGraphDAL


@pytest.mark.asyncio
async def test_relationship_table_and_updates(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    async with aiosqlite.connect(str(tmp_path / "sg.db")) as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'") as cur:
            row = await cur.fetchone()
        assert row is not None, "relationships table should exist"
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='mutual_affinity'") as cur:
            row = await cur.fetchone()
        assert row is not None, "mutual_affinity table should exist"

    await sg.log_interaction("u1", "u2", sentiment_score=0.3)
    await sg.log_interaction("u1", "u2", sentiment_score=0.2)
    await sg.log_interaction("u1")

    async with aiosqlite.connect(str(tmp_path / "sg.db")) as db:
        async with db.execute(
            "SELECT interaction_count, sentiment_sum, interaction_weight, last_interaction FROM relationships WHERE source_id=? AND target_id=?",
            ("u1", "u2"),
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 2 and row[1] == 0.5 and row[2] == 2
        assert row[3] is not None
        a, b = sorted(("u1", "u2"))
        async with db.execute(
            "SELECT score, interaction_weight FROM mutual_affinity WHERE user_a=? AND user_b=?",
            (a, b),
        ) as cur:
            mrow = await cur.fetchone()
        assert mrow == (2, 2.0)

    friendliness = await sg.get_friendliness("u1", "u2")
    assert pytest.approx(friendliness) == 0.25
    assert await sg.get_hostility("u1", "u2") == 0.0

    await sg.log_interaction("u2", "u1", sentiment_score=-1.0)
    assert await sg.get_friendliness("u2", "u1") == 0.0
    assert await sg.get_hostility("u2", "u1") == -1.0

    await sg.db_manager.close()


def test_user_graph_edges_and_stats(tmp_path):
    dal = UserGraphDAL(str(tmp_path / "g.json"))
    mem = SocialGraphMemory(dal)

    dal.add_message("a", "b", sentiment_score=0.5)
    dal.add_message("b", "a", sentiment_score=-0.2)

    # Both directional edges should be updated
    ab = dal.get_relationship("a", "b")
    ba = dal.get_relationship("b", "a")
    assert ab[0] == 2 and ab[2] == 2
    assert ba[0] == 2 and ba[2] == 2

    # mutual affinity counts actual messages between the pair
    assert dal.get_mutual_affinity("a", "b") == 2

    stats = mem.get_relationship_stats("a", "b")
    assert stats["mutual_affinity"] == 2
    assert stats["a_to_b"]["avg_sentiment"] == pytest.approx(0.15)
    assert stats["b_to_a"]["avg_sentiment"] == pytest.approx(0.15)
    assert stats["a_to_b"]["interaction_weight"] == 2
    assert stats["b_to_a"]["interaction_weight"] == 2
    assert stats["a_to_b"]["last_interaction"] > 0
