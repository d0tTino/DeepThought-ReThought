import pytest

pytest.importorskip("aiosqlite")
import aiosqlite

sg = pytest.importorskip("examples.social_graph_bot")
if not hasattr(sg, "TrustService"):
    pytest.skip("social_graph_bot optional dependencies not installed", allow_module_level=True)

pytest.importorskip("nats")
from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory


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


@pytest.mark.asyncio
async def test_relationship_stats(tmp_path):
    db = DBManager(str(tmp_path / "sg.db"))
    mem = SocialGraphMemory(db)

    await db.log_interaction("a", "b", sentiment_score=0.5)
    await db.log_interaction("b", "a", sentiment_score=-0.2)

    ab = await db.get_relationship("a", "b")
    ba = await db.get_relationship("b", "a")
    assert ab[0] == 1 and ab[2] == 1
    assert ba[0] == 1 and ba[2] == 1

    stats = await mem.get_relationship_stats("a", "b")
    assert stats["mutual_affinity"] == 2
    assert stats["a_to_b"]["avg_sentiment"] == pytest.approx(0.5)
    assert stats["b_to_a"]["avg_sentiment"] == pytest.approx(-0.2)
    assert stats["a_to_b"]["interaction_weight"] == pytest.approx(1.0)
    assert stats["b_to_a"]["interaction_weight"] == pytest.approx(1.0)
    assert stats["a_to_b"]["last_interaction"] is not None


@pytest.mark.asyncio
async def test_relationship_type_computation(tmp_path):
    db_path = tmp_path / "sg.db"
    mem = SocialGraphMemory(DBManager(str(db_path)))

    for _ in range(3):
        await mem.record_message("alice", "I love you", "bob")

    status = await mem.get_relationship_status("alice", "bob")
    assert status == "friend"

    for _ in range(3):
        await mem.record_message("eve", "I hate you", "mallory")

    status = await mem.get_relationship_status("eve", "mallory")
    assert status == "rival"

    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='relationship_types'") as cur:
            assert await cur.fetchone() is not None

    await mem.close()
