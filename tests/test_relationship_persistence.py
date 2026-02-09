import pytest

pytest.importorskip("aiosqlite")

from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory


@pytest.mark.asyncio
async def test_relationship_persistence(tmp_path):
    db_path = tmp_path / "sg.db"
    mem = SocialGraphMemory(DBManager(str(db_path)))

    await mem.record_message("alice", "hi", "bob")
    await mem.record_message("bob", "hi", "alice")
    await mem.close()

    mem2 = SocialGraphMemory(DBManager(str(db_path)))
    stats = await mem2.get_relationship_stats("alice", "bob")
    assert stats["mutual_affinity"] == 2
    assert stats["a_to_b"]["count"] == 1
    assert stats["b_to_a"]["count"] == 1
    await mem2.close()



@pytest.mark.asyncio
async def test_memory_persistence_isolated_per_user(tmp_path):
    db_path = tmp_path / "sg.db"
    db = DBManager(str(db_path))

    await db.store_memory("user-a", "alpha fact", topic="bio")
    await db.store_memory("user-b", "beta fact", topic="bio")

    a_rows = await db.recall_user("user-a")
    b_rows = await db.recall_user("user-b")

    assert [memory for _, memory in a_rows] == ["alpha fact"]
    assert [memory for _, memory in b_rows] == ["beta fact"]

    await db.close()
