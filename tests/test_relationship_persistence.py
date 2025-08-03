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

