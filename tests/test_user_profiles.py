import pytest

pytest.importorskip("aiosqlite")

from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory


@pytest.mark.asyncio
async def test_user_profile_persistence(tmp_path):
    db_path = tmp_path / "sg.db"
    mem = SocialGraphMemory(DBManager(str(db_path)))

    traits = {"openness": 0.7, "tags": ["curious", "analytical"]}
    await mem.set_personality("alice", traits)

    # Ensure data persists after reopening the database
    await mem.close()
    mem2 = SocialGraphMemory(DBManager(str(db_path)))
    stored = await mem2.get_personality("alice")
    assert stored == traits
    await mem2.close()
