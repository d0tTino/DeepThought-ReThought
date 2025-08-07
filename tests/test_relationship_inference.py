import pytest

pytest.importorskip("aiosqlite")

from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory


@pytest.mark.asyncio
async def test_friendship_status(tmp_path):
    db_path = tmp_path / "sg.db"
    mem = SocialGraphMemory(DBManager(str(db_path)))

    for text in ["You are awesome", "Great job", "I love you"]:
        await mem.record_message("alice", text, "bob")

    await mem.close()

    mem2 = SocialGraphMemory(DBManager(str(db_path)))
    status = await mem2.get_relationship_status("alice", "bob")
    assert status == "friend"
    await mem2.close()


@pytest.mark.asyncio
async def test_hostility_status(tmp_path):
    db_path = tmp_path / "sg.db"
    mem = SocialGraphMemory(DBManager(str(db_path)))

    for text in ["You are terrible", "Awful job", "I hate you"]:
        await mem.record_message("alice", text, "bob")

    await mem.close()

    mem2 = SocialGraphMemory(DBManager(str(db_path)))
    status = await mem2.get_relationship_status("alice", "bob")
    assert status == "rival"
    await mem2.close()
