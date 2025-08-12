import pytest

pytest.importorskip("aiosqlite")

from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory
from deepthought.services.prism_adapter import PrismAdapter


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


@pytest.mark.asyncio
async def test_prism_event_updates_affinity_and_edges(tmp_path):
    db_path = tmp_path / "sg.db"
    mem = SocialGraphMemory(DBManager(str(db_path)))
    adapter = PrismAdapter(mem)

    await adapter.ingest(
        {
            "source": "alice",
            "target": "bob",
            "sentiment": 0.7,
            "reply_latency": 1.0,
            "emoji_counts": {"❤️": 1},
        }
    )

    await adapter.ingest(
        {
            "source": "mallory",
            "target": "trent",
            "sentiment": -0.6,
            "reply_latency": 10.0,
            "emoji_counts": {"💔": 2},
        }
    )

    assert await mem.get_affinity("alice") > 0
    assert await mem.get_affinity("mallory") < 0
    assert await mem.get_edge_weight("alice", "bob", "ally") > 0
    assert await mem.get_edge_weight("mallory", "trent", "rival") > 0
    await mem.close()
