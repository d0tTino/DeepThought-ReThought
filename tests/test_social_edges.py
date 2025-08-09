import pytest

pytest.importorskip("aiosqlite")

from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory


@pytest.mark.asyncio
async def test_inferred_edges(tmp_path):
    db_path = tmp_path / "sg.db"
    mem = SocialGraphMemory(DBManager(str(db_path)))

    for text in ["I love you", "I love you", "I love you"]:
        await mem.record_message("alice", text, "bob")
    assert await mem.get_edge_weight("alice", "bob", "ally") > 0

    for text in ["I hate you", "I hate you", "I hate you"]:
        await mem.record_message("mallory", text, "trent")
    assert await mem.get_edge_weight("mallory", "trent", "rival") > 0

    await mem.close()


@pytest.mark.asyncio
async def test_edge_decay(tmp_path):
    db = DBManager(str(tmp_path / "sg.db"))
    mem = SocialGraphMemory(db)
    await db.set_decay_params(0.5, 1.0)

    await mem.update_edge("a", "b", "ally", 4.0)
    assert db._db is not None
    await db._db.execute(
        "UPDATE social_edges SET last_updated=datetime('now','-2 seconds')"
    )
    await db._db.commit()

    weight = await mem.get_edge_weight("a", "b", "ally")
    assert 0.0 < weight < 4.0
    await mem.close()


@pytest.mark.asyncio
async def test_faction_discovery(tmp_path):
    db = DBManager(str(tmp_path / "sg.db"))
    mem = SocialGraphMemory(db)

    await mem.update_edge("a", "b", "ally", 2.0)
    await mem.update_edge("b", "c", "ally", 2.0)
    await mem.update_edge("d", "e", "ally", 2.0)

    factions = await mem.discover_factions()
    factions = [set(f) for f in factions]
    assert set(["a", "b", "c"]) in factions
    assert set(["d", "e"]) in factions

    await mem.close()
