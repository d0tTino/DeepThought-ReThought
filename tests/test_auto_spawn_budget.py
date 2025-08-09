import pytest

from deepthought.quest.templates import (
    CooldownTracker,
    HorizonManager,
    HorizonRule,
    auto_spawn_quests,
)
from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory

pytest.importorskip("aiosqlite")


@pytest.mark.asyncio
async def test_auto_spawn_respects_budget(tmp_path):
    db = DBManager(str(tmp_path / "sg.db"))
    memory = SocialGraphMemory(db)
    await db.adjust_affinity("u1", 5)

    tracker = CooldownTracker()
    rules = {
        "short": HorizonRule(limit=1),
        "medium": HorizonRule(limit=1),
        "long": HorizonRule(limit=0),
    }
    manager = HorizonManager(rules)

    quests = await auto_spawn_quests(manager, memory, tracker)
    names = [q.name for q in quests]

    assert names == ["Investigation", "Side"]
    assert not manager.can_spawn("short")
    assert not manager.can_spawn("medium")

    await memory.close()
