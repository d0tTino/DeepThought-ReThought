import pytest

from deepthought.services import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory
from deepthought.quest.templates import CooldownTracker, auto_spawn_quests

pytest.importorskip("aiosqlite")


@pytest.mark.asyncio
async def test_auto_spawn_respects_budget(tmp_path):
    db = DBManager(str(tmp_path / "sg.db"))
    memory = SocialGraphMemory(db)
    await db.adjust_affinity("u1", 5)

    tracker = CooldownTracker()
    budget = {"short": 1, "medium": 1, "long": 0}

    quests = await auto_spawn_quests(budget, memory, tracker)
    names = [q.name for q in quests]

    assert names == ["Investigation", "Side"]
    assert budget["short"] == 0
    assert budget["medium"] == 0

    await db.close()
