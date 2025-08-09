from datetime import timedelta

import pytest

from deepthought.quest.templates import (
    BRIDGE_BUILDER,
    SIDE,
    CooldownTracker,
    HorizonManager,
    HorizonRule,
    auto_spawn_quests,
)
from deepthought.services.db_manager import DBManager
from deepthought.services.social_graph_memory import SocialGraphMemory


async def _make_memory():
    memory = SocialGraphMemory(DBManager(":memory:"))
    # ensure at least one user exists
    await memory.record_message("u1", "hello")
    return memory


@pytest.mark.asyncio
async def test_auto_spawn_respects_budget_and_ttl():
    memory = await _make_memory()
    tracker = CooldownTracker()
    rules = {"short": HorizonRule(limit=1, cooldown=timedelta(hours=1), ttl=timedelta(hours=2))}
    manager = HorizonManager(rules)

    quests1 = await auto_spawn_quests(manager, memory, tracker, templates=[SIDE])
    assert len(quests1) == 1

    # second spawn should be blocked by budget/cooldown
    quests2 = await auto_spawn_quests(manager, memory, tracker, templates=[SIDE])
    assert quests2 == []

    # Age the active quest beyond TTL and cooldown to free slot
    manager._active["short"][0] -= timedelta(hours=3)
    manager._last_spawn["short"] -= timedelta(hours=3)

    quests3 = await auto_spawn_quests(manager, memory, tracker, templates=[SIDE])
    assert len(quests3) == 1
    await memory.close()


@pytest.mark.asyncio
async def test_bridge_builder_uses_relationship_metrics():
    memory = SocialGraphMemory(DBManager(":memory:"))
    await memory._db.connect()
    assert memory._db._db is not None
    # two users with mutual affinity record
    await memory._db._db.execute("INSERT INTO affinity (user_id, score) VALUES ('a', 1)")
    await memory._db._db.execute("INSERT INTO affinity (user_id, score) VALUES ('b', 1)")
    await memory._db._db.execute("INSERT INTO mutual_affinity (user_a, user_b, score) VALUES ('a', 'b', -5)")
    await memory._db._db.commit()

    tracker = CooldownTracker()
    rules = {"medium": HorizonRule(limit=1, cooldown=timedelta(0), ttl=timedelta(hours=1))}
    manager = HorizonManager(rules)

    quests = await auto_spawn_quests(manager, memory, tracker, templates=[BRIDGE_BUILDER])
    assert len(quests) == 1
    # the description should mention the pair identified by mutual affinity
    desc = quests[0].description
    assert "a" in desc and "b" in desc
    await memory.close()
