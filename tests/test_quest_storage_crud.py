import pytest
from deepthought.services import DBManager
from deepthought.quest import (
    Evidence,
    Epiphany,
    LieRecord,
    Objective,
    Quest,
    QuestStorage,
)

pytest.importorskip("aiosqlite")


@pytest.mark.asyncio
async def test_quest_crud(tmp_path):
    db_file = tmp_path / "db.sqlite"
    db = DBManager(str(db_file))
    storage = QuestStorage(db)

    quest = Quest(
        id=None,
        name="Test Quest",
        description="desc",
        quest_type="main",
        priority=1,
        horizon="short",
        faction="alpha",
        cover_story="cover",
        secrecy="low",
        risk="minimal",
        status="pending",
    )

    quest_id = await storage.add_quest(quest)
    assert quest_id

    fetched = await storage.get_quest(quest_id)
    assert fetched is not None
    assert fetched.quest_type == "main"
    assert fetched.priority == 1
    assert fetched.horizon == "short"
    assert fetched.faction == "alpha"
    assert fetched.cover_story == "cover"
    assert fetched.secrecy == "low"
    assert fetched.risk == "minimal"
    assert fetched.created is not None
    assert fetched.updated is not None

    fetched.priority = 5
    fetched.status = "done"
    await storage.update_quest(fetched)

    updated = await storage.get_quest(quest_id)
    assert updated.priority == 5
    assert updated.status == "done"
    assert updated.updated is not None

    await storage.delete_quest(quest_id)
    assert await storage.get_quest(quest_id) is None

    await db.close()


@pytest.mark.asyncio
async def test_nested_entities(tmp_path):
    db_file = tmp_path / "db.sqlite"
    db = DBManager(str(db_file))
    storage = QuestStorage(db)

    quest = Quest(id=None, name="Q", description="desc")
    quest_id = await storage.add_quest(quest)

    obj = Objective(id=None, quest_id=quest_id, description="obj")
    obj_id = await storage.add_objective(obj)

    ev = Evidence(id=None, objective_id=obj_id, content="proof")
    await storage.add_evidence(ev)

    epi = Epiphany(id=None, quest_id=quest_id, insight="aha")
    await storage.add_epiphany(epi)

    lie = LieRecord(id=None, quest_id=quest_id, lie="fib")
    await storage.add_lie(lie)

    fetched = await storage.get_quest(quest_id)
    assert fetched is not None
    assert fetched.objectives and fetched.objectives[0].evidence[0].content == "proof"
    assert fetched.epiphanies and fetched.epiphanies[0].insight == "aha"
    assert fetched.lies and fetched.lies[0].lie == "fib"

    await storage.delete_quest(quest_id)
    assert await storage.get_quest(quest_id) is None

    await db.close()
