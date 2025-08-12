from datetime import UTC, datetime

import pytest

from deepthought.quest import (
    Epiphany,
    Evidence,
    LieRecord,
    Objective,
    Quest,
    QuestStorage,
)
from deepthought.services import DBManager

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

    obj = Objective(
        id=None,
        quest_id=quest_id,
        description="obj",
        preconditions=["prep"],
        success_criteria=["success"],
        fail_criteria=["fail"],
        fallbacks=["fallback"],
        cooldowns=["1h"],
    )
    obj_id = await storage.add_objective(obj)

    expiry = datetime.now(UTC).replace(microsecond=0)
    ev = Evidence(
        id=None,
        objective_id=obj_id,
        content="proof",
        who="agent",
        confidence_delta=0.5,
        expiry=expiry,
    )
    await storage.add_evidence(ev)

    epi = Epiphany(
        id=None,
        quest_id=quest_id,
        insight="aha",
        who="sage",
        confidence_delta=0.2,
        expiry=expiry,
    )
    await storage.add_epiphany(epi)

    lie = LieRecord(
        id=None,
        quest_id=quest_id,
        lie="fib",
        who="trickster",
        confidence_delta=-0.3,
        expiry=expiry,
    )
    await storage.add_lie(lie)

    fetched = await storage.get_quest(quest_id)
    assert fetched is not None
    obj_f = fetched.objectives[0]
    assert obj_f.preconditions == ["prep"]
    assert obj_f.success_criteria == ["success"]
    assert obj_f.fail_criteria == ["fail"]
    assert obj_f.fallbacks == ["fallback"]
    assert obj_f.cooldowns == ["1h"]
    ev_f = obj_f.evidence[0]
    assert ev_f.content == "proof" and ev_f.who == "agent" and ev_f.confidence_delta == 0.5
    assert ev_f.expiry == expiry
    epi_f = fetched.epiphanies[0]
    assert epi_f.insight == "aha" and epi_f.who == "sage" and epi_f.confidence_delta == 0.2
    assert epi_f.expiry == expiry
    lie_f = fetched.lies[0]
    assert lie_f.lie == "fib" and lie_f.who == "trickster" and lie_f.confidence_delta == -0.3
    assert lie_f.expiry == expiry

    await storage.delete_quest(quest_id)
    assert await storage.get_quest(quest_id) is None

    await db.close()
