from datetime import UTC, datetime

from deepthought.quest import Epiphany, Evidence, LieRecord, Objective, Quest
from deepthought.quest.dsl import load_quests, save_quests


def test_load_and_save_roundtrip(tmp_path):
    quest = Quest(
        id=1,
        name="Quest",
        description="desc",
        quest_type="main",
        priority=1,
        horizon="short",
        faction="alpha",
        cover_story="cover",
        secrecy="low",
        risk="low",
        status="pending",
        created=datetime.now(UTC),
    )
    expiry = datetime.now(UTC).replace(microsecond=0)
    obj = Objective(
        id=10,
        quest_id=1,
        description="Do thing",
        status="done",
        preconditions=["prep"],
        success_criteria=["success"],
        fail_criteria=["fail"],
        fallbacks=["fallback"],
        cooldowns=["1h"],
    )
    obj.evidence.append(
        Evidence(
            id=None,
            objective_id=10,
            content="proof",
            who="agent",
            confidence_delta=0.5,
            expiry=expiry,
        )
    )
    quest.objectives.append(obj)
    quest.epiphanies.append(
        Epiphany(
            id=None,
            quest_id=1,
            insight="idea",
            who="sage",
            confidence_delta=0.2,
            expiry=expiry,
        )
    )
    quest.lies.append(
        LieRecord(
            id=None,
            quest_id=1,
            lie="fib",
            who="trickster",
            confidence_delta=-0.3,
            expiry=expiry,
        )
    )

    path = tmp_path / "quests.json"
    save_quests(path, [quest])

    loaded = load_quests(path)
    assert len(loaded) == 1
    q = loaded[0]
    assert q.name == "Quest"
    obj_f = q.objectives[0]
    assert obj_f.preconditions == ["prep"]
    assert obj_f.success_criteria == ["success"]
    assert obj_f.fail_criteria == ["fail"]
    assert obj_f.evidence[0].who == "agent" and obj_f.evidence[0].confidence_delta == 0.5
    assert obj_f.evidence[0].expiry == expiry
    epi_f = q.epiphanies[0]
    assert epi_f.who == "sage" and epi_f.confidence_delta == 0.2
    assert epi_f.expiry == expiry
    lie_f = q.lies[0]
    assert lie_f.who == "trickster" and lie_f.confidence_delta == -0.3
    assert lie_f.expiry == expiry
