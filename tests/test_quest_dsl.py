from datetime import datetime

from deepthought.quest.dsl import load_quests, save_quests
from deepthought.quest import Evidence, Epiphany, LieRecord, Objective, Quest


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
        created=datetime.utcnow(),
    )
    obj = Objective(id=10, quest_id=1, description="Do thing", status="done")
    obj.evidence.append(Evidence(id=None, objective_id=10, content="proof"))
    quest.objectives.append(obj)
    quest.epiphanies.append(Epiphany(id=None, quest_id=1, insight="idea"))
    quest.lies.append(LieRecord(id=None, quest_id=1, lie="fib"))

    path = tmp_path / "quests.json"
    save_quests(path, [quest])

    loaded = load_quests(path)
    assert len(loaded) == 1
    q = loaded[0]
    assert q.name == "Quest"
    assert q.objectives[0].evidence[0].content == "proof"
    assert q.epiphanies[0].insight == "idea"
    assert q.lies[0].lie == "fib"
