import pytest

from deepthought.goal_scheduler import GoalScheduler
from deepthought.services.db_manager import DBManager


@pytest.mark.asyncio
async def test_expand_goal_persists_sub_goals(tmp_path):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    sched = GoalScheduler(manager)
    sched.add_goal("parent", priority=1, sub_goals=["a", "b"])
    ids = await sched.expand_goal("parent")
    assert len(ids) == 2

    await manager.close()
    manager = DBManager(str(db_file))
    await manager.init_db()
    sched2 = GoalScheduler(manager)
    loaded = await sched2.load_pending_intentions()
    assert loaded == 2
    assert sched2.next_goal() == "a"
    assert sched2.next_goal() == "b"
