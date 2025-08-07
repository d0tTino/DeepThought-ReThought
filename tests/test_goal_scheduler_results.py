from datetime import datetime

import pytest

from deepthought.goal_scheduler import GoalScheduler
from deepthought.services.db_manager import DBManager


@pytest.mark.asyncio
async def test_record_result_tracks_sub_goals(tmp_path):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    sched = GoalScheduler(manager)
    parent_id = await sched.queue_intention("parent", priority=1, sub_goals=["child"])
    sub_ids = await sched.expand_goal("parent")
    assert sub_ids and sub_ids[0] is not None

    record_parent = sched.get_record(parent_id)
    assert record_parent.sub_goal_ids == sub_ids

    child_id = sub_ids[0]
    sched.record_result(child_id, "done")
    record_child = sched.get_record(child_id)
    assert record_child.outcome == "done"
    assert isinstance(record_child.completed_at, datetime)
