import asyncio

import pytest

pytest.importorskip("nats")

pytest_plugins = ["tests.helpers"]

from deepthought.eda.events import BDIIntentionPayload, EventSubjects
from deepthought.eda.publisher import connect
from deepthought.goal_scheduler import GoalScheduler
from deepthought.services.db_manager import DBManager

pytestmark = pytest.mark.nats


@pytest.mark.asyncio
async def test_goal_scheduler_persistence(tmp_path, nats_server):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    sched = GoalScheduler(manager)
    await sched.queue_intention("alpha", priority=1)
    await sched.queue_intention("beta", priority=5)

    await manager.close()
    manager = DBManager(str(db_file))
    await manager.init_db()
    sched = GoalScheduler(manager)
    loaded = await sched.load_pending_intentions()
    assert loaded == 2

    publisher = await connect(nats_server)
    nc = publisher._nc
    received: list[BDIIntentionPayload] = []

    async def handler(msg):
        received.append(BDIIntentionPayload.from_json(msg.data.decode()))

    sub = await nc.subscribe(EventSubjects.BDI_INTENTION, cb=handler)
    await asyncio.sleep(0.1)

    count = await sched.publish_intentions(publisher)
    await asyncio.sleep(0.5)

    await sub.unsubscribe()
    await nc.close()
    rows = await manager.list_pending_intentions()
    await manager.close()

    assert count == 2
    assert [p.goal for p in received] == ["beta", "alpha"]
    assert rows == []
