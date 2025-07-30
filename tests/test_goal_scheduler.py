import asyncio

import pytest

pytest.importorskip("nats")

from deepthought.eda.events import BDIIntentionPayload, EventSubjects
from deepthought.eda.publisher import connect
from deepthought.goal_scheduler import GoalScheduler

pytestmark = pytest.mark.nats


@pytest.mark.asyncio
async def test_goals_persist_and_publish(nats_server):
    publisher = await connect(nats_server)
    nc = publisher._nc
    received: list[BDIIntentionPayload] = []

    async def handler(msg):
        received.append(BDIIntentionPayload.from_json(msg.data.decode()))

    sub = await nc.subscribe(EventSubjects.BDI_INTENTION, cb=handler)
    await asyncio.sleep(0.1)

    sched = GoalScheduler()
    sched.add_goal("low", priority=1)
    sched.add_goal("high", priority=5)
    sched.add_goal("mid", priority=3)

    count = await sched.publish_intentions(publisher)
    await asyncio.sleep(0.5)

    await sub.unsubscribe()
    await nc.close()

    assert count == 3
    assert [p.goal for p in received] == ["high", "mid", "low"]
    assert sched.next_goal() is None
