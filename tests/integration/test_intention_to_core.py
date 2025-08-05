import asyncio
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("nats")

pytest_plugins = ["tests.helpers"]

from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.eda.publisher import connect
from deepthought.goal_scheduler import GoalScheduler
from deepthought.services.cognitive_core_service import CognitiveCoreService

pytestmark = pytest.mark.nats

STREAM_NAME = "deepthought_events"


class _DummyDB:
    async def close(self) -> None:  # pragma: no cover - simple stub
        return None


@pytest.mark.asyncio
async def test_intention_reaches_cognitive_core(nats_server, caplog):
    publisher = await connect(nats_server)
    nc = publisher._nc
    js = publisher._js

    try:
        await js.stream_info(STREAM_NAME)
    except Exception:
        config = StreamConfig(
            name=STREAM_NAME,
            subjects=["dtr.>"],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=100,
            discard=DiscardPolicy.OLD,
        )
        await js.add_stream(config)

    service = CognitiveCoreService(
        nats_client=nc,
        js_context=js,
        memory=SimpleNamespace(),
        db=_DummyDB(),
        search=None,
    )
    await service.start()

    sched = GoalScheduler()
    sched.add_goal("integration-goal", priority=1)

    with caplog.at_level(logging.INFO):
        await sched.publish_intentions(publisher)
        await asyncio.sleep(0.5)

    await service.stop()
    if nc.is_connected:
        await nc.close()

    assert any("integration-goal" in r.getMessage() for r in caplog.records)
