import asyncio
import json
import os
import tempfile

import pytest
pytest.importorskip("nats")
from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.eda.events import EventSubjects
from deepthought.modules.input_handler import InputHandler
from deepthought.services import DBManager
from deepthought.services.social_graph_service import SocialGraphService
from tests.helpers import nats_server_available

pytestmark = pytest.mark.nats


def get_nats_url() -> str:
    return os.getenv("NATS_URL", "nats://localhost:4222")


STREAM_NAME = "deepthought_events"


async def ensure_stream_exists(js):
    try:
        await js.stream_info(STREAM_NAME)
    except Exception:
        cfg = StreamConfig(
            name=STREAM_NAME,
            subjects=["dtr.>"],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=100,
            discard=DiscardPolicy.OLD,
        )
        await js.add_stream(cfg)


@pytest.mark.asyncio
async def test_social_graph_service_flow(tmp_path):
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")

    nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
    js = nc.jetstream(timeout=10.0)
    await ensure_stream_exists(js)

    db_file = tmp_path / "sg.db"
    db = DBManager(str(db_file))
    await db.connect()
    await db.init_db()

    service = SocialGraphService(nc, js, db)
    await service.start(durable_name="sg_listener")

    received = asyncio.Event()
    payload_holder = {}

    async def handler(msg):
        payload_holder.update(json.loads(msg.data.decode()))
        received.set()
        await msg.ack()

    sub = await js.subscribe(
        subject=EventSubjects.MEMORY_RETRIEVED,
        durable="sink",
        cb=handler,
        stream=STREAM_NAME,
    )

    handler_mod = InputHandler(nc, js)
    input_id = await handler_mod.process_input("hello world")

    await asyncio.wait_for(received.wait(), timeout=5.0)

    assert payload_holder.get("input_id") == input_id
    assert payload_holder.get("retrieved_knowledge", {}).get("source") == "social_graph_service"

    await sub.unsubscribe()
    await service.stop()
    await db.close()
    await nc.close()
