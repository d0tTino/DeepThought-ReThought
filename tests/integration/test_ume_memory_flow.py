import asyncio
import json
import os
import uuid

import pytest

pytest.importorskip("nats")
from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.services.memory_service import MemoryService
from deepthought.config import Settings
from tests.helpers import nats_server_available

pytestmark = pytest.mark.nats
STREAM_NAME = "deepthought_events"


def get_nats_url() -> str:
    return os.getenv("NATS_URL", "nats://localhost:4222")


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
async def test_ume_memory_roundtrip(monkeypatch):
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")

    monkeypatch.setenv("DT_MEMORY_BACKEND", "ume")
    nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
    js = nc.jetstream(timeout=10.0)
    await ensure_stream_exists(js)

    service = MemoryService(nc, js, Settings())
    await service.start(durable_name="ume_listener")

    received = asyncio.Event()
    holder = {}

    async def handler(msg):
        holder.update(json.loads(msg.data.decode()))
        received.set()
        await msg.ack()

    sub = await js.subscribe(
        subject=EventSubjects.MEMORY_RETRIEVED,
        durable="sink",
        cb=handler,
        stream=STREAM_NAME,
    )

    payload = InputReceivedPayload(user_input="hello world", input_id=str(uuid.uuid4()))
    await js.publish(EventSubjects.INPUT_RECEIVED, payload.to_json().encode())

    await asyncio.wait_for(received.wait(), timeout=10.0)
    assert holder.get("input_id") == payload.input_id

    ume_backend = service._memory.graph_backend
    assert getattr(ume_backend, "events", [])

    await sub.unsubscribe()
    await service.stop()
    await nc.drain()

