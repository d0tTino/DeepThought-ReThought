import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("nats")
from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.eda.events import EventSubjects
from tests.helpers import nats_server, nats_server_available

services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "deepthought" / "services")]
sys.modules.setdefault("deepthought.services", services_pkg)

fake_vid = types.ModuleType("deepthought.services.perception.worker_video")
fake_vid.VideoPerceptionWorker = object
sys.modules.setdefault("deepthought.services.perception.worker_video", fake_vid)

fake_audio = types.ModuleType("deepthought.services.perception.worker_audio")
fake_audio.AudioPerceptionWorker = object
sys.modules.setdefault("deepthought.services.perception.worker_audio", fake_audio)

cli = importlib.import_module("deepthought.services.perception.cli")

pytestmark = pytest.mark.nats


@pytest.mark.asyncio
async def test_perception_cli_replay_honors_consent(monkeypatch, nats_server):
    if not nats_server_available(nats_server):
        pytest.skip("NATS server not available")

    monkeypatch.setenv("PERCEPTION_REQUIRE_CONSENT", "1")

    nc = await NATS().connect(servers=[nats_server])
    js = nc.jetstream()

    cfg = StreamConfig(
        name="deepthought_events",
        subjects=["dtr.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.MEMORY,
        max_msgs_per_subject=100,
        discard=DiscardPolicy.OLD,
    )
    try:
        await js.add_stream(cfg)
    except Exception:
        pass

    await js.publish(
        EventSubjects.INPUT_RECEIVED,
        json.dumps({"user_input": "contact me at a@b.com", "input_id": "m1", "consent": True}).encode(),
    )
    await js.publish(
        EventSubjects.INPUT_RECEIVED,
        json.dumps({"user_input": "no consent here", "input_id": "m2"}).encode(),
    )

    received = []
    done = asyncio.Event()

    async def sink(msg):
        received.append(json.loads(msg.data.decode()))
        done.set()
        await msg.ack()

    sub = await js.subscribe(EventSubjects.PERCEPTION_EMBEDDINGS, durable="sink", cb=sink)

    argv = [
        "prog",
        "--nats-url",
        nats_server,
        "--listen",
        "--durable",
        "replay",
        "--replay",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    task = asyncio.create_task(cli._main())

    await asyncio.wait_for(done.wait(), timeout=5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await sub.unsubscribe()
    await nc.drain()

    assert len(received) == 1
    assert received[0]["message_id"] == "m1"
