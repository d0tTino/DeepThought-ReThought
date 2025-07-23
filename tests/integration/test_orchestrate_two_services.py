import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

for m in [
    "nats",
    "nats.aio",
    "nats.aio.client",
    "nats.aio.msg",
    "nats.js",
    "nats.js.client",
    "nats.js.api",
    "nats.errors",
]:
    sys.modules.pop(m, None)

from tests.helpers import nats_server_available

pytest.importorskip("nats")
from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.eda.events import EventSubjects, InputReceivedPayload

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


@pytest.fixture(scope="module")
def nats_container():
    compose = None
    if shutil.which("docker-compose"):
        compose = ["docker-compose"]
    elif shutil.which("docker"):
        compose = ["docker", "compose"]
    if compose is None:
        pytest.skip("Docker compose not available")

    already_running = nats_server_available(get_nats_url())

    if not already_running:
        cmd = compose + ["-f", "docker-compose.yml", "up", "-d", "nats"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"Could not start nats service: {proc.stderr}")

        for _ in range(30):
            if nats_server_available(get_nats_url()):
                break
            time.sleep(1)
        else:
            subprocess.run(compose + ["-f", "docker-compose.yml", "down"], capture_output=True)
            pytest.skip("NATS container did not start")

    try:
        yield
    finally:
        if not already_running:
            subprocess.run(compose + ["-f", "docker-compose.yml", "down"], capture_output=True)


@pytest.mark.asyncio
async def test_orchestrate_two_services(tmp_path, nats_container):
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    sitecustomize_code = """
import importlib.metadata as im
import json
from types import SimpleNamespace
from deepthought.eda import Publisher, Subscriber
from deepthought.eda.events import EventSubjects, ResponseGeneratedPayload

class UpperService:
    def __init__(self, nc, js):
        self._pub = Publisher(nc, js)
        self._sub = Subscriber(nc, js)

    async def _handle(self, msg):
        data = json.loads(msg.data.decode())
        text = data.get("user_input", "")
        inp_id = data.get("input_id")
        payload = ResponseGeneratedPayload(final_response=text.upper(), input_id=inp_id)
        await self._pub.publish(EventSubjects.RESPONSE_GENERATED, payload, use_jetstream=True)
        await msg.ack()

    async def start(self, durable_name="upper"):
        await self._sub.subscribe(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle,
            durable=durable_name,
            use_jetstream=True,
        )
        return True

    async def stop(self):
        await self._sub.unsubscribe_all()


class LowerService:
    def __init__(self, nc, js):
        self._pub = Publisher(nc, js)
        self._sub = Subscriber(nc, js)

    async def _handle(self, msg):
        data = json.loads(msg.data.decode())
        text = data.get("user_input", "")
        inp_id = data.get("input_id")
        payload = ResponseGeneratedPayload(final_response=text.lower(), input_id=inp_id)
        await self._pub.publish(EventSubjects.RESPONSE_GENERATED, payload, use_jetstream=True)
        await msg.ack()

    async def start(self, durable_name="lower"):
        await self._sub.subscribe(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle,
            durable=durable_name,
            use_jetstream=True,
        )
        return True

    async def stop(self):
        await self._sub.unsubscribe_all()


def _patched_entry_points():
    orig = im.entry_points()
    ep_u = SimpleNamespace(name="upper", load=lambda: UpperService)
    ep_l = SimpleNamespace(name="lower", load=lambda: LowerService)
    if hasattr(orig, "select"):
        class Wrapper:
            def select(self, **kw):
                group = kw.get("group")
                sel = orig.select(**kw)
                if group == "deepthought.services":
                    return list(sel) + [ep_u, ep_l]
                return sel
        return Wrapper()
    eps = dict(orig)
    eps.setdefault("deepthought.services", []).extend([ep_u, ep_l])
    return eps

im.entry_points = _patched_entry_points
"""
    (stub_dir / "sitecustomize.py").write_text(sitecustomize_code, encoding="utf-8")

    cfg = tmp_path / "orc.yaml"
    cfg.write_text("services:\n  - upper\n  - lower\n", encoding="utf-8")

    env = os.environ.copy()
    extra_path = os.pathsep.join([str(stub_dir), str(Path(__file__).resolve().parents[2] / "src")])
    env["PYTHONPATH"] = extra_path

    proc = subprocess.Popen([sys.executable, "-m", "deepthought.cli", "orchestrate", str(cfg)], env=env)
    try:
        nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
        js = nc.jetstream(timeout=10.0)
        await ensure_stream_exists(js)

        received_u = asyncio.Event()
        received_l = asyncio.Event()
        holder = {"upper": None, "lower": None}

        async def handler(msg):
            data = json.loads(msg.data.decode())
            text = data.get("final_response")
            if text.isupper():
                holder["upper"] = text
                received_u.set()
            else:
                holder["lower"] = text
                received_l.set()
            await msg.ack()

        sub = await js.subscribe(
            subject=EventSubjects.RESPONSE_GENERATED,
            durable="sink",
            cb=handler,
            stream=STREAM_NAME,
        )

        payload = InputReceivedPayload(user_input="HeLLo", input_id=str(uuid.uuid4()))
        await js.publish(EventSubjects.INPUT_RECEIVED, payload.to_json().encode())

        await asyncio.wait_for(asyncio.gather(received_u.wait(), received_l.wait()), timeout=10.0)
        assert holder["upper"] == "HELLO"
        assert holder["lower"] == "hello"

        await sub.unsubscribe()
        await nc.drain()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
