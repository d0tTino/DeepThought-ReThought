import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("nats")
from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.eda.events import EventSubjects, InputReceivedPayload
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
async def test_orchestrator_full(tmp_path, nats_container):
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")

    svc_dir = tmp_path / "stub" / "deepthought" / "services" / "demo"
    svc_dir.mkdir(parents=True)
    (svc_dir / "__init__.py").write_text("", encoding="utf-8")
    (svc_dir.parent / "__init__.py").write_text("", encoding="utf-8")
    (svc_dir.parent.parent / "__init__.py").write_text("", encoding="utf-8")
    service_code = '''
import json
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from deepthought.eda import Publisher, Subscriber
from deepthought.eda.events import EventSubjects, ResponseGeneratedPayload

class DemoService:
    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._pub = Publisher(nats_client, js_context)
        self._sub = Subscriber(nats_client, js_context)

    async def _handle(self, msg: Msg) -> None:
        data = json.loads(msg.data.decode())
        text = data.get("user_input", "")
        inp_id = data.get("input_id")
        payload = ResponseGeneratedPayload(final_response=text.upper(), input_id=inp_id)
        await self._pub.publish(EventSubjects.RESPONSE_GENERATED, payload, use_jetstream=True)
        await msg.ack()

    async def start(self, durable_name: str = "demo_listener") -> bool:
        await self._sub.subscribe(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle,
            use_jetstream=True,
            durable=durable_name,
        )
        return True

    async def stop(self) -> None:
        await self._sub.unsubscribe_all()
'''
    (svc_dir / "service.py").write_text(service_code, encoding="utf-8")

    cfg = tmp_path / "orch.yaml"
    cfg.write_text("services:\n  - memory\n  - demo\n", encoding="utf-8")

    env = os.environ.copy()
    extra_path = os.pathsep.join([str(tmp_path / "stub"), str(Path(__file__).resolve().parents[2] / "src")])
    env["PYTHONPATH"] = extra_path

    proc = subprocess.Popen([sys.executable, "-m", "deepthought.cli", "orchestrate", str(cfg)], env=env)
    try:
        nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
        js = nc.jetstream(timeout=10.0)
        await ensure_stream_exists(js)

        received = asyncio.Event()
        holder = {}

        async def handler(msg):
            holder.update(json.loads(msg.data.decode()))
            received.set()
            await msg.ack()

        sub = await js.subscribe(
            subject=EventSubjects.RESPONSE_GENERATED,
            durable="sink",
            cb=handler,
            stream=STREAM_NAME,
        )

        payload = InputReceivedPayload(user_input="hello", input_id=str(uuid.uuid4()))
        await js.publish(EventSubjects.INPUT_RECEIVED, payload.to_json().encode())

        await asyncio.wait_for(received.wait(), timeout=10.0)
        assert holder.get("input_id") == payload.input_id
        await sub.unsubscribe()
        await nc.drain()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
