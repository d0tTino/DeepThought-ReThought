import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("nats")
from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from tests.helpers import nats_server_available

pytestmark = pytest.mark.nats

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


def _compose_cmd():
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    if shutil.which("docker"):
        return ["docker", "compose"]
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["python", "go", "ts"])
async def test_bus_project(tmp_path: Path, language: str) -> None:
    compose = _compose_cmd()
    if compose is None:
        pytest.skip("Docker compose not available")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deepthought.cli",
            "bus",
            "init",
            "project",
            "demo",
            "--language",
            language,
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    compose_file = tmp_path / "demo" / "docker-compose.yml"

    proc = subprocess.run(
        compose + ["-f", str(compose_file), "up", "-d", "--build"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"Could not start compose: {proc.stderr}")
    try:
        for _ in range(30):
            if nats_server_available("localhost:4222"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("NATS did not start")

        nc = await NATS().connect(servers=["nats://localhost:4222"], connect_timeout=10)
        js = nc.jetstream(timeout=10.0)
        await ensure_stream_exists(js)

        received = asyncio.Event()

        async def handler(msg):
            received.set()
            await msg.ack()

        sub = await js.subscribe("dtr.template.output", durable="sink", cb=handler, stream=STREAM_NAME)
        await js.publish("dtr.template.input", b"test")
        await asyncio.wait_for(received.wait(), timeout=20.0)
        await sub.unsubscribe()
        await nc.drain()
    finally:
        subprocess.run(compose + ["-f", str(compose_file), "down"], capture_output=True)
