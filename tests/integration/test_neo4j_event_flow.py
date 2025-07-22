import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid

import pytest

pytest.importorskip("nats")
from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.modules.memory_kg import KnowledgeGraphMemory
from deepthought.graph import GraphDAL, Neo4jConnector
from tests.helpers import neo4j_available, nats_server_available

pytestmark = pytest.mark.neo4j

NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = int(os.getenv("NEO4J_PORT", 7687))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

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


@pytest.fixture(scope="module")
def neo4j_server():
    try:
        import neo4j  # noqa: F401
    except Exception:
        pytest.skip("neo4j-driver not installed")
    compose = None
    if shutil.which("docker-compose"):
        compose = ["docker-compose"]
    elif shutil.which("docker"):
        compose = ["docker", "compose"]
    if compose is None:
        pytest.skip("Docker compose not available")

    already_running = neo4j_available(NEO4J_HOST, NEO4J_PORT)

    if not already_running:
        cmd = compose + ["-f", "docker-compose.yml", "up", "-d", "neo4j"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"Could not start neo4j service: {proc.stderr}")

        for _ in range(30):
            if neo4j_available(NEO4J_HOST, NEO4J_PORT):
                break
            time.sleep(1)
        else:
            subprocess.run(compose + ["-f", "docker-compose.yml", "down"], capture_output=True)
            pytest.skip("Neo4j container did not start")

    try:
        yield
    finally:
        if not already_running:
            subprocess.run(compose + ["-f", "docker-compose.yml", "down"], capture_output=True)


@pytest.mark.asyncio
async def test_neo4j_event_flow(neo4j_server, nats_container):
    if not neo4j_available(NEO4J_HOST, NEO4J_PORT):
        pytest.skip("Neo4j not reachable")
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")

    nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
    js = nc.jetstream(timeout=10.0)
    await ensure_stream_exists(js)

    connector = Neo4jConnector(
        host=NEO4J_HOST,
        port=NEO4J_PORT,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
    dal = GraphDAL(connector)

    kg = KnowledgeGraphMemory(nc, js, dal)
    await kg.start_listening(durable_name="kg_neo4j_listener")

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

    payload = InputReceivedPayload(user_input="alpha beta", input_id=str(uuid.uuid4()))
    await js.publish(EventSubjects.INPUT_RECEIVED, payload.to_json().encode())

    await asyncio.wait_for(received.wait(), timeout=10.0)
    assert holder.get("input_id") == payload.input_id

    rows = dal.query_subgraph(
        "MATCH (a:Entity {name: $src})-[:NEXT]->(b:Entity {name: $dst}) RETURN a.name AS src, b.name AS dst",
        {"src": "alpha", "dst": "beta"},
    )
    assert rows

    await sub.unsubscribe()
    await kg.stop_listening()
    await nc.drain()
