import asyncio
import shutil
import socket
import subprocess
import time
from aiohttp import web
import httpx
import pytest
import pytest_asyncio

from tests.helpers import nats_server_available

pytestmark = pytest.mark.nats


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def nats_server(tmp_path_factory):
    if shutil.which("nats-server") is None:
        pytest.skip("nats-server executable not found")
    port = _find_free_port()
    data_dir = tmp_path_factory.mktemp("nats-data")
    proc = subprocess.Popen(
        ["nats-server", "-js", "-p", str(port), "-sd", str(data_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(20):
        if nats_server_available(f"nats://localhost:{port}"):
            break
        time.sleep(0.25)
    else:
        proc.terminate()
        raise RuntimeError("nats-server failed to start")
    yield f"nats://localhost:{port}"
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest_asyncio.fixture()
async def memory_app():
    port = _find_free_port()
    storage: dict[str, list[str]] = {}
    app = web.Application()

    async def add(request: web.Request) -> web.Response:
        data = await request.json()
        uid = str(data.get("user_id"))
        text = data.get("memory", "")
        storage.setdefault(uid, []).append(text)
        return web.json_response({"status": "ok"})

    async def query(request: web.Request) -> web.Response:
        data = await request.json()
        uid = str(data.get("user_id"))
        return web.json_response({"memories": storage.get(uid, [])})

    app.router.add_post("/memory/add", add)
    app.router.add_post("/memory/query", query)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", port)
    await site.start()
    try:
        yield f"http://localhost:{port}"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_memory_add_and_query(memory_app, nats_server):
    async with httpx.AsyncClient(base_url=memory_app) as client:
        resp = await client.post("/memory/add", json={"user_id": 1, "memory": "hello"})
        assert resp.status_code == 200
        resp = await client.post("/memory/query", json={"user_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["memories"] == ["hello"]

