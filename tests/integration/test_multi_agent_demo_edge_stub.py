import os
import shutil

import pytest
from aiohttp import web

from tests.helpers import nats_server_available

pytestmark = pytest.mark.nats


async def _start_stub_server():
    async def handle(request):
        data = await request.json()
        text = data.get("text", "")
        return web.json_response({"text": f"echo:{text}"})

    app = web.Application()
    app.router.add_post("/generate", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


@pytest.mark.asyncio
async def test_demo_with_stubbed_edge(monkeypatch):
    if shutil.which("docker") is None:
        pytest.skip("Docker not available")
    if not nats_server_available():
        pytest.skip("NATS server not available")

    monkeypatch.delenv("EDGE_IMAGE", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    runner, port = await _start_stub_server()
    monkeypatch.setenv("LLM_ENDPOINT", f"http://localhost:{port}/generate")
    monkeypatch.setenv("LANGGRAPH_RECURSION_LIMIT", "1000")

    try:
        import examples.multi_agent_demo as demo

        await demo.main()
        for handler in demo.output_handlers:
            assert handler.get_all_responses(), "handler received no messages"
    finally:
        await runner.cleanup()
