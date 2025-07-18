"""Run MemoryService, RemoteLLM, OutputHandler and RewardManager via the orchestrator."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import nats

from deepthought import orchestrator
from deepthought.modules import InputHandler


def _nats_available(url: str) -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url if "://" in url else f"//{url}")
    host = parsed.hostname
    port = parsed.port or 4222
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False


async def _start_nats(port: int) -> subprocess.Popen:
    if shutil.which("nats-server") is None:
        raise RuntimeError("nats-server executable not found")
    proc = subprocess.Popen(["nats-server", "-js", "-p", str(port)])
    for _ in range(20):
        if _nats_available(f"nats://localhost:{port}"):
            break
        time.sleep(0.25)
    else:
        proc.terminate()
        raise RuntimeError("nats-server failed to start")
    return proc


async def main() -> None:
    port = 4222
    nats_proc = await _start_nats(port)
    cfg = Path(tempfile.gettempdir()) / "orchestrator.yaml"
    cfg.write_text(
        """services:\n  - memory\n  - remote_llm\n  - output_handler\n  - reward_manager\n""",
        encoding="utf-8",
    )
    orch_task = asyncio.create_task(orchestrator.run(str(cfg)))
    await asyncio.sleep(2.0)
    nc = await nats.connect(f"nats://localhost:{port}")
    js = nc.jetstream()
    handler = InputHandler(nc, js)
    await handler.process_input("Hello orchestrator")
    await asyncio.sleep(2.0)
    orch_task.cancel()
    try:
        await orch_task
    except asyncio.CancelledError:
        pass
    if nc.is_connected:
        await nc.drain()
    nats_proc.terminate()
    try:
        nats_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        nats_proc.kill()
    print("End-to-end demo complete")


if __name__ == "__main__":
    asyncio.run(main())
