import os
import shutil
import socket
import subprocess
import time
from typing import Optional
from urllib.parse import urlparse

import pytest

DEFAULT_NATS_PORT = 4222


def nats_server_available(url: Optional[str] = None) -> bool:
    """Return ``True`` if a NATS server can be reached at ``url``.

    ``url`` may be a bare ``host:port`` pair or a full URL with scheme. If not
    provided, the ``NATS_URL`` environment variable or ``nats://localhost:4222``
    is used.
    """
    if url is None:
        url = os.getenv("NATS_URL", f"nats://localhost:{DEFAULT_NATS_PORT}")

    parsed = urlparse(url if "://" in url else f"//{url}")
    host = parsed.hostname
    port = parsed.port or DEFAULT_NATS_PORT
    if not host:
        return False

    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except Exception:
        return False


def memgraph_available(host: str | None = None, port: int | None = None) -> bool:
    """Return ``True`` if a Memgraph server is reachable."""
    host = host or os.getenv("MG_HOST", "localhost")
    port = int(port or os.getenv("MG_PORT", 7687))
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False


def chroma_available(host: str | None = None, port: int | None = None) -> bool:
    """Return ``True`` if a Chroma service is reachable."""
    host = host or os.getenv("CHROMA_HOST", "localhost")
    port = int(port or os.getenv("CHROMA_PORT", 8000))
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False


def neo4j_available(host: str | None = None, port: int | None = None) -> bool:
    """Return ``True`` if a Neo4j server is reachable."""
    host = host or os.getenv("NEO4J_HOST", "localhost")
    port = int(port or os.getenv("NEO4J_PORT", 7687))
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def nats_server(tmp_path_factory):
    """Run a temporary NATS server with JetStream enabled."""
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
