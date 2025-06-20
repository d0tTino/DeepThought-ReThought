import os
import socket
from typing import Optional
from urllib.parse import urlparse

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
