import os
import socket


def nats_server_available(url=None):
    """Check if a NATS server is reachable."""
    if url is None:
        url = os.getenv("NATS_URL", "nats://localhost:4222")
    try:
        # Extract host and port
        if "://" in url:
            url = url.split("://", 1)[1]
        host, port = url.split(":")
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except Exception:
        return False
    return True
