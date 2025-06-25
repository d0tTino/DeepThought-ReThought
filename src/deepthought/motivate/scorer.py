"""Simple deterministic caption scorer."""

from hashlib import sha256


def score_caption(caption: str, nonce: str) -> int:
    """Return a pseudo-random score 1-7 based on caption and nonce."""
    digest = sha256((nonce + caption).encode()).digest()
    return 1 + digest[0] % 7
