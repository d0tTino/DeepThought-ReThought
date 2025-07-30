import pytest

from deepthought.utils.ratelimit import UserRateLimiter


def test_user_rate_limiter(monkeypatch):
    current = 0.0

    def mono():
        return current

    monkeypatch.setattr("deepthought.utils.ratelimit.time.monotonic", mono)

    limiter = UserRateLimiter(1, 1)

    assert limiter.allow("u1")
    assert not limiter.allow("u1")

    current += 0.5
    assert not limiter.allow("u1")

    current += 0.5
    assert limiter.allow("u1")
