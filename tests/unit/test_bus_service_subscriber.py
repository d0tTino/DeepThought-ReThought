import pytest

from deepthought.templates.bus_service import subscriber as bus_subscriber


@pytest.mark.asyncio
async def test_rate_limit(monkeypatch):
    current = 0.0

    def mono():
        return current

    async def fake_sleep(seconds):
        nonlocal current
        current += seconds

    monkeypatch.setattr(bus_subscriber.time, "monotonic", mono)
    monkeypatch.setattr(bus_subscriber.asyncio, "sleep", fake_sleep)

    events = []

    @bus_subscriber.rate_limit(2, 1)
    async def handler(val):
        events.append((val, current))

    await handler(1)
    await handler(2)
    await handler(3)

    assert events == [(1, 0.0), (2, 0.0), (3, 1.0)]
