import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from deepthought.eda.events import EventSubjects
from deepthought.modules.input_handler import InputHandler


class DummyNATS:
    def __init__(self):
        self.is_connected = True
        self.published = []

    async def publish(self, subject, data):
        self.published.append((subject, data))


class DummyJS:
    def __init__(self):
        self.published = []

    async def publish(self, subject, data, timeout=10.0):
        self.published.append((subject, data))
        return SimpleNamespace(seq=1, stream="test")


@pytest.mark.asyncio
async def test_process_input_success():
    js = DummyJS()
    nc = DummyNATS()
    handler = InputHandler(nc, js)
    input_id = await handler.process_input("hello")

    assert js.published
    subject, data = js.published[0]
    assert subject == EventSubjects.INPUT_RECEIVED
    payload = json.loads(data.decode())
    assert payload["user_input"] == "hello"
    assert payload["input_id"] == input_id
    # Timestamp should be timezone-aware UTC
    ts = payload["timestamp"]
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo == timezone.utc


class FailingJS(DummyJS):
    async def publish(self, subject, data, timeout=10.0):
        raise RuntimeError("publish error")


@pytest.mark.asyncio
async def test_process_input_error():
    js = FailingJS()
    nc = DummyNATS()
    handler = InputHandler(nc, js)
    with pytest.raises(RuntimeError):
        await handler.process_input("boom")
