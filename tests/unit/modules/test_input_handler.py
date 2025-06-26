import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from deepthought.eda.events import EventSubjects
from deepthought.modules.input_handler import InputHandler


class DummyMemory:
    def __init__(self):
        self.prompt = None

    def retrieve_context(self, prompt):
        self.prompt = prompt
        return ["fact1", "fact2"]

    async def start(self):
        return True

    async def stop(self):
        pass


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
    memory = DummyMemory()
    handler = InputHandler(nc, js, memory_service=memory)
    input_id = await handler.process_input("hello")

    assert js.published
    # First publish: INPUT_RECEIVED
    subject, data = js.published[0]
    assert subject == EventSubjects.INPUT_RECEIVED
    payload = json.loads(data.decode())
    assert payload["user_input"] == "hello"
    assert payload["input_id"] == input_id
    ts = payload["timestamp"]
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo == timezone.utc

    # Second publish: MEMORY_RETRIEVED
    subject, data = js.published[1]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    memory_payload = json.loads(data.decode())
    assert memory_payload["input_id"] == input_id
    assert memory_payload["retrieved_knowledge"]["retrieved_knowledge"]["facts"] == [
        "fact1",
        "fact2",
    ]
    assert memory.prompt == "hello"


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


@pytest.mark.asyncio
async def test_process_input_invalid_type():
    js = DummyJS()
    nc = DummyNATS()
    handler = InputHandler(nc, js)
    with pytest.raises(ValueError):
        await handler.process_input(123)


@pytest.mark.asyncio
async def test_process_input_no_memory():
    js = DummyJS()
    nc = DummyNATS()
    handler = InputHandler(nc, js)
    input_id = await handler.process_input("hello")
    assert len(js.published) == 1
    subject, _ = js.published[0]
    assert subject == EventSubjects.INPUT_RECEIVED
    assert input_id
