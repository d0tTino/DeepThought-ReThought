import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import deepthought.modules.memory_basic as memory_basic
from deepthought.eda.events import EventSubjects, InputReceivedPayload


class DummyNATS:
    def __init__(self):
        self.is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))
        return SimpleNamespace(seq=1, stream="test")


class FailingPublisher(DummyPublisher):
    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        raise RuntimeError("boom")


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


def create_memory(monkeypatch, memory_file, publisher_cls=DummyPublisher):
    monkeypatch.setattr(memory_basic, "Publisher", publisher_cls)
    monkeypatch.setattr(memory_basic, "Subscriber", DummySubscriber)
    return memory_basic.BasicMemory(DummyNATS(), DummyJS(), memory_file=memory_file)


@pytest.mark.asyncio
async def test_handle_input_success(tmp_path, monkeypatch):
    mem_file = tmp_path / "mem.json"
    mem = create_memory(monkeypatch, mem_file)
    payload = InputReceivedPayload(user_input="hello", input_id="42")
    msg = DummyMsg(payload.to_json())
    await mem._handle_input_event(msg)

    assert msg.acked
    pub = mem._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "42"
    with open(mem_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    assert history[-1]["user_input"] == "hello"
    ts = sent_payload.timestamp
    assert datetime.fromisoformat(ts).tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_handle_input_error(tmp_path, monkeypatch):
    mem_file = tmp_path / "mem.json"
    mem = create_memory(monkeypatch, mem_file, FailingPublisher)
    payload = InputReceivedPayload(user_input="boom", input_id="99")
    msg = DummyMsg(payload.to_json())
    await mem._handle_input_event(msg)

    assert msg.acked  # Even on error, ack() should be called
    assert mem._publisher.published == []
    with open(mem_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    assert history[-1]["user_input"] == "boom"


def test_read_memory_invalid_json_logs_error(tmp_path, monkeypatch, caplog):
    mem_file = tmp_path / "mem.json"
    mem_file.write_text("{ invalid json")
    mem = create_memory(monkeypatch, mem_file)

    with caplog.at_level(logging.ERROR):
        data = mem._read_memory()

    assert data == []
    assert any("Corrupted memory file" in record.getMessage() for record in caplog.records)


def test_read_memory_invalid_json_backup(tmp_path, monkeypatch):
    mem_file = tmp_path / "mem.json"
    mem_file.write_text("{ invalid json")
    mem = create_memory(monkeypatch, mem_file)

    data = mem._read_memory()

    assert data == []
    backup_path = tmp_path / "mem.json.bak"
    assert backup_path.exists()
    assert mem_file.exists()
    assert json.load(open(mem_file, "r", encoding="utf-8")) == []
    assert "{ invalid json" in backup_path.read_text()
