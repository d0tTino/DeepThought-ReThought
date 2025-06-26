import json

import pytest

import deepthought.harness.trace as trace


class DummyNATS:
    is_connected = True


class DummyJS:
    pass


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def subscribe(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    async def unsubscribe_all(self):
        self.calls.clear()


class DummyMsg:
    def __init__(self, data: str) -> None:
        self.data = data.encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


@pytest.mark.asyncio
async def test_handle_input_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(trace, "Subscriber", DummySubscriber)
    outfile = tmp_path / "trace.jsonl"
    recorder = trace.TraceRecorder(DummyNATS(), DummyJS(), str(outfile))
    msg = DummyMsg('{"foo": "bar"}')

    await recorder._handle_input(msg)
    assert msg.acked
    with open(outfile, "r", encoding="utf-8") as f:
        line = f.readline()
    obj = json.loads(line)
    assert obj["event"] == "INPUT_RECEIVED"
    assert obj["payload"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_handle_response_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(trace, "Subscriber", DummySubscriber)
    outfile = tmp_path / "trace.jsonl"
    recorder = trace.TraceRecorder(DummyNATS(), DummyJS(), str(outfile))
    msg = DummyMsg('{"final_response": "ok"}')

    await recorder._handle_response(msg)
    assert msg.acked
    with open(outfile, "r", encoding="utf-8") as f:
        line = f.readline()
    obj = json.loads(line)
    assert obj["event"] == "RESPONSE_GENERATED"
    assert obj["payload"] == {"final_response": "ok"}


@pytest.mark.asyncio
async def test_start_subscribes_to_subjects(monkeypatch, tmp_path):
    sub = DummySubscriber()

    def fake_subscriber(*args, **kwargs):
        return sub

    monkeypatch.setattr(trace, "Subscriber", fake_subscriber)
    recorder = trace.TraceRecorder(DummyNATS(), DummyJS(), str(tmp_path / "f"))
    started = await recorder.start(durable_name="d")
    assert started
    subjects = [kwargs["subject"] for _, kwargs in sub.calls]
    assert trace.EventSubjects.INPUT_RECEIVED in subjects
    assert trace.EventSubjects.RESPONSE_GENERATED in subjects
