import json
import sys
import types

import pytest

torch_stub = types.ModuleType("torch")
torch_stub.no_grad = lambda: None
torch_stub.softmax = lambda t, dim=0: [type("T", (), {"tolist": lambda self: [0.0, 0.0, 0.0]})()]
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForSequenceClassification = object
transformers_stub.AutoTokenizer = object
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("transformers", transformers_stub)

# Provide minimal stubs for the nats package used in trace module imports
fake_nats = types.ModuleType("nats")
fake_nats.aio = types.ModuleType("aio")
fake_nats.aio.client = types.ModuleType("client")
fake_nats.aio.msg = types.ModuleType("msg")
fake_nats.js = types.ModuleType("js")
fake_nats.js.client = types.ModuleType("client_js")
fake_nats.aio.client.Client = object
fake_nats.aio.msg.Msg = object
fake_nats.js.client.JetStreamContext = object
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_nats.aio.client)
sys.modules.setdefault("nats.aio.msg", fake_nats.aio.msg)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_nats.js.client)

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
    monkeypatch.setattr(
        trace,
        "analyze_social",
        lambda text: {"flirtation": 0.2, "avoidance": 0.1, "manipulation": 0.0},
    )
    outfile = tmp_path / "trace.jsonl"
    recorder = trace.TraceRecorder(DummyNATS(), DummyJS(), str(outfile))
    msg = DummyMsg('{"user_input": "hi"}')

    await recorder._handle_input(msg)
    assert msg.acked
    with open(outfile, "r", encoding="utf-8") as f:
        line = f.readline()
    obj = json.loads(line)
    assert obj["event"] == "INPUT_RECEIVED"
    assert obj["payload"] == {"user_input": "hi"}
    assert obj["perception"] == {
        "flirtation": 0.2,
        "avoidance": 0.1,
        "manipulation": 0.0,
    }
    assert abs(obj["affinity"] - 0.1) < 1e-6


@pytest.mark.asyncio
async def test_handle_chat_raw(monkeypatch, tmp_path):
    monkeypatch.setattr(trace, "Subscriber", DummySubscriber)
    monkeypatch.setattr(
        trace,
        "analyze_social",
        lambda text: {"flirtation": 0.2, "avoidance": 0.1, "manipulation": 0.0},
    )
    outfile = tmp_path / "trace.jsonl"
    recorder = trace.TraceRecorder(DummyNATS(), DummyJS(), str(outfile))
    msg = DummyMsg("hello")

    await recorder._handle_chat_raw(msg)
    assert msg.acked
    with open(outfile, "r", encoding="utf-8") as f:
        line = f.readline()
    obj = json.loads(line)
    assert obj["event"] == "CHAT_RAW"
    assert obj["payload"] == "hello"
    assert obj["perception"] == {
        "flirtation": 0.2,
        "avoidance": 0.1,
        "manipulation": 0.0,
    }
    assert abs(obj["affinity"] - 0.1) < 1e-6
