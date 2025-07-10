import importlib.util
import json
import sys

import pytest

pytest.importorskip("nats")
pytest.importorskip("aiohttp")

spec = importlib.util.spec_from_file_location("deepthought.modules.llm_remote", "src/deepthought/modules/llm_remote.py")
llm_remote = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = llm_remote
assert spec.loader is not None
spec.loader.exec_module(llm_remote)

from deepthought.eda.events import EventSubjects, MemoryRetrievedPayload


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
        return None


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyResponse:
    def __init__(self, data=None):
        self.data = data or {"text": "ok"}
        self.status = 200
        self.raise_called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        self.raise_called = True

    async def json(self):
        return self.data


class DummySession:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None):
        self.calls.append((url, json))
        return self.resp


def create_llm(monkeypatch, session=None):
    monkeypatch.setattr(llm_remote, "Publisher", DummyPublisher)
    monkeypatch.setattr(llm_remote, "Subscriber", DummySubscriber)
    if session is not None:
        monkeypatch.setattr(llm_remote.aiohttp, "ClientSession", lambda: session)
    return llm_remote.RemoteLLM(DummyNATS(), DummyJS(), endpoint="http://api")


@pytest.mark.asyncio
async def test_generate_posts(monkeypatch):
    resp = DummyResponse({"text": "generated"})
    session = DummySession(resp)
    llm = create_llm(monkeypatch, session)

    result = await llm._generate("hello")

    assert result == "generated"
    assert session.calls == [("http://api", {"text": "hello"})]
    assert resp.raise_called


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        pass


@pytest.mark.asyncio
async def test_handle_memory_event_publishes(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate(self, prompt):
        return "answer"

    monkeypatch.setattr(llm, "_generate", fake_generate.__get__(llm, type(llm)))

    payload = MemoryRetrievedPayload(retrieved_knowledge={"facts": ["f1"]}, input_id="42")
    msg = DummyMsg(payload.to_json())

    await llm._handle_memory_event(msg)

    assert msg.acked
    pub = llm._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.RESPONSE_GENERATED
    assert sent_payload.final_response == "answer"
    assert sent_payload.input_id == "42"
