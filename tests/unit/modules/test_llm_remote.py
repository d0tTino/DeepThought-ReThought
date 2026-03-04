import asyncio
import importlib.util
import sys
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

pytest.importorskip("nats")
pytest.importorskip("aiohttp")

spec = importlib.util.spec_from_file_location("deepthought.modules.llm_remote", "src/deepthought/modules/llm_remote.py")
llm_remote = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = llm_remote
assert spec.loader is not None
spec.loader.exec_module(llm_remote)

from deepthought.eda.events import ContextAssembledPayload, EventSubjects  # noqa: E402


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
        self.headers = None
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


@pytest.mark.asyncio
async def test_handle_context_event_publishes(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate(self, prompt):
        return "answer"

    monkeypatch.setattr(llm, "_generate", fake_generate.__get__(llm, type(llm)))

    payload = ContextAssembledPayload(input_id="42", user_input="hello", retrieved_facts=["f1"])
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.acked
    pub = llm._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.RESPONSE_CANDIDATES
    assert sent_payload.candidates[0].text == "answer"
    assert sent_payload.input_id == "42"


@pytest.mark.asyncio
async def test_generate_timeout(monkeypatch):
    class TimeoutSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            raise asyncio.TimeoutError

    llm = create_llm(monkeypatch, TimeoutSession())

    with pytest.raises(asyncio.TimeoutError):
        await llm._generate("hello")


@pytest.mark.asyncio
async def test_generate_malformed_json(monkeypatch):
    class BadJSONResponse(DummyResponse):
        async def json(self):
            raise ValueError("bad json")

    resp = BadJSONResponse()
    session = DummySession(resp)
    llm = create_llm(monkeypatch, session)

    with pytest.raises(ValueError):
        await llm._generate("hello")


@pytest.mark.asyncio
async def test_handle_context_event_timeout(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate(self, prompt):
        raise asyncio.TimeoutError

    monkeypatch.setattr(llm, "_generate", fake_generate.__get__(llm, type(llm)))

    payload = ContextAssembledPayload(input_id="99", user_input="hello", retrieved_facts=["f1"])
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.nacked
    assert not msg.acked


@pytest.mark.asyncio
async def test_handle_context_event_bad_json(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate(self, prompt):
        raise ValueError("bad json")

    monkeypatch.setattr(llm, "_generate", fake_generate.__get__(llm, type(llm)))

    payload = ContextAssembledPayload(input_id="88", user_input="hello", retrieved_facts=["f1"])
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.nacked
    assert not msg.acked


@pytest.mark.asyncio
async def test_generate_http_error(monkeypatch):
    err = aiohttp.ClientResponseError(
        request_info=MagicMock(real_url="http://api"),
        history=(),
        status=500,
        message="boom",
    )

    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = False
    mock_response.raise_for_status = MagicMock(side_effect=err)
    mock_response.json.return_value = {"text": "x"}

    session = MagicMock()
    session.post = MagicMock(return_value=mock_response)

    llm = create_llm(monkeypatch, session)

    with pytest.raises(aiohttp.ClientResponseError):
        await llm._generate("hello")


@pytest.mark.asyncio
async def test_handle_context_event_http_error(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate(self, prompt):
        raise aiohttp.ClientResponseError(
            request_info=MagicMock(real_url="http://api"),
            history=(),
            status=500,
            message="bad",
        )

    monkeypatch.setattr(llm, "_generate", fake_generate.__get__(llm, type(llm)))

    payload = ContextAssembledPayload(input_id="77", user_input="hello", retrieved_facts=["f1"])
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.nacked
    assert not msg.acked


@pytest.mark.asyncio
async def test_handle_context_event_with_mock_session(monkeypatch):
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = False
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"text": "resp"}

    session = MagicMock()
    session.post = MagicMock(return_value=mock_response)

    llm = create_llm(monkeypatch, session)

    payload = ContextAssembledPayload(input_id="55", user_input="hello", retrieved_facts=["fact"])
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.acked
    assert llm._publisher.published
    subject, sent_payload = llm._publisher.published[0]
    assert subject == EventSubjects.RESPONSE_CANDIDATES
    assert sent_payload.candidates[0].text == "resp"


@pytest.mark.asyncio
async def test_generate_uses_dspy_pipeline(monkeypatch):
    monkeypatch.setenv("USE_DSPY", "1")
    monkeypatch.setattr(llm_remote, "build_qa_pipeline", lambda: lambda q: "pipe")
    llm = create_llm(monkeypatch)
    result = await llm._generate("query")
    assert result == "pipe"
    await llm._session.close()


def test_build_generation_prompt_includes_sections():
    prompt = llm_remote._build_generation_prompt(
        user_input="How are you?",
        facts=["Fact A", "Fact B"],
        author_name="Ada",
        channel_context="discord/#general",
        recent_turn_summary="The user asked about status.",
    )

    assert "[SYSTEM PERSONA]" in prompt
    assert "[RELEVANT FACTS]" in prompt
    assert "- Fact A" in prompt
    assert "[LATEST USER MESSAGE]" in prompt
    assert "How are you?" in prompt
    assert "[SOCIAL/PERCEPTION HINTS]" in prompt
    assert "- Author name: Ada" in prompt


def test_build_generation_prompt_without_optional_hints():
    prompt = llm_remote._build_generation_prompt(user_input="Hi", facts=[])
    assert "[RELEVANT FACTS]" in prompt
    assert "- None" in prompt


@pytest.mark.asyncio
async def test_handle_context_event_missing_user_input_naks(monkeypatch):
    llm = create_llm(monkeypatch)
    payload = ContextAssembledPayload(input_id="no-input", user_input="", retrieved_facts=["f1"])
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.nacked
    assert not msg.acked
    assert not llm._publisher.published
