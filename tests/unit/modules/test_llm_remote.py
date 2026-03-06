import asyncio
import json
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

from deepthought.eda.contracts import EventEnvelope  # noqa: E402
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

    result = await llm._generate_candidates("hello")

    assert result[0].text == "generated"
    assert session.calls == [("http://api", {"text": "hello", "n": 3})]
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

    async def fake_generate_candidates(self, prompt):
        return [llm_remote.ResponseCandidate(text="answer", confidence=0.8, source="stub", safety_passed=True)]

    monkeypatch.setattr(llm, "_generate_candidates", fake_generate_candidates.__get__(llm, type(llm)))

    payload = ContextAssembledPayload(input_id="42", user_input="hello", retrieved_facts=["f1"])
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.acked
    pub = llm._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.RESPONSE_CANDIDATES
    assert sent_payload["payload"]["candidates"][0]["text"] == "answer"
    assert sent_payload["payload"]["input_id"] == "42"


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
        await llm._generate_candidates("hello")


@pytest.mark.asyncio
async def test_generate_malformed_json(monkeypatch):
    class BadJSONResponse(DummyResponse):
        async def json(self):
            raise ValueError("bad json")

    resp = BadJSONResponse()
    session = DummySession(resp)
    llm = create_llm(monkeypatch, session)

    with pytest.raises(ValueError):
        await llm._generate_candidates("hello")


@pytest.mark.asyncio
async def test_handle_context_event_timeout(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate_candidates(self, prompt):
        raise asyncio.TimeoutError

    monkeypatch.setattr(llm, "_generate_candidates", fake_generate_candidates.__get__(llm, type(llm)))

    payload = ContextAssembledPayload(input_id="99", user_input="hello", retrieved_facts=["f1"])
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.nacked
    assert not msg.acked


@pytest.mark.asyncio
async def test_handle_context_event_bad_json(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate_candidates(self, prompt):
        raise ValueError("bad json")

    monkeypatch.setattr(llm, "_generate_candidates", fake_generate_candidates.__get__(llm, type(llm)))

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
        await llm._generate_candidates("hello")


@pytest.mark.asyncio
async def test_handle_context_event_http_error(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate_candidates(self, prompt):
        raise aiohttp.ClientResponseError(
            request_info=MagicMock(real_url="http://api"),
            history=(),
            status=500,
            message="bad",
        )

    monkeypatch.setattr(llm, "_generate_candidates", fake_generate_candidates.__get__(llm, type(llm)))

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
    assert sent_payload["payload"]["candidates"][0]["text"] == "resp"


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
        multimodal_interpretations={
            "notes": [
                {
                    "modality": "image",
                    "what": "2 embedding vectors across 1 spans",
                    "where": "attachment regions",
                    "who": "unknown",
                    "confidence": 0.81,
                }
            ],
            "confidence": {"aggregate": 0.81, "low_confidence": False},
        },
    )

    assert "[SYSTEM PERSONA]" in prompt
    assert "[RELEVANT FACTS]" in prompt
    assert "- Fact A" in prompt
    assert "[LATEST USER MESSAGE]" in prompt
    assert "How are you?" in prompt
    assert "[SOCIAL/PERCEPTION HINTS]" in prompt
    assert "- Author name: Ada" in prompt
    assert "[MULTIMODAL INTERPRETATIONS]" in prompt
    assert "[image]" in prompt
    assert "[UNCERTAINTY CUES]" in prompt


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


@pytest.mark.asyncio
async def test_generate_candidates_uses_scores_and_safety(monkeypatch):
    resp = DummyResponse(
        {
            "candidates": [
                {"text": "safe response", "avg_logprob": 0.2, "score": 0.7, "source": "sampler"},
                {"text": "unsafe kill plan", "avg_logprob": 0.1, "score": 0.6},
            ]
        }
    )
    session = DummySession(resp)
    llm = create_llm(monkeypatch, session)

    result = await llm._generate_candidates("hello")

    assert len(result) == 2
    assert result[0].source.endswith(":sampler")
    assert result[0].confidence_components["model_score"] == 0.7
    assert result[0].safety_passed is True
    assert result[1].safety_passed is False
    assert "kill" in result[1].safety_metadata["matched_terms"]


@pytest.mark.asyncio
async def test_handle_context_event_falls_back_to_clarifying_question_on_low_confidence(monkeypatch):
    llm = create_llm(monkeypatch)
    called = {"generate": False}

    async def fake_generate_candidates(self, prompt):
        called["generate"] = True
        return [llm_remote.ResponseCandidate(text="answer", confidence=0.8, source="stub", safety_passed=True)]

    monkeypatch.setattr(llm, "_generate_candidates", fake_generate_candidates.__get__(llm, type(llm)))

    payload = ContextAssembledPayload(
        input_id="low-conf",
        user_input="what's in this clip?",
        retrieved_facts=["f1"],
        multimodal_interpretations={
            "notes": [{"modality": "audio"}],
            "confidence": {"aggregate": 0.2, "low_confidence": True},
            "fallback": {"ask_clarifying_question": True, "reason": "low multimodal confidence"},
        },
    )
    msg = DummyMsg(payload.to_json())

    await llm._handle_context_event(msg)

    assert msg.acked
    assert called["generate"] is False
    _, sent_payload = llm._publisher.published[0]
    assert "Could you clarify" in sent_payload["payload"]["candidates"][0]["text"]


def test_build_generation_prompt_includes_image_and_audio_interpretations():
    prompt = llm_remote._build_generation_prompt(
        user_input="Please summarize these attachments",
        facts=[],
        multimodal_interpretations={
            "notes": [
                {
                    "modality": "image",
                    "what": "1 embedding vectors across 1 spans",
                    "where": "attachment regions",
                    "who": "unknown",
                    "confidence": 0.73,
                },
                {
                    "modality": "audio",
                    "what": "3 embedding vectors across 2 spans",
                    "where": "temporal spans",
                    "who": "speaker unknown",
                    "confidence": 0.52,
                },
            ],
            "confidence": {"aggregate": 0.62, "low_confidence": False},
        },
    )

    assert "[image] what=1 embedding vectors across 1 spans" in prompt
    assert "[audio] what=3 embedding vectors across 2 spans" in prompt


@pytest.mark.asyncio
async def test_handle_context_event_accepts_enveloped_payload(monkeypatch):
    llm = create_llm(monkeypatch)

    async def fake_generate_candidates(self, prompt):
        return [llm_remote.ResponseCandidate(text="answer", confidence=0.8, source="stub", safety_passed=True)]

    monkeypatch.setattr(llm, "_generate_candidates", fake_generate_candidates.__get__(llm, type(llm)))

    payload = ContextAssembledPayload(input_id="42-env", user_input="hello", retrieved_facts=["f1"])
    envelope = EventEnvelope.build(
        subject=EventSubjects.CONTEXT_ASSEMBLED,
        payload=json.loads(payload.to_json()),
        producer="context_assembler",
    )
    msg = DummyMsg(json.dumps(envelope.__dict__))

    await llm._handle_context_event(msg)

    assert msg.acked
    subject, sent_payload = llm._publisher.published[0]
    assert subject == EventSubjects.RESPONSE_CANDIDATES
    assert sent_payload["payload"]["input_id"] == "42-env"


@pytest.mark.asyncio
async def test_generate_candidates_local_quantized_backend(monkeypatch):
    class DummySettings:
        llm_backend = "local_quantized"
        llm_model_path = "dummy/model"
        model_path = "fallback/model"
        llm_quantization_bits = 4
        llm_local_max_new_tokens = 128
        llm_remote_endpoint = "http://unused"

    monkeypatch.setattr(llm_remote, "Publisher", DummyPublisher)
    monkeypatch.setattr(llm_remote, "Subscriber", DummySubscriber)
    monkeypatch.setattr(llm_remote, "get_settings", lambda: DummySettings())

    def fake_build_pipeline(self):
        def _gen(prompt, num_return_sequences, max_new_tokens, do_sample, return_full_text):
            assert num_return_sequences == 3
            assert max_new_tokens == 128
            assert do_sample is True
            assert return_full_text is False
            return [{"generated_text": "local answer"}]

        return _gen

    monkeypatch.setattr(llm_remote.LocalQuantizedResponderBackend, "_build_pipeline", fake_build_pipeline)
    llm = llm_remote.RemoteLLM(DummyNATS(), DummyJS())

    result = await llm._generate_candidates("hello")

    assert result[0].text == "local answer"
    assert ":local_quantized:" in result[0].source
    assert isinstance(result[0].confidence, float)
    assert result[0].safety_metadata["rule"] == "keyword_v1"
    await llm.stop_listening()
