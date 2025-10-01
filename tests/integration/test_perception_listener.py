import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

if "deepthought" not in sys.modules:
    deepthought_pkg = types.ModuleType("deepthought")
    deepthought_pkg.__path__ = [str(SRC_ROOT / "deepthought")]
    sys.modules["deepthought"] = deepthought_pkg
else:
    deepthought_pkg = sys.modules["deepthought"]

if "deepthought.services" not in sys.modules:
    services_pkg = types.ModuleType("deepthought.services")
    services_pkg.__path__ = [str(SRC_ROOT / "deepthought" / "services")]
    sys.modules["deepthought.services"] = services_pkg
else:
    services_pkg = sys.modules["deepthought.services"]

setattr(deepthought_pkg, "services", services_pkg)

if "deepthought.services.perception" not in sys.modules:
    perception_pkg = types.ModuleType("deepthought.services.perception")
    perception_pkg.__path__ = [str(SRC_ROOT / "deepthought" / "services" / "perception")]
    sys.modules["deepthought.services.perception"] = perception_pkg
else:
    perception_pkg = sys.modules["deepthought.services.perception"]

setattr(services_pkg, "perception", perception_pkg)

if "deepthought.services.perception.cli" not in sys.modules:
    cli_stub = types.ModuleType("deepthought.services.perception.cli")
    sys.modules["deepthought.services.perception.cli"] = cli_stub
    setattr(perception_pkg, "cli", cli_stub)

if "deepthought.services.perception.service" not in sys.modules:
    service_stub = types.ModuleType("deepthought.services.perception.service")

    class _StubPerceptionService:
        def __init__(self, publisher, *args, **kwargs):
            self.publisher = publisher

        async def run(self, message_id, user_id, **kwargs):
            await self.publisher.publish(
                message_id,
                user_id,
                fused=kwargs.get("embeddings"),
                by_modality=kwargs.get("by_modality"),
                spans=kwargs.get("spans"),
                modality_mask=kwargs.get("modality_mask"),
                provenance=kwargs.get("provenance"),
            )

    service_stub.PerceptionService = _StubPerceptionService
    sys.modules["deepthought.services.perception.service"] = service_stub
    setattr(perception_pkg, "service", service_stub)

from deepthought.eda.events import EventSubjects
from deepthought.services.perception.listener import PerceptionServiceListener
from deepthought.services.perception.publisher import PerceptionPublisher
from deepthought.services.perception.service import PerceptionService
from deepthought.services.perception.text_utils import hop_aligned_tokens, scrub_tokens


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.publish = AsyncMock(return_value={"seq": 1})


class FakeNATS:
    def __init__(self) -> None:
        self.is_connected = True


@pytest.mark.asyncio
async def test_perception_listener_publishes_embeddings(monkeypatch):
    """The listener should publish PERCEPTION_EMBEDDINGS events."""

    monkeypatch.setattr("deepthought.services.perception.publisher.Publisher", DummyPublisher)

    pub = PerceptionPublisher(nats_client=object(), js_context=object())
    service = PerceptionService(pub)

    payload = {
        "message_id": "m1",
        "user_id": "u1",
        "embeddings": [[0.1, 0.2]],
    }
    msg = SimpleNamespace(data=json.dumps(payload).encode(), ack=AsyncMock())

    async def listener(message):
        data = json.loads(message.data.decode())
        await service.run(**data)
        await message.ack()

    await listener(msg)

    msg.ack.assert_awaited_once()
    pub._publisher.publish.assert_awaited_once()
    subject, event = pub._publisher.publish.call_args[0]
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert event.payload is not None
    assert event.payload.message_id == "m1"
    assert event.payload.user_id == "u1"
    assert event.payload.spans == []
    assert event.payload.modality_mask == {}


@pytest.mark.asyncio
async def test_listener_generates_tokens_from_user_input(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.listener.PerceptionConfig",
        lambda: SimpleNamespace(enable_asr_transcription=False, text_hop_size=0.05),
    )

    service = SimpleNamespace(run=AsyncMock())
    listener = PerceptionServiceListener(
        service,
        FakeNATS(),
        object(),
        default_user_id="user",
    )

    payload = {"input_id": "m5", "user_input": "Email me at foo@example.com"}
    msg = SimpleNamespace(data=json.dumps(payload).encode(), ack=AsyncMock(), headers=None)

    await listener._handle(msg)

    msg.ack.assert_awaited_once()
    assert service.run.await_count == 1
    _, kwargs = service.run.await_args
    assert kwargs["message_id"] == "m5"
    expected_tokens = hop_aligned_tokens(payload["user_input"], 0.05)
    assert scrub_tokens(kwargs["text_tokens"]) == expected_tokens


@pytest.mark.asyncio
async def test_listener_transcribes_audio_when_tokens_missing(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.listener.PerceptionConfig",
        lambda: SimpleNamespace(enable_asr_transcription=True, text_hop_size=0.05),
    )
    asr = SimpleNamespace(transcribe=AsyncMock(return_value=[("hello", 0.0, 1.0)]))
    service = SimpleNamespace(run=AsyncMock())
    listener = PerceptionServiceListener(
        service,
        FakeNATS(),
        object(),
        default_user_id="user",
        asr=asr,
    )

    payload = {
        "message_id": "m2",
        "user_id": "u2",
        "audio_path": "audio.wav",
        "consent": {"general": True, "audio": True},
    }
    msg = SimpleNamespace(data=json.dumps(payload).encode(), ack=AsyncMock(), headers=None)

    await listener._handle(msg)

    asr.transcribe.assert_awaited_once_with("audio.wav")
    msg.ack.assert_awaited_once()
    assert service.run.await_count == 1
    _, kwargs = service.run.await_args
    assert kwargs["text_tokens"] == [("hello", 0.0, 1.0)]
    assert kwargs["audio_path"] == "audio.wav"


@pytest.mark.asyncio
async def test_listener_skips_asr_without_flag(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.listener.PerceptionConfig",
        lambda: SimpleNamespace(enable_asr_transcription=False, text_hop_size=0.05),
    )

    asr = SimpleNamespace(transcribe=AsyncMock(return_value=[("ignored", 0.0, 1.0)]))
    service = SimpleNamespace(run=AsyncMock())
    listener = PerceptionServiceListener(
        service,
        FakeNATS(),
        object(),
        default_user_id="user",
        asr=asr,
    )

    payload = {
        "message_id": "m3",
        "user_id": "u3",
        "audio_path": "audio.wav",
        "consent": {"general": True, "audio": True},
    }
    msg = SimpleNamespace(data=json.dumps(payload).encode(), ack=AsyncMock(), headers=None)

    await listener._handle(msg)

    asr.transcribe.assert_not_awaited()
    msg.ack.assert_awaited_once()
    assert service.run.await_count == 1
    _, kwargs = service.run.await_args
    assert "text_tokens" not in kwargs


@pytest.mark.asyncio
async def test_listener_skips_asr_without_audio_consent(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.listener.PerceptionConfig",
        lambda: SimpleNamespace(enable_asr_transcription=True, text_hop_size=0.05),
    )

    asr = SimpleNamespace(transcribe=AsyncMock(return_value=[("hello", 0.0, 1.0)]))
    service = SimpleNamespace(run=AsyncMock())
    listener = PerceptionServiceListener(
        service,
        FakeNATS(),
        object(),
        default_user_id="user",
        asr=asr,
    )

    payload = {
        "message_id": "m4",
        "user_id": "u4",
        "audio_path": "audio.wav",
        "consent": {"general": True, "audio": False},
    }
    msg = SimpleNamespace(data=json.dumps(payload).encode(), ack=AsyncMock(), headers=None)

    await listener._handle(msg)

    asr.transcribe.assert_not_awaited()
    msg.ack.assert_awaited_once()
    assert service.run.await_count == 1
    _, kwargs = service.run.await_args
    assert "text_tokens" not in kwargs


@pytest.mark.asyncio
async def test_listener_skips_asr_when_audio_consent_missing(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.listener.PerceptionConfig",
        lambda: SimpleNamespace(enable_asr_transcription=True, text_hop_size=0.05),
    )

    asr = SimpleNamespace(transcribe=AsyncMock(return_value=[("hello", 0.0, 1.0)]))
    service = SimpleNamespace(run=AsyncMock())
    listener = PerceptionServiceListener(
        service,
        FakeNATS(),
        object(),
        default_user_id="user",
        asr=asr,
    )

    payload = {
        "message_id": "m5",
        "user_id": "u5",
        "audio_path": "audio.wav",
        "consent": {"general": True},
    }
    msg = SimpleNamespace(data=json.dumps(payload).encode(), ack=AsyncMock(), headers=None)

    await listener._handle(msg)

    asr.transcribe.assert_not_awaited()
    msg.ack.assert_awaited_once()
    assert service.run.await_count == 1
    _, kwargs = service.run.await_args
    assert "text_tokens" not in kwargs
