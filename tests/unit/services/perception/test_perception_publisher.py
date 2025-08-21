from unittest.mock import AsyncMock

import pytest

from deepthought.eda.events import (
    EventSubjects,
    PerceptionEmbeddingsEvent,
)
from deepthought.services.perception import publisher as perception_publisher


@pytest.mark.asyncio
async def test_publish_embeddings(monkeypatch):
    """Publish a perception event and verify payload formatting."""

    captured: dict = {}

    async def fake_publish(subject, payload, use_jetstream=True, retries=3):
        captured["subject"] = subject
        captured["payload"] = payload
        captured["use_jetstream"] = use_jetstream
        captured["retries"] = retries
        return {"seq": 1}

    class FakePublisher:
        def __init__(self, nats, js):  # pragma: no cover - unused
            self.publish = AsyncMock(side_effect=fake_publish)

    monkeypatch.setattr(perception_publisher, "Publisher", FakePublisher)

    publisher = perception_publisher.PerceptionPublisher(object(), object())
    ack = await publisher.publish(
        message_id="msg1",
        user_id="user1",
        fused=[[0.0, 0.1]],
        by_modality={
            "text": {
                "spans": [[0, 1]],
                "embeddings": [[0.0, 0.1]],
                "encoders": [{"name": "test", "dim": 2, "modality": "text"}],
            }
        },
        provenance={"source": "unit"},
    )

    assert ack == {"seq": 1}
    assert captured["subject"] == EventSubjects.PERCEPTION_EMBEDDINGS
    event = captured["payload"]
    assert isinstance(event, PerceptionEmbeddingsEvent)
    assert event.payload is not None
    assert event.payload.message_id == "msg1"
    assert event.payload.fused == [[0.0, 0.1]]
    text_mod = event.payload.by_modality["text"]

    assert text_mod.encoders[0].name == "test"
    assert event.provenance == {"source": "unit"}
    assert event.encoders[0].name == "test"
    assert captured["use_jetstream"] is True
    assert captured["retries"] == 3


@pytest.mark.asyncio
async def test_publish_deduplicates_encoders(monkeypatch):
    """Encoders should be unique at the event top level."""

    captured: dict = {}

    async def fake_publish(subject, payload, use_jetstream=True, retries=3):
        captured["payload"] = payload
        return {"seq": 1}

    class FakePublisher:
        def __init__(self, nats, js):
            self.publish = AsyncMock(side_effect=fake_publish)

    monkeypatch.setattr(perception_publisher, "Publisher", FakePublisher)

    publisher = perception_publisher.PerceptionPublisher(object(), object())
    await publisher.publish(
        message_id="m1",
        user_id="u1",
        by_modality={
            "a": {
                "spans": [],
                "embeddings": [],
                "encoders": [{"name": "enc", "modality": "text"}],
            },
            "b": {
                "spans": [],
                "embeddings": [],
                "encoders": [{"name": "enc", "modality": "text"}],
            },
        },
    )

    event = captured["payload"]
    assert isinstance(event, PerceptionEmbeddingsEvent)
    assert len(event.encoders) == 1


@pytest.mark.asyncio
async def test_publish_forwards_retries(monkeypatch):
    """Ensure PerceptionPublisher forwards retry count to Publisher."""

    captured: dict = {}

    async def fake_publish(subject, payload, use_jetstream=True, retries=3):
        captured["retries"] = retries
        return {"seq": 2}

    class FakePublisher:
        def __init__(self, nats, js):  # pragma: no cover - unused
            self.publish = AsyncMock(side_effect=fake_publish)

    monkeypatch.setattr(perception_publisher, "Publisher", FakePublisher)

    publisher = perception_publisher.PerceptionPublisher(object(), object())
    await publisher.publish("msg1", "user1", retries=5)

    assert captured["retries"] == 5


@pytest.mark.asyncio
async def test_publish_raises_after_retries(monkeypatch):
    """Ensure publish raises if underlying Publisher fails."""

    async def always_fail(subject, payload, use_jetstream=True, retries=1):
        raise RuntimeError("fail")

    class FakePublisher:
        def __init__(self, nats, js):  # pragma: no cover - unused
            self.publish = AsyncMock(side_effect=always_fail)

    monkeypatch.setattr(perception_publisher, "Publisher", FakePublisher)

    publisher = perception_publisher.PerceptionPublisher(object(), object())

    with pytest.raises(RuntimeError):
        await publisher.publish("msg1", "user1", retries=2)
