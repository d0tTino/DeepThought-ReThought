from unittest.mock import AsyncMock

import pytest

from deepthought.eda.events import (
    EventSubjects,
    PerceptionEmbeddingsPayload,
)
from deepthought.services.perception import publisher as perception_publisher


@pytest.mark.asyncio
async def test_publish_embeddings(monkeypatch):
    """Publish a perception event and verify payload formatting."""

    captured: dict = {}

    async def fake_publish(subject, payload, use_jetstream=True):
        captured["subject"] = subject
        captured["payload"] = payload
        captured["use_jetstream"] = use_jetstream
        return {"seq": 1}

    class FakePublisher:
        def __init__(self, nats, js):  # pragma: no cover - unused
            self.publish = AsyncMock(side_effect=fake_publish)

    monkeypatch.setattr(perception_publisher, "Publisher", FakePublisher)

    publisher = perception_publisher.PerceptionPublisher(object(), object())
    ack = await publisher.publish(
        message_id="msg1",
        user_id="user1",
        spans=[(0, 1)],
        embeddings=[[0.0, 0.1]],
        encoders=[{"name": "test", "dim": 2}],
        provenance={"source": "unit"},
    )

    assert ack == {"seq": 1}
    assert captured["subject"] == EventSubjects.PERCEPTION_EMBEDDINGS
    assert isinstance(captured["payload"], PerceptionEmbeddingsPayload)
    assert captured["payload"].message_id == "msg1"
    assert captured["payload"].encoders == [{"name": "test", "dim": 2}]
    assert captured["payload"].provenance == {"source": "unit"}
    assert captured["use_jetstream"] is True


@pytest.mark.asyncio
async def test_publish_retries(monkeypatch):
    """Verify that publish retries on failure."""

    attempts = 0

    async def failing_publish(subject, payload, use_jetstream=True):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("fail")
        return {"seq": 2}

    class FakePublisher:
        def __init__(self, nats, js):  # pragma: no cover - unused
            self.publish = AsyncMock(side_effect=failing_publish)

    monkeypatch.setattr(perception_publisher, "Publisher", FakePublisher)

    publisher = perception_publisher.PerceptionPublisher(object(), object())
    ack = await publisher.publish("msg1", "user1", retries=3)

    assert ack == {"seq": 2}
    assert attempts == 2


@pytest.mark.asyncio
async def test_publish_raises_after_retries(monkeypatch):
    """Ensure publish raises after exhausting retries."""

    async def always_fail(subject, payload, use_jetstream=True):
        raise RuntimeError("fail")

    class FakePublisher:
        def __init__(self, nats, js):  # pragma: no cover - unused
            self.publish = AsyncMock(side_effect=always_fail)

    monkeypatch.setattr(perception_publisher, "Publisher", FakePublisher)

    publisher = perception_publisher.PerceptionPublisher(object(), object())

    with pytest.raises(RuntimeError):
        await publisher.publish("msg1", "user1", retries=2)
