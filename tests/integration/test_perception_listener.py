import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deepthought.eda.events import EventSubjects
from deepthought.services.perception.publisher import PerceptionPublisher
from deepthought.services.perception.service import PerceptionService


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.publish = AsyncMock(return_value={"seq": 1})


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
