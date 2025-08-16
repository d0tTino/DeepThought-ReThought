import asyncio
from unittest.mock import AsyncMock

import pytest

from deepthought.eda.events import EventSubjects, PerceptionEmbeddingsPayload
from deepthought.services.perception.publisher import PerceptionPublisher


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.publish = AsyncMock(return_value={"seq": 1})


@pytest.mark.asyncio
async def test_perception_publisher(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.publisher.Publisher",
        DummyPublisher,
    )
    pub = PerceptionPublisher(nats_client=object(), js_context=object())
    result = await pub.publish(
        "msg1",
        "user1",
        spans=[[0, 1]],
        embeddings=[[0.1, 0.2]],
        encoders=[{"name": "enc"}],
        provenance={"p": 1},
    )
    assert result == {"seq": 1}
    pub._publisher.publish.assert_awaited_once()
    args, kwargs = pub._publisher.publish.call_args
    subject, payload = args
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert isinstance(payload, PerceptionEmbeddingsPayload)
    assert payload.message_id == "msg1"
