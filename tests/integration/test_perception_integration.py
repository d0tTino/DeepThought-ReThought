from unittest.mock import AsyncMock

import pytest

from deepthought.eda.events import EventSubjects, PerceptionEmbeddingsEvent
from deepthought.services.perception.publisher import PerceptionPublisher



class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.publish = AsyncMock(return_value={"seq": 1})


@pytest.mark.asyncio
async def test_service_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "deepthought.services.perception.publisher.Publisher",
        DummyPublisher,
    )

    pub = PerceptionPublisher(nats_client=object(), js_context=object())
    await pub.publish(
        "m1",
        "u1",
        by_modality={
            "text": {
                "spans": [[0, 1]],
                "embeddings": [[0.1, 0.2]],
                "encoders": [],
            }
        },
    )
    pub._publisher.publish.assert_awaited_once()
    subject, payload = pub._publisher.publish.call_args[0]
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert isinstance(payload, PerceptionEmbeddingsEvent)
    assert payload.payload is not None
    assert payload.payload.message_id == "m1"
