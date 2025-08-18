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
        fused=[0.1, 0.2],
        by_modality={
            "text": {
                "spans": [[0, 1]],
                "embeddings": [[0.1, 0.2]],
                "encoders": [{"name": "enc", "modality": "text"}],
            }
        },
        provenance={"p": 1},
    )
    assert result == {"seq": 1}
    pub._publisher.publish.assert_awaited_once()
    args, kwargs = pub._publisher.publish.call_args
    subject, payload = args
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert isinstance(payload, PerceptionEmbeddingsPayload)
    assert payload.message_id == "msg1"
    assert payload.fused == [0.1, 0.2]
    assert "text" in payload.by_modality
    text_mod = payload.by_modality["text"]
    assert text_mod.spans == [(0, 1)]
    assert text_mod.encoders[0].name == "enc"


@pytest.mark.asyncio
async def test_perception_publisher_defaults(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.publisher.Publisher",
        DummyPublisher,
    )
    pub = PerceptionPublisher(nats_client=object(), js_context=object())
    result = await pub.publish("m2", "u2")
    assert result == {"seq": 1}
    args, kwargs = pub._publisher.publish.call_args
    subject, payload = args
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert payload.fused is None
    assert payload.by_modality == {}
    assert payload.provenance == {}
