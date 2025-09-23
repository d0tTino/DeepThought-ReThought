import asyncio
from unittest.mock import AsyncMock

import pytest

from deepthought.eda.events import (
    EventSubjects,
    PerceptionEmbeddingsEvent,
)
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
        fused=[[0.1, 0.2]],
        by_modality={
            "text": {
                "spans": [[0, 1]],
                "embeddings": [[0.1, 0.2]],
                "encoders": [
                    {
                        "name": "enc",
                        "modality": "text",
                        "dim": 2,
                        "parameters": {
                            "hop_size": 0.5,
                            "config_source": "PerceptionConfig.text_model",
                        },
                    }
                ],
            }
        },
        provenance={"p": 1},
    )
    assert result == {"seq": 1}
    pub._publisher.publish.assert_awaited_once()
    args, kwargs = pub._publisher.publish.call_args
    subject, event = args
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert isinstance(event, PerceptionEmbeddingsEvent)
    assert event.payload is not None
    assert event.payload.message_id == "msg1"
    assert event.payload.fused == [[0.1, 0.2]]
    assert "text" in event.payload.by_modality
    text_mod = event.payload.by_modality["text"]

    assert text_mod.spans == [[0, 1]]
    assert text_mod.encoders[0].name == "enc"
    assert text_mod.encoders[0].parameters["hop_size"] == 0.5
    assert event.encoders[0].name == "enc"
    assert event.encoders[0].parameters["config_source"] == "PerceptionConfig.text_model"
    assert event.provenance == {"p": 1}


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
    assert isinstance(payload, PerceptionEmbeddingsEvent)
    assert payload.payload is not None
    assert payload.payload.fused is None
    assert payload.payload.by_modality == {}
    assert payload.provenance == {}


@pytest.mark.asyncio
async def test_perception_publisher_deduplicates_encoders(monkeypatch):
    """Ensure top-level encoders are unique across modalities."""

    monkeypatch.setattr(
        "deepthought.services.perception.publisher.Publisher",
        DummyPublisher,
    )
    pub = PerceptionPublisher(nats_client=object(), js_context=object())
    await pub.publish(
        "m1",
        "u1",
        by_modality={
            "a": {
                "spans": [],
                "embeddings": [],
                "encoders": [
                    {
                        "name": "enc",
                        "modality": "text",
                        "dim": 2,
                        "parameters": {
                            "hop_size": 0.5,
                            "config_source": "PerceptionConfig.text_model",
                        },
                    }
                ],
            },
            "b": {
                "spans": [],
                "embeddings": [],
                "encoders": [
                    {
                        "name": "enc",
                        "modality": "text",
                        "dim": 2,
                        "parameters": {
                            "hop_size": 0.5,
                            "config_source": "PerceptionConfig.text_model",
                        },
                    }
                ],
            },
        },
    )
    subject, payload = pub._publisher.publish.call_args[0]
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert len(payload.encoders) == 1
