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
                "encoders": [{"name": "enc", "modality": "text", "parameters": {}}],
            }
        },
        spans=[[0, 1]],
        modality_mask={"text": [True]},
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
    assert event.payload.spans == [[0, 1]]
    assert event.payload.modality_mask == {"text": [True]}
    assert "text" in event.payload.by_modality
    text_mod = event.payload.by_modality["text"]

    assert text_mod.spans == [[0, 1]]
    assert text_mod.encoders[0].name == "enc"
    assert event.encoders[0].name == "enc"
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
    assert payload.payload.spans == []
    assert payload.payload.modality_mask == {}
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
                "encoders": [{"name": "enc", "modality": "text", "parameters": {}}],
            },
            "b": {
                "spans": [],
                "embeddings": [],
                "encoders": [{"name": "enc", "modality": "text", "parameters": {}}],
            },
        },
        spans=[],
        modality_mask={},
    )
    subject, payload = pub._publisher.publish.call_args[0]
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert len(payload.encoders) == 1


@pytest.mark.asyncio
async def test_perception_publisher_emits_modality_subjects_and_correlation(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.publisher.Publisher",
        DummyPublisher,
    )
    pub = PerceptionPublisher(nats_client=object(), js_context=object())
    await pub.publish(
        "msg3",
        "user3",
        input_id="input-3",
        author_id="author-3",
        channel_id="channel-3",
        confidence=0.91,
        modality_confidence={"image": 0.8, "audio": 0.7},
        by_modality={
            "image": {"spans": [[0, 1]], "embeddings": [[0.1, 0.2]], "encoders": []},
            "audio": {"spans": [[0, 1]], "embeddings": [[0.3, 0.4]], "encoders": []},
        },
    )

    calls = pub._publisher.publish.await_args_list
    subjects = [call.args[0] for call in calls]
    assert subjects == [
        EventSubjects.PERCEPTION_EMBEDDINGS,
        EventSubjects.PERCEPTION_IMAGE_EMBED,
        EventSubjects.PERCEPTION_AUDIO_EMBED,
    ]
    event = calls[0].args[1]
    assert event.payload is not None
    assert event.payload.input_id == "input-3"
    assert event.payload.author_id == "author-3"
    assert event.payload.channel_id == "channel-3"
    assert event.payload.confidence == 0.91
    assert event.payload.modality_confidence == {"image": 0.8, "audio": 0.7}
