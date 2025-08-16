import numpy as np
import pytest
from unittest.mock import AsyncMock

from deepthought.eda.events import EventSubjects, PerceptionEmbeddingsPayload
from deepthought.services.perception.publisher import PerceptionPublisher
from deepthought.services.perception.service import PerceptionService


class DummyTextWorker:
    def __call__(self, tokens, memmap_path):
        data = np.array([[1.0, 2.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        return mm


class DummyAudioWorker:
    def __call__(self, audio_path):
        feats = np.array([[0.5, 1.5]], dtype="float32")
        times = np.array([0.0], dtype="float32")
        return feats, times


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.publish = AsyncMock(return_value={"seq": 1})


@pytest.mark.asyncio
async def test_service_end_to_end(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.publisher.Publisher",
        DummyPublisher,
    )

    publisher = PerceptionPublisher(nats_client=object(), js_context=object())
    service = PerceptionService(
        publisher,
        text_worker=DummyTextWorker(),
        audio_worker=DummyAudioWorker(),
    )

    await service.run(
        message_id="m1",
        user_id="u1",
        text_tokens=[("hi", 0.0, 0.1)],
        audio_path="a.wav",
        provenance={"source": "integration"},
    )

    publisher._publisher.publish.assert_awaited_once()
    args, _ = publisher._publisher.publish.call_args
    subject, payload = args
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert isinstance(payload, PerceptionEmbeddingsPayload)
    assert payload.embeddings == [[1.0, 2.0], [0.5, 1.5]]
    assert payload.spans == [[0, 1], [1, 2]]
    assert payload.encoders == [
        {"name": "DummyTextWorker"},
        {"name": "DummyAudioWorker"},
    ]
    assert payload.provenance == {
        "source": "integration",
        "modalities": ["text", "audio"],
    }
