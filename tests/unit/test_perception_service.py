from __future__ import annotations

import asyncio
from typing import Dict, Sequence, Tuple

import numpy as np

from deepthought.services.perception.service import PerceptionService


class DummyPublisher:
    def __init__(self) -> None:
        self.kwargs: Dict | None = None

    async def publish(self, **kwargs) -> None:  # pragma: no cover - simple async stub
        self.kwargs = kwargs


class DummyTextWorker:
    def __call__(self, tokens: Sequence[Tuple[str, float, float]], memmap_path: str):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        times = np.array([[0.0, 0.05], [0.05, 0.1]], dtype=np.float32)
        return mm, times


def test_service_publishes_raw_embeddings_and_metadata():
    publisher = DummyPublisher()
    service = PerceptionService(publisher, text_worker=DummyTextWorker())

    asyncio.run(
        service.run(
            message_id="m1",
            user_id="u1",
            text_tokens=[("hi", 0.0, 0.1)],
            provenance={"test": True},
        )
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["message_id"] == "m1"
    assert publisher.kwargs["user_id"] == "u1"
    assert publisher.kwargs["fused"] is None
    assert "text" in publisher.kwargs["by_modality"]
    text_meta = publisher.kwargs["by_modality"]["text"]
    assert text_meta["spans"] == [[0, 1], [1, 2]]
    assert text_meta["embeddings"] == [[1.0, 2.0], [3.0, 4.0]]
    assert text_meta["encoders"] == [{"name": "DummyTextWorker"}] * 2
    assert publisher.kwargs["provenance"] == {"test": True, "modalities": ["text"]}


class DummyVideoWorker:
    def __call__(self, path: str):
        feats = np.array([[1.0, 1.0], [2.0, 2.0]], dtype="float32")
        times = np.array([0.0, 1.0], dtype=np.float32)
        return feats, times


def test_service_handles_video_modality():
    publisher = DummyPublisher()
    service = PerceptionService(publisher, video_worker=DummyVideoWorker())

    asyncio.run(
        service.run(
            message_id="m1",
            user_id="u1",
            video_path="video.mp4",
            provenance={"test": True},
        )
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["fused"] is None
    assert "video" in publisher.kwargs["by_modality"]
    video_meta = publisher.kwargs["by_modality"]["video"]
    assert video_meta["spans"] == [[0, 1], [1, 2]]
    assert video_meta["encoders"] == [{"name": "DummyVideoWorker"}] * 2
    assert publisher.kwargs["provenance"] == {"test": True, "modalities": ["video"]}
