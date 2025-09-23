from __future__ import annotations

import asyncio
from typing import Dict

import numpy as np

from deepthought.services.perception.service import PerceptionService


class DummyPublisher:
    def __init__(self) -> None:
        self.kwargs: Dict | None = None

    async def publish(self, **kwargs) -> None:  # pragma: no cover - simple async stub
        self.kwargs = kwargs


class DummyVideoWorker:
    def __call__(self, path: str):
        data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
        times = np.array([[0.0, 0.05], [0.05, 0.1]], dtype=np.float32)
        return data, times


def test_service_publishes_video_embeddings_and_metadata():
    publisher = DummyPublisher()
    service = PerceptionService(publisher, video_worker=DummyVideoWorker())

    asyncio.run(
        service.run(
            message_id="m1",
            user_id="u1",
            video_path="v.mp4",
            video_opt_in=True,
            provenance={"test": True},
        )
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["message_id"] == "m1"
    assert publisher.kwargs["user_id"] == "u1"
    assert publisher.kwargs["fused"] == [[5.0, 6.0], [7.0, 8.0]]
    assert "video" in publisher.kwargs["by_modality"]
    video_meta = publisher.kwargs["by_modality"]["video"]
    assert video_meta["spans"] == [[0, 50], [50, 100]]
    assert video_meta["embeddings"] == [[5.0, 6.0], [7.0, 8.0]]
    assert len(video_meta["encoders"]) == 2
    video_encoder = video_meta["encoders"][0]
    assert video_encoder["name"] == "DummyVideoWorker"
    assert video_encoder["modality"] == "video"
    assert video_encoder["dim"] == 2
    assert video_encoder["parameters"]["config_source"] == "PerceptionConfig.video_model"
    assert "hop_size" in video_encoder["parameters"]
    assert publisher.kwargs["provenance"]["test"] is True
    assert publisher.kwargs["provenance"]["modalities"] == ["video"]
    assert "timestamp" in publisher.kwargs["provenance"]
