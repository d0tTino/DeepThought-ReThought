from __future__ import annotations

import asyncio
from typing import Dict, Sequence, Tuple

import numpy as np
import pytest
import torch

from deepthought.services.perception.service import PerceptionService
from deepthought.services.perception.user_embeddings import UserEmbeddings


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


class DummyAudioWorker:
    cache_dir = None
    model = "dummy"
    window_size = 1
    step_size = 1

    def __call__(self, path: str, cache_dir=None):  # pragma: no cover - simple stub
        feats = np.array([[0.5, 0.5], [0.25, 0.25]], dtype="float32")
        times = np.array([[0.0, 0.05], [0.05, 0.1]], dtype=np.float32)
        return feats, times


class SumFuser:
    def __init__(self):
        self.modality_dims = {"text": 2, "audio": 2}
        self.user_dim = 0

    def __call__(
        self,
        modalities: Dict[str, torch.Tensor],
        user_embedding=None,
        user_id: str | None = None,
        embedding_store=None,
    ) -> torch.Tensor:  # pragma: no cover - deterministic stub
        ordered = [modalities[name] for name in sorted(modalities.keys())]
        stacked = torch.stack(ordered, dim=0)
        return stacked.sum(dim=0)


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
    assert publisher.kwargs["fused"] == [[1.0, 2.0], [3.0, 4.0]]
    assert "text" in publisher.kwargs["by_modality"]
    text_meta = publisher.kwargs["by_modality"]["text"]
    assert text_meta["spans"] == [[0, 50], [50, 100]]
    assert text_meta["embeddings"] == [[1.0, 2.0], [3.0, 4.0]]
    assert len(text_meta["encoders"]) == 2
    text_encoder = text_meta["encoders"][0]
    assert text_encoder["name"] == "DummyTextWorker"
    assert text_encoder["modality"] == "text"
    assert text_encoder["dim"] == 2
    assert text_encoder["parameters"]["config_source"] == "PerceptionConfig.text_model"
    assert "hop_size" in text_encoder["parameters"]
    assert publisher.kwargs["provenance"]["test"] is True
    assert publisher.kwargs["provenance"]["modalities"] == ["text"]
    assert "timestamp" in publisher.kwargs["provenance"]


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
            video_opt_in=True,
            provenance={"test": True},
        )
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["fused"] == [[1.0, 1.0], [2.0, 2.0]]
    assert "video" in publisher.kwargs["by_modality"]
    video_meta = publisher.kwargs["by_modality"]["video"]
    assert video_meta["spans"] == [[0, 1000], [1000, 2000]]
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


def test_service_requires_fuser_for_multiple_modalities(tmp_path):
    publisher = DummyPublisher()
    service = PerceptionService(
        publisher,
        text_worker=DummyTextWorker(),
        audio_worker=DummyAudioWorker(),
    )

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"00")

    with pytest.raises(ValueError, match="fuser"):
        asyncio.run(
            service.run(
                message_id="m1",
                user_id="u1",
                text_tokens=[("hi", 0.0, 0.1)],
                audio_path=audio_path,
                audio_opt_in=True,
                retain_media=True,
            )
        )


def test_service_updates_user_embeddings(tmp_path):
    publisher = DummyPublisher()
    store_path = tmp_path / "store.json"
    user_store = UserEmbeddings(store_path)
    service = PerceptionService(
        publisher,
        text_worker=DummyTextWorker(),
        audio_worker=DummyAudioWorker(),
        fuser=SumFuser(),
        user_embeddings=user_store,
    )

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"00")

    asyncio.run(
        service.run(
            message_id="m1",
            user_id="u1",
            text_tokens=[("hi", 0.0, 0.1)],
            audio_path=audio_path,
            audio_opt_in=True,
            retain_media=True,
        )
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["fused"] == [[1.5, 2.5], [3.25, 4.25]]

    stored = user_store.get("u1")
    assert stored is not None
    expected = torch.tensor([2.375, 3.375], dtype=torch.float32)
    assert torch.allclose(stored, expected)


class UserAwareFuser:
    def __init__(self):
        self.modality_dims = {"text": 2}
        self.user_dim = 2

    def __call__(
        self,
        modalities: Dict[str, torch.Tensor],
        user_embedding=None,
        user_id: str | None = None,
        embedding_store=None,
    ) -> torch.Tensor:
        assert user_embedding is not None
        assert user_embedding.shape == modalities["text"].shape
        return modalities["text"] + user_embedding


def test_service_expands_stored_user_embedding(tmp_path):
    publisher = DummyPublisher()
    store_path = tmp_path / "store.json"
    user_store = UserEmbeddings(store_path)
    user_store.set("u1", torch.tensor([0.5, 1.5], dtype=torch.float32))
    service = PerceptionService(
        publisher,
        text_worker=DummyTextWorker(),
        fuser=UserAwareFuser(),
        user_embeddings=user_store,
    )

    asyncio.run(
        service.run(
            message_id="m1",
            user_id="u1",
            text_tokens=[("hi", 0.0, 0.1)],
        )
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["fused"] == [[1.5, 3.5], [3.5, 5.5]]

    stored = user_store.get("u1")
    assert stored is not None
    expected = torch.tensor([2.5, 4.5], dtype=torch.float32)
    assert torch.allclose(stored, expected)
