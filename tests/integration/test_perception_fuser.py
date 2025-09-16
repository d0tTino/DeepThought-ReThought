"""Integration test for the perception fuser pipeline."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import numpy as np
import pytest
import torch

os.environ.setdefault("DEEPTHOUGHT_LIGHT_IMPORT", "1")

modules_pkg = types.ModuleType("deepthought.modules")
modules_pkg.__path__ = ["src/deepthought/modules"]
sys.modules.setdefault("deepthought.modules", modules_pkg)

services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = ["src/deepthought/services"]
sys.modules.setdefault("deepthought.services", services_pkg)

perception_pkg = types.ModuleType("deepthought.services.perception")
perception_pkg.__path__ = ["src/deepthought/services/perception"]
sys.modules.setdefault("deepthought.services.perception", perception_pkg)
services_pkg.perception = perception_pkg  # type: ignore[attr-defined]

worker_text_stub = types.ModuleType("deepthought.services.perception.worker_text")
worker_text_stub.TextPerceptionWorker = object  # type: ignore[attr-defined]
worker_text_stub.Token = Tuple[str, float, float]  # type: ignore[attr-defined]
sys.modules.setdefault("deepthought.services.perception.worker_text", worker_text_stub)

worker_audio_stub = types.ModuleType("deepthought.services.perception.worker_audio")
worker_audio_stub.AudioPerceptionWorker = object  # type: ignore[attr-defined]
sys.modules.setdefault(
    "deepthought.services.perception.worker_audio", worker_audio_stub
)

worker_video_stub = types.ModuleType("deepthought.services.perception.worker_video")
worker_video_stub.VideoPerceptionWorker = object  # type: ignore[attr-defined]
sys.modules.setdefault(
    "deepthought.services.perception.worker_video", worker_video_stub
)

fuser_module = importlib.import_module("deepthought.modules.fuser")
ModalityFuser = fuser_module.ModalityFuser
modules_pkg.ModalityFuser = ModalityFuser

import deepthought  # type: ignore  # noqa: E402

deepthought.modules = modules_pkg  # type: ignore[attr-defined]
deepthought.services = services_pkg  # type: ignore[attr-defined]
from deepthought.services.perception.service import PerceptionService
from deepthought.services.perception.user_embeddings import UserEmbeddings

if not hasattr(torch, "SymBool"):
    pytest.skip("PyTorch lacks SymBool", allow_module_level=True)


class DummyPublisher:
    """Capture publish calls for assertion."""

    def __init__(self) -> None:
        self.kwargs: Dict | None = None

    async def publish(self, **kwargs) -> None:  # pragma: no cover - simple async stub
        self.kwargs = kwargs


class DummyTextWorker:
    """Return deterministic text embeddings for two spans."""

    def __call__(self, tokens: Sequence[Tuple[str, float, float]], memmap_path: str):
        data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        times = np.array([[0.0, 0.05], [0.05, 0.1]], dtype=np.float32)
        return mm, times


class DummyAudioWorker:
    """Return deterministic audio embeddings for two spans."""

    def __init__(self) -> None:
        self.model = "dummy"
        self.window_size = 0.05
        self.step_size = 0.05
        self.cache_dir = None

    def __call__(self, audio_path: str, cache_dir: str | None = None):
        feats = np.array([[0.2, 0.4], [0.6, 0.8]], dtype="float32")
        times = np.array([[0.0, 0.05], [0.05, 0.1]], dtype=np.float32)
        return feats, times


@pytest.mark.asyncio
async def test_perception_fuser_publishes_fused_vectors(tmp_path, monkeypatch):
    """Fusing multiple modalities publishes fused vectors and updates user embeddings."""

    monkeypatch.setattr(
        "deepthought.services.perception.service.get_settings",
        lambda: SimpleNamespace(
            wandb_enabled=False,
            wandb_project=None,
            wandb_sweep_id=None,
            wandb_upload_artifacts=False,
        ),
    )

    fake_cfg = SimpleNamespace(
        grid_hop_size=None,
        text_cache_dir=None,
        audio_cache_dir=None,
        video_cache_dir=None,
        audio_window_size=0.05,
        audio_hop_size=0.05,
        audio_model="dummy",
        audio_model_path=None,
        text_model="dummy",
        text_hop_size=0.05,
        video_model="dummy",
        video_hop_size=1.0,
    )
    monkeypatch.setattr(
        "deepthought.services.perception.service.PerceptionConfig", lambda: fake_cfg
    )

    publisher = DummyPublisher()
    text_worker = DummyTextWorker()
    audio_worker = DummyAudioWorker()

    fuser = ModalityFuser({"text": 2, "audio": 2}, fused_dim=4, user_dim=3)
    fuser.eval()

    user_id = "user-123"
    message_id = "msg-1"
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")

    embedding_file = tmp_path / "embeddings.json"
    embedding_store = UserEmbeddings(embedding_file)
    initial_user_embedding = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.1, 0.5]], dtype=torch.float32
    )
    embedding_store.set(user_id, initial_user_embedding)

    expected_modalities = {
        "text": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "audio": torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float32),
    }
    with torch.no_grad():
        expected_fused = fuser(
            expected_modalities, user_embedding=initial_user_embedding
        ).tolist()

    service = PerceptionService(
        publisher,
        text_worker=text_worker,
        audio_worker=audio_worker,
        fuser=fuser,
        user_embeddings=embedding_store,
    )

    await service.run(
        message_id=message_id,
        user_id=user_id,
        text_tokens=[("hello", 0.0, 0.05), ("world", 0.05, 0.1)],
        audio_path=str(audio_path),
        audio_opt_in=True,
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["message_id"] == message_id
    assert publisher.kwargs["user_id"] == user_id

    fused_payload = publisher.kwargs["fused"]
    assert fused_payload is not None
    np.testing.assert_allclose(
        np.asarray(fused_payload, dtype=np.float32),
        np.asarray(expected_fused, dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )

    modalities = publisher.kwargs["by_modality"]
    assert set(modalities.keys()) >= {"text", "audio"}
    np.testing.assert_allclose(
        np.asarray(modalities["text"]["embeddings"], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(modalities["audio"]["embeddings"], dtype=np.float32),
        np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
    )

    saved = json.loads(embedding_file.read_text())
    assert user_id in saved
    expected_mean = initial_user_embedding.mean(dim=0).tolist()
    np.testing.assert_allclose(
        np.asarray(saved[user_id], dtype=np.float32),
        np.asarray(expected_mean, dtype=np.float32),
    )
