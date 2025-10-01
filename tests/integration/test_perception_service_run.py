"""Integration coverage for :class:`PerceptionService.run`."""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import numpy as np
import pytest

os.environ.setdefault("DEEPTHOUGHT_LIGHT_IMPORT", "1")

modules_pkg = types.ModuleType("deepthought.modules")
modules_pkg.__path__ = ["src/deepthought/modules"]
modules_pkg.ModalityFuser = object  # type: ignore[attr-defined]
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

publisher_stub = types.ModuleType("deepthought.services.perception.publisher")


class _BasePublisher:
    async def publish(self, **kwargs) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError


publisher_stub.PerceptionPublisher = _BasePublisher  # type: ignore[attr-defined]
sys.modules.setdefault("deepthought.services.perception.publisher", publisher_stub)

import deepthought  # type: ignore  # noqa: E402

deepthought.modules = modules_pkg  # type: ignore[attr-defined]
deepthought.services = services_pkg  # type: ignore[attr-defined]
from deepthought.services.perception.service import PerceptionService  # noqa: E402


class RecordingPublisher(_BasePublisher):
    """Capture arguments passed to ``publish``."""

    def __init__(self) -> None:
        self.calls: list[Dict] = []

    async def publish(self, **kwargs) -> None:  # pragma: no cover - simple async stub
        self.calls.append(kwargs)


class StubTextWorker:
    """Return deterministic features and timestamps for two tokens."""

    def __call__(
        self,
        tokens: Sequence[Tuple[str, float, float]],
        memmap_path: str,
    ):
        data = np.array([[1.0, 2.0], [2.0, 3.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        times = np.array([[0.0, 0.05], [0.05, 0.1]], dtype=np.float32)
        return mm, times


@pytest.mark.asyncio
async def test_perception_service_publishes_expected_spans_and_masks(monkeypatch, tmp_path):
    """The real service should publish aligned spans and modality masks."""

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
        modality_dims={"text": 2},
    )
    monkeypatch.setattr(
        "deepthought.services.perception.service.PerceptionConfig", lambda: fake_cfg
    )

    publisher = RecordingPublisher()
    worker = StubTextWorker()
    service = PerceptionService(publisher, text_worker=worker)

    await service.run(
        message_id="msg-123",
        user_id="user-456",
        text_tokens=[("hello", 0.0, 0.05), ("world", 0.05, 0.1)],
    )

    assert len(publisher.calls) == 1
    kwargs = publisher.calls[0]
    assert kwargs["message_id"] == "msg-123"
    assert kwargs["user_id"] == "user-456"

    spans = kwargs["spans"]
    assert spans == [[0, 50], [50, 100]]

    modality_mask = kwargs["modality_mask"]
    assert modality_mask == {"text": [True, True]}

    contribution_mask = kwargs["contribution_mask"]
    assert contribution_mask == {"text": [True, True]}

    by_modality = kwargs["by_modality"]
    assert set(by_modality.keys()) == {"text"}
    text_payload = by_modality["text"]
    assert text_payload["spans"] == spans
    np.testing.assert_allclose(
        np.asarray(text_payload["embeddings"], dtype=np.float32),
        np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
    )
    assert text_payload["mask"] == [True, True]
    assert len(text_payload["encoders"]) == 2
