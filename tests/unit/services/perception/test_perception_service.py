from __future__ import annotations
import os
import sys
import types
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import numpy as np
import pytest

os.environ.setdefault("DEEPTHOUGHT_LIGHT_IMPORT", "1")
sys.modules.setdefault("deepthought.services", types.ModuleType("deepthought.services"))
sys.modules["deepthought.services"].__path__ = ["src/deepthought/services"]
sys.modules.setdefault(
    "deepthought.services.perception", types.ModuleType("deepthought.services.perception")
)
sys.modules["deepthought.services.perception"].__path__ = [
    "src/deepthought/services/perception"
]
worker_video_stub = types.ModuleType("deepthought.services.perception.worker_video")
worker_video_stub.VideoPerceptionWorker = object
sys.modules["deepthought.services.perception.worker_video"] = worker_video_stub
from deepthought.services.perception import service as service_module  # noqa: E402
from deepthought.services.perception.service import PerceptionService  # noqa: E402


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


@pytest.mark.asyncio
async def test_grid_hop_size_overrides_default(monkeypatch):
    fake_cfg = SimpleNamespace(
        grid_hop_size=0.1,
        text_cache_dir=None,
        audio_cache_dir=None,
        video_cache_dir=None,
        audio_window_size=0.02,
        audio_hop_size=0.01,
        audio_model="",
        audio_model_path=None,
        text_model="",
        text_hop_size=0.03,
        video_model="",
        video_hop_size=1.0,
        wandb_project=None,
        wandb_sweep_id=None,
        nats_url="",
    )
    monkeypatch.setattr(service_module, "PerceptionConfig", lambda: fake_cfg)
    publisher = DummyPublisher()
    service = PerceptionService(publisher, text_worker=DummyTextWorker())

    await service.run(
        message_id="m1",
        user_id="u1",
        text_tokens=[("hi", 0.0, 0.1)],
    )

    assert publisher.kwargs is not None
    assert publisher.kwargs["fused"] == [[2.0, 3.0]]
    text_meta = publisher.kwargs["by_modality"]["text"]
    assert text_meta["spans"] == [[0, 100]]
    assert text_meta["embeddings"] == [[2.0, 3.0]]
