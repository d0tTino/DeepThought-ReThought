from __future__ import annotations

import asyncio
import os
import sys
import types
from typing import Dict

import numpy as np
import pytest

os.environ.setdefault("DEEPTHOUGHT_LIGHT_IMPORT", "1")
services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = ["src/deepthought/services"]
sys.modules.setdefault("deepthought.services", services_pkg)

perception_pkg = types.ModuleType("deepthought.services.perception")
perception_pkg.__path__ = ["src/deepthought/services/perception"]
sys.modules.setdefault("deepthought.services.perception", perception_pkg)

worker_video_stub = types.ModuleType("deepthought.services.perception.worker_video")
worker_video_stub.VideoPerceptionWorker = object
sys.modules.setdefault("deepthought.services.perception.worker_video", worker_video_stub)

core_video_stub = types.ModuleType("deepthought.perception.worker_video")
core_video_stub.video_to_feature_grid = lambda *args, **kwargs: (np.zeros((1, 1)), np.zeros((1,)))
sys.modules.setdefault("deepthought.perception.worker_video", core_video_stub)

from deepthought.services.perception.service import PerceptionService


@pytest.fixture(autouse=True)
def _stub_build_metadata(monkeypatch):
    service_module = sys.modules[PerceptionService.__module__]
    monkeypatch.setattr(service_module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(service_module, "get_package_version", lambda: "0.0.test")
    monkeypatch.setattr(service_module, "get_container_tag", lambda: "unit-test")


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
    encoder = video_meta["encoders"][0]
    assert encoder["name"] == "DummyVideoWorker"
    assert encoder["modality"] == "video"
    assert encoder["dim"] == 2
    params = encoder["parameters"]
    assert params.get("model")
    assert "decode_fps" in params and "grid_fps" in params
    provenance = publisher.kwargs["provenance"]
    assert provenance["test"] is True
    assert provenance["modalities"] == ["video"]
    assert isinstance(provenance["timestamp"], float)
    assert provenance["git_commit"] == "deadbeef"
    assert provenance["package_version"] == "0.0.test"
    assert provenance["container_tag"] == "unit-test"
