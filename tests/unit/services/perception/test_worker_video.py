import os
import tempfile

import cv2
import numpy as np

from deepthought.perception import worker_video as util
from deepthought.services.perception.worker_video import VideoPerceptionWorker


def _make_test_video() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = tmp.name
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 10, (32, 32))
    for i in range(20):
        frame = np.full((32, 32, 3), i, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def test_video_perception_worker(monkeypatch):
    path = _make_test_video()

    def fake_embed(frames, model_type="siglip", device=None):
        return np.arange(len(frames), dtype=float).reshape(-1, 1)

    monkeypatch.setattr(util, "embed_frames", fake_embed)

    worker = VideoPerceptionWorker(decode_fps=2, model_type="siglip", grid_fps=4)
    features, times = worker(path)
    os.remove(path)

    assert features.shape == (7, 1)
    assert np.allclose(times, np.arange(0.0, 1.51, 0.25))
    assert np.allclose(features[:, 0], [0, 0.5, 1, 1.5, 2, 2.5, 3])
