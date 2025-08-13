import os
import tempfile

import cv2
import numpy as np

from deepthought.perception.worker_video import decode_video, interpolate_features


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


def test_decode_and_interpolate():
    path = _make_test_video()
    frames, ts = decode_video(path, fps=2)
    os.remove(path)
    assert len(frames) == 4
    assert np.allclose(ts, [0.0, 0.5, 1.0, 1.5])

    features = np.stack([ts, ts**2], axis=1)
    grid_times = np.arange(0.0, 1.51, 0.25)
    grid_feats = interpolate_features(features, ts, grid_times)
    assert grid_feats.shape == (len(grid_times), 2)
    # Check interpolation at a midpoint
    assert np.allclose(grid_feats[2], [0.5, 0.25])
