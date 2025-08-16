import numpy as np

from deepthought.services.perception.worker_video import VideoPerceptionWorker


def test_video_perception_worker(monkeypatch):
    feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    times = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "deepthought.services.perception.worker_video.video_to_feature_grid",
        lambda path, decode_fps, model_type, grid_fps: (feats, times),
    )

    worker = VideoPerceptionWorker(decode_fps=1, model_type="siglip")
    out_feats, out_times = worker("dummy.mp4")

    assert np.array_equal(out_feats, feats)
    assert np.array_equal(out_times, times)
