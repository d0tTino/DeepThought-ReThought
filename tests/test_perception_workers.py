import numpy as np
import pytest
from scipy.io import wavfile

from deepthought.services.perception.worker_audio import AudioPerceptionWorker
from deepthought.services.perception.worker_text import TextPerceptionWorker
from deepthought.services.perception.worker_video import VideoPerceptionWorker


class _DummySentenceModel:
    def encode(self, text: str) -> np.ndarray:
        length = len(text)
        return np.asarray([length, length + 1], dtype=np.float32)


def test_audio_perception_worker(tmp_path):
    sr = 16000
    data = np.ones(int(0.05 * sr), dtype=np.int16)
    path = tmp_path / "test.wav"
    wavfile.write(path, sr, data)

    worker = AudioPerceptionWorker(window_size=0.02, step_size=0.01)
    features, times = worker(path)

    assert features.shape == (4, 4)
    assert times.shape == (4, 2)


def test_text_perception_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "deepthought.services.perception.worker_text.SentenceTransformer",
        lambda name: _DummySentenceModel(),
    )
    tokens = [("hi", 0.0, 0.05), ("there", 0.05, 0.1)]
    memmap_path = tmp_path / "tokens.dat"

    worker = TextPerceptionWorker(model_name="dummy", hop_seconds=0.05)
    feats, times = worker(tokens, memmap_path)

    assert feats.shape == (2, 2)
    assert times.shape == (2, 2)
    assert np.allclose(feats[0], [2, 3])
    assert np.allclose(feats[1], [5, 6])


def test_video_perception_worker(monkeypatch):
    dummy_feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dummy_times = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "deepthought.services.perception.worker_video.video_to_feature_grid",
        lambda path, decode_fps, model_type, grid_fps: (dummy_feats, dummy_times),
    )

    worker = VideoPerceptionWorker(decode_fps=1, model_type="siglip")
    feats, times = worker("dummy.mp4")

    assert np.array_equal(feats, dummy_feats)
    assert np.array_equal(times, dummy_times)
