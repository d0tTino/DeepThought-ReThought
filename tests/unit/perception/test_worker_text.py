import numpy as np
import pytest

from deepthought.services.perception.worker_text import TextPerceptionWorker


class _DummySentenceModel:
    def encode(self, text: str) -> np.ndarray:
        length = len(text)
        return np.asarray([length, length + 1], dtype=np.float32)


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
