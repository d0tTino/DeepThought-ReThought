import numpy as np

from deepthought.services.perception import TextPerceptionWorker


def test_worker_text_memmap(tmp_path):
    tokens = [("hi", 0.0, 0.1), ("world", 0.1, 0.2)]
    worker = TextPerceptionWorker(hop_seconds=0.05)
    memmap_file = tmp_path / "features.dat"
    feats = worker(tokens, str(memmap_file))

    assert feats.shape == (4, 1)
    assert np.allclose(feats[0], feats[1])
    assert np.allclose(feats[2], feats[3])
    assert not np.allclose(feats[0], feats[2])
