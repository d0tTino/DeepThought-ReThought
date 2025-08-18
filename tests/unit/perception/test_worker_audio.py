import numpy as np
from scipy.io import wavfile

from deepthought.perception.worker_audio import extract_windowed_features


def test_extract_windowed_features(tmp_path):
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_file = tmp_path / "test.wav"
    wavfile.write(audio_file, sr, data)

    features, timestamps = extract_windowed_features(audio_file, window_size=0.1, step_size=0.05, cache_dir=tmp_path)

    assert features.shape == (19, 4)
    assert timestamps.shape == (19, 2)
    assert np.allclose(timestamps[0], [0.0, 0.1])
    assert np.allclose(timestamps[1], [0.05, 0.15])

    # Second call should reuse memmap and produce identical results
    features2, timestamps2 = extract_windowed_features(audio_file, window_size=0.1, step_size=0.05, cache_dir=tmp_path)
    assert np.allclose(features, features2)
    assert np.allclose(timestamps, timestamps2)
    memmap_path = tmp_path / "test_wavlm_ws0.1_ss0.05.dat"
    assert memmap_path.exists()
