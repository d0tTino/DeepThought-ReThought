from __future__ import annotations

"""Audio worker utilities.

This module extracts simple windowed features from ``.wav`` files and caches
results using memory-mapped arrays. Each window emits a ``[start, end]``
 timestamp that aligns with the text worker grid.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile

from deepthought.config import get_settings


def extract_windowed_features(
    audio_path: str | Path,
    window_size: float = 0.02,
    step_size: float = 0.01,
    cache_dir: str | Path | None = None,
) -> Tuple[np.memmap, np.ndarray]:
    """Return RMS features and per-window timestamps for ``audio_path``.

    Parameters
    ----------
    audio_path:
        Path to a ``.wav`` file.
    window_size:
        Size of each analysis window in seconds.
    step_size:
        Stride between analysis windows in seconds.
    cache_dir:
        Optional directory in which to store the memmap file. Defaults to the
        parent directory of ``audio_path``.
    """

    audio_path = Path(audio_path)
    if cache_dir is None:
        cache_dir = audio_path.parent
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sr, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    win_samples = int(window_size * sr)
    step_samples = int(step_size * sr)
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("window_size and step_size must be positive")

    n_samples = len(data)
    if n_samples < win_samples:
        num_windows = 0
    else:
        num_windows = 1 + (n_samples - win_samples) // step_samples

    memmap_path = cache_dir / f"{audio_path.stem}_ws{window_size}_ss{step_size}.dat"
    shape = (num_windows, 1)

    if memmap_path.exists():
        features = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=shape)
    else:
        features = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=shape)
        for i in range(num_windows):
            start = i * step_samples
            end = start + win_samples
            window = data[start:end]
            features[i, 0] = np.sqrt(np.mean(window.astype(np.float32) ** 2))
        features.flush()

    starts = np.arange(num_windows) * step_size
    ends = starts + window_size
    timestamps = np.column_stack((starts, ends)).astype(np.float32)

    settings = get_settings()
    if settings.wandb_enabled:
        try:  # pragma: no cover - optional dependency
            import wandb

            wandb.log({"audio_windows": num_windows})
            if settings.wandb_upload_artifacts:
                art = wandb.Artifact(
                    name=f"audio_features_{memmap_path.stem}",
                    type="features",
                )
                art.add_file(str(memmap_path))
                wandb.log_artifact(art)
        except Exception:  # pragma: no cover - wandb may be missing
            pass

    return features, timestamps
