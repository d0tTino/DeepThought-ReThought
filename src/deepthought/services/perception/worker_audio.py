from __future__ import annotations

"""Audio perception worker producing windowed RMS features."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from deepthought.perception.worker_audio import extract_windowed_features


@dataclass
class AudioPerceptionWorker:
    """Extract windowed RMS features from an audio file.

    Parameters
    ----------
    window_size:
        Size of each analysis window in seconds.
    step_size:
        Step between windows in seconds.
    """

    window_size: float = 0.02
    step_size: float = 0.01

    def __call__(self, audio_path: str | Path, cache_dir: str | Path | None = None) -> Tuple[np.memmap, np.ndarray]:
        """Return features and per-window timestamps for ``audio_path``.

        Parameters
        ----------
        audio_path:
            Path to a ``.wav`` file.
        cache_dir:
            Optional directory in which to store the memmap file. Defaults to the
            parent directory of ``audio_path``.
        """

        return extract_windowed_features(
            audio_path,
            window_size=self.window_size,
            step_size=self.step_size,
            cache_dir=cache_dir,
        )
