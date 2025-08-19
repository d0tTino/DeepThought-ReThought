from __future__ import annotations

"""Audio perception worker producing windowed audio embeddings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from deepthought.perception.worker_audio import extract_windowed_features

from .config import PerceptionConfig


@dataclass
class AudioPerceptionWorker:
    """Extract windowed embeddings from an audio file.

    Parameters
    ----------
    window_size:
        Size of each analysis window in seconds.
    step_size:
        Step between windows in seconds.
    model:
        Embedding model to use (``"wavlm"`` or ``"clap"``).
    model_path:
        Optional path or identifier for the embedding model.
    cache_dir:
        Default directory in which to store memmap files.
    """

    _cfg = PerceptionConfig()

    window_size: float = _cfg.audio_window_size
    step_size: float = _cfg.audio_hop_size
    model: str = _cfg.audio_model
    model_path: str | None = _cfg.audio_model_path
    cache_dir: str | Path | None = _cfg.audio_cache_dir

    def __call__(self, audio_path: str | Path, cache_dir: str | Path | None = None) -> Tuple[np.memmap, np.ndarray]:
        """Return features and per-window timestamps for ``audio_path``."""

        return extract_windowed_features(
            audio_path,
            window_size=self.window_size,
            step_size=self.step_size,
            cache_dir=cache_dir or self.cache_dir,
            model=self.model,
            model_path=self.model_path,
        )
