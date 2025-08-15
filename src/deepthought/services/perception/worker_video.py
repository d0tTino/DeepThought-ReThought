from __future__ import annotations

"""Video perception worker converting clips into time-aligned features.

This worker leverages the utility functions in
:mod:`deepthought.perception.worker_video` to decode frames from a video file,
embed the frames using either SigLIP or OpenCLIP models, and interpolate the
resulting features onto a uniform time grid.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from deepthought.perception.worker_video import video_to_feature_grid


@dataclass
class VideoPerceptionWorker:
    """Decode ``path`` and emit features on a uniform time grid.

    Parameters
    ----------
    decode_fps:
        Target frame sampling rate between 1 and 3 frames-per-second.
    model_type:
        Embedding model to use, either ``"siglip"`` or ``"openclip"``.
    grid_fps:
        Optional output grid sampling rate. Defaults to ``decode_fps``.
    """

    decode_fps: int = 1
    model_type: str = "siglip"
    grid_fps: int | None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if not 1 <= self.decode_fps <= 3:
            raise ValueError("decode_fps must be between 1 and 3 fps")
        if self.grid_fps is not None and self.grid_fps <= 0:
            raise ValueError("grid_fps must be positive")

    def __call__(self, path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """Process ``path`` and return ``(features, times)`` arrays.

        Returns
        -------
        features: np.ndarray
            Grid of frame embeddings with shape ``[T, D]``.
        times: np.ndarray
            Uniform timestamps in seconds with shape ``[T]``.
        """

        return video_to_feature_grid(str(path), self.decode_fps, self.model_type, self.grid_fps)
