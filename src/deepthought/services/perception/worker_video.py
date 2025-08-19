from __future__ import annotations

"""Video perception worker converting clips into time-aligned features.

This worker leverages the utility functions in
:mod:`deepthought.perception.worker_video` to decode frames from a video file,
embed the frames using either SigLIP or OpenCLIP models, and interpolate the
resulting features onto a uniform time grid.
"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np

from deepthought.config import get_settings
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
    cache_dir:
        Optional directory for caching computed feature and timestamp arrays.
    """

    decode_fps: int = 1
    model_type: str = "siglip"
    grid_fps: int | None = None
    cache_dir: Path | None = None

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
        path = Path(path)
        feats = times = None
        cache_hit = False
        feats_file = times_file = embed_file = None
        if self.cache_dir is not None:
            cache_dir = Path(self.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"{self.decode_fps}_{self.model_type}_{self.grid_fps or self.decode_fps}"
            stem = path.stem
            feats_file = cache_dir / f"{stem}_{suffix}_feats.npy"
            times_file = cache_dir / f"{stem}_{suffix}_times.npy"
            embed_file = cache_dir / f"{stem}_{suffix}_embed.npy"
            if feats_file.exists() and times_file.exists():
                feats = np.load(feats_file, mmap_mode="r")
                times = np.load(times_file, mmap_mode="r")
                cache_hit = True

        if feats is None or times is None:
            if embed_file is not None:
                feats, times = video_to_feature_grid(
                    str(path),
                    self.decode_fps,
                    self.model_type,
                    self.grid_fps,
                    embed_cache=embed_file,
                )
            else:
                feats, times = video_to_feature_grid(
                    str(path),
                    self.decode_fps,
                    self.model_type,
                    self.grid_fps,
                )
            if feats_file and times_file:
                ff = np.lib.format.open_memmap(
                    feats_file, mode="w+", dtype=feats.dtype, shape=feats.shape
                )
                ff[:] = feats
                ff.flush()
                tf = np.lib.format.open_memmap(
                    times_file, mode="w+", dtype=times.dtype, shape=times.shape
                )
                tf[:] = times
                tf.flush()
                feats, times = ff, tf

        settings = get_settings()
        if settings.wandb_enabled:
            try:  # pragma: no cover - optional dependency
                import wandb

                wandb.log({"video_frames": feats.shape[0], "cache_hit": cache_hit})
                if settings.wandb_upload_artifacts and not cache_hit:
                    with NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                        np.save(tmp.name, feats)
                    art = wandb.Artifact(
                        name=f"video_features_{path.stem}",
                        type="features",
                    )
                    art.add_file(tmp.name)
                    wandb.log_artifact(art)
                    Path(tmp.name).unlink(missing_ok=True)
            except Exception:  # pragma: no cover - wandb may be missing
                pass

        return feats, times
