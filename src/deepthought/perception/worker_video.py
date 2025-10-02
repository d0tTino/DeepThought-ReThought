from __future__ import annotations

"""Utilities for decoding video and embedding frames.

This module provides helper functions to sample frames from a video at a
controlled rate, embed the frames using either SigLIP or OpenCLIP models,
and interpolate the resulting features onto a common time grid.
"""

from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from deepthought.config import get_settings
from deepthought.utils.model_specs import split_model_revision

try:  # pragma: no cover - optional dependency
    import open_clip
except Exception:  # pragma: no cover - optional dependency
    open_clip = None


FrameT = np.ndarray


def decode_video(path: str, fps: int = 1) -> Tuple[List[FrameT], np.ndarray]:
    """Decode ``path`` and return RGB frames and timestamps.

    Parameters
    ----------
    path:
        Path to the input video file.
    fps:
        Target frames-per-second rate between 1 and 3. Values outside the
        range are clamped.
    """
    fps = int(np.clip(fps, 1, 3))
    cap = cv2.VideoCapture(path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(int(round(video_fps / fps)), 1)
    frames: List[FrameT] = []
    timestamps: List[float] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            timestamps.append(idx / video_fps)
        idx += 1
    cap.release()
    return frames, np.asarray(timestamps, dtype=float)


def _siglip(device: torch.device):
    from transformers import SiglipImageProcessor, SiglipModel

    processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    return processor, model


def _openclip(device: torch.device):  # pragma: no cover - optional dependency
    if open_clip is None:
        raise ImportError("open_clip is not installed")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device)
    return preprocess, model


def embed_frames(
    frames: Iterable[FrameT],
    model_type: str = "siglip",
    device: torch.device | None = None,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Embed ``frames`` using ``model_type``.

    Parameters
    ----------
    frames:
        Iterable of RGB numpy arrays.
    model_type:
        Either ``"siglip"`` or ``"openclip"``.
    device:
        Torch device for model inference.
    cache_path:
        Optional path to a ``.npy`` file used to cache embeddings. When the
        file exists, it is loaded via :func:`numpy.load` with ``mmap_mode='r'``
        and returned directly. Otherwise, embeddings are computed and saved to
        this location using :func:`numpy.lib.format.open_memmap`.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_list = list(frames)
    if not frames_list:
        return np.empty((0, 0))

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return np.load(cache_path, mmap_mode="r")

    if model_type.lower() == "siglip":
        processor, model = _siglip(device)
        inputs = processor(images=frames_list, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
    elif model_type.lower() == "openclip":
        preprocess, model = _openclip(device)
        tensor = torch.stack([preprocess(Image.fromarray(f)) for f in frames_list]).to(device)
        with torch.no_grad():  # pragma: no cover - optional dependency
            feats = model.encode_image(tensor)
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"unknown model_type: {model_type}")

    feats_np = feats.cpu().numpy()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=feats_np.dtype, shape=feats_np.shape
        )
        mm[:] = feats_np
        mm.flush()
        return mm
    return feats_np


def interpolate_features(
    features: np.ndarray,
    timestamps: np.ndarray,
    grid_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate ``features`` onto ``grid_times``.

    ``features`` must be shaped ``(n_frames, dim)`` with corresponding
    ``timestamps`` in seconds.
    """
    if len(features) == 0:
        return np.empty((len(grid_times), 0))
    dim = features.shape[1]
    grid = np.empty((len(grid_times), dim), dtype=features.dtype)
    for i in range(dim):
        grid[:, i] = np.interp(grid_times, timestamps, features[:, i])
    return grid


def video_to_feature_grid(
    path: str,
    decode_fps: int = 1,
    model_type: str = "siglip",
    grid_fps: int | None = None,
    embed_cache: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode ``path`` and return features on a uniform time grid.

    Parameters
    ----------
    path:
        Video file to process.
    decode_fps:
        Frame sampling rate used during decoding.
    model_type:
        Embedding model to use.
    grid_fps:
        Optional output grid sampling rate.
    embed_cache:
        Optional path to a ``.npy`` file for caching frame embeddings.
    """
    frames, timestamps = decode_video(path, decode_fps)
    model_key, _ = split_model_revision(model_type)
    feats = embed_frames(frames, model_type=model_key, cache_path=embed_cache)
    if grid_fps is None:
        grid_fps = decode_fps
    if len(timestamps) == 0:
        return np.empty((0, feats.shape[1])), np.empty(0)
    start, end = timestamps[0], timestamps[-1]
    step = 1.0 / grid_fps
    grid_times = np.arange(start, end + step, step)
    grid_feats = interpolate_features(feats, timestamps, grid_times)

    settings = get_settings()
    if settings.wandb_enabled:
        try:  # pragma: no cover - optional dependency
            import wandb

            wandb.log({"video_frames": len(frames)})
        except Exception:  # pragma: no cover - wandb may be missing
            pass

    return grid_feats, grid_times
