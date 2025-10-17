"""Audio worker utilities.

This module extracts windowed audio embeddings and caches results using
memory-mapped arrays. Each window emits a ``[start, end]`` timestamp that
aligns with the text worker grid. Embeddings are produced from either
``WavLM`` or ``CLAP`` models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from scipy.io import wavfile

from deepthought.config import get_settings


def _parse_model_spec(spec: str | Path | None) -> tuple[str | Path | None, str | None]:
    """Return a normalized (identifier, revision) tuple for ``spec``."""

    if spec is None:
        return None, None
    if isinstance(spec, Path):
        return spec, None
    name = spec.strip()
    if not name:
        return None, None
    if "@" not in name:
        return name, None
    base, revision = name.split("@", 1)
    base = base.strip() or None
    revision = revision.strip() or None
    return base, revision


def _select_embedding_fn(
    model: str,
    model_path: str | Path | None,
    sampling_rate: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that maps a waveform to an embedding.

    Parameters
    ----------
    model:
        Which embedding model to use (``"wavlm"`` or ``"clap"``).
    model_path:
        Optional Hugging Face model identifier.
    sampling_rate:
        Sampling rate of the audio in Hertz.
    """

    model_name, model_revision = _parse_model_spec(model)
    if not model_name:
        raise ValueError(f"Unsupported model: {model}")
    model_key = str(model_name).lower()

    path_name, path_revision = _parse_model_spec(model_path)

    if model_key == "wavlm":
        try:
            import torch
            from transformers import WavLMFeatureExtractor, WavLMModel  # type: ignore

            name = str(path_name) if path_name is not None else "microsoft/wavlm-base-plus"
            revision = path_revision or (model_revision if path_name is None else None)
            load_kwargs = {"revision": revision} if revision else {}

            extractor = WavLMFeatureExtractor.from_pretrained(name, **load_kwargs)
            mdl = WavLMModel.from_pretrained(name, **load_kwargs)
            mdl.eval()

            def embed(window: np.ndarray) -> np.ndarray:
                inputs = extractor(window, sampling_rate=sampling_rate, return_tensors="pt")
                with torch.no_grad():
                    hidden = mdl(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
                return hidden.cpu().numpy().astype(np.float32)

            return embed
        except Exception:  # pragma: no cover - optional dependency
            pass

    elif model_key == "clap":
        try:
            import torch
            from transformers import ClapModel, ClapProcessor  # type: ignore

            name = str(path_name) if path_name is not None else "laion/clap-htsat-unfused"
            revision = path_revision or (model_revision if path_name is None else None)
            load_kwargs = {"revision": revision} if revision else {}

            processor = ClapProcessor.from_pretrained(name, **load_kwargs)
            mdl = ClapModel.from_pretrained(name, **load_kwargs)
            mdl.eval()

            def embed(window: np.ndarray) -> np.ndarray:
                inputs = processor(audios=window, sampling_rate=sampling_rate, return_tensors="pt")
                with torch.no_grad():
                    hidden = mdl.get_audio_features(**inputs).squeeze(0)
                return hidden.cpu().numpy().astype(np.float32)

            return embed
        except Exception:  # pragma: no cover - optional dependency
            pass
    else:
        raise ValueError(f"Unsupported model: {model}")

    def embed(window: np.ndarray) -> np.ndarray:
        """Fallback embedding using simple statistics."""

        return np.asarray(
            [
                float(window.mean()),
                float(window.std()),
                float(window.min()),
                float(window.max()),
            ],
            dtype=np.float32,
        )

    return embed


def extract_windowed_features(
    audio_path: str | Path,
    window_size: float = 0.02,
    step_size: float = 0.01,
    cache_dir: str | Path | None = None,
    *,
    model: str = "wavlm",
    model_path: str | Path | None = None,
) -> Tuple[np.memmap, np.ndarray]:
    """Return embedding features and per-window timestamps for ``audio_path``.

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
    model:
        Name of the embedding model to use (``"wavlm"`` or ``"clap"``).
    model_path:
        Optional Hugging Face identifier or local path for the model.
    """

    audio_path = Path(audio_path)
    if cache_dir is None:
        cache_dir = audio_path.parent
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sr, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind in "iu":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)

    win_samples = int(window_size * sr)
    step_samples = int(step_size * sr)
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("window_size and step_size must be positive")

    n_samples = len(data)
    if n_samples < win_samples:
        num_windows = 0
    else:
        num_windows = 1 + (n_samples - win_samples) // step_samples

    model_name, _ = _parse_model_spec(model)
    memmap_suffix = model_name or model
    memmap_path = cache_dir / f"{audio_path.stem}_{memmap_suffix}_ws{window_size}_ss{step_size}.dat"

    if memmap_path.exists():
        file_size = memmap_path.stat().st_size
        emb_dim = 0 if num_windows == 0 else int(file_size // (4 * num_windows))
        features = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(num_windows, emb_dim))
    else:
        embed = _select_embedding_fn(model, model_path, sr)
        if num_windows == 0:
            features = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(0, 0))
        else:
            first = embed(data[:win_samples])
            emb_dim = int(first.shape[0])
            features = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(num_windows, emb_dim))
            features[0] = first
            for i in range(1, num_windows):
                start = i * step_samples
                end = start + win_samples
                window = data[start:end]
                features[i] = embed(window)
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
