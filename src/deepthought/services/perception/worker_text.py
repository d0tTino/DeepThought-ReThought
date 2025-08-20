"""Text perception worker producing time-aligned embeddings.

This worker uses a BGE/E5 encoder from :mod:`sentence_transformers` to
transform text tokens into vector representations. Tokens are aligned to a
fixed temporal hop (between 25 and 50 milliseconds) and the resulting
features are stored in a NumPy ``memmap`` for efficient downstream
processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from deepthought.config import get_settings

# A token is represented as ``(text, start_time, end_time)`` where times are in seconds.
Token = Tuple[str, float, float]


@dataclass
class TextPerceptionWorker:
    """Convert tokens into hop-aligned embeddings and store in a memmap.

    Parameters
    ----------
    model_name:
        Name of the BGE or E5 model to load via :class:`SentenceTransformer`.
    hop_seconds:
        Temporal hop in seconds. Must be between 25 and 50 milliseconds.
    """

    model_name: str = "intfloat/e5-small-v2"
    hop_seconds: float = 0.03

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if not 0.025 <= self.hop_seconds <= 0.05:
            raise ValueError("hop_seconds must be between 0.025 and 0.05 seconds")
        self._model = SentenceTransformer(self.model_name)

    def __call__(self, tokens: Sequence[Token], memmap_path: str | Path) -> Tuple[np.memmap, np.ndarray]:
        """Process ``tokens`` and write embeddings and timestamps to ``memmap_path``.

        Each hop is filled with the embedding of the token active during that
        time frame. Gaps remain zero-filled. The returned timestamps array has
        shape ``(num_steps, 2)`` with ``[start, end]`` times in seconds.
        """

        if not tokens:
            raise ValueError("tokens must not be empty")

        memmap_path = Path(memmap_path)

        # Pre-compute embeddings to determine dimensionality
        embeddings = [np.asarray(self._model.encode(text)) for text, _, _ in tokens]
        emb_dim = embeddings[0].shape[0]

        duration = max(end for _, _, end in tokens)
        num_steps = int(np.ceil(duration / self.hop_seconds))
        features = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(num_steps, emb_dim))
        features[:] = 0.0

        for emb, (_, start, end) in zip(embeddings, tokens):
            start_idx = int(start / self.hop_seconds)
            end_idx = int(np.ceil(end / self.hop_seconds))
            features[start_idx:end_idx] = emb

        features.flush()

        starts = np.arange(num_steps, dtype=np.float32) * self.hop_seconds
        ends = starts + self.hop_seconds
        timestamps = np.column_stack((starts, ends))

        settings = get_settings()
        if settings.wandb_enabled:
            try:  # pragma: no cover - optional dependency
                import wandb

                wandb.log({"text_tokens": len(tokens)})
                if settings.wandb_upload_artifacts:
                    art = wandb.Artifact(
                        name=f"text_features_{memmap_path.stem}",
                        type="features",
                    )
                    art.add_file(memmap_path)
                    wandb.log_artifact(art)
            except Exception:  # pragma: no cover - wandb may be missing
                pass

        return features, timestamps
