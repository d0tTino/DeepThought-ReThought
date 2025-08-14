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

    def __call__(self, tokens: Sequence[Token], memmap_path: str) -> np.memmap:
        """Process ``tokens`` and write embeddings to ``memmap_path``.

        Each hop is filled with the embedding of the token active during that
        time frame. Gaps remain zero-filled.
        """

        if not tokens:
            raise ValueError("tokens must not be empty")

        # Determine embedding dimension using the first token
        first_emb = np.asarray(self._model.encode(tokens[0][0]))
        duration = max(end for _, _, end in tokens)
        num_steps = int(np.ceil(duration / self.hop_seconds))
        features = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(num_steps, len(first_emb)))
        features[:] = 0.0

        # Fill with first token embedding
        start_idx = int(tokens[0][1] / self.hop_seconds)
        end_idx = int(np.ceil(tokens[0][2] / self.hop_seconds))
        features[start_idx:end_idx] = first_emb

        for text, start, end in tokens[1:]:
            embedding = np.asarray(self._model.encode(text))
            start_idx = int(start / self.hop_seconds)
            end_idx = int(np.ceil(end / self.hop_seconds))
            features[start_idx:end_idx] = embedding

        features.flush()

        settings = get_settings()
        if settings.wandb_enabled:
            try:  # pragma: no cover - optional dependency
                import wandb

                wandb.log({"text_tokens": len(tokens)})
                if settings.wandb_upload_artifacts:
                    art = wandb.Artifact(
                        name=f"text_features_{Path(memmap_path).stem}",
                        type="features",
                    )
                    art.add_file(memmap_path)
                    wandb.log_artifact(art)
            except Exception:  # pragma: no cover - wandb may be missing
                pass

        return features
