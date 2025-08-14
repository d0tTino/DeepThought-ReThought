from __future__ import annotations

"""Utilities for persisting per-user embedding vectors."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    import torch


class UserEmbeddings:
    """Persist and retrieve embedding vectors keyed by ``user_id``.

    Parameters
    ----------
    path:
        Location on disk where embeddings are stored as JSON. The file is
        created if it does not already exist.
    """

    def __init__(self, path: str | Path) -> None:
        import torch  # Lazy import to avoid eager torch initialization

        self.path = Path(path)
        self._store: Dict[str, "torch.Tensor"]
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._store = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
        else:
            self._store = {}

    def get(self, user_id: str) -> Optional["torch.Tensor"]:
        """Return the embedding vector for ``user_id`` if present."""

        return self._store.get(user_id)

    def set(self, user_id: str, embedding: Sequence[float] | "torch.Tensor") -> None:
        """Store ``embedding`` for ``user_id`` and persist to disk."""

        import torch  # Lazy import to avoid eager torch initialization

        if isinstance(embedding, torch.Tensor):
            tensor = embedding.detach().cpu().float()
        else:
            tensor = torch.tensor(list(embedding), dtype=torch.float32)
        self._store[user_id] = tensor
        self.save()

    def save(self) -> None:
        """Persist the current embeddings to the configured path."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.tolist() for k, v in self._store.items()}
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
