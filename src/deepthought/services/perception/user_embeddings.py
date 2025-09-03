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

    def __contains__(self, user_id: str) -> bool:
        """Return ``True`` if an embedding for ``user_id`` exists."""

        return user_id in self._store

    def __len__(self) -> int:
        """Return the number of stored user embeddings."""

        return len(self._store)

    def delete(self, user_id: str) -> None:
        """Remove ``user_id`` from the store and persist changes."""

        if user_id in self._store:
            del self._store[user_id]
            self.save()

    def set(self, user_id: str, embedding: Sequence[float] | "torch.Tensor") -> None:
        """Store ``embedding`` for ``user_id`` and persist to disk."""

        import torch  # Lazy import to avoid eager torch initialization

        if isinstance(embedding, torch.Tensor):
            tensor = embedding.detach().cpu().float()
        else:
            tensor = torch.tensor(list(embedding), dtype=torch.float32)
        self._store[user_id] = tensor
        self.save()

    def update_from_gradient(
        self,
        user_id: str,
        gradient: Sequence[float] | "torch.Tensor",
        *,
        lr: float = 1e-3,
    ) -> "torch.Tensor":
        """Apply gradient descent update to ``user_id``'s embedding.

        A new zero-initialized embedding is created if ``user_id`` has not been
        seen before. The updated embedding is saved to disk and returned.
        """

        import torch  # Lazy import to avoid eager torch initialization

        if isinstance(gradient, torch.Tensor):
            grad = gradient.detach().cpu().float()
        else:
            grad = torch.tensor(list(gradient), dtype=torch.float32)

        if user_id not in self._store:
            self._store[user_id] = torch.zeros_like(grad)

        self._store[user_id] = self._store[user_id] - lr * grad
        self.save()
        return self._store[user_id]

    def update_from_bandit(
        self,
        user_id: str,
        reward: float,
        context: Sequence[float] | "torch.Tensor",
        *,
        lr: float = 1e-2,
    ) -> "torch.Tensor":
        """Update ``user_id``'s embedding using bandit feedback.

        ``context`` defines the direction of the update which is scaled by the
        observed ``reward``. The updated embedding is persisted and returned.
        """

        import torch  # Lazy import to avoid eager torch initialization

        if isinstance(context, torch.Tensor):
            ctx = context.detach().cpu().float()
        else:
            ctx = torch.tensor(list(context), dtype=torch.float32)

        if user_id not in self._store:
            self._store[user_id] = torch.zeros_like(ctx)

        self._store[user_id] = self._store[user_id] + lr * reward * ctx
        self.save()
        return self._store[user_id]

    def save(self) -> None:
        """Persist the current embeddings to the configured path."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.tolist() for k, v in self._store.items()}
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
