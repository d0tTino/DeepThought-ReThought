from __future__ import annotations

"""Utilities for fusing multi-modal embeddings."""

from typing import Dict, Optional, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from deepthought.services.perception.user_embeddings import UserEmbeddings


class ModalityFuser(nn.Module):
    """Fuse modality-specific embeddings into a single representation.

    Parameters
    ----------
    modality_dims:
        Mapping of modality name to the size of its embedding vector.
    fused_dim:
        Output dimension of the fused embedding.
    dropout_prob:
        Probability of dropping an entire modality during training.
    user_dim:
        Size of an optional user embedding appended during fusion.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        fused_dim: int,
        *,
        dropout_prob: float = 0.0,
        user_dim: int = 0,
    ) -> None:
        super().__init__()
        self.modality_dims = modality_dims
        self.dropout_prob = dropout_prob
        self.user_dim = user_dim

        input_dim = sum(modality_dims.values()) + user_dim
        self.project = nn.Linear(input_dim, fused_dim)

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        user_embedding: Optional[torch.Tensor] = None,
        *,
        user_id: str | None = None,
        embedding_store: "UserEmbeddings" | None = None,
    ) -> torch.Tensor:
        """Return a fused embedding from provided modality tensors.

        Each modality tensor should have shape ``(batch, dim)``. If
        ``user_embedding`` is provided it must have shape ``(batch, user_dim)``.
        When ``embedding_store`` and ``user_id`` are given, the store is queried
        for a matching embedding which is appended when present. If a
        ``user_embedding`` is provided it is persisted back to ``embedding_store``
        for future calls. Modality dropout randomly zeroes whole modality
        vectors during training with probability ``dropout_prob``.
        """

        if not modalities:
            raise ValueError("No modalities provided for fusion")

        pieces = []
        for name, tensor in modalities.items():
            if self.training and self.dropout_prob > 0.0:
                mask = (torch.rand(tensor.size(0), 1, device=tensor.device) > self.dropout_prob).float()
                tensor = tensor * mask
            pieces.append(tensor)

        if embedding_store is not None and user_id is not None:
            if self.user_dim <= 0:
                raise ValueError("user_dim must be > 0 when using user embeddings")

            if user_embedding is None:
                stored = embedding_store.get(user_id)
                if stored is not None:
                    if stored.shape[-1] != self.user_dim:
                        raise ValueError(
                            f"Stored embedding for user '{user_id}' has dimension {stored.shape[-1]} "
                            f"but expected {self.user_dim}"
                        )
                    base = next(iter(modalities.values()))
                    stored = stored.to(base.device)
                    if stored.ndim == 1:
                        stored = stored.unsqueeze(0)
                    user_embedding = stored.expand(base.size(0), -1)
            else:
                if user_embedding.shape[-1] != self.user_dim:
                    raise ValueError(
                        f"Provided user_embedding has dimension {user_embedding.shape[-1]} "
                        f"but expected {self.user_dim}"
                    )
                embedding_store.set(user_id, user_embedding.mean(dim=0))

        if user_embedding is not None:
            pieces.append(user_embedding)

        combined = torch.cat(pieces, dim=-1)
        return self.project(combined)

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - stubbed method
        """Placeholder for training logic."""
        raise NotImplementedError("Training is not implemented for ModalityFuser")
