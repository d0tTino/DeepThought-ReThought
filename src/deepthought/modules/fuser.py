from __future__ import annotations

"""Utilities for fusing multi-modal embeddings."""

from typing import Dict, Optional

import torch
from torch import nn


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
        self, modalities: Dict[str, torch.Tensor], user_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return a fused embedding from provided modality tensors.

        Each modality tensor should have shape ``(batch, dim)``. If
        ``user_embedding`` is provided it must have shape ``(batch, user_dim)``.
        Modality dropout randomly zeroes whole modality vectors during
        training with probability ``dropout_prob``.
        """

        if not modalities:
            raise ValueError("No modalities provided for fusion")

        pieces = []
        for name, tensor in modalities.items():
            if self.training and self.dropout_prob > 0.0:
                mask = (torch.rand(tensor.size(0), 1, device=tensor.device) > self.dropout_prob).float()
                tensor = tensor * mask
            pieces.append(tensor)

        if user_embedding is not None:
            pieces.append(user_embedding)

        combined = torch.cat(pieces, dim=-1)
        return self.project(combined)

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - stubbed method
        """Placeholder for training logic."""
        raise NotImplementedError("Training is not implemented for ModalityFuser")
