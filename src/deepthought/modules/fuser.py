from __future__ import annotations

"""Utilities for fusing multi-modal embeddings."""

from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING

import os
import torch
from torch import nn

if not hasattr(torch, "SymBool"):  # pragma: no cover - torch<2.0 compatibility
    torch.SymBool = bool  # type: ignore[attr-defined]

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
            if user_embedding is not None:
                embedding_store.set(user_id, user_embedding.mean(dim=0))
            elif self.user_dim > 0:
                stored = embedding_store.get(user_id)
                if stored is not None:
                    base = next(iter(modalities.values()))
                    stored = stored.to(base.device)
                    if stored.ndim == 1:
                        stored = stored.unsqueeze(0)
                    user_embedding = stored.expand(base.size(0), -1)

        if user_embedding is not None:
            pieces.append(user_embedding)

        combined = torch.cat(pieces, dim=-1)
        return self.project(combined)

    def fit(
        self,
        dataloader: Iterable[Dict[str, Any]] | Iterable[Any],
        *,
        epochs: int = 1,
        lr: float = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        user_conditioner: Optional[
            Callable[[Dict[str, Any], Optional[torch.Tensor]], Optional[torch.Tensor]]
        ] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Train the projection layer using provided training data.

        Parameters
        ----------
        dataloader:
            Iterable yielding batches. Each batch can either be a dictionary
            with the keys ``modalities`` and ``target`` (tensors), plus optional
            ``user_embedding``, ``user_id`` and ``embedding_store`` entries, or a
            tuple ``(modalities, target)``.
        epochs:
            Number of training epochs.
        lr:
            Learning rate used when ``optimizer`` is not provided.
        optimizer:
            Optional optimizer. If ``None`` an :class:`Adam` optimizer is
            created for the projection layer.
        criterion:
            Loss function applied to the fused embedding and ``target``.
            Defaults to mean squared error.
        user_conditioner:
            Optional callback invoked for every batch. It receives the batch and
            an optional existing ``user_embedding`` and should return the
            ``user_embedding`` to use for that batch.
        checkpoint_dir:
            When provided, model checkpoints are stored under this directory at
            the end of each epoch using ``fuser_epoch{n}.pt`` naming.
        device:
            Device to which tensors are moved during training.
        """

        self.train()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if criterion is None:
            criterion = nn.MSELoss()

        for epoch in range(epochs):
            for batch in dataloader:
                if isinstance(batch, dict):
                    modalities = batch["modalities"]
                    target = batch["target"]
                    user_embedding = batch.get("user_embedding")
                    user_id = batch.get("user_id")
                    embedding_store = batch.get("embedding_store")
                else:
                    modalities, target = batch  # type: ignore[assignment]
                    user_embedding = None
                    user_id = None
                    embedding_store = None

                if user_conditioner is not None:
                    user_embedding = user_conditioner(batch, user_embedding)

                if device is not None:
                    modalities = {k: v.to(device) for k, v in modalities.items()}
                    target = target.to(device)
                    if user_embedding is not None:
                        user_embedding = user_embedding.to(device)

                output = self.forward(
                    modalities,
                    user_embedding=user_embedding,
                    user_id=user_id,
                    embedding_store=embedding_store,
                )

                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"fuser_epoch{epoch + 1}.pt")
                torch.save(self.state_dict(), path)
