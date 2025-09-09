from __future__ import annotations

"""Utilities for fusing multi-modal embeddings."""

from typing import Dict, Optional, Sequence, TYPE_CHECKING

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

    def fit(
        self,
        batches: "Sequence[tuple[Dict[str, torch.Tensor], torch.Tensor, str | None]]",
        *,
        embedding_store: "UserEmbeddings" | None = None,
        epochs: int = 1,
        lr: float = 1e-3,
        criterion: nn.Module | None = None,
    ) -> None:
        """Train the projection layer using provided data.

        Parameters
        ----------
        batches:
            Iterable yielding ``(modalities, target, user_id)`` tuples. Each
            ``modalities`` dict maps modality name to a tensor of shape
            ``(batch, dim)``. ``target`` is the supervision signal matching the
            output of the fuser. ``user_id`` identifies the user for the entire
            batch and is optional.
        embedding_store:
            Optional :class:`~deepthought.services.perception.user_embeddings.UserEmbeddings`
            instance used to retrieve and update per-user embeddings.
        epochs:
            Number of epochs to iterate over ``batches``.
        lr:
            Learning rate for both fusion weights and user embedding updates.
        criterion:
            Loss function used for training. Defaults to ``nn.MSELoss``.

        Notes
        -----
        Modality dropout is handled in :meth:`forward`. When ``embedding_store``
        is provided user embeddings are updated from the gradient of the loss
        and persisted via ``update_from_gradient``.
        """

        if criterion is None:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        device = next(self.parameters()).device

        for _ in range(epochs):
            for modalities, target, user_id in batches:
                modalities = {k: v.to(device) for k, v in modalities.items()}
                target = target.to(device)

                user_embedding = None
                if (
                    embedding_store is not None
                    and user_id is not None
                    and self.user_dim > 0
                ):
                    emb = embedding_store.get(user_id)
                    if emb is None:
                        emb = torch.zeros(self.user_dim, device=device)
                        embedding_store.set(user_id, emb)
                    if emb.ndim == 1:
                        emb = emb.unsqueeze(0)
                    base = next(iter(modalities.values()))
                    emb = emb.to(device).expand(base.size(0), -1).clone().detach()
                    emb.requires_grad_(True)
                    user_embedding = emb

                optimizer.zero_grad()
                output = self.forward(modalities, user_embedding=user_embedding)
                loss = criterion(output, target)
                loss.backward()

                if (
                    user_embedding is not None
                    and embedding_store is not None
                    and user_id is not None
                    and user_embedding.grad is not None
                ):
                    grad = user_embedding.grad.mean(dim=0)
                    embedding_store.update_from_gradient(user_id, grad, lr=lr)

                optimizer.step()

    def bandit_step(
        self,
        modalities: Dict[str, torch.Tensor],
        reward: float,
        context: Sequence[float] | torch.Tensor,
        user_id: str,
        embedding_store: "UserEmbeddings",
    ) -> torch.Tensor:
        """Fuse ``modalities`` and update ``user_id`` with bandit feedback.

        Parameters
        ----------
        modalities:
            Mapping from modality name to tensor inputs.
        reward:
            Scalar reward used to scale the update.
        context:
            Feature vector indicating the direction of the update.
        user_id:
            Identifier whose embedding should be updated.
        embedding_store:
            Store managing persistent user embeddings.

        Returns
        -------
        torch.Tensor
            The fused embedding produced from ``modalities``.
        """

        embedding_store.update_from_bandit(user_id, reward, context)
        return self.forward(modalities, user_id=user_id, embedding_store=embedding_store)
