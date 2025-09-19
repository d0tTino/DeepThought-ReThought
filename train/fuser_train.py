"""Train a :class:`~deepthought.modules.fuser.ModalityFuser` from cached features."""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from deepthought.modules.fuser import ModalityFuser


logger = logging.getLogger(__name__)


class CachedFeatureDataset(
    Dataset[
        tuple[
            dict[str, torch.Tensor],
            torch.Tensor,
            str | None,
            torch.Tensor | None,
        ]
    ]
):
    """Dataset backed by cached modality features stored in an ``.npz`` file."""

    def __init__(
        self,
        modality_arrays: dict[str, np.ndarray],
        target: np.ndarray,
        *,
        user_ids: Sequence[str | None] | None = None,
        user_embeddings: np.ndarray | None = None,
    ) -> None:
        if not modality_arrays:
            raise ValueError("Feature file does not contain any modality arrays")

        target = np.asarray(target)
        if target.ndim == 1:
            target = target[:, np.newaxis]
        if target.ndim != 2:
            raise ValueError("Target array must be 1D or 2D")
        if target.shape[0] == 0:
            raise ValueError("Feature file is empty")

        self._target = torch.as_tensor(target, dtype=torch.float32)

        self._modalities: dict[str, torch.Tensor] = {}
        for name, array in modality_arrays.items():
            modality = np.asarray(array)
            if modality.ndim != 2:
                raise ValueError(f"Modality '{name}' must be 2-dimensional")
            if modality.shape[0] != self._target.shape[0]:
                raise ValueError(
                    f"Modality '{name}' has {modality.shape[0]} samples "
                    f"but target has {self._target.shape[0]}"
                )
            self._modalities[name] = torch.as_tensor(modality, dtype=torch.float32)

        if user_embeddings is not None:
            embeddings = np.asarray(user_embeddings)
            if embeddings.ndim != 2:
                raise ValueError("user_embeddings must be a 2D array")
            if embeddings.shape[0] != len(self):
                raise ValueError(
                    "Number of user embeddings does not match number of samples"
                )
            self._user_embeddings: torch.Tensor | None = torch.as_tensor(
                embeddings, dtype=torch.float32
            )
        else:
            self._user_embeddings = None

        if user_ids is not None:
            if len(user_ids) != len(self):
                raise ValueError(
                    "Number of user_ids does not match number of samples"
                )
            self._user_ids: Sequence[str | None] | None = [
                None if uid is None else str(uid) for uid in user_ids
            ]
        else:
            self._user_ids = None

    def __len__(self) -> int:  # pragma: no cover - simple container
        return self._target.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        str | None,
        torch.Tensor | None,
    ]:
        modalities = {name: tensor[index] for name, tensor in self._modalities.items()}
        target = self._target[index]
        user_id = None if self._user_ids is None else self._user_ids[index]
        if self._user_embeddings is not None:
            user_embedding: torch.Tensor | None = self._user_embeddings[index]
        else:
            user_embedding = None
        return modalities, target, user_id, user_embedding

    @property
    def modality_dims(self) -> dict[str, int]:  # pragma: no cover - trivial accessors
        return {name: tensor.shape[1] for name, tensor in self._modalities.items()}

    @property
    def target_dim(self) -> int:  # pragma: no cover - trivial accessors
        return self._target.shape[1]

    @property
    def user_embedding_dim(self) -> int:  # pragma: no cover - trivial accessors
        if self._user_embeddings is None:
            return 0
        return self._user_embeddings.shape[1]


BatchItem = tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    str | None,
    torch.Tensor | None,
]


def _collate_batch(
    batch: Sequence[BatchItem],
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    Sequence[str | None] | None,
    torch.Tensor | None,
]:
    modalities, targets, user_ids, user_embeddings = zip(*batch)

    stacked_modalities = {
        name: torch.stack([sample[name] for sample in modalities], dim=0)
        for name in modalities[0]
    }
    stacked_targets = torch.stack(list(targets), dim=0)

    if all(uid is None for uid in user_ids):
        batch_user_ids: Sequence[str | None] | None = None
    else:
        batch_user_ids = list(user_ids)

    if all(embedding is None for embedding in user_embeddings):
        batch_user_embeddings: torch.Tensor | None = None
    else:
        if any(embedding is None for embedding in user_embeddings):
            raise ValueError("Mixed presence of user embeddings within a batch")
        stacked_embeddings = torch.stack(
            [cast(torch.Tensor, embedding) for embedding in user_embeddings], dim=0
        )
        batch_user_embeddings = stacked_embeddings

    return stacked_modalities, stacked_targets, batch_user_ids, batch_user_embeddings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the training script."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a modality fuser using cached features saved in an .npz file. "
            "The file must contain a 'target' array and one array per modality."
        )
    )
    parser.add_argument("--features", type=Path, required=True, help="Path to .npz file")
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination path for the trained model"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Mini-batch size (default: 32)"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam optimizer (default: 0.0)",
    )
    parser.add_argument(
        "--dropout-prob",
        type=float,
        default=0.0,
        help="Probability of dropping an entire modality during training",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to train on (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for python, NumPy, and torch RNGs",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable shuffling before each epoch",
    )
    parser.set_defaults(shuffle=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for the DataLoader (default: 0)",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin DataLoader memory. Useful when training on GPU",
    )
    parser.add_argument(
        "--no-user-embeddings",
        action="store_true",
        help="Ignore per-sample user embeddings even if present in the feature file",
    )
    parser.add_argument(
        "--user-embedding-key",
        default="user_embeddings",
        help=(
            "Name of the array within the feature file containing per-sample "
            "user embeddings. Set to 'none' to disable automatic loading."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        help="Weights & Biases project name for logging (disabled by default)",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Optional W&B entity/organization for the run",
    )
    parser.add_argument("--wandb-group", help="Optional W&B group name")
    parser.add_argument("--wandb-run-name", help="Optional W&B run name")
    parser.add_argument("--wandb-job-type", help="Optional W&B job type tag")
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA not available in CI
        torch.cuda.manual_seed_all(seed)


def _load_dataset(path: Path, *, user_embedding_key: str | None) -> CachedFeatureDataset:
    with np.load(path, allow_pickle=True) as data:
        if "target" not in data:
            raise ValueError("Feature file must contain a 'target' array")

        reserved = {"target", "user_ids"}
        if user_embedding_key:
            reserved.add(user_embedding_key)

        modalities = {key: data[key] for key in data.files if key not in reserved}

        user_ids: Sequence[str | None] | None = None
        if "user_ids" in data.files:
            raw_ids = np.asarray(data["user_ids"]).tolist()
            user_ids = [None if uid is None else str(uid) for uid in raw_ids]

        embeddings_array: np.ndarray | None = None
        if user_embedding_key and user_embedding_key in data.files:
            embeddings_array = np.asarray(data[user_embedding_key])

        dataset = CachedFeatureDataset(
            modalities,
            data["target"],
            user_ids=user_ids,
            user_embeddings=embeddings_array,
        )
    return dataset


def _train_one_epoch(
    model: ModalityFuser,
    loader: DataLoader[
        tuple[
            dict[str, torch.Tensor],
            torch.Tensor,
            Sequence[str | None] | None,
            torch.Tensor | None,
        ]
    ],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    use_user_embeddings: bool,
) -> float:
    total_loss = 0.0
    total_examples = 0

    for modalities, target, _user_ids, user_embeddings in loader:
        batch_size = target.size(0)
        if batch_size == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        fused = model(
            {k: v.to(device) for k, v in modalities.items()},
            user_embedding=None
            if not use_user_embeddings or user_embeddings is None
            else user_embeddings.to(device),
        )
        loss = criterion(fused, target.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("No samples available for training")

    return total_loss / total_examples


@contextmanager
def _wandb_run(args: argparse.Namespace, config: dict[str, Any]) -> Generator[Any, None, None]:
    if not args.wandb_project:
        yield None
        return

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "wandb is required when --wandb-project is provided"
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        name=args.wandb_run_name,
        config=config,
    )
    try:
        yield wandb
    finally:  # pragma: no branch - clean up logging context
        if run is not None and hasattr(run, "finish"):
            run.finish()
        else:
            wandb.finish()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not logging.getLogger().handlers:  # pragma: no cover - depends on caller
        logging.basicConfig(level=logging.INFO)

    if args.seed is not None:
        _set_seed(args.seed)

    user_embedding_key: str | None
    if args.no_user_embeddings or args.user_embedding_key.lower() == "none":
        user_embedding_key = None
    else:
        user_embedding_key = args.user_embedding_key

    dataset = _load_dataset(args.features, user_embedding_key=user_embedding_key)
    logger.info(
        "Loaded %s with %d samples across %s",
        args.features,
        len(dataset),
        ", ".join(sorted(dataset.modality_dims)),
    )

    use_user_embeddings = (
        not args.no_user_embeddings and dataset.user_embedding_dim > 0
    )
    if use_user_embeddings:
        logger.info("Using user embeddings of dimension %d", dataset.user_embedding_dim)
    elif dataset.user_embedding_dim > 0:
        logger.info("Ignoring %d-dimensional user embeddings", dataset.user_embedding_dim)

    generator = None
    if args.shuffle and args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=False,
        generator=generator,
        collate_fn=_collate_batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    device = torch.device(args.device)
    model = ModalityFuser(
        dataset.modality_dims,
        fused_dim=dataset.target_dim,
        dropout_prob=args.dropout_prob,
        user_dim=dataset.user_embedding_dim if use_user_embeddings else 0,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion: nn.Module = nn.MSELoss()

    wandb_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout_prob": args.dropout_prob,
        "device": str(device),
        "modalities": sorted(dataset.modality_dims.keys()),
        "target_dim": dataset.target_dim,
        "use_user_embeddings": use_user_embeddings,
        "user_embedding_dim": dataset.user_embedding_dim if use_user_embeddings else 0,
        "feature_file": str(args.features),
        "seed": args.seed,
    }

    best_loss: float | None = None
    final_loss: float | None = None

    with _wandb_run(args, wandb_config) as wandb_run:
        for epoch in range(1, args.epochs + 1):
            mean_loss = _train_one_epoch(
                model,
                loader,
                optimizer,
                criterion,
                device=device,
                use_user_embeddings=use_user_embeddings,
            )
            final_loss = mean_loss
            if best_loss is None or mean_loss < best_loss:
                best_loss = mean_loss
            logger.info("epoch %d | loss=%.6f", epoch, mean_loss)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": mean_loss,
                        "train/epoch": epoch,
                        "train/best_loss": best_loss,
                    },
                    step=epoch,
                )
        if wandb_run is not None and final_loss is not None:
            wandb_run.log({"train/final_loss": final_loss}, step=args.epochs)
            if hasattr(wandb_run, "summary") and isinstance(wandb_run.summary, dict):
                wandb_run.summary.setdefault("train/best_loss", best_loss)
                wandb_run.summary.setdefault("train/final_loss", final_loss)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info("Saved fuser weights to %s", args.output)

    if best_loss is not None and args.wandb_project:
        try:  # pragma: no cover - optional dependency
            import wandb

            artifact = wandb.Artifact(
                name=f"{args.output.stem or 'modality-fuser'}-weights",
                type="model",
                metadata={
                    "modalities": sorted(dataset.modality_dims.keys()),
                    "user_embedding_dim": dataset.user_embedding_dim
                    if use_user_embeddings
                    else 0,
                    "dropout_prob": args.dropout_prob,
                    "epochs": args.epochs,
                    "best_loss": best_loss,
                },
            )
            artifact.add_file(str(args.output))
            wandb.log_artifact(artifact)
        except ImportError:  # pragma: no cover - wandb optional
            logger.warning(
                "wandb was requested but not available when logging artifact", exc_info=True
            )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

