"""Train a :class:`~deepthought.modules.fuser.ModalityFuser` from cached features."""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import importlib.util
import sys

import numpy as np


def _load_real_torch() -> object:
    """Return a fully featured ``torch`` module, bypassing lightweight stubs."""

    sys.modules.pop("torch", None)
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.loader is None:  # pragma: no cover - torch missing
        raise ImportError("torch is required for training utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["torch"] = module
    return module


try:  # pragma: no branch - executed at import time
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch not importable
    torch = _load_real_torch()  # type: ignore[assignment]
else:
    if not hasattr(torch, "Tensor"):
        torch = _load_real_torch()  # type: ignore[assignment]

import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader, Dataset

from deepthought.modules.fuser import ModalityFuser


logger = logging.getLogger(__name__)


class CachedFeatureDataset(
    Dataset[tuple[dict[str, torch.Tensor], torch.Tensor, str | None]]
):
    """Dataset backed by cached modality features stored in an ``.npz`` file."""

    def __init__(
        self,
        modality_arrays: dict[str, np.ndarray],
        target: np.ndarray,
        *,
        user_ids: Sequence[str | None] | None = None,
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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str | None]:
        modalities = {name: tensor[index] for name, tensor in self._modalities.items()}
        target = self._target[index]
        user_id = None if self._user_ids is None else self._user_ids[index]
        return modalities, target, user_id

    @property
    def modality_dims(self) -> dict[str, int]:  # pragma: no cover - trivial accessors
        return {name: tensor.shape[1] for name, tensor in self._modalities.items()}

    @property
    def target_dim(self) -> int:  # pragma: no cover - trivial accessors
        return self._target.shape[1]


BatchItem = tuple[dict[str, torch.Tensor], torch.Tensor, str | None]


def _collate_batch(
    batch: Sequence[BatchItem],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, Sequence[str | None] | None]:
    modalities, targets, user_ids = zip(*batch)

    stacked_modalities = {
        name: torch.stack([sample[name] for sample in modalities], dim=0)
        for name in modalities[0]
    }
    stacked_targets = torch.stack(list(targets), dim=0)

    if all(uid is None for uid in user_ids):
        batch_user_ids: Sequence[str | None] | None = None
    else:
        batch_user_ids = list(user_ids)

    return stacked_modalities, stacked_targets, batch_user_ids


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
        "--wandb-project",
        help="Weights & Biases project name for logging (disabled by default)",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Optional W&B entity/organization for the run",
    )
    parser.add_argument("--wandb-group", help="Optional W&B group name")
    parser.add_argument("--wandb-run-name", help="Optional W&B run name")
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA not available in CI
        torch.cuda.manual_seed_all(seed)


def _load_dataset(path: Path) -> CachedFeatureDataset:
    with np.load(path, allow_pickle=True) as data:
        if "target" not in data:
            raise ValueError("Feature file must contain a 'target' array")

        modalities = {
            key: data[key]
            for key in data.files
            if key not in {"target", "user_ids"}
        }
        user_ids: Sequence[str | None] | None = None
        if "user_ids" in data.files:
            raw_ids = np.asarray(data["user_ids"]).tolist()
            user_ids = [None if uid is None else str(uid) for uid in raw_ids]

        dataset = CachedFeatureDataset(modalities, data["target"], user_ids=user_ids)
    return dataset


def _train_one_epoch(
    model: ModalityFuser,
    loader: DataLoader[tuple[dict[str, torch.Tensor], torch.Tensor, Sequence[str | None]]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_examples = 0

    for modalities, target, _ in loader:
        batch_size = target.size(0)
        if batch_size == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        fused = model({k: v.to(device) for k, v in modalities.items()})
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

    dataset = _load_dataset(args.features)
    logger.info(
        "Loaded %s with %d samples across %s",
        args.features,
        len(dataset),
        ", ".join(sorted(dataset.modality_dims)),
    )

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
    )

    device = torch.device(args.device)
    model = ModalityFuser(
        dataset.modality_dims,
        fused_dim=dataset.target_dim,
        dropout_prob=args.dropout_prob,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion: nn.Module = nn.MSELoss()

    wandb_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout_prob": args.dropout_prob,
        "device": str(device),
        "modalities": sorted(dataset.modality_dims.keys()),
        "target_dim": dataset.target_dim,
    }

    with _wandb_run(args, wandb_config) as wandb_run:
        for epoch in range(1, args.epochs + 1):
            mean_loss = _train_one_epoch(
                model, loader, optimizer, criterion, device=device
            )
            logger.info("epoch %d | loss=%.6f", epoch, mean_loss)
            if wandb_run is not None:
                wandb_run.log({"train/loss": mean_loss, "train/epoch": epoch}, step=epoch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info("Saved fuser weights to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
