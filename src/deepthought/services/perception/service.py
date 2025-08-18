"""Minimal perception service leveraging a NATS publisher stub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Sequence

import numpy as np
import torch

from ...config import get_settings
from ...modules import ModalityFuser
from .publisher import PerceptionPublisher
from .worker_audio import AudioPerceptionWorker
from .worker_text import TextPerceptionWorker, Token


@dataclass
class PerceptionService:
    """Orchestrate worker outputs and publish fused embeddings."""

    publisher: PerceptionPublisher
    text_worker: TextPerceptionWorker | None = None
    audio_worker: AudioPerceptionWorker | None = None
    fuser: ModalityFuser | None = None

    async def run(
        self,
        message_id: str,
        user_id: str,
        *,
        spans: Sequence[Sequence[int]] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        encoders: Sequence[Dict[str, Any]] | None = None,
        provenance: Dict[str, Any] | None = None,
        text_tokens: Sequence[Token] | None = None,
        audio_path: str | Path | None = None,
    ) -> None:
        """Fuse worker outputs and publish via the ``publisher``."""
        settings = get_settings()
        wandb_run = None
        if settings.wandb_enabled:
            try:  # pragma: no cover - optional dependency
                import wandb

                wandb_run = wandb.init(
                    project=settings.wandb_project,
                    id=settings.wandb_sweep_id,
                )
            except Exception:  # pragma: no cover - wandb may be missing
                wandb_run = None

        if embeddings is None:
            embeddings = []
            spans = list(spans or [])
            encoders = list(encoders or [])
            provenance = dict(provenance or {})
            modalities: Dict[str, torch.Tensor] = {}
            idx = 0

            if self.text_worker is not None and text_tokens:
                with NamedTemporaryFile(suffix=".mm", delete=False) as tmp:
                    mem = self.text_worker(text_tokens, tmp.name)
                arr = np.asarray(mem)
                try:
                    embeddings.extend(arr.tolist())
                    spans.extend([[i + idx, i + idx + 1] for i in range(arr.shape[0])])
                    encoders.extend(
                        [{"name": self.text_worker.__class__.__name__}] * arr.shape[0]
                    )
                    modalities["text"] = torch.from_numpy(arr)
                    idx += arr.shape[0]
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

            if self.audio_worker is not None and audio_path is not None:
                feats, _ = self.audio_worker(audio_path)
                arr = np.asarray(feats)
                embeddings.extend(arr.tolist())
                spans.extend([[i + idx, i + idx + 1] for i in range(arr.shape[0])])
                encoders.extend(
                    [{"name": self.audio_worker.__class__.__name__}] * arr.shape[0]
                )
                modalities["audio"] = torch.from_numpy(arr)
                idx += arr.shape[0]

            if not embeddings:
                raise ValueError("No modalities available for publication")

            provenance.setdefault("modalities", list(modalities.keys()))

            if self.fuser is not None:
                fused = self.fuser({k: v.mean(0, keepdim=True) for k, v in modalities.items()})
                _ = fused  # pragma: no cover - fused embedding currently unused

        modality_payload = {}
        if embeddings is not None:
            modality_payload["generic"] = {
                "spans": list(spans or []),
                "embeddings": [list(map(float, e)) for e in embeddings],
                "encoders": [dict(meta) for meta in (encoders or [])],
            }

        await self.publisher.publish(
            message_id=message_id,
            user_id=user_id,
            fused=embeddings[0] if embeddings else None,
            by_modality=modality_payload,
            provenance=provenance,
        )

        if wandb_run is not None:
            try:  # pragma: no cover - optional dependency
                import wandb

                wandb.log({"embeddings_published": len(embeddings or [])})
                if settings.wandb_upload_artifacts and embeddings is not None:
                    with NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                        np.save(tmp.name, np.asarray(embeddings))
                    art = wandb.Artifact(
                        name=f"embeddings_{message_id}",
                        type="checkpoint",
                    )
                    art.add_file(tmp.name)
                    wandb.log_artifact(art)
                    Path(tmp.name).unlink(missing_ok=True)
            finally:  # pragma: no cover - wandb may be missing
                wandb_run.finish()


async def run(*args: Any, **kwargs: Any) -> None:
    """Entry point for ``dtrt perception run``."""

    service: PerceptionService | None = kwargs.pop("service", None)
    if service is not None:
        await service.run(*args, **kwargs)
