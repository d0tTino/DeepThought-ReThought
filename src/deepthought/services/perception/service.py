"""Minimal perception service leveraging a NATS publisher stub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Sequence

import numpy as np
import torch

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

        if embeddings is None and self.fuser is not None:
            modalities: Dict[str, torch.Tensor] = {}
            if self.text_worker is not None and text_tokens:
                with NamedTemporaryFile(suffix=".mm", delete=False) as tmp:
                    mem = self.text_worker(text_tokens, tmp.name)
                try:
                    modalities["text"] = torch.from_numpy(np.asarray(mem)).mean(0, keepdim=True)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            if self.audio_worker is not None and audio_path is not None:
                feats, _ = self.audio_worker(audio_path)
                modalities["audio"] = torch.from_numpy(np.asarray(feats)).mean(0, keepdim=True)
            if not modalities:
                raise ValueError("No modalities available for fusion")
            fused = self.fuser(modalities)
            embeddings = [fused.squeeze(0).tolist()]
            encoders = encoders or [{"name": self.fuser.__class__.__name__}]

        await self.publisher.publish(
            message_id=message_id,
            user_id=user_id,
            spans=spans,
            embeddings=embeddings,
            encoders=encoders,
            provenance=provenance,
        )


async def run(*args: Any, **kwargs: Any) -> None:
    """Entry point for ``dtrt perception run``."""

    service: PerceptionService | None = kwargs.pop("service", None)
    if service is not None:
        await service.run(*args, **kwargs)
