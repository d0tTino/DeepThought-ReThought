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
from .user_embeddings import UserEmbeddings
from .worker_audio import AudioPerceptionWorker
from .worker_text import TextPerceptionWorker, Token
from .worker_video import VideoPerceptionWorker


@dataclass
class PerceptionService:
    """Orchestrate worker outputs and publish fused embeddings."""

    publisher: PerceptionPublisher
    text_worker: TextPerceptionWorker | None = None
    audio_worker: AudioPerceptionWorker | None = None
    video_worker: VideoPerceptionWorker | None = None
    fuser: ModalityFuser | None = None
    user_embeddings: UserEmbeddings | None = None

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
        video_path: str | Path | None = None,
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
            provenance = dict(provenance or {})
            modality_arrays: Dict[str, np.ndarray] = {}
            modality_times: Dict[str, np.ndarray] = {}
            encoder_meta: Dict[str, Dict[str, Any]] = {}

            if self.text_worker is not None and text_tokens:
                with NamedTemporaryFile(suffix=".mm", delete=False) as tmp:
                    feats, times = self.text_worker(text_tokens, tmp.name)
                modality_arrays["text"] = np.asarray(feats)
                modality_times["text"] = np.asarray(times)
                encoder_meta["text"] = {"name": self.text_worker.__class__.__name__}
                Path(tmp.name).unlink(missing_ok=True)

            if self.audio_worker is not None and audio_path is not None:
                feats, times = self.audio_worker(audio_path)
                modality_arrays["audio"] = np.asarray(feats)
                modality_times["audio"] = np.asarray(times)
                encoder_meta["audio"] = {"name": self.audio_worker.__class__.__name__}

            if self.video_worker is not None and video_path is not None:
                feats, times = self.video_worker(video_path)
                times_arr = np.asarray(times)
                if times_arr.ndim == 1:
                    if len(times_arr) > 1:
                        step = float(np.min(np.diff(times_arr)))
                    else:
                        fps = self.video_worker.grid_fps or self.video_worker.decode_fps
                        step = 1.0 / float(fps)
                    times_arr = np.column_stack((times_arr, times_arr + step))
                modality_arrays["video"] = np.asarray(feats)
                modality_times["video"] = times_arr

                encoder_meta["video"] = {"name": self.video_worker.__class__.__name__}

            if not modality_arrays:
                raise ValueError("No modalities available for publication")

            provenance.setdefault("modalities", list(modality_arrays.keys()))

            hop = min(np.min(t[:, 1] - t[:, 0]) for t in modality_times.values())
            start = min(t[0, 0] for t in modality_times.values())
            end = max(t[-1, 1] for t in modality_times.values())
            num_spans = int(np.ceil((end - start) / hop))
            grid_starts = start + np.arange(num_spans) * hop
            grid_ends = grid_starts + hop
            spans = [[i, i + 1] for i in range(num_spans)]

            aligned_modalities: Dict[str, torch.Tensor] = {}
            modality_payload: Dict[str, Dict[str, Any]] = {}
            for name, arr in modality_arrays.items():
                times = modality_times[name]
                dim = arr.shape[1]
                aligned = np.zeros((num_spans, dim), dtype=np.float32)
                for i, (gs, ge) in enumerate(zip(grid_starts, grid_ends)):
                    mask = (gs < times[:, 1]) & (ge > times[:, 0])
                    if mask.any():
                        aligned[i] = arr[mask].mean(axis=0)
                aligned_modalities[name] = torch.from_numpy(aligned)
                modality_payload[name] = {
                    "spans": spans,
                    "embeddings": aligned.tolist(),
                    "encoders": [encoder_meta[name]] * num_spans,
                }

            fused_list: Sequence[Sequence[float]] | None = None
            if self.fuser is not None:
                existing_user_embedding = (
                    self.user_embeddings.get(user_id) if self.user_embeddings is not None else None
                )
                fused_tensor = self.fuser(
                    aligned_modalities,
                    user_embedding=existing_user_embedding,
                    user_id=user_id,
                    embedding_store=self.user_embeddings,
                )
                fused_list = fused_tensor.tolist()
            else:
                first = next(iter(aligned_modalities.values()))
                fused_list = first.tolist()

        else:
            modality_payload = {}
            fused_list = embeddings  # type: ignore[assignment]

        await self.publisher.publish(
            message_id=message_id,
            user_id=user_id,
            fused=fused_list,
            by_modality=modality_payload,
            provenance=provenance,
        )

        if wandb_run is not None:
            try:  # pragma: no cover - optional dependency
                import wandb

                wandb.log({"embeddings_published": len(fused_list or [])})
                if settings.wandb_upload_artifacts and fused_list is not None:
                    with NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                        np.save(tmp.name, np.asarray(fused_list))
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
