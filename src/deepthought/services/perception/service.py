"""Minimal perception service leveraging a NATS publisher stub."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Sequence

import logging
import numpy as np
import torch

from ...config import get_settings
from ...metrics.prometheus import (
    INPUT_LATENCY_SECONDS,
    INPUTS_TOTAL,
    MODALITY_INFERENCE_LATENCY_SECONDS,
    MISSING_MODALITY_TOTAL,

)
from ...modules import ModalityFuser
from .config import PerceptionConfig
from .publisher import PerceptionPublisher
from .user_embeddings import UserEmbeddings
from .worker_audio import AudioPerceptionWorker
from .worker_text import TextPerceptionWorker, Token
from .worker_video import VideoPerceptionWorker


logger = logging.getLogger(__name__)


def _consent_granted(kind: str) -> bool:
    """Return ``True`` if consent for ``kind`` processing is granted."""

    required = os.getenv(f"DT_REQUIRE_{kind.upper()}_CONSENT", "false").lower()
    if required in {"1", "true", "yes"}:
        consent = os.getenv(f"DT_{kind.upper()}_CONSENT", "false").lower()
        return consent in {"1", "true", "yes"}
    return True


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
        retain_media: bool = False,
    ) -> None:
        """Fuse worker outputs and publish via the ``publisher``."""
        settings = get_settings()
        start_time = time.perf_counter()
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
            cfg = PerceptionConfig()

            if self.text_worker is not None and text_tokens:
                feats = times = None
                cache_dir = Path(cfg.text_cache_dir) if cfg.text_cache_dir else None
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    token_bytes = json.dumps(list(text_tokens), sort_keys=True).encode()
                    key = hashlib.sha1(token_bytes).hexdigest()
                    feats_file = cache_dir / f"{key}_feats.dat"
                    meta_file = cache_dir / f"{key}_meta.json"
                    if feats_file.exists() and meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        feats = np.memmap(feats_file, dtype="float32", mode="r", shape=tuple(meta["shape"]))
                        times = np.asarray(meta["timestamps"], dtype=np.float32)
                        encoder_meta["text"] = meta["encoder"]
                    else:
                        start = time.perf_counter()
                        feats, times = self.text_worker(text_tokens, str(feats_file))
                        MODALITY_INFERENCE_LATENCY_SECONDS.labels(
                            service="perception_service", modality="text"
                        ).observe(time.perf_counter() - start)
                        meta = {
                            "shape": list(feats.shape),
                            "timestamps": times.tolist(),
                            "encoder": {"name": self.text_worker.__class__.__name__},
                            "created": time.time(),
                        }
                        meta_file.write_text(json.dumps(meta))
                        encoder_meta["text"] = meta["encoder"]
                else:
                    with NamedTemporaryFile(suffix=".mm", delete=False) as tmp:
                        start = time.perf_counter()
                        feats, times = self.text_worker(text_tokens, tmp.name)
                        MODALITY_INFERENCE_LATENCY_SECONDS.labels(
                            service="perception_service", modality="text"
                        ).observe(time.perf_counter() - start)
                    encoder_meta["text"] = {"name": self.text_worker.__class__.__name__}
                    Path(tmp.name).unlink(missing_ok=True)
                modality_arrays["text"] = np.asarray(feats)
                modality_times["text"] = np.asarray(times)

            if self.audio_worker is not None and audio_path is not None:
                if not _consent_granted("audio"):
                    raise PermissionError("Audio consent not granted")
                audio_path = Path(audio_path)
                cache_dir = Path(self.audio_worker.cache_dir or cfg.audio_cache_dir or audio_path.parent)
                cache_dir.mkdir(parents=True, exist_ok=True)
                base = (
                    f"{audio_path.stem}_{self.audio_worker.model}_ws{self.audio_worker.window_size}"
                    f"_ss{self.audio_worker.step_size}"
                )
                feats_file = cache_dir / f"{base}.dat"
                meta_file = cache_dir / f"{base}_meta.json"
                if feats_file.exists() and meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    feats = np.memmap(feats_file, dtype="float32", mode="r", shape=tuple(meta["shape"]))
                    times = np.asarray(meta["timestamps"], dtype=np.float32)
                    encoder_meta["audio"] = meta["encoder"]
                else:
                    start = time.perf_counter()
                    feats, times = self.audio_worker(audio_path, cache_dir=cache_dir)
                    MODALITY_INFERENCE_LATENCY_SECONDS.labels(service="perception_service", modality="audio").observe(
                        time.perf_counter() - start
                    )
                    meta = {
                        "shape": list(feats.shape),
                        "timestamps": times.tolist(),
                        "encoder": {"name": self.audio_worker.__class__.__name__},
                        "created": time.time(),
                    }
                    meta_file.write_text(json.dumps(meta))
                    encoder_meta["audio"] = meta["encoder"]
                modality_arrays["audio"] = np.asarray(feats)
                modality_times["audio"] = np.asarray(times)
                if not retain_media:
                    audio_path.unlink(missing_ok=True)

            if self.video_worker is not None and video_path is not None:
                if not _consent_granted("video"):
                    raise PermissionError("Video consent not granted")
                video_path = Path(video_path)
                cache_dir = Path(
                    getattr(self.video_worker, "cache_dir", None) or cfg.video_cache_dir or video_path.parent
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                decode_fps = getattr(self.video_worker, "decode_fps", 1)
                model_type = getattr(self.video_worker, "model_type", "model")
                grid_fps = getattr(self.video_worker, "grid_fps", None)
                suffix = f"{decode_fps}_{model_type}_{grid_fps or decode_fps}"
                base = f"{video_path.stem}_{suffix}"
                feats_file = cache_dir / f"{base}_feats.npy"
                meta_file = cache_dir / f"{base}_meta.json"
                if feats_file.exists() and meta_file.exists():
                    feats = np.load(feats_file, mmap_mode="r")
                    meta = json.loads(meta_file.read_text())
                    times_arr = np.asarray(meta["timestamps"], dtype=np.float32)
                    encoder_meta["video"] = meta["encoder"]
                else:
                    start = time.perf_counter()
                    feats, times_arr = self.video_worker(video_path)
                    MODALITY_INFERENCE_LATENCY_SECONDS.labels(service="perception_service", modality="video").observe(
                        time.perf_counter() - start
                    )
                    if not feats_file.exists():
                        np.save(feats_file, np.asarray(feats))
                    meta = {
                        "shape": list(feats.shape),
                        "timestamps": times_arr.tolist(),
                        "encoder": {"name": self.video_worker.__class__.__name__},
                        "created": time.time(),
                    }
                    meta_file.write_text(json.dumps(meta))
                    encoder_meta["video"] = meta["encoder"]
                times = np.asarray(meta["timestamps"], dtype=np.float32)
                if times.ndim == 1:
                    if len(times) > 1:
                        step = float(np.min(np.diff(times)))
                    else:
                        fps = grid_fps or decode_fps
                        step = 1.0 / float(fps)
                    times = np.column_stack((times, times + step))
                modality_arrays["video"] = np.asarray(feats)
                modality_times["video"] = times
                if not retain_media:
                    video_path.unlink(missing_ok=True)

            expected_modalities = set()
            if self.fuser is not None:
                expected_modalities.update(self.fuser.modality_dims.keys())
            for name, worker in (
                ("text", self.text_worker),
                ("audio", self.audio_worker),
                ("video", self.video_worker),
            ):
                if worker is not None:
                    expected_modalities.add(name)
            missing_modalities = expected_modalities.difference(modality_arrays.keys())
            for name in sorted(missing_modalities):
                logger.warning("%s modality absent for message %s", name, message_id)
                MISSING_MODALITY_TOTAL.labels(modality=name).inc()

            if not modality_arrays:
                raise ValueError("No modalities available for publication")

            provenance.setdefault("modalities", list(modality_arrays.keys()))

            hop = (
                cfg.grid_hop_size
                if cfg.grid_hop_size is not None
                else min(np.min(t[:, 1] - t[:, 0]) for t in modality_times.values())
            )
            start = min(t[0, 0] for t in modality_times.values())
            end = max(t[-1, 1] for t in modality_times.values())
            num_spans = int(np.ceil((end - start) / hop))
            grid_starts = start + np.arange(num_spans) * hop
            grid_ends = grid_starts + hop
            spans = [[int(gs * 1000), int(ge * 1000)] for gs, ge in zip(grid_starts, grid_ends)]

            if self.fuser is not None:
                expected_order = list(self.fuser.modality_dims.keys())
            else:
                expected_order = list(modality_arrays.keys())

            aligned_modalities: Dict[str, torch.Tensor] = {}
            modality_payload: Dict[str, Dict[str, Any]] = {}
            for name in expected_order:
                if name in modality_arrays:
                    arr = modality_arrays[name]
                    times = modality_times[name]
                    dim = arr.shape[1]
                    aligned = np.zeros((num_spans, dim), dtype=np.float32)
                    for i, (gs, ge) in enumerate(zip(grid_starts, grid_ends)):
                        mask = (gs < times[:, 1]) & (ge > times[:, 0])
                        if mask.any():
                            aligned[i] = arr[mask].mean(axis=0)
                    try:
                        aligned_modalities[name] = torch.from_numpy(aligned)
                    except RuntimeError:  # pragma: no cover - torch built without numpy
                        aligned_modalities[name] = torch.tensor(aligned.tolist(), dtype=torch.float32)
                    modality_payload[name] = {
                        "spans": spans,
                        "embeddings": aligned.tolist(),
                        "encoders": [encoder_meta[name]] * num_spans,
                    }
                elif self.fuser is not None and name in self.fuser.modality_dims:
                    dim = self.fuser.modality_dims[name]
                    aligned_modalities[name] = torch.zeros((num_spans, dim), dtype=torch.float32)

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

        duration = time.perf_counter() - start_time
        INPUTS_TOTAL.labels(service="perception_service").inc()
        INPUT_LATENCY_SECONDS.labels(service="perception_service").observe(duration)


async def run(*args: Any, **kwargs: Any) -> None:
    """Entry point for ``dtrt perception run``."""

    service: PerceptionService | None = kwargs.pop("service", None)
    if service is not None:
        await service.run(*args, **kwargs)
