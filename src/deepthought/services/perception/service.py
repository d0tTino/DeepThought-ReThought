"""Minimal perception service leveraging a NATS publisher stub."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
from .text_utils import Token
from .worker_text import TextPerceptionWorker
from .worker_video import VideoPerceptionWorker


logger = logging.getLogger(__name__)


def _split_model_identifier(identifier: str | None) -> tuple[str | None, str | None]:
    """Return ``(model, revision)`` for ``identifier`` split on ``@``."""

    if not identifier:
        return None, None
    if "@" not in identifier:
        return identifier, None
    model, revision = identifier.rsplit("@", 1)
    return model, revision


def _clean_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ``None`` values and stringify paths for metadata parameters."""

    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, Path):
            cleaned[key] = str(value)
        elif isinstance(value, (np.floating, np.integer)):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


@dataclass
class PerceptionService:
    """Orchestrate worker outputs and publish fused embeddings.

    When multiple modalities are processed a :class:`~deepthought.modules.fuser.ModalityFuser`
    must be provided. Configuring :class:`~.user_embeddings.UserEmbeddings` enables persistence of the
    fused representation for the associated ``user_id`` after each run.
    """

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
        audio_opt_in: bool | None = None,
        video_opt_in: bool | None = None,
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

        store_embedding: torch.Tensor | None = None
        provenance = dict(provenance or {})

        if embeddings is None:
            modality_arrays: Dict[str, np.ndarray] = {}
            modality_times: Dict[str, np.ndarray] = {}
            encoder_meta: Dict[str, Dict[str, Any]] = {}
            cfg = PerceptionConfig()

            if self.text_worker is not None and text_tokens:
                feats = times = None
                meta: Dict[str, Any] | None = None
                cache_dir = Path(cfg.text_cache_dir) if cfg.text_cache_dir else None
                meta_file: Path | None = None
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
                    else:
                        start = time.perf_counter()
                        feats, times = self.text_worker(text_tokens, str(feats_file))
                        MODALITY_INFERENCE_LATENCY_SECONDS.labels(
                            service="perception_service", modality="text"
                        ).observe(time.perf_counter() - start)
                        meta = {
                            "shape": list(feats.shape),
                            "timestamps": times.tolist(),
                            "created": time.time(),
                        }
                        meta_file.write_text(json.dumps(meta))
                else:
                    with NamedTemporaryFile(suffix=".mm", delete=False) as tmp:
                        start = time.perf_counter()
                        feats, times = self.text_worker(text_tokens, tmp.name)
                        MODALITY_INFERENCE_LATENCY_SECONDS.labels(
                            service="perception_service", modality="text"
                        ).observe(time.perf_counter() - start)
                    Path(tmp.name).unlink(missing_ok=True)
                if feats is None or times is None:
                    raise RuntimeError("Failed to compute text embeddings")
                text_array = np.asarray(feats)
                text_times = np.asarray(times)
                text_model, text_revision = _split_model_identifier(cfg.text_model)
                text_parameters = _clean_parameters(
                    {
                        "model": text_model or getattr(self.text_worker, "model_name", None),
                        "revision": text_revision,
                        "config_source": "PerceptionConfig.text_model",
                        "config_value": cfg.text_model,
                        "hop_size": float(cfg.text_hop_size),
                        "cache_enabled": cache_dir is not None,
                        "cache_path": cache_dir,
                    }
                )
                text_dim = int(text_array.shape[-1]) if text_array.ndim > 1 else int(text_array.shape[0])
                text_metadata = {
                    "name": self.text_worker.__class__.__name__,
                    "modality": "text",
                    "dim": text_dim,
                    "parameters": text_parameters,
                }
                if meta is not None and meta_file is not None:
                    if meta.get("encoder") != text_metadata:
                        meta["encoder"] = text_metadata
                        meta_file.write_text(json.dumps(meta))
                modality_arrays["text"] = text_array
                modality_times["text"] = text_times
                encoder_meta["text"] = text_metadata

            if self.audio_worker is not None and audio_path is not None:
                if audio_opt_in is not True:
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
                else:
                    start = time.perf_counter()
                    feats, times = self.audio_worker(audio_path, cache_dir=cache_dir)
                    MODALITY_INFERENCE_LATENCY_SECONDS.labels(service="perception_service", modality="audio").observe(
                        time.perf_counter() - start
                    )
                    meta = {
                        "shape": list(feats.shape),
                        "timestamps": times.tolist(),
                        "created": time.time(),
                    }
                    meta_file.write_text(json.dumps(meta))
                audio_array = np.asarray(feats)
                audio_times = np.asarray(times)
                audio_model, audio_revision = _split_model_identifier(cfg.audio_model)
                audio_parameters = _clean_parameters(
                    {
                        "model": audio_model or getattr(self.audio_worker, "model", None),
                        "revision": audio_revision,
                        "config_source": "PerceptionConfig.audio_model",
                        "config_value": cfg.audio_model,
                        "window_size": float(getattr(self.audio_worker, "window_size", cfg.audio_window_size)),
                        "hop_size": float(getattr(self.audio_worker, "step_size", cfg.audio_hop_size)),
                        "cache_enabled": bool(getattr(self.audio_worker, "cache_dir", None) or cfg.audio_cache_dir),
                        "cache_path": cache_dir,
                        "model_path": getattr(self.audio_worker, "model_path", None) or cfg.audio_model_path,
                    }
                )
                audio_dim = int(audio_array.shape[-1]) if audio_array.ndim > 1 else int(audio_array.shape[0])
                audio_metadata = {
                    "name": self.audio_worker.__class__.__name__,
                    "modality": "audio",
                    "dim": audio_dim,
                    "parameters": audio_parameters,
                }
                if meta.get("encoder") != audio_metadata:
                    meta["encoder"] = audio_metadata
                    meta_file.write_text(json.dumps(meta))
                modality_arrays["audio"] = audio_array
                modality_times["audio"] = audio_times
                encoder_meta["audio"] = audio_metadata
                if not retain_media:
                    audio_path.unlink(missing_ok=True)

            if self.video_worker is not None and video_path is not None:
                if video_opt_in is not True:
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
                        "created": time.time(),
                    }
                    meta_file.write_text(json.dumps(meta))
                times = np.asarray(meta["timestamps"], dtype=np.float32)
                if times.ndim == 1:
                    if len(times) > 1:
                        step = float(np.min(np.diff(times)))
                    else:
                        fps = grid_fps or decode_fps
                        step = 1.0 / float(fps)
                    times = np.column_stack((times, times + step))
                video_array = np.asarray(feats)
                modality_times["video"] = times
                video_model, video_revision = _split_model_identifier(cfg.video_model)
                video_parameters = _clean_parameters(
                    {
                        "model": video_model or getattr(self.video_worker, "model_type", None),
                        "revision": video_revision,
                        "config_source": "PerceptionConfig.video_model",
                        "config_value": cfg.video_model,
                        "hop_size": float(cfg.video_hop_size),
                        "cache_enabled": bool(getattr(self.video_worker, "cache_dir", None) or cfg.video_cache_dir),
                        "cache_path": cache_dir,
                        "decode_fps": decode_fps,
                        "grid_fps": grid_fps,
                        "model_variant": getattr(self.video_worker, "model_type", None),
                    }
                )
                video_dim = int(video_array.shape[-1]) if video_array.ndim > 1 else int(video_array.shape[0])
                video_metadata = {
                    "name": self.video_worker.__class__.__name__,
                    "modality": "video",
                    "dim": video_dim,
                    "parameters": video_parameters,
                }
                if meta.get("encoder") != video_metadata:
                    meta["encoder"] = video_metadata
                    meta_file.write_text(json.dumps(meta))
                modality_arrays["video"] = video_array
                encoder_meta["video"] = video_metadata
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
                logger.warning("No modalities available for message %s; skipping", message_id)
                return

            if len(modality_arrays) > 1 and self.fuser is None:
                raise ValueError("Multiple modalities available but no fuser configured")

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
                    dim = int(arr.shape[-1]) if arr.ndim > 1 else int(arr.shape[0])
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
                embedding_store = None
                existing_user_embedding = None
                if self.user_embeddings is not None and getattr(self.fuser, "user_dim", 0) > 0:
                    embedding_store = self.user_embeddings
                    stored_embedding = embedding_store.get(user_id)
                    if stored_embedding is not None:
                        if stored_embedding.shape[-1] == self.fuser.user_dim:
                            expanded = stored_embedding.detach().clone()
                            if expanded.ndim == 1:
                                expanded = expanded.unsqueeze(0)
                            if expanded.ndim == 2:
                                span_count = len(spans)
                                if span_count == 0:
                                    logger.warning("No spans generated; skipping stored embedding for user %s", user_id)
                                else:
                                    if expanded.size(0) == 1 and span_count > 1:
                                        expanded = expanded.expand(span_count, -1)
                                    if expanded.size(0) == span_count:
                                        existing_user_embedding = expanded
                                    else:
                                        logger.warning(
                                            "Stored embedding for user %s has %s spans but %s expected; ignoring",
                                            user_id,
                                            expanded.size(0),
                                            span_count,
                                        )
                            else:
                                logger.warning(
                                    "Stored embedding for user %s has %s dimensions; expected 1 or 2", user_id, expanded.ndim
                                )
                        else:
                            logger.warning(
                                "Stored embedding for user %s has dimension %s but expected %s; ignoring",
                                user_id,
                                stored_embedding.shape[-1],
                                self.fuser.user_dim,
                            )
                fused_tensor = self.fuser(
                    aligned_modalities,
                    user_embedding=existing_user_embedding,
                    user_id=user_id,
                    embedding_store=embedding_store,
                )
                fused_list = fused_tensor.tolist()
                if fused_tensor.numel() > 0:
                    if fused_tensor.ndim > 1:
                        store_embedding = fused_tensor.mean(dim=0).detach()
                    else:
                        store_embedding = fused_tensor.detach()
            else:
                first = next(iter(aligned_modalities.values()))
                fused_list = first.tolist()
                if first.numel() > 0:
                    if first.ndim > 1:
                        store_embedding = first.mean(dim=0).detach()
                    else:
                        store_embedding = first.detach()

        else:
            modality_payload = {}
            fused_list = embeddings  # type: ignore[assignment]
            if self.user_embeddings is not None:
                fused_tensor = torch.tensor(embeddings, dtype=torch.float32)
                if fused_tensor.numel() > 0:
                    if fused_tensor.ndim > 1:
                        store_embedding = fused_tensor.mean(dim=0)
                    else:
                        store_embedding = fused_tensor

        if self.user_embeddings is not None and store_embedding is not None and store_embedding.numel() > 0:
            self.user_embeddings.set(user_id, store_embedding)

        if "timestamp" not in provenance:
            provenance["timestamp"] = datetime.now(timezone.utc).isoformat()

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
