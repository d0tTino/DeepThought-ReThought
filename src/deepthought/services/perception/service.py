"""Minimal perception service leveraging a NATS publisher stub."""

from __future__ import annotations

import hashlib
import json
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
from .text_utils import Token
from .worker_text import TextPerceptionWorker
from .worker_video import VideoPerceptionWorker


logger = logging.getLogger(__name__)


def _split_model_revision(value: str | None) -> tuple[str | None, str | None]:
    """Return model name and optional revision from a ``model@revision`` string."""

    if not value:
        return None, None
    if "@" in value:
        name, revision = value.split("@", 1)
        return name, revision
    return value, None


def _build_parameters(model_value: str | None, **extras: Any) -> Dict[str, Any]:
    """Construct encoder parameter metadata with normalized values."""

    params: Dict[str, Any] = {}
    model_name, revision = _split_model_revision(model_value)
    if model_name:
        params["model"] = model_name
    if revision:
        params["revision"] = revision
    for key, value in extras.items():
        if value is None:
            continue
        if isinstance(value, (np.generic,)):
            params[key] = value.item()
        else:
            params[key] = value
    return params


def _ensure_encoder_metadata(
    existing: Dict[str, Any] | None,
    *,
    name: str,
    modality: str,
    dim: int | None,
    parameters: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Merge encoder metadata with defaults ensuring required fields exist."""

    meta = dict(existing or {})
    meta.setdefault("name", name)
    meta.setdefault("modality", modality)

    if "dim" in meta and meta["dim"] is not None:
        try:
            meta["dim"] = int(meta["dim"])
        except (TypeError, ValueError):
            if dim is not None:
                meta["dim"] = int(dim)
            else:
                meta.pop("dim", None)
    elif dim is not None:
        meta["dim"] = int(dim)

    params = dict(meta.get("parameters") or {})
    if parameters:
        for key, value in parameters.items():
            if value is None or key in params:
                continue
            params[key] = value
    meta["parameters"] = params
    return meta


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
        provenance.setdefault("timestamp", time.time())

        if embeddings is None:
            modality_arrays: Dict[str, np.ndarray] = {}
            modality_times: Dict[str, np.ndarray] = {}
            encoder_meta: Dict[str, Dict[str, Any]] = {}
            cfg = PerceptionConfig()

            if self.text_worker is not None and text_tokens:
                feats = times = None
                cache_dir = Path(cfg.text_cache_dir) if cfg.text_cache_dir else None
                text_params = _build_parameters(
                    getattr(cfg, "text_model", None),
                    hop_size=getattr(cfg, "text_hop_size", None),
                )
                default_dim = None
                modality_dims = getattr(cfg, "modality_dims", {}) or {}
                if isinstance(modality_dims, dict):
                    default_dim = modality_dims.get("text")
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
                        dim = feats.shape[1] if feats.ndim > 1 else default_dim
                        encoder_info = _ensure_encoder_metadata(
                            meta.get("encoder"),
                            name=self.text_worker.__class__.__name__,
                            modality="text",
                            dim=dim if dim is not None else default_dim,
                            parameters=text_params,
                        )
                        if meta.get("encoder") != encoder_info:
                            meta["encoder"] = encoder_info
                            meta_file.write_text(json.dumps(meta))
                        encoder_meta["text"] = encoder_info
                    else:
                        start = time.perf_counter()
                        feats, times = self.text_worker(text_tokens, str(feats_file))
                        MODALITY_INFERENCE_LATENCY_SECONDS.labels(
                            service="perception_service", modality="text"
                        ).observe(time.perf_counter() - start)
                        dim = feats.shape[1] if feats.ndim > 1 else default_dim
                        encoder_info = _ensure_encoder_metadata(
                            None,
                            name=self.text_worker.__class__.__name__,
                            modality="text",
                            dim=dim if dim is not None else default_dim,
                            parameters=text_params,
                        )
                        meta = {
                            "shape": list(feats.shape),
                            "timestamps": times.tolist(),
                            "encoder": encoder_info,
                            "created": time.time(),
                        }
                        meta_file.write_text(json.dumps(meta))
                        encoder_meta["text"] = encoder_info
                else:
                    with NamedTemporaryFile(suffix=".mm", delete=False) as tmp:
                        start = time.perf_counter()
                        feats, times = self.text_worker(text_tokens, tmp.name)
                        MODALITY_INFERENCE_LATENCY_SECONDS.labels(
                            service="perception_service", modality="text"
                        ).observe(time.perf_counter() - start)
                    dim = feats.shape[1] if feats.ndim > 1 else default_dim
                    encoder_meta["text"] = _ensure_encoder_metadata(
                        None,
                        name=self.text_worker.__class__.__name__,
                        modality="text",
                        dim=dim if dim is not None else default_dim,
                        parameters=text_params,
                    )
                    Path(tmp.name).unlink(missing_ok=True)
                modality_arrays["text"] = np.asarray(feats)
                modality_times["text"] = np.asarray(times)

            if self.audio_worker is not None and audio_path is not None:
                if audio_opt_in is not True:
                    raise PermissionError("Audio consent not granted")
                audio_path = Path(audio_path)
                cache_dir = Path(self.audio_worker.cache_dir or cfg.audio_cache_dir or audio_path.parent)
                cache_dir.mkdir(parents=True, exist_ok=True)
                model_value = getattr(self.audio_worker, "model", None) or getattr(cfg, "audio_model", None)
                extras: Dict[str, Any] = {
                    "window_size": getattr(self.audio_worker, "window_size", getattr(cfg, "audio_window_size", None)),
                    "step_size": getattr(self.audio_worker, "step_size", getattr(cfg, "audio_hop_size", None)),
                }
                model_path_value = getattr(self.audio_worker, "model_path", None) or getattr(
                    cfg, "audio_model_path", None
                )
                if model_path_value is not None:
                    extras["model_path"] = str(model_path_value)
                audio_params = _build_parameters(model_value, **extras)
                default_dim = None
                if isinstance(getattr(cfg, "modality_dims", None), dict):
                    default_dim = cfg.modality_dims.get("audio")
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
                    dim = feats.shape[1] if feats.ndim > 1 else default_dim
                    encoder_info = _ensure_encoder_metadata(
                        meta.get("encoder"),
                        name=self.audio_worker.__class__.__name__,
                        modality="audio",
                        dim=dim if dim is not None else default_dim,
                        parameters=audio_params,
                    )
                    if meta.get("encoder") != encoder_info:
                        meta["encoder"] = encoder_info
                        meta_file.write_text(json.dumps(meta))
                    encoder_meta["audio"] = encoder_info
                else:
                    start = time.perf_counter()
                    feats, times = self.audio_worker(audio_path, cache_dir=cache_dir)
                    MODALITY_INFERENCE_LATENCY_SECONDS.labels(service="perception_service", modality="audio").observe(
                        time.perf_counter() - start
                    )
                    dim = feats.shape[1] if feats.ndim > 1 else default_dim
                    encoder_info = _ensure_encoder_metadata(
                        None,
                        name=self.audio_worker.__class__.__name__,
                        modality="audio",
                        dim=dim if dim is not None else default_dim,
                        parameters=audio_params,
                    )
                    meta = {
                        "shape": list(feats.shape),
                        "timestamps": times.tolist(),
                        "encoder": encoder_info,
                        "created": time.time(),
                    }
                    meta_file.write_text(json.dumps(meta))
                    encoder_meta["audio"] = encoder_info
                modality_arrays["audio"] = np.asarray(feats)
                modality_times["audio"] = np.asarray(times)
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
                model_value = getattr(cfg, "video_model", None) or model_type
                video_params = _build_parameters(
                    model_value,
                    decode_fps=int(decode_fps),
                    grid_fps=int(grid_fps) if grid_fps is not None else int(decode_fps),
                )
                default_dim = None
                if isinstance(getattr(cfg, "modality_dims", None), dict):
                    default_dim = cfg.modality_dims.get("video")
                suffix = f"{decode_fps}_{model_type}_{grid_fps or decode_fps}"
                base = f"{video_path.stem}_{suffix}"
                feats_file = cache_dir / f"{base}_feats.npy"
                meta_file = cache_dir / f"{base}_meta.json"
                if feats_file.exists() and meta_file.exists():
                    feats = np.load(feats_file, mmap_mode="r")
                    meta = json.loads(meta_file.read_text())
                    times_arr = np.asarray(meta["timestamps"], dtype=np.float32)
                    dim = feats.shape[1] if feats.ndim > 1 else default_dim
                    encoder_info = _ensure_encoder_metadata(
                        meta.get("encoder"),
                        name=self.video_worker.__class__.__name__,
                        modality="video",
                        dim=dim if dim is not None else default_dim,
                        parameters=video_params,
                    )
                    if meta.get("encoder") != encoder_info:
                        meta["encoder"] = encoder_info
                        meta_file.write_text(json.dumps(meta))
                    encoder_meta["video"] = encoder_info
                else:
                    start = time.perf_counter()
                    feats, times_arr = self.video_worker(video_path)
                    MODALITY_INFERENCE_LATENCY_SECONDS.labels(service="perception_service", modality="video").observe(
                        time.perf_counter() - start
                    )
                    if not feats_file.exists():
                        np.save(feats_file, np.asarray(feats))
                    dim = feats.shape[1] if feats.ndim > 1 else default_dim
                    encoder_info = _ensure_encoder_metadata(
                        None,
                        name=self.video_worker.__class__.__name__,
                        modality="video",
                        dim=dim if dim is not None else default_dim,
                        parameters=video_params,
                    )
                    meta = {
                        "shape": list(feats.shape),
                        "timestamps": times_arr.tolist(),
                        "encoder": encoder_info,
                        "created": time.time(),
                    }
                    meta_file.write_text(json.dumps(meta))
                    encoder_meta["video"] = encoder_info
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
