"""Command-line interface for the perception service."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from nats.aio.client import Client as NATS
from nats.js.api import DeliverPolicy

from ...eda.events import EventSubjects
from ...modules import ModalityFuser
from .config import PerceptionConfig
from .listener import PerceptionServiceListener
from .publisher import PerceptionPublisher
from .service import PerceptionService
from .service import run as run_service
from .text_utils import hop_aligned_tokens, scrub_tokens
from .user_embeddings import UserEmbeddings
from .worker_audio import AudioPerceptionWorker
from .worker_text import TextPerceptionWorker
from .worker_video import VideoPerceptionWorker


def _parse_modality_dim(value: str) -> tuple[str, int]:
    """Return ``(name, dim)`` from ``value`` formatted as ``name=dim``."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("modality dims must be provided as name=dimension")
    name, dim_str = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("modality name must not be empty")
    try:
        dim = int(dim_str)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"invalid dimension for modality '{name}': {dim_str}") from exc
    if dim <= 0:
        raise argparse.ArgumentTypeError("modality dimension must be positive")
    return name, dim


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the perception service")
    defaults = PerceptionConfig()
    parser.add_argument("--nats-url", default=defaults.nats_url)
    parser.add_argument("--listen", action="store_true", help="Listen for INPUT_RECEIVED events")
    parser.add_argument("--durable", default="perception_service", help="Durable consumer name for listen mode")
    parser.add_argument("--replay", action="store_true", help="Replay existing stream messages from start")
    parser.add_argument("--message-id")
    parser.add_argument("--user-id", default="user")
    parser.add_argument(
        "--grid-hop-size",
        type=float,
        default=defaults.grid_hop_size,
        help="Override grid hop size in seconds",
    )
    parser.add_argument("--text-model", default=defaults.text_model)
    parser.add_argument("--text-hop-size", type=float, default=defaults.text_hop_size)
    parser.add_argument("--text-cache-dir", default=defaults.text_cache_dir)
    parser.add_argument("--audio-model", default=defaults.audio_model)
    parser.add_argument("--audio-hop-size", type=float, default=defaults.audio_hop_size)
    parser.add_argument("--audio-cache-dir", default=defaults.audio_cache_dir)
    parser.add_argument("--audio-path")
    parser.add_argument("--video-model", default=defaults.video_model)
    parser.add_argument("--video-hop-size", type=float, default=defaults.video_hop_size)
    parser.add_argument("--video-cache-dir", default=defaults.video_cache_dir)
    parser.add_argument("--video-path")
    parser.add_argument("--text-path")
    parser.add_argument("--tokens-json")
    parser.add_argument("--fused-dim", type=int, default=defaults.fused_dim)
    parser.add_argument("--dropout-prob", type=float, default=defaults.dropout_prob)
    parser.add_argument(
        "--modality-dim",
        dest="modality_dims",
        action="append",
        type=_parse_modality_dim,
        default=None,
        metavar="NAME=DIM",
        help="Expected embedding dimension for a modality (e.g. text=384).",
    )
    parser.add_argument(
        "--user-embeddings-path",
        default=defaults.user_embeddings_path,
        help="Location on disk for storing per-user embeddings.",
    )
    parser.add_argument("--wandb-project", default=defaults.wandb_project)
    parser.add_argument("--wandb-sweep-id", default=defaults.wandb_sweep_id)
    args = parser.parse_args()

    if not args.listen and (not args.message_id or not args.user_id):
        parser.error("--message-id and --user-id are required unless --listen is specified")

    modality_dims = dict(defaults.modality_dims)
    for name, dim in args.modality_dims or []:
        modality_dims[name] = dim

    cfg_kwargs = {
        key: getattr(args, key)
        for key in (
            "nats_url",
            "text_model",
            "text_hop_size",
            "text_cache_dir",
            "grid_hop_size",
            "audio_model",
            "audio_hop_size",
            "audio_cache_dir",
            "video_model",
            "video_hop_size",
            "video_cache_dir",
            "fused_dim",
            "dropout_prob",
            "user_embeddings_path",
            "wandb_project",
            "wandb_sweep_id",
        )
    }
    cfg_kwargs["modality_dims"] = modality_dims
    cfg = PerceptionConfig(**cfg_kwargs)

    if cfg.wandb_project:
        os.environ["DT_WANDB_PROJECT"] = cfg.wandb_project
    if cfg.wandb_sweep_id:
        os.environ["DT_WANDB_SWEEP_ID"] = cfg.wandb_sweep_id

    nc = NATS()
    await nc.connect(cfg.nats_url)
    js = nc.jetstream()

    publisher = PerceptionPublisher(nc, js)

    text_worker = None
    text_tokens = None
    if args.text_path or args.tokens_json:
        text_worker = TextPerceptionWorker(
            model_name=cfg.text_model,
            hop_seconds=cfg.text_hop_size,
        )
        if args.tokens_json:
            with open(args.tokens_json, "r", encoding="utf8") as f:
                raw_tokens = json.load(f)
            text_tokens = scrub_tokens(raw_tokens)
        elif args.text_path:
            with open(args.text_path, "r", encoding="utf8") as f:
                text_content = f.read()
            text_tokens = hop_aligned_tokens(text_content, cfg.text_hop_size)

    audio_worker = None
    if args.audio_path:
        audio_worker = AudioPerceptionWorker(
            window_size=cfg.audio_window_size,
            step_size=cfg.audio_hop_size,
            model=cfg.audio_model,
            model_path=cfg.audio_model_path,
            cache_dir=cfg.audio_cache_dir,
        )

    video_worker = None
    if args.video_path:
        fps = max(1, min(3, int(round(1 / cfg.video_hop_size))))
        video_worker = VideoPerceptionWorker(
            decode_fps=fps,
            model_type=cfg.video_model_key,
            model_revision=cfg.video_model_revision,
            grid_fps=fps,
            cache_dir=cfg.video_cache_dir,
        )

    fuser = None
    user_embeddings = None
    active_workers = {
        "text": text_worker,
        "audio": audio_worker,
        "video": video_worker,
    }
    active_modalities = [name for name, worker in active_workers.items() if worker is not None]
    if len(active_modalities) > 1:
        available_dims = dict(cfg.modality_dims)
        missing_dims = [name for name in active_modalities if name not in available_dims]
        if missing_dims:
            parser.error(
                "Missing modality dimensions for: "
                + ", ".join(sorted(missing_dims))
                + ". Provide values via --modality-dim or configuration."
            )
        if cfg.fused_dim is None:
            parser.error("--fused-dim must be specified when fusing multiple modalities")

        modality_config = {name: available_dims[name] for name in active_modalities}
        user_dim = 0
        embeddings_path = cfg.user_embeddings_path
        if embeddings_path:
            resolved = Path(os.path.expanduser(embeddings_path)).resolve()
            user_embeddings = UserEmbeddings(resolved)
            user_dim = cfg.fused_dim

        fuser = ModalityFuser(
            modality_config,
            fused_dim=cfg.fused_dim,
            dropout_prob=cfg.dropout_prob,
            user_dim=user_dim,
        )
        fuser.eval()

    service = PerceptionService(
        publisher,
        text_worker=text_worker,
        audio_worker=audio_worker,
        video_worker=video_worker,
        fuser=fuser,
        user_embeddings=user_embeddings,
    )

    if args.listen:
        listener = PerceptionServiceListener(service, nc, js, default_user_id=args.user_id)
        deliver_policy = DeliverPolicy.ALL if args.replay else DeliverPolicy.NEW
        await js.subscribe(
            EventSubjects.INPUT_RECEIVED,
            durable=args.durable,
            deliver_policy=deliver_policy,
            cb=listener._handle,
            manual_ack=True,
        )
        try:
            await asyncio.Future()
        finally:
            await nc.drain()
        return

    await run_service(
        message_id=args.message_id,
        user_id=args.user_id,
        text_tokens=text_tokens,
        audio_path=args.audio_path,
        video_path=args.video_path,
        service=service,
    )
    await nc.drain()


def main() -> None:
    """Synchronous entry point for ``python -m`` execution."""

    asyncio.run(_main())


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
