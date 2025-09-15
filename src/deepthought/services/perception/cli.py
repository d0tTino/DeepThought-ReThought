"""Command-line interface for the perception service."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from nats.aio.client import Client as NATS
from nats.js.api import DeliverPolicy

from ...eda.events import EventSubjects
from .config import PerceptionConfig
from .listener import PerceptionServiceListener
from .publisher import PerceptionPublisher
from .service import PerceptionService
from .service import run as run_service
from .worker_audio import AudioPerceptionWorker
from .worker_text import TextPerceptionWorker
from .worker_video import VideoPerceptionWorker


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
    parser.add_argument("--wandb-project", default=defaults.wandb_project)
    parser.add_argument("--wandb-sweep-id", default=defaults.wandb_sweep_id)
    args = parser.parse_args()

    if not args.listen and (not args.message_id or not args.user_id):
        parser.error("--message-id and --user-id are required unless --listen is specified")

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
            "wandb_project",
            "wandb_sweep_id",
        )
    }
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
            text_tokens = [(t[0], float(t[1]), float(t[2])) for t in raw_tokens]
        elif args.text_path:
            with open(args.text_path, "r", encoding="utf8") as f:
                words = f.read().strip().split()
            hop = cfg.text_hop_size
            text_tokens = [(w, i * hop, (i + 1) * hop) for i, w in enumerate(words)]

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
            model_type=cfg.video_model,
            grid_fps=fps,
            cache_dir=cfg.video_cache_dir,
        )

    service = PerceptionService(
        publisher,
        text_worker=text_worker,
        audio_worker=audio_worker,
        video_worker=video_worker,
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
