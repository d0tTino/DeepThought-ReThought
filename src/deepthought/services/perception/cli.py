"""Command-line interface for the perception service."""

from __future__ import annotations

import argparse
import asyncio
import os

from nats.aio.client import Client as NATS

from .config import PerceptionConfig
from .publisher import PerceptionPublisher
from .service import PerceptionService
from .service import run as run_service


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the perception service")
    defaults = PerceptionConfig()
    parser.add_argument("--nats-url", default=defaults.nats_url)
    parser.add_argument("--message-id", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--text-model", default=defaults.text_model)
    parser.add_argument("--text-hop-size", type=float, default=defaults.text_hop_size)
    parser.add_argument("--text-cache-dir", default=defaults.text_cache_dir)
    parser.add_argument("--audio-model", default=defaults.audio_model)
    parser.add_argument("--audio-hop-size", type=float, default=defaults.audio_hop_size)
    parser.add_argument("--audio-cache-dir", default=defaults.audio_cache_dir)
    parser.add_argument("--video-model", default=defaults.video_model)
    parser.add_argument("--video-hop-size", type=float, default=defaults.video_hop_size)
    parser.add_argument("--video-cache-dir", default=defaults.video_cache_dir)
    parser.add_argument("--wandb-project", default=defaults.wandb_project)
    parser.add_argument("--wandb-sweep-id", default=defaults.wandb_sweep_id)
    args = parser.parse_args()

    cfg_kwargs = {
        key: getattr(args, key)
        for key in (
            "nats_url",
            "text_model",
            "text_hop_size",
            "text_cache_dir",
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
    service = PerceptionService(publisher)
    await run_service(
        message_id=args.message_id,
        user_id=args.user_id,
        service=service,
    )
    await nc.drain()


def main() -> None:
    """Synchronous entry point for ``python -m`` execution."""

    asyncio.run(_main())


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
