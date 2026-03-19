from __future__ import annotations

"""Utilities for removing a user's perception data."""

import asyncio
import json
import os
import shutil
from pathlib import Path

from nats.aio.client import Client as NATS

from ...eda.contracts import EventEnvelope
from .config import PerceptionConfig
from .user_embeddings import UserEmbeddings


async def trigger_replay_jobs(user_id: str, nats_url: str) -> None:
    """Publish a request to purge perception events for ``user_id``."""

    nc = NATS()
    await nc.connect(servers=[nats_url])
    js = nc.jetstream()
    envelope = EventEnvelope.build(
        subject="dtr.perception.replay.delete_user",
        payload={"user_id": user_id},
        producer="perception.delete_user_data",
    )
    await js.publish(
        "dtr.perception.replay.delete_user",
        json.dumps(envelope.__dict__).encode(),
    )
    await nc.drain()


def _remove_cache_entries(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    root = Path(cache_dir)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)


def delete_user_data(user_id: str, *, nats_url: str = "nats://localhost:4222") -> None:
    """Remove cached perception data and embeddings for ``user_id``."""

    cfg = PerceptionConfig()
    _remove_cache_entries(cfg.text_cache_dir)
    _remove_cache_entries(cfg.audio_cache_dir)
    _remove_cache_entries(cfg.video_cache_dir)

    emb_path = os.getenv("DT_USER_EMBEDDINGS_PATH")
    if emb_path:
        store = UserEmbeddings(emb_path)
        store.delete(user_id)

    asyncio.run(trigger_replay_jobs(user_id, nats_url))
