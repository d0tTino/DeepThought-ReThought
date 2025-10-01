#!/usr/bin/env python3
"""Replay stored perception events and republish updated embeddings."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from typing import Any, Dict

from nats.aio.client import Client as NATS
from nats.errors import TimeoutError
from nats.js.api import ConsumerConfig, DeliverPolicy
from nats.js.client import JetStreamContext

from deepthought.eda.events import EventSubjects, PerceptionEmbeddingsEvent
from deepthought.services.perception.publisher import PerceptionPublisher
from deepthought.services.perception.service import PerceptionService


async def _replay(
    *,
    user_id: str | None = None,
    start: float | None = None,
    end: float | None = None,
    nats_url: str = "nats://localhost:4222",
) -> None:
    nc = NATS()
    await nc.connect(servers=[nats_url])
    js: JetStreamContext = nc.jetstream()

    publisher = PerceptionPublisher(nc, js)
    service = PerceptionService(publisher)

    sub = await js.pull_subscribe(
        EventSubjects.PERCEPTION_EMBEDDINGS,
        durable="replay-perception",
        stream="PERCEPTION",
        config=ConsumerConfig(deliver_policy=DeliverPolicy.ALL),
    )

    try:
        while True:
            try:
                msgs = await sub.fetch(10, timeout=1)
            except TimeoutError:
                break
            for msg in msgs:
                event = PerceptionEmbeddingsEvent.from_json(msg.data.decode())
                payload = event.payload
                if payload is None:
                    await msg.ack()
                    continue
                if user_id is not None and payload.user_id != user_id:
                    await msg.ack()
                    continue
                ts = event.provenance.get("timestamp")
                if ts is not None:
                    try:
                        ts = float(ts)
                    except (TypeError, ValueError):
                        ts = None
                if start is not None and (ts is None or ts < start):
                    await msg.ack()
                    continue
                if end is not None and (ts is None or ts > end):
                    await msg.ack()
                    continue
                encoders: list[Dict[str, Any]] = [
                    {"name": enc.name, "modality": enc.modality} for enc in event.encoders
                ]
                await service.run(
                    message_id=payload.message_id,
                    user_id=payload.user_id,
                    embeddings=payload.fused,
                    spans=payload.spans,
                    modality_mask=payload.modality_mask,
                    encoders=encoders,
                    provenance=event.provenance,
                )
                await msg.ack()
    finally:
        await nc.drain()


def _parse_time(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).timestamp()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", help="Only replay events for this user")
    parser.add_argument("--start", help="Earliest timestamp (unix or ISO)")
    parser.add_argument("--end", help="Latest timestamp (unix or ISO)")
    parser.add_argument("--nats-url", default="nats://localhost:4222")
    args = parser.parse_args()

    start = _parse_time(args.start)
    end = _parse_time(args.end)

    asyncio.run(_replay(user_id=args.user_id, start=start, end=end, nats_url=args.nats_url))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
