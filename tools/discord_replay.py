#!/usr/bin/env python3
"""Replay chat logs through NATS and collect bot replies."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg

from deepthought.eda.events import EventSubjects
from deepthought.eda.publisher import Publisher
from deepthought.eda.subscriber import Subscriber
from deepthought.harness.record import TraceEvent, record_event
from deepthought.metrics import actions_per_second, average_latency, bleu, rouge_l


def _load_events(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    events: List[dict] = []
    if text.startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            events.extend(data)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


async def _replay(path: Path, output: Path, metrics: Path, nats_url: str) -> None:
    events = _load_events(path)
    inputs = [e["payload"] for e in events if e.get("event") == "CHAT_RAW"]
    expected = [
        e["payload"].get("final_response", "")
        for e in events
        if e.get("event") == "RESPONSE_GENERATED"
    ]

    nc = NATS()
    await nc.connect(servers=[nats_url])
    js = nc.jetstream()
    publisher = Publisher(nc, js)
    subscriber = Subscriber(nc, js)

    responses: List[str] = []
    trace: List[TraceEvent] = []
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def handle_response(msg: Msg) -> None:
        data = json.loads(msg.data.decode())
        await msg.ack()
        queue.put_nowait(data.get("final_response", ""))

    await subscriber.subscribe(
        subject=EventSubjects.RESPONSE_GENERATED,
        handler=handle_response,
        use_jetstream=True,
        durable="discord_replay_response",
    )

    try:
        for state in inputs:
            start = datetime.utcnow()
            await publisher.publish(
                EventSubjects.CHAT_RAW, state, use_jetstream=True, timeout=10.0
            )
            action = await queue.get()
            latency = (datetime.utcnow() - start).total_seconds()
            record_event(trace, state, action, 0.0, latency)
            responses.append(action)
    finally:
        await subscriber.unsubscribe_all()
        await nc.drain()

    with output.open("w", encoding="utf-8") as f:
        for evt in trace:
            json.dump(asdict(evt), f, default=str)
            f.write("\n")

    bleu_scores = [bleu(r, e) for r, e in zip(responses, expected)]
    rouge_scores = [rouge_l(r, e) for r, e in zip(responses, expected)]
    metrics_data = {
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "avg_latency": average_latency(trace),
        "actions_per_second": actions_per_second(trace),
    }
    metrics.write_text(json.dumps(metrics_data))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Recorded trace JSONL file")
    parser.add_argument(
        "--nats",
        "-n",
        default="nats://localhost:4222",
        help="NATS server URL",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("replay.jsonl"),
        help="Output file for collected replies",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        type=Path,
        default=Path("metrics.json"),
        help="Metrics output JSON",
    )
    args = parser.parse_args()

    await _replay(args.trace, args.output, args.metrics, args.nats)


if __name__ == "__main__":
    asyncio.run(main())
