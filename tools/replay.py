#!/usr/bin/env python3
"""Replay a recorded trace and compute text similarity metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from nats.aio.client import Client as NATS

from deepthought.eda.events import EventSubjects
from deepthought.eda.publisher import Publisher
from deepthought.metrics import bleu, rouge_l


def _load_jsonl(path: Path) -> List[dict]:
    events: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _extract_responses(events: Iterable[dict]) -> List[str]:
    responses = []
    for rec in events:
        if rec.get("event") == "RESPONSE_GENERATED":
            payload = rec.get("payload", {})
            responses.append(str(payload.get("final_response", "")))
    return responses


def _compute_metrics(golden: Iterable[dict], trial: Iterable[dict]) -> dict[str, float]:
    g_responses = _extract_responses(golden)
    t_responses = _extract_responses(trial)
    bleu_scores = [bleu(t, g) for t, g in zip(t_responses, g_responses)]
    rouge_scores = [rouge_l(t, g) for t, g in zip(t_responses, g_responses)]
    return {
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
    }


def _map_subject(event: str) -> str | None:
    if event == "INPUT_RECEIVED":
        return EventSubjects.INPUT_RECEIVED
    if event == "RESPONSE_GENERATED":
        return EventSubjects.RESPONSE_GENERATED
    return None


async def _publish_events(events: Iterable[dict], publisher: Publisher) -> None:
    prev_ts: datetime | None = None
    for rec in events:
        payload = rec.get("payload", {})
        ts_str = payload.get("timestamp")
        ts = datetime.fromisoformat(ts_str) if ts_str else None
        if prev_ts and ts:
            delta = (ts - prev_ts).total_seconds()
            if delta > 0:
                await asyncio.sleep(delta)
        subj = _map_subject(rec.get("event", ""))
        if subj:
            await publisher.publish(subj, payload)
        if ts:
            prev_ts = ts


async def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="Path to trial trace JSONL")
    parser.add_argument("golden", type=Path, help="Path to golden trace JSONL")
    parser.add_argument("--nats", default="", help="NATS server URL; empty to disable publishing")
    args = parser.parse_args(argv)

    trial_events = _load_jsonl(args.trial)
    golden_events = _load_jsonl(args.golden)
    metrics = _compute_metrics(golden_events, trial_events)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if args.nats:
        nc = NATS()
        await nc.connect(servers=[args.nats])
        js = nc.jetstream()
        publisher = Publisher(nc, js)
        await _publish_events(trial_events, publisher)
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())
