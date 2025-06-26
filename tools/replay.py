#!/usr/bin/env python3
"""Compare bot output trace against a golden reference."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from deepthought.harness.record import TraceEvent
from deepthought.metrics import (actions_per_second, average_latency, bleu,
                                 rouge_l)


def _load_trace(path: Path) -> List[TraceEvent]:
    events: List[TraceEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "payload" in item:
                item = item["payload"]
            events.append(
                TraceEvent(
                    state=item.get("state", ""),
                    action=item.get("action", ""),
                    reward=float(item.get("reward", 0.0)),
                    latency=float(item.get("latency", 0.0)),
                    timestamp=datetime.fromisoformat(item.get("timestamp")),
                    timestamp_delta=float(item.get("timestamp_delta", 0.0)),
                )
            )
    return events


def _compare(
    golden: Iterable[TraceEvent], trial: Iterable[TraceEvent]
) -> dict[str, float]:
    bleu_scores = []
    rouge_scores = []
    for g, t in zip(golden, trial):
        bleu_scores.append(bleu(t.action, g.action))
        rouge_scores.append(rouge_l(t.action, g.action))
    return {
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "avg_latency": average_latency(trial),
        "actions_per_second": actions_per_second(trial),
    }


class _DummyNATS:
    is_connected = True


class _DummyJS:
    pass


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="Path to trial trace JSON")
    parser.add_argument("golden", type=Path, help="Path to golden trace JSON")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="Directory to write graph.dot using HierarchicalService",
    )
    args = parser.parse_args(argv)

    trial = _load_trace(args.trial)
    golden = _load_trace(args.golden)
    metrics = _compare(golden, trial)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if args.dump_dir:
        from deepthought.graph import GraphConnector, GraphDAL
        from deepthought.services import HierarchicalService

        connector = GraphConnector(
            host=os.getenv("MG_HOST", "localhost"),
            port=int(os.getenv("MG_PORT", "7687")),
            username=os.getenv("MG_USER", ""),
            password=os.getenv("MG_PASSWORD", ""),
        )
        dal = GraphDAL(connector)
        service = HierarchicalService(_DummyNATS(), _DummyJS(), None, dal)
        service.dump_graph(str(args.dump_dir))


if __name__ == "__main__":
    main()
