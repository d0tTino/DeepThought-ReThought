#!/usr/bin/env python3
"""Plot BLEU, ROUGE-L and latency metrics from multiple runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def _parse_metrics(path: Path) -> Dict[str, float]:
    """Return metrics dict from ``path``.

    The file may be JSON or contain ``key: value`` pairs per line.
    """
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            metrics = {
                k: float(v)
                for k, v in data.items()
                if k in {"bleu", "rouge_l", "avg_latency", "latency"}
            }
        else:
            metrics = {}
    except json.JSONDecodeError:
        metrics = {}
        for line in text.splitlines():
            match = re.match(r"(\w+):\s*([0-9.]+)", line.strip())
            if match:
                metrics[match.group(1)] = float(match.group(2))
    metrics.setdefault("avg_latency", metrics.get("latency", 0.0))
    metrics.setdefault("bleu", 0.0)
    metrics.setdefault("rouge_l", 0.0)
    return metrics


def _collect_metrics(paths: Iterable[Path]) -> List[Dict[str, float]]:
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
            files.extend(sorted(p.glob("*.txt")))
        else:
            files.append(p)
    metrics_list = []
    for f in files:
        try:
            metrics_list.append(_parse_metrics(f))
        except Exception as e:  # pragma: no cover - defensive
            print(f"Failed to parse {f}: {e}")
    return metrics_list


def _plot(metrics_list: List[Dict[str, float]], output: Path, show: bool) -> None:
    if not metrics_list:
        print("No metrics to plot")
        return
    steps = list(range(1, len(metrics_list) + 1))
    bleu = [m.get("bleu", 0.0) for m in metrics_list]
    rouge = [m.get("rouge_l", 0.0) for m in metrics_list]
    latency = [m.get("avg_latency", 0.0) for m in metrics_list]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(steps, bleu, marker="o")
    axes[0].set_ylabel("BLEU")
    axes[0].grid(True)

    axes[1].plot(steps, rouge, marker="o", color="tab:orange")
    axes[1].set_ylabel("ROUGE-L")
    axes[1].grid(True)

    axes[2].plot(steps, latency, marker="o", color="tab:green")
    axes[2].set_ylabel("Avg Latency (s)")
    axes[2].set_xlabel("Run")
    axes[2].grid(True)

    fig.tight_layout()
    plt.savefig(output)
    if show:
        plt.show()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Metrics files or directories"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dashboard.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )
    args = parser.parse_args(argv)

    metrics_list = _collect_metrics(args.paths)
    _plot(metrics_list, args.output, args.show)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
