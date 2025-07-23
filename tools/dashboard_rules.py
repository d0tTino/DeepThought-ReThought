#!/usr/bin/env python3
"""Plot rule evaluation metrics from Prometheus snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set

import matplotlib.pyplot as plt
from prometheus_client.parser import text_string_to_metric_families


def _parse_metrics(path: Path) -> Dict[str, float]:
    """Return rule counts from a Prometheus metrics file."""
    text = path.read_text(encoding="utf-8")
    counters: Dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        if family.name == "rule_evaluations_total":
            for sample in family.samples:
                rule = sample.labels.get("rule", "unknown")
                counters[rule] = sample.value
    return counters


def _collect_metrics(paths: Iterable[Path]) -> tuple[List[str], List[Dict[str, float]]]:
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.prom")))
            files.extend(sorted(p.glob("*.txt")))
        else:
            files.append(p)
    metrics_list: List[Dict[str, float]] = []
    rules: Set[str] = set()
    for f in files:
        try:
            metrics = _parse_metrics(f)
            metrics_list.append(metrics)
            rules.update(metrics.keys())
        except Exception as e:  # pragma: no cover - defensive
            print(f"Failed to parse {f}: {e}")
    return sorted(rules), metrics_list


def _plot(
    rules: List[str], metrics_list: List[Dict[str, float]], output: Path, show: bool
) -> None:
    if not metrics_list:
        print("No metrics to plot")
        return
    steps = list(range(1, len(metrics_list) + 1))
    fig, ax = plt.subplots()
    for rule in rules:
        values = [m.get(rule, 0.0) for m in metrics_list]
        ax.plot(steps, values, marker="o", label=rule)
    ax.set_xlabel("Snapshot")
    ax.set_ylabel("Evaluations Total")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(output)
    if show:
        plt.show()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Prometheus metrics files or directories"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("rules_dashboard.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )
    args = parser.parse_args(argv)

    rules, metrics_list = _collect_metrics(args.paths)
    _plot(rules, metrics_list, args.output, args.show)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
