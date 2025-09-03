#!/usr/bin/env python3
"""Benchmark perception workers and report P95 latency for each modality."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np

from deepthought.services.perception.worker_audio import AudioPerceptionWorker
from deepthought.services.perception.worker_text import TextPerceptionWorker, Token
from deepthought.services.perception.worker_video import VideoPerceptionWorker


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95))


def benchmark_text(runs: int) -> float:
    tokens: list[Token] = [("hello world", 0.0, 1.0)]
    worker = TextPerceptionWorker()
    durations: list[float] = []
    for _ in range(runs):
        with NamedTemporaryFile(suffix=".mm") as tmp:
            start = time.perf_counter()
            worker(tokens, tmp.name)
            durations.append(time.perf_counter() - start)
    return _p95(durations)


def benchmark_audio(path: Path, runs: int) -> float:
    worker = AudioPerceptionWorker()
    durations: list[float] = []
    for _ in range(runs):
        with TemporaryDirectory() as tmpdir:
            start = time.perf_counter()
            worker(path, cache_dir=tmpdir)
            durations.append(time.perf_counter() - start)
    return _p95(durations)


def benchmark_video(path: Path, runs: int) -> float:
    worker = VideoPerceptionWorker()
    durations: list[float] = []
    for _ in range(runs):
        with TemporaryDirectory() as tmpdir:
            worker.cache_dir = Path(tmpdir)
            start = time.perf_counter()
            worker(path)
            durations.append(time.perf_counter() - start)
    return _p95(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per modality")
    parser.add_argument("--audio", type=Path, required=True, help="Path to an audio file")
    parser.add_argument("--video", type=Path, required=True, help="Path to a video file")
    args = parser.parse_args()

    print(f"Text P95 latency: {benchmark_text(args.runs)*1000:.2f} ms")
    print(f"Audio P95 latency: {benchmark_audio(args.audio, args.runs)*1000:.2f} ms")
    print(f"Video P95 latency: {benchmark_video(args.video, args.runs)*1000:.2f} ms")


if __name__ == "__main__":
    main()

