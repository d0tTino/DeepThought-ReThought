from __future__ import annotations

"""Simple metrics for DeepThought reThought."""

import math
from collections import Counter
from statistics import mean
from typing import Iterable, List, Tuple

from ..harness.record import TraceEvent


def _ngrams(tokens: List[str], n: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))  # noqa: E203


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Compute a simple BLEU score for a single pair of strings."""
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    precisions = []
    for n in range(1, max_n + 1):
        cand_grams = _ngrams(cand_tokens, n)
        ref_grams = _ngrams(ref_tokens, n)
        if not cand_grams:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, cand_grams.get(g, 0)) for g, count in ref_grams.items())
        precisions.append(overlap / sum(cand_grams.values()))
    if not precisions or any(p == 0 for p in precisions):
        return 0.0
    log_prec = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    return math.exp(log_prec) * bp


def rouge_l(candidate: str, reference: str) -> float:
    """Return ROUGE-L F1 score between ``candidate`` and ``reference``."""
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    m, n = len(ref_tokens), len(cand_tokens)
    if not m or not n:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == cand_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    recall = lcs / m
    precision = lcs / n
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def average_latency(trace: Iterable[TraceEvent]) -> float:
    """Return the average latency of events in ``trace``."""
    latencies = [e.latency for e in trace]
    return mean(latencies) if latencies else 0.0


def actions_per_second(trace: Iterable[TraceEvent]) -> float:
    """Return throughput (actions per second) for ``trace``."""
    latencies = [e.latency for e in trace]
    total = sum(latencies)
    return len(latencies) / total if total > 0 else 0.0


__all__ = ["bleu", "rouge_l", "average_latency", "actions_per_second"]
