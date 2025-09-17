"""Prometheus metric helpers with graceful degradation when unavailable."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency may be missing in tests
    from prometheus_client import REGISTRY, Counter, Histogram
except ImportError:  # pragma: no cover - allow running without prometheus_client
    class _NoopCollector:
        """Lightweight stand-in matching the parts of the Prometheus API we use."""

        def __init__(self, name: str, *args, **kwargs) -> None:  # noqa: D401 - simple proxy
            self._name = name
            REGISTRY._names_to_collectors[name] = self  # type: ignore[attr-defined]

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs) -> None:
            return None

        def observe(self, *args, **kwargs) -> None:
            return None

    class _NoopRegistry:
        def __init__(self) -> None:
            self._names_to_collectors: dict[str, _NoopCollector] = {}

    REGISTRY = _NoopRegistry()  # type: ignore[assignment]
    Counter = _NoopCollector  # type: ignore[assignment]
    Histogram = _NoopCollector  # type: ignore[assignment]

if "inputs_total" not in REGISTRY._names_to_collectors:
    INPUTS_TOTAL = Counter(
        "inputs_total",
        "Total number of input events processed",
        labelnames=["service"],
    )
else:  # pragma: no cover - already registered in another module
    INPUTS_TOTAL = REGISTRY._names_to_collectors["inputs_total"]  # type: ignore

if "input_latency_seconds" not in REGISTRY._names_to_collectors:
    INPUT_LATENCY_SECONDS = Histogram(
        "input_latency_seconds",
        "Latency for processing input events",
        labelnames=["service"],
    )
else:  # pragma: no cover - already registered
    INPUT_LATENCY_SECONDS = REGISTRY._names_to_collectors["input_latency_seconds"]  # type: ignore

if "modality_inference_latency_seconds" not in REGISTRY._names_to_collectors:
    MODALITY_INFERENCE_LATENCY_SECONDS = Histogram(
        "modality_inference_latency_seconds",
        "Latency for processing individual modality inputs",
        labelnames=["service", "modality"],
    )
else:  # pragma: no cover - already registered
    MODALITY_INFERENCE_LATENCY_SECONDS = REGISTRY._names_to_collectors[
        "modality_inference_latency_seconds"
    ]  # type: ignore

if "rule_evaluations_total" not in REGISTRY._names_to_collectors:
    RULE_EVALUATIONS_TOTAL = Counter(
        "rule_evaluations_total",
        "Total number of rule evaluations",
        labelnames=["rule"],
    )
else:  # pragma: no cover - already registered
    RULE_EVALUATIONS_TOTAL = REGISTRY._names_to_collectors["rule_evaluations_total"]  # type: ignore

if "rule_evaluation_errors_total" not in REGISTRY._names_to_collectors:
    RULE_EVALUATION_ERRORS_TOTAL = Counter(
        "rule_evaluation_errors_total",
        "Total number of rule evaluation errors",
        labelnames=["rule"],
    )
else:  # pragma: no cover - already registered
    RULE_EVALUATION_ERRORS_TOTAL = REGISTRY._names_to_collectors["rule_evaluation_errors_total"]  # type: ignore

if "missing_modality_total" not in REGISTRY._names_to_collectors:
    MISSING_MODALITY_TOTAL = Counter(
        "missing_modality_total",
        "Total number of times a modality was absent",
        labelnames=["modality"],
    )
else:  # pragma: no cover - already registered
    MISSING_MODALITY_TOTAL = REGISTRY._names_to_collectors["missing_modality_total"]  # type: ignore

__all__ = [
    "INPUTS_TOTAL",
    "INPUT_LATENCY_SECONDS",
    "MODALITY_INFERENCE_LATENCY_SECONDS",
    "RULE_EVALUATIONS_TOTAL",
    "RULE_EVALUATION_ERRORS_TOTAL",
    "MISSING_MODALITY_TOTAL",
]
