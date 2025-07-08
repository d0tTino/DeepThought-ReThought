from prometheus_client import Counter, Histogram, REGISTRY

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

__all__ = ["INPUTS_TOTAL", "INPUT_LATENCY_SECONDS"]
