from prometheus_client import Counter, Histogram

INPUTS_TOTAL = Counter(
    "inputs_total",
    "Total number of input events processed",
    labelnames=["service"],
)

INPUT_LATENCY_SECONDS = Histogram(
    "input_latency_seconds",
    "Latency for processing input events",
    labelnames=["service"],
)

__all__ = ["INPUTS_TOTAL", "INPUT_LATENCY_SECONDS"]
