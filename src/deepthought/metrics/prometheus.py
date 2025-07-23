from prometheus_client import REGISTRY, Counter, Histogram

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

__all__ = [
    "INPUTS_TOTAL",
    "INPUT_LATENCY_SECONDS",
    "RULE_EVALUATIONS_TOTAL",
    "RULE_EVALUATION_ERRORS_TOTAL",
]
